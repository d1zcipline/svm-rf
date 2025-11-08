import os
import numpy as np
import librosa
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import optuna
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    ROOT_DATA_DIR = "datasets/A3Car_1s"
    DATASETS = [ROOT_DATA_DIR]  # Корневая папка с segments_audio-ch{i}
    POSSIBLE_PAV_TYPES = {"asphalt", "cobblestones"}
    N_MFCC = 13
    HOP_LENGTH = 512
    N_FFT = 2048
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_OPTUNA_TRIALS = 50
    CV_FOLDS = 5
    MODEL_DIR = "trained_models"
    
    @staticmethod
    def ensure_model_dir():
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

# ==================== ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ====================
def extract_features_optimized(file_path, n_mfcc=13, hop_length=512, n_fft=2048):
    """
    Оптимизированная функция извлечения признаков с кэшированием вычислений
    """
    try:
        # Загрузка с явным sr для консистентности
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        
        # Предварительная нормализация
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        features = []
        
        # MFCC + Delta + Delta-Delta (оптимизация: вычисляем один раз)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
                                     hop_length=hop_length, n_fft=n_fft)
        mfcc_d = librosa.feature.delta(mfcc)
        mfcc_dd = librosa.feature.delta(mfcc, order=2)
        
        for feat in [mfcc, mfcc_d, mfcc_dd]:
            features.extend(np.mean(feat, axis=1))
            features.extend(np.std(feat, axis=1))
        
        # Спектральные признаки (вычисляем за один проход)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                               hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, 
                                                             hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, 
                                                                 hop_length=hop_length)[0]
        zero_crossing = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Новые признаки для повышения качества
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, 
                                                               hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        for arr in [spectral_centroid, spectral_rolloff, spectral_bandwidth, 
                    zero_crossing, rms]:
            features.append(np.mean(arr))
            features.append(np.std(arr))
            features.append(np.median(arr))  # Новая статистика
            features.append(np.max(arr) - np.min(arr))  # Диапазон
        
        # Добавляем спектральный контраст и хрома
        features.extend(np.mean(spectral_contrast, axis=1))
        features.extend(np.std(spectral_contrast, axis=1))
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return None

# ==================== ПАРСИНГ ИМЕН ФАЙЛОВ ====================
def get_base_filename(filename):
    """
    Извлекает базовое имя файла без префикса канала.
    
    Пример:
    audio-ch1-20221011-1_segment_0001_town_asphalt_dry_residential_closed.wav
    -> segment_0001_town_asphalt_dry_residential_closed.wav
    """
    # Ищем первое вхождение "segment_"
    if "segment_" in filename:
        idx = filename.find("segment_")
        return filename[idx:]
    
    # Если не найдено, возвращаем оригинальное имя
    return filename


def parse_pav_type_from_filename(filename):
    """
    Извлечение типа покрытия из имени файла (работает с префиксами).
    """
    # Получаем базовое имя
    base_name = get_base_filename(filename)
    parts = base_name.replace(".wav", "").split("_")
    
    for part in parts:
        if part in Config.POSSIBLE_PAV_TYPES:
            return part
    return None

# ==================== ЗАГРУЗКА ДАННЫХ ====================
def load_dataset(root_dir, verbose=True):
    """
    Загрузка данных из новой структуры:
    root_dir/segments_audio-ch1/*.wav
    root_dir/segments_audio-ch2/*.wav
    ...
    root_dir/segments_audio-ch8/*.wav
    
    Имена файлов: audio-ch{i}-YYYYMMDD-N_segment_XXXX_info.wav
    """
    X = []
    y = []
    
    if verbose:
        print(f"Сбор данных из датасета: {root_dir}\n")
    
    # Проверяем, что директория существует
    if not os.path.exists(root_dir):
        print(f"Ошибка: {root_dir} не найден")
        return np.array([]), np.array([])
    
    # Получаем список файлов из первого канала
    ch1_dir = os.path.join(root_dir, "segments_audio-ch1")
    
    if not os.path.exists(ch1_dir):
        print(f"Ошибка: {ch1_dir} не найден")
        return np.array([]), np.array([])
    
    files_ch1 = [f for f in os.listdir(ch1_dir) if f.endswith(".wav")]
    
    if verbose:
        print(f"Найдено {len(files_ch1)} сегментов в канале 1")
    
    # Создаем словарь базовых имен файлов для быстрого поиска
    files_by_base = {}
    for file in files_ch1:
        base_name = get_base_filename(file)
        files_by_base[base_name] = file
    
    if verbose:
        print(f"Уникальных базовых имен: {len(files_by_base)}")
    
    # Обрабатываем каждый базовый сегмент
    for base_name, ch1_file in tqdm(files_by_base.items(), 
                                     desc=f" {os.path.basename(root_dir)}", 
                                     disable=not verbose):
        label = parse_pav_type_from_filename(base_name)
        if label is None:
            continue
        
        # Собираем признаки из 8 каналов
        all_feats = []
        valid = True
        
        for i in range(1, 9):
            ch_dir = os.path.join(root_dir, f"segments_audio-ch{i}")
            
            if not os.path.exists(ch_dir):
                valid = False
                break
            
            # Ищем файл с таким же базовым именем
            expected_files = [f for f in os.listdir(ch_dir) 
                            if get_base_filename(f) == base_name]
            
            if len(expected_files) == 0:
                valid = False
                break
            
            full_path = os.path.join(ch_dir, expected_files[0])
            
            feats = extract_features_optimized(
                full_path, 
                n_mfcc=Config.N_MFCC,
                hop_length=Config.HOP_LENGTH,
                n_fft=Config.N_FFT
            )
            
            if feats is None:
                valid = False
                break
            
            all_feats.append(feats)
        
        if valid:
            X.append(np.concatenate(all_feats))
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Всего сегментов: {len(X)}")
        print(f"Признаков на сегмент: {X.shape[1] if len(X) > 0 else 0}")
        print(f"Уникальные метки: {np.unique(y)}")
        if len(y) > 0:
            print(f"Распределение классов: {dict(zip(*np.unique(y, return_counts=True)))}")
        print('='*70)
    
    return X, y

# ==================== OPTUNA ДЛЯ SVM ====================
def objective_svm(trial, X_train, y_train):
    """Целевая функция для оптимизации SVM"""
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 1e0, log=True)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=C,
            gamma=gamma,
            kernel='rbf',
            class_weight='balanced',
            random_state=Config.RANDOM_STATE
        ))
    ])
    
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, 
                         random_state=Config.RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, 
                             scoring='f1_weighted', n_jobs=-1)
    
    return scores.mean()

# ==================== OPTUNA ДЛЯ RANDOM FOREST ====================
def objective_rf(trial, X_train, y_train):
    """Целевая функция для оптимизации Random Forest"""
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 10, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, 
                         random_state=Config.RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, 
                             scoring='f1_weighted', n_jobs=-1)
    
    return scores.mean()

# ==================== ОБУЧЕНИЕ И ОЦЕНКА ====================
def train_and_evaluate_model(model_name, model, X_train, X_test, 
                             y_train, y_test, le):
    """Обучение и оценка модели с детальным отчетом"""
    print(f"\n{'='*70}")
    print(f"Обучение модели: {model_name}")
    print('='*70)
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Метрики
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\n{model_name} - TRAIN SET:")
    print(classification_report(y_train, y_pred_train, target_names=le.classes_))
    print(f"F1-weighted (train): {f1_train:.4f}")
    
    print(f"\n{model_name} - TEST SET:")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))
    print(f"F1-weighted (test): {f1_test:.4f}")
    
    # Проверка переобучения
    overfitting_gap = f1_train - f1_test
    if overfitting_gap > 0.05:
        print(f"\n⚠️ Возможное переобучение (разница: {overfitting_gap:.4f})")
    else:
        print(f"\n✓ Нет значительного переобучения (разница: {overfitting_gap:.4f})")
    
    # Матрица ошибок
    print(f"\nMatрица ошибок (TEST):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    return model, f1_test

# ==================== СОХРАНЕНИЕ МОДЕЛЕЙ ====================
def save_model(model, model_name, le, metadata=None):
    """
    Сохранение обученной модели с метаданными
    """
    Config.ensure_model_dir()
    
    model_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pkl")
    le_path = os.path.join(Config.MODEL_DIR, f"{model_name}_label_encoder.pkl")
    metadata_path = os.path.join(Config.MODEL_DIR, f"{model_name}_metadata.pkl")
    
    # Сохранение модели
    joblib.dump(model, model_path)
    print(f"✓ Модель сохранена: {model_path}")
    
    # Сохранение энкодера меток
    joblib.dump(le, le_path)
    print(f"✓ Label Encoder сохранен: {le_path}")
    
    # Сохранение метаданных
    if metadata:
        joblib.dump(metadata, metadata_path)
        print(f"✓ Метаданные сохранены: {metadata_path}")

# ==================== ЗАГРУЗКА МОДЕЛЕЙ ====================
def load_model(model_name):
    """
    Загрузка обученной модели с метаданными
    """
    model_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pkl")
    le_path = os.path.join(Config.MODEL_DIR, f"{model_name}_label_encoder.pkl")
    metadata_path = os.path.join(Config.MODEL_DIR, f"{model_name}_metadata.pkl")
    
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    
    metadata = None
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
    
    print(f"✓ Модель загружена: {model_path}")
    return model, le, metadata

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
def main():
    """Основной pipeline обучения"""
    
    # 1. Загрузка данных
    X, y = load_dataset(Config.ROOT_DATA_DIR, verbose=True)
    
    if len(X) == 0:
        print("\n❌ ОШИБКА: Не загружено ни одного образца!")
        print(f"Проверьте путь: {Config.ROOT_DATA_DIR}")
        print("Ожидаемая структура:")
        print(f"  {Config.ROOT_DATA_DIR}/segments_audio-ch1/*.wav")
        print(f"  {Config.ROOT_DATA_DIR}/segments_audio-ch2/*.wav")
        print(f"  ...")
        print(f"  {Config.ROOT_DATA_DIR}/segments_audio-ch8/*.wav")
        return
    
    # 2. Подготовка данных
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=Config.TEST_SIZE, 
        stratify=y_encoded, 
        random_state=Config.RANDOM_STATE
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # 3. Оптимизация и обучение SVM
    print("\n" + "="*70)
    print("ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ SVM")
    print("="*70)
    
    study_svm = optuna.create_study(direction="maximize")
    study_svm.optimize(
        lambda trial: objective_svm(trial, X_train, y_train),
        n_trials=Config.N_OPTUNA_TRIALS,
        show_progress_bar=True
    )
    
    print(f"\n✓ Лучшие параметры SVM: {study_svm.best_params}")
    print(f"✓ Лучший F1 (CV): {study_svm.best_value:.4f}")
    
    best_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            **study_svm.best_params,
            kernel='rbf',
            class_weight='balanced',
            random_state=Config.RANDOM_STATE
        ))
    ])
    
    svm_model, svm_f1 = train_and_evaluate_model(
        "SVM", best_svm, X_train, X_test, y_train, y_test, le
    )
    
    # 4. Оптимизация и обучение Random Forest
    print("\n" + "="*70)
    print("ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ RANDOM FOREST")
    print("="*70)
    
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(
        lambda trial: objective_rf(trial, X_train, y_train),
        n_trials=Config.N_OPTUNA_TRIALS,
        show_progress_bar=True
    )
    
    print(f"\n✓ Лучшие параметры RF: {study_rf.best_params}")
    print(f"✓ Лучший F1 (CV): {study_rf.best_value:.4f}")
    
    best_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            **study_rf.best_params,
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    rf_model, rf_f1 = train_and_evaluate_model(
        "Random Forest", best_rf, X_train, X_test, y_train, y_test, le
    )
    
    # 5. Сохранение моделей
    print("\n" + "="*70)
    print("СОХРАНЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    
    svm_metadata = {
        'best_params': study_svm.best_params,
        'cv_f1_score': study_svm.best_value,
        'test_f1_score': svm_f1,
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'classes': le.classes_.tolist(),
        'dataset': Config.ROOT_DATA_DIR
    }
    
    rf_metadata = {
        'best_params': study_rf.best_params,
        'cv_f1_score': study_rf.best_value,
        'test_f1_score': rf_f1,
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'classes': le.classes_.tolist(),
        'dataset': Config.ROOT_DATA_DIR
    }
    
    save_model(svm_model, "svm_road_surface_1s", le, svm_metadata)
    save_model(rf_model, "rf_road_surface_1s", le, rf_metadata)
    
    # 6. Сравнение моделей
    print("\n" + "="*70)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    print(f"SVM F1 (test): {svm_f1:.4f}")
    print(f"Random Forest F1 (test): {rf_f1:.4f}")
    print(f"Лучшая модель: {'SVM' if svm_f1 > rf_f1 else 'Random Forest'}")

if __name__ == "__main__":
    main()
