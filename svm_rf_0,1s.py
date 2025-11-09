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
    ROOT_DATA_DIR = "datasets/A3Car_0,1s"
    DATASETS = [ROOT_DATA_DIR]
    POSSIBLE_PAV_TYPES = {"asphalt", "cobblestones"}
    N_MFCC = 13
    HOP_LENGTH = 512
    N_FFT = 2048
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_OPTUNA_TRIALS = 50
    CV_FOLDS = 5
    MODEL_DIR = "trained_models"
    
    # НОВЫЙ ПАРАМЕТР: обрабатывать каналы независимо
    TREAT_CHANNELS_SEPARATELY = True  # True = каждый канал отдельно
    
    @staticmethod
    def ensure_model_dir():
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

# ==================== ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ====================
def extract_features_optimized(file_path, n_mfcc=13, hop_length=512, n_fft=2048):
    """Извлечение признаков из ОДНОГО аудио файла"""
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        features = []
        
        # MFCC + Delta + Delta-Delta
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
                                     hop_length=hop_length, n_fft=n_fft)
        mfcc_d = librosa.feature.delta(mfcc)
        mfcc_dd = librosa.feature.delta(mfcc, order=2)
        
        for feat in [mfcc, mfcc_d, mfcc_dd]:
            features.extend(np.mean(feat, axis=1))
            features.extend(np.std(feat, axis=1))
        
        # Спектральные признаки
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        zero_crossing = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        for arr in [spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing, rms]:
            features.append(np.mean(arr))
            features.append(np.std(arr))
            features.append(np.median(arr))
            features.append(np.max(arr) - np.min(arr))
        
        features.extend(np.mean(spectral_contrast, axis=1))
        features.extend(np.std(spectral_contrast, axis=1))
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return None

# ==================== ПАРСИНГ ====================
def get_base_filename(filename):
    """Извлекает базовое имя файла"""
    if "segment_" in filename:
        idx = filename.find("segment_")
        return filename[idx:]
    return filename

def parse_pav_type_from_filename(filename):
    """Извлечение типа покрытия из имени файла"""
    base_name = get_base_filename(filename)
    parts = base_name.replace(".wav", "").split("_")
    
    for part in parts:
        if part in Config.POSSIBLE_PAV_TYPES:
            return part
    return None

def extract_channel_number(folder_name):
    """Извлекает номер канала из имени папки"""
    try:
        ch_part = folder_name.split('audio-ch')[1].split('-')[0]
        return int(ch_part)
    except (IndexError, ValueError):
        return None

# ==================== ЗАГРУЗКА ДАННЫХ (НЕЗАВИСИМЫЕ КАНАЛЫ) ====================
def load_dataset_independent_channels(root_dir, verbose=True):
    """
    Загрузка данных, где каждый канал - это ОТДЕЛЬНЫЙ независимый образец.
    
    Результат: 
    - Файл из канала 1 = образец 1
    - Файл из канала 2 = образец 2
    - И так далее для всех 8 каналов
    """
    X = []
    y = []
    metadata = []  # Для отслеживания источника каждого образца
    
    if verbose:
        print(f"Сбор данных из датасета: {root_dir}")
        print("Режим: НЕЗАВИСИМЫЕ КАНАЛЫ (каждый канал = отдельный образец)\n")
    
    if not os.path.exists(root_dir):
        print(f"Ошибка: {root_dir} не найден")
        return np.array([]), np.array([]), []
    
    # Получаем список всех папок с каналами
    all_channel_folders = []
    for item in os.listdir(root_dir):
        if 'audio-ch' in item and os.path.isdir(os.path.join(root_dir, item)):
            all_channel_folders.append(item)
    
    all_channel_folders = sorted(all_channel_folders)
    
    if verbose:
        print(f"Найдено папок с каналами: {len(all_channel_folders)}")
    
    total_files = 0
    
    # Обрабатываем каждую папку канала НЕЗАВИСИМО
    for folder_name in tqdm(all_channel_folders, desc="Обработка папок"):
        folder_path = os.path.join(root_dir, folder_name)
        channel_num = extract_channel_number(folder_name)
        
        if channel_num is None:
            continue
        
        # Получаем все wav файлы в папке
        files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        
        # Обрабатываем каждый файл КАК ОТДЕЛЬНЫЙ ОБРАЗЕЦ
        for file in files:
            label = parse_pav_type_from_filename(file)
            if label is None:
                continue
            
            full_path = os.path.join(folder_path, file)
            
            # Извлекаем признаки
            feats = extract_features_optimized(
                full_path,
                n_mfcc=Config.N_MFCC,
                hop_length=Config.HOP_LENGTH,
                n_fft=Config.N_FFT
            )
            
            if feats is not None:
                X.append(feats)  # НЕ ОБЪЕДИНЯЕМ! Просто добавляем
                y.append(label)
                metadata.append({
                    'channel': channel_num,
                    'folder': folder_name,
                    'filename': file,
                    'base_name': get_base_filename(file)
                })
                total_files += 1
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Всего образцов (файлов): {len(X)}")
        print(f"Признаков на образец: {X.shape[1] if len(X) > 0 else 0}")
        print(f"Уникальные метки: {np.unique(y)}")
        
        if len(y) > 0:
            print(f"Распределение классов: {dict(zip(*np.unique(y, return_counts=True)))}")
            
            # Статистика по каналам
            if metadata:
                channels = [m['channel'] for m in metadata]
                unique_channels = sorted(set(channels))
                print(f"\nОбразцов по каналам:")
                for ch in unique_channels:
                    count = channels.count(ch)
                    print(f"  Канал {ch}: {count} образцов")
        
        print('='*70)
    
    return X, y, metadata

# ==================== ЗАГРУЗКА ДАННЫХ (ОБЪЕДИНЕННЫЕ КАНАЛЫ) ====================
def load_dataset_combined_channels(root_dir, verbose=True):
    """
    СТАРЫЙ СПОСОБ: объединение всех 8 каналов в один вектор признаков
    """
    X = []
    y = []
    
    if verbose:
        print(f"Сбор данных из датасета: {root_dir}")
        print("Режим: ОБЪЕДИНЕННЫЕ КАНАЛЫ (8 каналов → 1 образец)\n")
    
    # ... (ваш старый код из файла)
    # Не буду дублировать, так как вам нужен независимый режим
    
    return X, y, []

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
    """Обучение и оценка модели"""
    print(f"\n{'='*70}")
    print(f"Обучение модели: {model_name}")
    print('='*70)
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\n{model_name} - TRAIN SET:")
    print(classification_report(y_train, y_pred_train, target_names=le.classes_))
    print(f"F1-weighted (train): {f1_train:.4f}")
    
    print(f"\n{model_name} - TEST SET:")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))
    print(f"F1-weighted (test): {f1_test:.4f}")
    
    overfitting_gap = f1_train - f1_test
    if overfitting_gap > 0.05:
        print(f"\n Возможное переобучение (разница: {overfitting_gap:.4f})")
    else:
        print(f"\nНет значительного переобучения (разница: {overfitting_gap:.4f})")
    
    print(f"\nMatрица ошибок (TEST):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    return model, f1_test

# ==================== СОХРАНЕНИЕ МОДЕЛЕЙ ====================
def save_model(model, model_name, le, metadata=None):
    """Сохранение модели"""
    Config.ensure_model_dir()
    
    model_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pkl")
    le_path = os.path.join(Config.MODEL_DIR, f"{model_name}_label_encoder.pkl")
    metadata_path = os.path.join(Config.MODEL_DIR, f"{model_name}_metadata.pkl")
    
    joblib.dump(model, model_path)
    print(f"✓ Модель сохранена: {model_path}")
    
    joblib.dump(le, le_path)
    print(f"✓ Label Encoder сохранен: {le_path}")
    
    if metadata:
        joblib.dump(metadata, metadata_path)
        print(f"✓ Метаданные сохранены: {metadata_path}")

def load_model(model_name):
    """Загрузка модели"""
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
    
    print("="*70)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ НА НЕЗАВИСИМЫХ КАНАЛАХ")
    print("="*70)
    
    # 1. Загрузка данных
    if Config.TREAT_CHANNELS_SEPARATELY:
        X, y, metadata = load_dataset_independent_channels(
            Config.ROOT_DATA_DIR, verbose=True
        )
    else:
        X, y, metadata = load_dataset_combined_channels(
            Config.ROOT_DATA_DIR, verbose=True
        )
    
    if len(X) == 0:
        print("\nОШИБКА: Не загружено ни одного образца!")
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
    
    # 3-6. Обучение SVM и RF
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
    
    mode_suffix = "independent" if Config.TREAT_CHANNELS_SEPARATELY else "combined"
    
    svm_metadata = {
        'best_params': study_svm.best_params,
        'cv_f1_score': study_svm.best_value,
        'test_f1_score': svm_f1,
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'classes': le.classes_.tolist(),
        'dataset': Config.ROOT_DATA_DIR,
        'channel_mode': mode_suffix
    }
    
    rf_metadata = {
        'best_params': study_rf.best_params,
        'cv_f1_score': study_rf.best_value,
        'test_f1_score': rf_f1,
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'classes': le.classes_.tolist(),
        'dataset': Config.ROOT_DATA_DIR,
        'channel_mode': mode_suffix
    }
    
    save_model(svm_model, f"svm_road_surface_1s_{mode_suffix}", le, svm_metadata)
    save_model(rf_model, f"rf_road_surface_1s_{mode_suffix}", le, rf_metadata)
    
    # 6. Сравнение моделей
    print("\n" + "="*70)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    print(f"SVM F1 (test): {svm_f1:.4f}")
    print(f"Random Forest F1 (test): {rf_f1:.4f}")
    print(f"Лучшая модель: {'SVM' if svm_f1 > rf_f1 else 'Random Forest'}")
    print(f"\nРежим: {'Независимые каналы' if Config.TREAT_CHANNELS_SEPARATELY else 'Объединенные каналы'}")

if __name__ == "__main__":
    main()
