import os
import numpy as np
import librosa
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    MODEL_DIR = "trained_models/rf_1s"
    MODEL_NAME = "rf_road_surface_1s_independent"
    
    TEST_DATA_DIR = "datasets/A3Car_4s"
    
    N_MFCC = 13
    HOP_LENGTH = 512
    N_FFT = 2048
    
    POSSIBLE_PAV_TYPES = {"asphalt", "cobblestones"}
    
    OUTPUT_CSV = "predictions_csv/predictions_4s.csv"

# ==================== ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ====================
def extract_features_optimized(file_path, n_mfcc=13, hop_length=512, n_fft=2048):
    """
    Извлечение признаков для коротких аудио (0.1s)
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        features = []
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
                                     hop_length=hop_length, n_fft=n_fft)
        
        n_frames = mfcc.shape[1]
        
        # Дельты с правильным mode для коротких аудио
        if n_frames < 9:
            # Для коротких сегментов используем width=3 и mode='nearest'
            width = 3
            mfcc_d = librosa.feature.delta(mfcc, width=width, mode='nearest')
            mfcc_dd = librosa.feature.delta(mfcc, width=width, order=2, mode='nearest')
        else:
            # Для нормальных сегментов - стандартные параметры
            mfcc_d = librosa.feature.delta(mfcc)
            mfcc_dd = librosa.feature.delta(mfcc, order=2)
        
        for feat in [mfcc, mfcc_d, mfcc_dd]:
            features.extend(np.mean(feat, axis=1))
            features.extend(np.std(feat, axis=1))
        
        # Спектральные признаки
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                               hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, 
                                                             hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, 
                                                                 hop_length=hop_length)[0]
        zero_crossing = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        for arr in [spectral_centroid, spectral_rolloff, spectral_bandwidth, 
                    zero_crossing, rms]:
            features.append(np.mean(arr))
            features.append(np.std(arr))
            features.append(np.median(arr))
            features.append(np.max(arr) - np.min(arr))
        
        # Spectral contrast и Chroma
        if n_frames >= 7:
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, 
                                                                   hop_length=hop_length)
            features.extend(np.mean(spectral_contrast, axis=1))
            features.extend(np.std(spectral_contrast, axis=1))
        else:
            # Для очень коротких - нули
            features.extend([0.0] * 14)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        # Молча пропускаем файлы с ошибками
        return None

# ==================== ПАРСИНГ ====================
def parse_label_from_path(folder_name):
    folder_lower = folder_name.lower()
    for pav_type in Config.POSSIBLE_PAV_TYPES:
        if pav_type in folder_lower:
            return pav_type
    return None

def parse_condition_from_path(folder_name):
    folder_lower = folder_name.lower()
    if 'dry' in folder_lower:
        return 'dry'
    elif 'wet' in folder_lower:
        return 'wet'
    return None

# ==================== ЗАГРУЗКА МОДЕЛИ ====================
def load_model_safe(model_name):
    import joblib
    
    model_path = os.path.join(Config.MODEL_DIR, f"{model_name}.pkl")
    le_path = os.path.join(Config.MODEL_DIR, f"{model_name}_label_encoder.pkl")
    metadata_path = os.path.join(Config.MODEL_DIR, f"{model_name}_metadata.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Модель не найдена: {model_path}")
    
    print(f"Загрузка: {model_path}")
    
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    
    metadata = None
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
    
    print(f"✓ Загружено!")
    
    if metadata:
        print(f"  • Классы: {metadata.get('classes', 'N/A')}")
        print(f"  • Признаков: {metadata.get('n_features', 'N/A')}")
        print(f"  • F1: {metadata.get('test_f1_score', 0):.4f}")
    
    return model, le, metadata

# ==================== ПРЕДСКАЗАНИЕ ====================
def predict_on_new_dataset(model, le, root_dir, verbose=True):
    """Предсказания"""
    predictions = []
    true_labels = []
    conditions = []
    filenames = []
    folder_names = []
    file_paths = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ПРЕДСКАЗАНИЕ: {root_dir}")
        print('='*70)
    
    if not os.path.exists(root_dir):
        print(f"❌ {root_dir} не найден")
        return None, None, None, None, None, None
    
    subfolders = [d for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d))]
    
    if verbose:
        print(f"\nПапок: {len(subfolders)}")
        for folder in sorted(subfolders):
            folder_path = os.path.join(root_dir, folder)
            num_files = len([f for f in os.listdir(folder_path) if f.endswith('.wav')])
            print(f"  • {folder}: {num_files} файлов")
        print()
    
    total_processed = 0
    total_skipped = 0
    
    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        
        true_label = parse_label_from_path(folder)
        condition = parse_condition_from_path(folder)
        
        if true_label is None:
            continue
        
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        for filename in tqdm(wav_files, desc=f"  {folder}", disable=not verbose):
            
            full_path = os.path.join(folder_path, filename)
            
            features = extract_features_optimized(
                full_path,
                n_mfcc=Config.N_MFCC,
                hop_length=Config.HOP_LENGTH,
                n_fft=Config.N_FFT
            )
            
            if features is None:
                total_skipped += 1
                continue
            
            try:
                X = features.reshape(1, -1)
                pred_encoded = model.predict(X)[0]
                pred_label = le.inverse_transform([pred_encoded])[0]
                
                predictions.append(pred_label)
                true_labels.append(true_label)
                conditions.append(condition)
                filenames.append(filename)
                folder_names.append(folder)
                file_paths.append(full_path)
                total_processed += 1
            except Exception as e:
                total_skipped += 1
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Обработано: {total_processed}, Пропущено: {total_skipped}")
    
    return predictions, true_labels, conditions, filenames, folder_names, file_paths

# ==================== ОЦЕНКА ====================
def evaluate_predictions(predictions, true_labels, conditions=None):
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ")
    print("="*70)
    
    acc = accuracy_score(true_labels, predictions)
    print(f"\n✓ Точность: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nОтчет:")
    print(classification_report(true_labels, predictions))
    
    print("\nМатрица ошибок:")
    print(confusion_matrix(true_labels, predictions))
    
    if conditions:
        print("\n" + "="*70)
        print("ПО УСЛОВИЯМ")
        print("="*70)
        
        for condition in sorted(set(conditions)):
            if not condition:
                continue
            
            indices = [i for i, c in enumerate(conditions) if c == condition]
            if not indices:
                continue
            
            cond_true = [true_labels[i] for i in indices]
            cond_pred = [predictions[i] for i in indices]
            
            cond_acc = accuracy_score(cond_true, cond_pred)
            print(f"\n{condition.upper()}: {cond_acc:.4f} ({cond_acc*100:.2f}%) - {len(indices)} образцов")

# ==================== СОХРАНЕНИЕ ====================
def save_results(predictions, true_labels, conditions, filenames, 
                folder_names, file_paths, output_path):
    df = pd.DataFrame({
        'filename': filenames,
        'folder': folder_names,
        'condition': conditions,
        'file_path': file_paths,
        'predicted_label': predictions,
        'true_label': true_labels,
        'correct': [pred == true for pred, true in zip(predictions, true_labels)]
    })
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ Сохранено: {output_path} ({len(df)} записей)")
    
    print(f"\nТочность по папкам:")
    for folder in sorted(df['folder'].unique()):
        folder_df = df[df['folder'] == folder]
        folder_acc = folder_df['correct'].mean()
        print(f"  • {folder}: {folder_acc:.4f} ({folder_acc*100:.1f}%)")
    
    return df

# ==================== MAIN ====================
def main():
    print("="*70)
    print("ПРЕДСКАЗАНИЕ НА 0.1s ДАТАСЕТЕ")
    print("="*70)
    
    print("\n[1/3] Загрузка модели...")
    try:
        model, le, metadata = load_model_safe(Config.MODEL_NAME)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    print("\n[2/3] Предсказания...")
    results = predict_on_new_dataset(model, le, Config.TEST_DATA_DIR, verbose=True)
    
    predictions, true_labels, conditions, filenames, folder_names, file_paths = results
    
    if not predictions:
        print("❌ Нет результатов")
        return
    
    print("\n[3/3] Оценка...")
    evaluate_predictions(predictions, true_labels, conditions)
    
    print("\n" + "="*70)
    print("СОХРАНЕНИЕ")
    print("="*70)
    
    save_results(predictions, true_labels, conditions, filenames, 
                folder_names, file_paths, Config.OUTPUT_CSV)
    
    print("\n✓ ГОТОВО!")

if __name__ == "__main__":
    main()
