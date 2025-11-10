import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    MODEL_DIR = "trained_models/rf_0,1s_kaggle"
    MODEL_NAME = "kaggle_rf_road_surface_0,1s_independent" 
    
    # Путь к датасету 1s
    TEST_DATA_DIR = "datasets/A3Car_1s"
    
    N_MFCC = 13
    HOP_LENGTH = 512
    N_FFT = 2048
    
    POSSIBLE_PAV_TYPES = {"asphalt", "cobblestones"}
    
    OUTPUT_CSV = "predicitons_result/predictions_1s_pretrained_0,1s_model.csv"

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
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                               hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, 
                                                             hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, 
                                                                 hop_length=hop_length)[0]
        zero_crossing = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, 
                                                               hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        for arr in [spectral_centroid, spectral_rolloff, spectral_bandwidth, 
                    zero_crossing, rms]:
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
        return None

# ==================== ПАРСИНГ ИМЕНИ ФАЙЛА ====================
def parse_filename_info(filename):
    """
    Парсит информацию из имени файла.
    
    Пример: audio-ch8-20221011-1_segment_0001_town_asphalt_dry_residential_closed.wav
    
    Возвращает словарь с полями:
    - channel: номер канала (8)
    - date: дата записи (20221011)
    - session: номер сессии (1)
    - segment: номер сегмента (0001)
    - location: локация (town)
    - pavement: тип покрытия (asphalt)
    - condition: состояние (dry)
    - area_type: тип области (residential)
    - window_state: состояние окон (closed)
    """
    info = {
        'channel': None,
        'date': None,
        'session': None,
        'segment': None,
        'location': None,
        'pavement': None,
        'condition': None,
        'area_type': None,
        'window_state': None
    }
    
    # Убираем расширение
    name = filename.replace('.wav', '')
    
    # Парсим префикс (audio-ch8-20221011-1)
    if name.startswith('audio-ch'):
        prefix_parts = name.split('_')[0]  # audio-ch8-20221011-1
        prefix_split = prefix_parts.split('-')
        
        if len(prefix_split) >= 4:
            info['channel'] = int(prefix_split[1].replace('ch', ''))
            info['date'] = prefix_split[2]
            info['session'] = int(prefix_split[3])
    
    # Парсим остальное
    parts = name.split('_')
    
    for i, part in enumerate(parts):
        if part == 'segment' and i + 1 < len(parts):
            info['segment'] = int(parts[i + 1])
        
        elif part in ['town', 'highway', 'country']:
            info['location'] = part
        
        elif part in Config.POSSIBLE_PAV_TYPES:
            info['pavement'] = part
        
        elif part in ['dry', 'wet']:
            info['condition'] = part
        
        elif part in ['residential', 'commercial', 'industrial']:
            info['area_type'] = part
        
        elif part in ['closed', 'open', 'half']:
            info['window_state'] = part
    
    return info

# ==================== ЗАГРУЗКА МОДЕЛИ ====================
def load_model_safe(model_name):
    """Загрузка модели"""
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
    """Предсказания на датасете 1s с каналами"""
    predictions = []
    true_labels = []
    metadata_list = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ПРЕДСКАЗАНИЕ: {root_dir}")
        print('='*70)
    
    if not os.path.exists(root_dir):
        print(f"❌ {root_dir} не найден")
        return None, None, None
    
    # Находим все папки с каналами
    channel_folders = []
    for item in os.listdir(root_dir):
        if 'segments_audio-ch' in item and os.path.isdir(os.path.join(root_dir, item)):
            channel_folders.append(item)
    
    channel_folders = sorted(channel_folders)
    
    if verbose:
        print(f"\nНайдено папок с каналами: {len(channel_folders)}")
        for folder in channel_folders:
            folder_path = os.path.join(root_dir, folder)
            num_files = len([f for f in os.listdir(folder_path) if f.endswith('.wav')])
            print(f"  • {folder}: {num_files} файлов")
        print()
    
    total_processed = 0
    total_skipped = 0
    
    # Обрабатываем каждую папку канала
    for folder in channel_folders:
        folder_path = os.path.join(root_dir, folder)
        
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        for filename in tqdm(wav_files, desc=f"  {folder}", disable=not verbose):
            
            full_path = os.path.join(folder_path, filename)
            
            # Извлекаем информацию из имени файла
            file_info = parse_filename_info(filename)
            
            # Истинная метка из имени файла
            true_label = file_info['pavement']
            
            if true_label is None:
                total_skipped += 1
                continue
            
            # Извлекаем признаки
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
                
                # Сохраняем всю метаинформацию
                file_info['filename'] = filename
                file_info['folder'] = folder
                file_info['file_path'] = full_path
                file_info['predicted'] = pred_label
                metadata_list.append(file_info)
                
                total_processed += 1
                
            except Exception as e:
                total_skipped += 1
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Обработано: {total_processed}, Пропущено: {total_skipped}")
    
    return predictions, true_labels, metadata_list

# ==================== ОЦЕНКА ====================
def evaluate_predictions(predictions, true_labels, metadata_list=None):
    """Оценка с детальной статистикой"""
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ")
    print("="*70)
    
    acc = accuracy_score(true_labels, predictions)
    print(f"\n✓ Точность: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nОтчет:")
    print(classification_report(true_labels, predictions))
    
    print("\nМатрица ошибок:")
    print(confusion_matrix(true_labels, predictions))
    
    if metadata_list:
        # Анализ по различным параметрам
        import pandas as pd
        df = pd.DataFrame(metadata_list)
        df['correct'] = [pred == true for pred, true in zip(predictions, true_labels)]
        
        print("\n" + "="*70)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ")
        print("="*70)
        
        # По условиям (dry/wet)
        if 'condition' in df.columns and df['condition'].notna().any():
            print("\nПо условиям:")
            for condition in sorted(df['condition'].dropna().unique()):
                subset = df[df['condition'] == condition]
                acc = subset['correct'].mean()
                print(f"  • {condition}: {acc:.4f} ({acc*100:.1f}%) - {len(subset)} образцов")
        
        # По каналам
        if 'channel' in df.columns:
            print("\nПо каналам:")
            for channel in sorted(df['channel'].dropna().unique()):
                subset = df[df['channel'] == channel]
                acc = subset['correct'].mean()
                print(f"  • Канал {int(channel)}: {acc:.4f} ({acc*100:.1f}%) - {len(subset)} образцов")
        
        # По локациям
        if 'location' in df.columns and df['location'].notna().any():
            print("\nПо локациям:")
            for location in sorted(df['location'].dropna().unique()):
                subset = df[df['location'] == location]
                acc = subset['correct'].mean()
                print(f"  • {location}: {acc:.4f} ({acc*100:.1f}%) - {len(subset)} образцов")

# ==================== СОХРАНЕНИЕ ====================
def save_results(predictions, true_labels, metadata_list, output_path):
    """Сохранение результатов в CSV"""
    import pandas as pd
    
    df = pd.DataFrame(metadata_list)
    df['true_label'] = true_labels
    df['predicted_label'] = predictions
    df['correct'] = [pred == true for pred, true in zip(predictions, true_labels)]
    
    # Переупорядочиваем колонки
    cols = ['filename', 'channel', 'date', 'session', 'segment', 'location', 
            'pavement', 'condition', 'area_type', 'window_state', 
            'true_label', 'predicted_label', 'correct', 'folder', 'file_path']
    
    # Оставляем только те, что есть
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ Сохранено: {output_path} ({len(df)} записей)")
    
    return df

# ==================== MAIN ====================
def main():
    print("="*70)
    print("ПРЕДСКАЗАНИЕ НА 1s ДАТАСЕТЕ")
    print("="*70)
    
    print("\n[1/3] Загрузка модели...")
    try:
        model, le, metadata = load_model_safe(Config.MODEL_NAME)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    print("\n[2/3] Предсказания...")
    predictions, true_labels, metadata_list = predict_on_new_dataset(
        model, le, Config.TEST_DATA_DIR, verbose=True
    )
    
    if not predictions:
        print("❌ Нет результатов")
        return
    
    print("\n[3/3] Оценка...")
    evaluate_predictions(predictions, true_labels, metadata_list)
    
    print("\n" + "="*70)
    print("СОХРАНЕНИЕ")
    print("="*70)
    
    save_results(predictions, true_labels, metadata_list, Config.OUTPUT_CSV)
    
    print("\n✓ ГОТОВО!")

if __name__ == "__main__":
    main()
