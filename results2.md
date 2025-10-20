# Results of raw_audio of 0.1s dataset

## SVM

Всего сегментов: 59594 | Размер признака: 58  
Запуск Optuna для SVM (RBF): trials=30, CV folds=3…  
Лучшие параметры: {'C': 1.9068442481765508, 'gamma': 0.034586112996797244}  
Лучший F1 (CV): 0.8549  

### SVM (RBF) — ОТЧЁТЫ  

Отчёт (TRAIN):
```
              precision    recall  f1-score   support

     asphalt       1.00      0.96      0.98     31665
cobblestones       0.92      0.99      0.96     16010

    accuracy                           0.97     47675
   macro avg       0.96      0.98      0.97     47675
weighted avg       0.97      0.97      0.97     47675
```

Отчёт (TEST):
```
              precision    recall  f1-score   support

     asphalt       0.92      0.86      0.89      7917
cobblestones       0.75      0.86      0.80      4002

    accuracy                           0.86     11919
   macro avg       0.84      0.86      0.85     11919
weighted avg       0.87      0.86      0.86     11919
```

ИТОГО:  
Accuracy: train=0.9691 | test=0.8585  
F1-weighted: train=0.9694 | test=0.8606  
Разница (train-test) по F1: 0.1087  

## RF

Всего сегментов: 59594 | Размер признака: 58  
Запуск Optuna: trials=30, CV folds=3…  
Лучшие параметры: {'n_estimators': 503, 'max_depth': 31, 'min_samples_split': 9, 'min_samples_leaf': 3, 'max_features': None, 'bootstrap': True}  
Лучший F1 (CV): 0.8756  

### RF — ОТЧЁТЫ  

Отчёт (TRAIN):
```
              precision    recall  f1-score   support

     asphalt       1.00      0.98      0.99     31665
cobblestones       0.96      0.99      0.98     16010

    accuracy                           0.98     47675
   macro avg       0.98      0.99      0.98     47675
weighted avg       0.98      0.98      0.98     47675
```

Отчёт (TEST):  
```
              precision    recall  f1-score   support

     asphalt       0.93      0.89      0.91      7917
cobblestones       0.79      0.87      0.83      4002

    accuracy                           0.88     11919
   macro avg       0.86      0.88      0.87     11919
weighted avg       0.88      0.88      0.88     11919
```

ИТОГО:  
Accuracy: train=0.9846 | test=0.8797  
F1-weighted: train=0.9846 | test=0.8809  
Разница (train-test) по F1: 0.1037  
