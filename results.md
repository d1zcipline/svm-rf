## RF

Подбор гиперпараметров для Random Forest:
Лучшие параметры: `{'n_estimators': 479, 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 10}`
Лучший F1 (CV): 0.9803

### RANDOM FOREST (dry + wet, мультиканальный) — CLASSIFICATION REPORT (TRAIN)
```
              precision    recall  f1-score   support
     asphalt       1.00      0.99      1.00      1676
cobblestones       0.78      1.00      0.88        51

    accuracy                           0.99      1727
   macro avg       0.89      1.00      0.94      1727
weighted avg       0.99      0.99      0.99      1727
```

### RANDOM FOREST (dry + wet, мультиканальный) — CLASSIFICATION REPORT (TEST)
```
              precision    recall  f1-score   support
     asphalt       1.00      1.00      1.00       419
cobblestones       1.00      0.85      0.92        13

    accuracy                           1.00       432
   macro avg       1.00      0.92      0.96       432
weighted avg       1.00      1.00      1.00       432
```

F1-weighted (train): 0.9924
F1-weighted (test):  0.9952

Нет признаков значительного переобучения

## SVM

Подбор гиперпараметров для SVM:
Лучшие параметры: `{'C': 3.4684554466506023, 'gamma': 0.00023310381598388974}`
Лучший F1 (CV): 0.9830

### SVM (dry + wet, мультиканальный) — CLASSIFICATION REPORT (TRAIN)
```
              precision    recall  f1-score   support
     asphalt       1.00      1.00      1.00      1676
cobblestones       0.88      1.00      0.94        51

    accuracy                           1.00      1727
   macro avg       0.94      1.00      0.97      1727
weighted avg       1.00      1.00      1.00      1727
```

### SVM (dry + wet, мультиканальный) — CLASSIFICATION REPORT (TEST)
```
              precision    recall  f1-score   support
     asphalt       1.00      0.99      0.99       419
cobblestones       0.71      0.92      0.80        13

    accuracy                           0.99       432
   macro avg       0.85      0.96      0.90       432
weighted avg       0.99      0.99      0.99       432
```

F1-weighted (train): 0.9961
F1-weighted (test):  0.9870

Нет признаков значительного переобучения

## SVM + RF

Подбор гиперпараметров для SVM:
Лучшие параметры SVM: `{'C': 13.083051327582735, 'gamma': 0.00010021523102968868}`
Подбор гиперпараметров для Random Forest...
Лучшие параметры RF: `{'n_estimators': 237, 'max_depth': 5, 'min_samples_split': 7, 'min_samples_leaf': 4}`
Обучаем ансамбль SVM + Random Forest...

### ENSEMBLE (SVM + RF) — CLASSIFICATION REPORT (TRAIN)
```
              precision    recall  f1-score   support
     asphalt       1.00      1.00      1.00      1676
cobblestones       0.96      1.00      0.98        51

    accuracy                           1.00      1727
   macro avg       0.98      1.00      0.99      1727
weighted avg       1.00      1.00      1.00      1727
```

### ENSEMBLE (SVM + RF) — CLASSIFICATION REPORT (TEST)
```
              precision    recall  f1-score   support
     asphalt       1.00      1.00      1.00       419
cobblestones       1.00      0.85      0.92        13

    accuracy                           1.00       432
   macro avg       1.00      0.92      0.96       432
weighted avg       1.00      1.00      1.00       432
```

F1-weighted (train): 0.9989
F1-weighted (test):  0.9952

Нет признаков значительного переобучения
