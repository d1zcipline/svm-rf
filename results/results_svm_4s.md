Всего сегментов: 2159
Признаков на сегмент: 1088
Уникальные метки: ['asphalt' 'cobblestones']
Распределение классов: {np.str*('asphalt'): np.int64(2095), np.str*('cobblestones'): np.int64(64)}

Train: 1727, Test: 432

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ SVM

✓ Лучшие параметры SVM: {'C': 4.6957056507727195, 'gamma': 0.000335499224734649}
✓ Лучший F1 (CV): 0.9858

Обучение модели: SVM

SVM - TRAIN SET:

```
              precision    recall  f1-score   support
     asphalt       1.00      1.00      1.00      1676
cobblestones       1.00      1.00      1.00        51

    accuracy                           1.00      1727
   macro avg       1.00      1.00      1.00      1727
weighted avg       1.00      1.00      1.00      1727
```

F1-weighted (train): 1.0000

SVM - TEST SET:

```
              precision    recall  f1-score   support

     asphalt       0.99      1.00      1.00       419
cobblestones       1.00      0.69      0.82        13

    accuracy                           0.99       432
   macro avg       1.00      0.85      0.91       432
weighted avg       0.99      0.99      0.99       432
```

F1-weighted (test): 0.9899

✓ Нет значительного переобучения (разница: 0.0101)

Matрица ошибок (TEST):
[[419   0]
 [  4   9]]

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ RANDOM FOREST

✓ Лучшие параметры RF: {'n_estimators': 400, 'max_depth': 18, 'min_samples_split': 6, 'min_samples_leaf': 5, 'max_features': 'sqrt'}
✓ Лучший F1 (CV): 0.9816

Обучение модели: Random Forest

Random Forest - TRAIN SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      1.00      1.00      1676
cobblestones       1.00      1.00      1.00        51

    accuracy                           1.00      1727
   macro avg       1.00      1.00      1.00      1727
weighted avg       1.00      1.00      1.00      1727
```

F1-weighted (train): 1.0000

Random Forest - TEST SET:

```
              precision    recall  f1-score   support

     asphalt       0.99      1.00      0.99       419
cobblestones       1.00      0.54      0.70        13

    accuracy                           0.99       432
   macro avg       0.99      0.77      0.85       432
weighted avg       0.99      0.99      0.98       432
```

F1-weighted (test): 0.9841

✓ Нет значительного переобучения (разница: 0.0159)

Matрица ошибок (TEST):
[[419   0]
 [  6   7]]

ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ
SVM F1 (test): 0.9899
Random Forest F1 (test): 0.9841
Лучшая модель: SVM
