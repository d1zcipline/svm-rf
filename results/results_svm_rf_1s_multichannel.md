Всего сегментов: 3338  
Признаков на сегмент: 1088  
Уникальные метки: ['asphalt' 'cobblestones']  
Распределение классов: {np.str*('asphalt'): np.int64(3148), np.str*('cobblestones'): np.int64(190)}

Train: 2670, Test: 668

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ SVM

✓ Лучшие параметры SVM: {'C': 9.36573870625742, 'gamma': 0.0006952333704387059}  
✓ Лучший F1 (CV): 0.9880

Обучение модели: SVM

SVM - TRAIN SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      1.00      1.00      2518
cobblestones       1.00      1.00      1.00       152

    accuracy                           1.00      2670
   macro avg       1.00      1.00      1.00      2670
weighted avg       1.00      1.00      1.00      2670
```

F1-weighted (train): 1.0000

SVM - TEST SET:

```
              precision    recall  f1-score   support

     asphalt       0.99      1.00      1.00       630
cobblestones       0.97      0.87      0.92        38

    accuracy                           0.99       668
   macro avg       0.98      0.93      0.96       668
weighted avg       0.99      0.99      0.99       668
```

F1-weighted (test): 0.9908

✓ Нет значительного переобучения (разница: 0.0092)

Matрица ошибок (TEST):  
[[629   1]  
 [  5  33]]

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ RANDOM FOREST

✓ Лучшие параметры RF: {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 6, 'min_samples_leaf': 4, 'max_features': 'sqrt'}  
✓ Лучший F1 (CV): 0.9852

Обучение модели: Random Forest

Random Forest - TRAIN SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      1.00      1.00      2518
cobblestones       0.98      1.00      0.99       152

    accuracy                           1.00      2670
   macro avg       0.99      1.00      0.99      2670
weighted avg       1.00      1.00      1.00      2670
```

F1-weighted (train): 0.9989

Random Forest - TEST SET:

```
              precision    recall  f1-score   support

     asphalt       0.99      1.00      1.00       630
cobblestones       0.94      0.89      0.92        38

    accuracy                           0.99       668
   macro avg       0.97      0.95      0.96       668
weighted avg       0.99      0.99      0.99       668
```

F1-weighted (test): 0.9909

✓ Нет значительного переобучения (разница: 0.0080)

Matрица ошибок (TEST):  
[[628   2]  
 [  4  34]]

ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ  
SVM F1 (test): 0.9908  
Random Forest F1 (test): 0.9909  
Лучшая модель: Random Forest
