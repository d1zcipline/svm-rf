Всего образцов (файлов): 42400  
Признаков на образец: 136  
Уникальные метки: ['asphalt' 'cobblestones']  
Распределение классов: {'asphalt': 41880, 'cobblestones': 520}

Train: 33920, Test: 8480

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ SVM

✓ Лучшие параметры SVM: {'C': 8.584975979794343, 'gamma': 0.005050894650563541}  
✓ Лучший F1 (CV): 0.9914

Обучение модели: SVM

SVM - TRAIN SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      1.00      1.00     33504
cobblestones       0.96      1.00      0.98       416

    accuracy                           1.00     33920
   macro avg       0.98      1.00      0.99     33920
weighted avg       1.00      1.00      1.00     33920
```

F1-weighted (train): 0.9995

SVM - TEST SET:

```
              precision    recall  f1-score   support

     asphalt       0.99      1.00      1.00      8376
cobblestones       0.79      0.56      0.66       104

    accuracy                           0.99      8480
   macro avg       0.89      0.78      0.83      8480
weighted avg       0.99      0.99      0.99      8480
```

F1-weighted (test): 0.9922

Нет значительного переобучения (разница: 0.0073)

Matрица ошибок (TEST):  
[[8361   15]  
 [  46   58]]

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ RANDOM FOREST

✓ Лучшие параметры RF: {'n_estimators': 400, 'max_depth': 33, 'min_samples_split': 8, 'min_samples_leaf': 5, 'max_features': 'sqrt'}  
✓ Лучший F1 (CV): 0.9886

Обучение модели: Random Forest

Random Forest - TRAIN SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      1.00      1.00     33504
cobblestones       0.85      1.00      0.92       416

    accuracy                           1.00     33920
   macro avg       0.93      1.00      0.96     33920
weighted avg       1.00      1.00      1.00     33920
```

F1-weighted (train): 0.9980

Random Forest - TEST SET:

```
              precision    recall  f1-score   support

     asphalt       0.99      1.00      1.00      8376
cobblestones       0.68      0.38      0.48       104

    accuracy                           0.99      8480
   macro avg       0.84      0.69      0.74      8480
weighted avg       0.99      0.99      0.99      8480
```

F1-weighted (test): 0.9888

Нет значительного переобучения (разница: 0.0092)

Matрица ошибок (TEST):  
[[8358   18]  
 [  65   39]]

ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ  
SVM F1 (test): 0.9922  
Random Forest F1 (test): 0.9888  
Лучшая модель: SVM
