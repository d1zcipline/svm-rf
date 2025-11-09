Всего образцов (файлов): 26704  
Признаков на образец: 136  
Уникальные метки: ['asphalt' 'cobblestones']  
Распределение классов: {'asphalt': 25184, 'cobblestones': 1520}

Train: 21363, Test: 5341

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ SVM

✓ Лучшие параметры SVM: {'C': 3.6350994223189024, 'gamma': 0.007127913487469911}  
✓ Лучший F1 (CV): 0.9716

Обучение модели: SVM

SVM - TRAIN SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      0.99      1.00     20147
cobblestones       0.91      1.00      0.95      1216

    accuracy                           0.99     21363
   macro avg       0.96      1.00      0.98     21363
weighted avg       0.99      0.99      0.99     21363
```

F1-weighted (train): 0.9946

SVM - TEST SET:

```
              precision    recall  f1-score   support

     asphalt       0.98      0.98      0.98      5037
cobblestones       0.66      0.74      0.70       304

    accuracy                           0.96      5341
   macro avg       0.82      0.86      0.84      5341
weighted avg       0.97      0.96      0.96      5341
```

F1-weighted (test): 0.9647

Нет значительного переобучения (разница: 0.0299)

Matрица ошибок (TEST):  
[[4924  113]  
[  80  224]]

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ RANDOM FOREST  
✓ Лучшие параметры RF: {'n_estimators': 450, 'max_depth': 21, 'min_samples_split': 7, 'min_samples_leaf': 5, 'max_features': 'sqrt'}  
✓ Лучший F1 (CV): 0.9683

Обучение модели: Random Forest

Random Forest - TRAIN SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      0.99      1.00     20147
cobblestones       0.91      1.00      0.95      1216

    accuracy                           0.99     21363
   macro avg       0.96      1.00      0.98     21363
weighted avg       1.00      0.99      0.99     21363
```

F1-weighted (train): 0.9947

Random Forest - TEST SET:

```
   precision    recall  f1-score   support

     asphalt       0.98      0.99      0.98      5037
cobblestones       0.76      0.63      0.69       304

    accuracy                           0.97      5341
   macro avg       0.87      0.81      0.84      5341
weighted avg       0.97      0.97      0.97      5341
```

F1-weighted (test): 0.9664

Нет значительного переобучения (разница: 0.0284)

Matрица ошибок (TEST):  
[[4977   60]  
[ 112  192]]

ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ  
SVM F1 (test): 0.9647  
Random Forest F1 (test): 0.9664  
Лучшая модель: Random Forest
