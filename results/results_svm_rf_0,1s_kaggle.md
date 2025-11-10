Всего образцов (файлов): 68064
Признаков на образец: 136
Уникальные метки: ['asphalt' 'cobblestones']
Распределение классов: {'asphalt': 67544, 'cobblestones': 520}

Train: 54451, Test: 13613

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ SVM

✓ Лучшие параметры SVM: {'C': 14.08834317764575, 'gamma': 0.004921312831254389}  
✓ Лучший F1 (CV): 0.9945

Обучение модели: SVM

SVM - TRAIN SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      1.00      1.00     54035
cobblestones       0.98      1.00      0.99       416

    accuracy                           1.00     54451
   macro avg       0.99      1.00      1.00     54451
weighted avg       1.00      1.00      1.00     54451
```

F1-weighted (train): 0.9999

SVM - TEST SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      1.00      1.00     13509
cobblestones       0.74      0.52      0.61       104

    accuracy                           0.99     13613
   macro avg       0.87      0.76      0.80     13613
weighted avg       0.99      0.99      0.99     13613
```

F1-weighted (test): 0.9945

Нет значительного переобучения (разница: 0.0054)

Matрица ошибок (TEST):  
[[13490    19]  
 [   50    54]]

ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ RANDOM FOREST

✓ Лучшие параметры RF: {'n_estimators': 200, 'max_depth': 30, 'min_samples_split': 8, 'min_samples_leaf': 5, 'max_features': 'sqrt'}  
✓ Лучший F1 (CV): 0.9923

Обучение модели: Random Forest

Random Forest - TRAIN SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      1.00      1.00     54035
cobblestones       0.78      1.00      0.88       416

    accuracy                           1.00     54451
   macro avg       0.89      1.00      0.94     54451
weighted avg       1.00      1.00      1.00     54451
```

F1-weighted (train): 0.9980

Random Forest - TEST SET:

```
              precision    recall  f1-score   support

     asphalt       1.00      1.00      1.00     13509
cobblestones       0.59      0.47      0.52       104

    accuracy                           0.99     13613
   macro avg       0.79      0.73      0.76     13613
weighted avg       0.99      0.99      0.99     13613
```

F1-weighted (test): 0.9931

Нет значительного переобучения (разница: 0.0049)

Matрица ошибок (TEST):  
[[13475    34]  
 [   55    49]]

ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ  
SVM F1 (test): 0.9945  
Random Forest F1 (test): 0.9931
Лучшая модель: SVM
