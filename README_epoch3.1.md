# Titanic Kaggle – Epoch 3.1

В этой эпохе мы обновили pipeline обработки данных:

- **Возраст (Age)** заполняется через KNN на основе схожих пассажиров.  
- Сохраняем признаки:
  - Пол (`Sex`)  
  - Возраст (`Age`)  
  - Семейное положение (`IsAlone`)  
  - Титул (`Title_*` — one-hot encoding)  
  - Порт посадки (`Embarked_*` — one-hot encoding)  

---  

## Обработка данных

| Шаг | Описание |
|-----|----------|
| KNN Age | Заполнение пропусков возраста по ближайшим соседям |
| Sex | Маппинг male=0, female=1 |
| IsAlone | 1 если пассажир был один, 0 иначе |
| Title | One-hot encoding титулов (Mr, Mrs, Miss, Master, Rare) |
| Embarked | One-hot encoding C, Q, S |


---

## Результаты обучения

### Manual Logistic Regression

| L2 Lambda | Epochs | Validation Accuracy |
|-----------|--------|-------------------|
| 0.0       | 1000   | 0.8268            |
| 0.01      | 1000   | 0.8268            |
| 0.1       | 1000   | 0.8156            |

**Best Manual Model:** L2=0.0, Validation Accuracy=0.8268  

### SKLearn Logistic Regression Wrapper

| C (Inverse Regularization) | Validation Accuracy |
|----------------------------|-------------------|
| 0.1                        | 0.8268            |
| 1.0                        | 0.8268            |
| 10.0                       | 0.8268            |

**Best SKLearn Model:** C=0.1, Validation Accuracy=0.8268   

---

**Вывод:**  

Логистическая регрессия показала стабильную точность ~0.8268 на валидации. Дальнейшее улучшение требует других моделей или feature engineering.
