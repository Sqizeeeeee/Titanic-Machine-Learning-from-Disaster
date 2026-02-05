# Epoch 2 — Feature Expansion & Regularization

## Goal of Epoch 2

Цель второй эпохи — аккуратно расширить baseline модель:

- добавить новые осмысленные признаки
- сохранить простую и интерпретируемую модель
- проверить, даёт ли feature engineering прирост без усложнения алгоритма

Модель и pipeline намеренно остаются простыми, без бустинга и сложных ансамблей.

---

## New Features Added

По сравнению с Epoch 1 были добавлены следующие признаки:

- **Fare** — стоимость билета  
- **FamilySize** — размер семьи пассажир

`FamilySize = SibSp + Parch + 1`


Используемый набор признаков в Epoch 2:

- Age  
- Sex (binary encoding)  
- Pclass  
- Fare  
- FamilySize  

---

## Preprocessing

- Пропуски в `Age` заполняются **медианой**
- `Fare` не содержит пропусков и используется без изменений
- Категориальные признаки закодированы численно
- Числовые признаки масштабируются с помощью **StandardScaler**
- Train / Validation split: **80 / 20**, `random_state=42`

---

## Models Used

### 1. Manual Logistic Regression

- Реализация логистической регрессии **с нуля**
- Batch Gradient Descent
- Добавлена **L2-регуляризация**
- Подбор коэффициента регуляризации `lambda`

### 2. SKLearn Logistic Regression (Wrapper)

- `sklearn.linear_model.LogisticRegression`
- L2-регуляризация
- Подбор параметра `C` (обратный коэффициент регуляризации)

---

## Hyperparameter Tuning (Manual Model)

| L2 Lambda | Validation Accuracy |
|----------|---------------------|
| 0.0      | 0.8045              |
| 0.01     | **0.8101**          |
| 0.1      | 0.7542              |

**Best manual model:**  
- `L2 lambda = 0.01`  
- Validation accuracy = **0.8101**

---

## Hyperparameter Tuning (SKLearn Model)

| C (inverse regularization) | Validation Accuracy |
|---------------------------|---------------------|
| 0.1                       | **0.8101**          |
| 1.0                       | 0.8045              |
| 10.0                      | 0.8045              |

**Best sklearn model:**  
- `C = 0.1`  
- Validation accuracy = **0.8101**

---

## Model Comparison

- Лучшие manual и sklearn модели дают **одинаковую validation accuracy**
- Количество различий в предсказаниях на validation: `0`

Это означает, что обе реализации сошлись к практически идентичному решению.

---

## Kaggle Submission Results

| Model            | Kaggle Score |
|------------------|--------------|
| Manual Logistic  | 0.76315      |
| SKLearn Logistic | 0.76315      |

Результат совпадает с Epoch 1 — добавленные признаки не дали прироста на leaderboard.

---

## Key Takeaways of Epoch 2

- Добавление `Fare` и `FamilySize` **не улучшило итоговый score**
- Регуляризация помогла стабилизировать веса, но не дала прироста качества
- Потолок логистической регрессии с текущими признаками достигнут
- Основной путь улучшения — **более сильный feature engineering**, а не тюнинг модели

---

## Next Steps (Epoch 3)

В следующей эпохе планируется:

- извлечение **Title** из имени пассажира
- работа с более информативными категориальными признаками
- сохранение интерпретируемости моделей
