# Epoch 6.1 — Model Softening & Generalization Breakthrough

## Контекст
После эпохи 6.0 была получена сильная модель CatBoost с высоким CV score
(~0.85), однако Kaggle submission показывал заметно более низкий результат
(~0.75–0.76), что указывало на переобучение.

В рамках Epoch 6.1 основной фокус был смещён с feature engineering
(признаки уже были насыщенные и стабильные) на:
- регуляризацию,
- калибровку вероятностей,
- threshold tuning,
- ослабление модели.

---

## Используемые признаки
Модель обучалась на следующих признаках (epoch6_train.csv):

- Pclass
- Sex
- Age
- Fare
- Embarked
- Title
- FamilySize
- IsAlone
- Age_group
- Fare_bin
- Title_Pclass
- Sex_Pclass
- IsAlone_AgeGroup


---

## Эксперименты Epoch 6.1

### 1. Threshold tuning
- Перебор threshold в диапазоне `[0.30 … 0.70]`
- Лучший threshold по validation: ~`0.34–0.37`
- Улучшения на Kaggle не наблюдалось → подтверждение, что проблема не в threshold

---

### 2. Multi-model CatBoost (3 экземпляра)
- Обучение 3 моделей CatBoost на подвыборках
- Soft-voting по вероятностям
- Threshold tuning поверх ансамбля

Результат:
- CV стал более стабильным
- Kaggle score остался на уровне ~0.76–0.77

Вывод:
Ансамбль сильных моделей не решает проблему переобучения.

---

### 3. Deep Grid Search (5-fold CV)
Проведён широкий и глубокий grid search:

Параметры:
- iterations: [200, 400, 800, 1200]
- learning_rate: [0.01, 0.03, 0.05, 0.1]
- depth: [4, 5, 6, 8]
- l2_leaf_reg: [1, 3, 5, 7, 9]
- bagging_temperature: [0.0, 0.5, 1.0, 2.0]
- rsm: [0.6, 0.8, 1.0]

Всего комбинаций: **11 520**  
CV: **StratifiedKFold (5 folds)**

Лучшие параметры по CV:
```json
{
  "iterations": 800,
  "learning_rate": 0.1,
  "depth": 4,
  "l2_leaf_reg": 9,
  "bagging_temperature": 0.0,
  "rsm": 1.0,
  "cv_accuracy": 0.85297
}
```

Вывод: Grid research не дал хорошего улучшения score, остался на уровне `0.75837 - 0.76555`

---


### 4. Смягчение параметров

Гипотеза: из-за слишком строгих параметров модель переобучается на тестовых данных, на данных которые она не видела ломается. Значит нужно смягчить параметры.


```json
{
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 4,
    "l2_leaf_reg": 9,
    "bagging_temperature": 1.0,
    "rsm": 1.0,
    "random_seed": 42
}
```

Вывод: Гипотеза верна. Итоговый score: `0.78229`
