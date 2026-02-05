# Epoch 4 – Feature Engineering Patch

В этой эпохе мы добавили новые признаки для логистической регрессии и SKLearn LogisticRegression.  
Цель – попробовать поднять точность модели за счет синтетических и комбинированных признаков.

---

## Обработанные признаки

1. **Sex** – бинарный признак (`0` = male, `1` = female)  
2. **Age** – возраст, два варианта заполнения пропусков:  
   - `median` – медианой  
   - `knn` – KNN по другим признакам  
3. **IsAlone** – был ли пассажир один (`1`) или нет (`0`)  
4. **Title** – выделенные титулы из имени (`Mr`, `Miss`, `Mrs`, `Master`, `Rare`)  
5. **Embarked** – one-hot кодирование порта посадки (`C`, `Q`, `S`)  
6. **Синтетические признаки (Epoch 4)**:
   - `Sex_Age` – комбинация пола и возраста  
   - `Age_squared` – квадрат возраста  
   - `Embarked_*_x_Pclass` – взаимодействие порта посадки и класса  
   - `IsChild` – возраст ≤ 15  
   - `IsSenior` – возраст ≥ 60  

---

## Используемые данные

| Dataset | Age fill method | Rows | Columns |
|---------|----------------|------|---------|
| Train   | Median          | 891  | все обработанные признаки |
| Test    | Median          | 418  | согласованные признаки |
| Train   | KNN             | 891  | все обработанные признаки |
| Test    | KNN             | 418  | согласованные признаки |

---

## Модели

- **Manual Logistic Regression** – реализация собственной модели с L2 регуляризацией.  
- **SKLearn LogisticRegression** – стандартная реализация с L2.  

Гиперпараметры для обеих моделей:

| Model | lr | epochs | L2 / C |
|-------|----|--------|--------|
| Manual | 0.01 | 3000 | 0.01 |
| SKLearn | – | – | 1.0 |

---

## Результаты на валидации

**Median Age**

| Step | Manual val_acc | SKLearn val_acc |
|------|----------------|----------------|
| 1 (базовые признаки) | 0.8212 | 0.8212 |
| 2 (+ Sex_Age)         | 0.8212 | 0.8212 |
| 3 (+ Age_squared)     | 0.8212 | 0.8212 |
| 4 (+ Embarked*Pclass, IsChild, IsSenior) | 0.8212 | 0.8212 |

**KNN Age**

| Step | Manual val_acc | SKLearn val_acc |
|------|----------------|----------------|
| 1 (базовые признаки) | 0.8212 | 0.8212 |
| 2 (+ Sex_Age)         | 0.8212 | 0.8212 |
| 3 (+ Age_squared)     | 0.8212 | 0.8212 |
| 4 (+ Embarked*Pclass, IsChild, IsSenior) | 0.8212 | 0.8212 |

---

## Submissions

- `submissions/epoch4_submission_manual_median.csv`  
- `submissions/epoch4_submission_skl_median.csv`  
- `submissions/epoch4_submission_manual_knn.csv`  
- `submissions/epoch4_submission_skl_knn.csv`  


---

**Выводы**

- На текущем наборе признаков точность остановилась на ~0.8212 на валидации.  
- Следующие патчи будут исследовать **новые модели**, так как логистическая регрессия достигла потолка.
