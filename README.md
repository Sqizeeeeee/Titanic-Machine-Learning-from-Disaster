# Titanic Kaggle — EDA

## Что мы сделали на этапе анализа данных
- Просмотрели все колонки датасета train.csv и test.csv
-	Определили типы признаков:
    -	Числовые: Age, Pclass, SibSp, Parch, Fare
    -	Категориальные: Sex, Embarked
    -	Прочие: Name, Ticket, Cabin, PassengerId
-	Проверили пропуски:
    -	Age имеет пропуски, заполнили медианой
    -	Другие признаки пока оставлены как есть
-	Начали визуальный анализ:
    -	Посмотрели распределение Age, Pclass, Sex
-	Замечено, что:
    -	Женщины выживали чаще
    -	Пассажиры 1 класса выживали чаще
    -	Молодые пассажиры имели чуть больше шансов выжить

---

## Результаты:


| Модель (version)          | Score Manual | Score SKLR | Epoch Readme                   |
|---------------------------|--------------|------------|--------------------------------|
|Logic regression  (1)      |0.76076       | 0.76315    | [description](about/README_epoch1.md)|
|Logic regression  (2)      |no test       | 0.76315    | [description](about/README_epoch2.md)|
|Logic regression  (3)      |0.77033       | 0.76794    | [description](about/README_epoch3.md)|
|Logic regression  (3.1)    |0.77033       | no test    | [description](about/README_epoch3.1.md)|
|Logic regression  (4)      |no test       | 0.77033   | [description](about/README_epoch4.md)|

