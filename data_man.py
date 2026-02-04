import pandas as pd
import numpy as np
import os

df_train = pd.read_csv("data/raw/train.csv")

# Обработка признаков
df_train["Sex"] = df_train["Sex"].map({"male": 0, "female": 1})
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())
df_train["Fare"] = df_train["Fare"].fillna(df_train["Fare"].median())

# Создаём новый признак FamilySize
df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"]

# Выбираем нужные колонки для модели
train_features = ["Age", "Sex", "Pclass", "Fare", "FamilySize"]

# Сохраняем processed train
os.makedirs("data/processed", exist_ok=True)
df_train[train_features + ["Survived"]].to_csv("data/processed/epoch2_train.csv", index=False)
print("Processed train saved to data/processed/epoch2_train.csv")

# ---------------------- TEST ----------------------
df_test = pd.read_csv("data/raw/test.csv")

# Сохраняем PassengerId отдельно
passenger_ids = df_test["PassengerId"].copy()

# Обработка признаков
df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})
df_test["Age"] = df_test["Age"].fillna(df_train["Age"].median())  # используем медиану train
df_test["Fare"] = df_test["Fare"].fillna(df_train["Fare"].median())  # на всякий случай
df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"]

# Сохраняем только признаки для модели + PassengerId для submission
test_features = ["Age", "Sex", "Pclass", "Fare", "FamilySize"]
df_test[["PassengerId"] + test_features].to_csv("data/processed/epoch2_test.csv", index=False)
print("Processed test saved to data/processed/epoch2_test.csv")