import os
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.impute import KNNImputer

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TRAIN_OUT = os.path.join(PROCESSED_DIR, "epoch5_train.csv")
TEST_OUT = os.path.join(PROCESSED_DIR, "epoch5_test.csv")



def extract_title(name: str) -> str:
    title = name.split(',')[1].split('.')[0].strip()
    if title in {"Mr", "Miss", "Mrs", "Master"}:
        return title
    return "Rare"


def encode_title(series: pd.Series) -> pd.Series:
    mapping = {
        "Mr": 0,
        "Miss": 1,
        "Mrs": 2,
        "Master": 3,
        "Rare": 4,
    }
    return series.map(mapping).astype(int)


def encode_embarked(series: pd.Series) -> pd.Series:
    mapping = {"S": 0, "C": 1, "Q": 2}
    return series.map(mapping).astype(int)



def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))
    return train, test


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # concatenate for consistent processing
    full = pd.concat([train.drop(columns=["Survived"]), test], axis=0, ignore_index=True)

    # Sex
    full["Sex"] = full["Sex"].map({"male": 0, "female": 1}).astype(int)

    # Title
    full["Title"] = full["Name"].apply(extract_title)
    full["Title"] = encode_title(full["Title"])

    # Family features
    full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
    full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

    # Embarked
    embarked_mode = train["Embarked"].mode()[0]
    full["Embarked"] = full["Embarked"].fillna(embarked_mode)
    full["Embarked"] = encode_embarked(full["Embarked"])

    # Fare
    fare_median = train["Fare"].median()
    full["Fare"] = full["Fare"].fillna(fare_median)
    full["Fare_log"] = np.log(full["Fare"] + 1)

    # Age — KNN imputation
    age_features = ["Pclass", "Sex", "SibSp", "Parch", "Fare_log"]

    train_idx = np.arange(len(train))
    test_idx = np.arange(len(train), len(full))

    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(full.loc[train_idx, age_features + ["Age"]])

    age_imputed = imputer.transform(full[age_features + ["Age"]])
    full["Age"] = age_imputed[:, -1]

    # Select final features
    features = [
        "Sex",
        "Title",
        "Pclass",
        "Age",
        "Fare_log",
        "FamilySize",
        "IsAlone",
        "Embarked",
    ]

    full = full[features]

    X_train = full.iloc[:len(train)].copy()
    X_test = full.iloc[len(train):].copy()

    return X_train, X_test


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_raw, test_raw = load_raw()
    X_train, X_test = build_features(train_raw, test_raw)

    y_train = train_raw["Survived"].astype(int)

    train_out = X_train.copy()
    train_out["Survived"] = y_train.values

    train_out.to_csv(TRAIN_OUT, index=False)
    X_test.to_csv(TEST_OUT, index=False)

    print("[datapipe] Saved:")
    print(f" - {TRAIN_OUT}")
    print(f" - {TEST_OUT}")


if __name__ == "__main__":
    main()