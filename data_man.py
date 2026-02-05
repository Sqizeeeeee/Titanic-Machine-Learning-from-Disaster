import pandas as pd
import numpy as np
from pathlib import Path

import pandas as pd
import numpy as np
from pathlib import Path

def process_single_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка одного DataFrame:
    - Sex, Age
    - FamilySize, IsAlone
    - Title → one-hot
    - Embarked → one-hot
    - удаление лишних колонок
    """
    # -----------------------
    # Пол
    # -----------------------
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # -----------------------
    # Возраст
    # -----------------------
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # -----------------------
    # FamilySize и IsAlone
    # -----------------------
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # -----------------------
    # Title
    # -----------------------
    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")
    title_dummies = pd.get_dummies(df["Title"], prefix="Title")
    df = pd.concat([df, title_dummies], axis=1)

    # -----------------------
    # Embarked
    # -----------------------
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
    df = pd.concat([df, embarked_dummies], axis=1)

    # -----------------------
    # Удаляем ненужные колонки
    # -----------------------
    drop_cols = ["Name", "Title", "Embarked", "Pclass", "Ticket", "Cabin", "SibSp", "Parch", "Fare"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df = df.astype(float)

    return df

def process_titanic_data(train_path: str, test_path: str, processed_dir: str) -> None:
    """
    Полный pipeline обработки данных для Epoch 3.
    Сохраняет train и test в processed_dir
    """

    processed_dir_path = Path(processed_dir)
    processed_dir_path.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Загружаем данные
    # -----------------------
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # -----------------------
    # Обработка
    # -----------------------
    train_df = process_single_df(train_df)
    test_df = process_single_df(test_df)

    # -----------------------
    # Сохраняем processed
    # -----------------------
    train_processed_path = processed_dir_path / "epoch3_train.csv"
    test_processed_path = processed_dir_path / "epoch3_test.csv"

    train_df.to_csv(train_processed_path, index=False)
    test_df.to_csv(test_processed_path, index=False)

    print(f"Processed train saved to {train_processed_path}")
    print(f"Processed test saved to {test_processed_path}")



if __name__ == "__main__":
    process_titanic_data(
        train_path="data/raw/train.csv",
        test_path="data/raw/test.csv",
        processed_dir="data/processed"
    )