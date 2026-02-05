import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def knn_impute_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    title_map = {
        "Mr": 0,
        "Miss": 1,
        "Mrs": 2,
        "Master": 3
    }
    df["Title_ord"] = df["Title"].map(title_map).fillna(4)

    knn_features = [
        "Sex",
        "Pclass",
        "SibSp",
        "Parch",
        "Title_ord"
    ]

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        knn_features.append("Fare")


    known_age = df[df["Age"].notna()]
    unknown_age = df[df["Age"].isna()]

    scaler = StandardScaler()
    X_known = scaler.fit_transform(known_age[knn_features])
    y_known = known_age["Age"]

    X_unknown = scaler.transform(unknown_age[knn_features])

    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
    knn.fit(X_known, y_known)

    df.loc[df["Age"].isna(), "Age"] = knn.predict(X_unknown)

    df.drop(columns=["Title_ord"], inplace=True)
    return df


def process_titanic_epoch3_1(
    train_path: str,
    test_path: str,
    processed_dir: str
) -> None:

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_len = len(train_df)

    # ======================
    # 1. Объединяем
    # ======================
    df = pd.concat([train_df, test_df], ignore_index=True)

    # ======================
    # 2. Sex
    # ======================
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # ======================
    # 3. Title
    # ======================
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")

    # ======================
    # 4. KNN → Age
    # ======================
    df = knn_impute_age(df)

    # ======================
    # 5. Family / IsAlone
    # ======================
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # ======================
    # 6. Embarked
    # ======================
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # ======================
    # 7. One-hot
    # ======================
    title_dummies = pd.get_dummies(df["Title"], prefix="Title", dtype=int)
    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", dtype=int)

    df = pd.concat([df, title_dummies, embarked_dummies], axis=1)

    # ======================
    # 8. Drop columns
    # ======================
    drop_cols = [
        "PassengerId",
        "Name",
        "Title",
        "Embarked",
        "Ticket",
        "Cabin",
        "Fare",
        "Pclass",
        "SibSp",
        "Parch",
        "FamilySize"
    ]

    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ======================
    # 9. Split back
    # ======================
    train_processed = df.iloc[:train_len].copy()
    test_processed = df.iloc[train_len:].copy()

    train_processed.to_csv(processed_dir / "epoch3_1_train.csv", index=False)
    test_processed.to_csv(processed_dir / "epoch3_1_test.csv", index=False)

    print("Epoch 3.1 processed data saved")
    print(train_processed.head())
    print(train_processed.dtypes)


if __name__ == '__main__':
    process_titanic_epoch3_1('data/raw/train.csv', 'data/raw/test.csv', 'data/processed/')
