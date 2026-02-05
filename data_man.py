import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer

def process_titanic_epoch4(train_path: str, test_path: str, processed_dir: str) -> None:
    """
    Epoch 4: базовые признаки + два варианта Age (median и KNN)
    Сохраняем train и test для двух баз
    """

    processed_dir_path = Path(processed_dir)
    processed_dir_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for df_name, df in zip(["train", "test"], [train_df, test_df]):

        # -----------------------
        # 1. Пол
        # -----------------------
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

        # -----------------------
        # 2. FamilySize и IsAlone
        # -----------------------
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

        # -----------------------
        # 3. Title
        # -----------------------
        common_titles = ["Mr", "Miss", "Mrs", "Master"]
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
        df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")
        title_dummies = pd.get_dummies(df["Title"], prefix="Title")
        df = pd.concat([df, title_dummies], axis=1)

        # -----------------------
        # 4. Embarked
        # -----------------------
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
        df = pd.concat([df, embarked_dummies], axis=1)

        # -----------------------
        # 5. Удаляем ненужные колонки
        # -----------------------
        drop_cols = ["Name", "Title", "Ticket", "Cabin", "SibSp", "Parch", "Fare", "Pclass"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # -----------------------
        # 6. Age - два варианта
        # -----------------------
        # 6a. Median
        df_median = df.copy()
        df_median["Age"] = df_median["Age"].fillna(df_median["Age"].median())

        # 6b. KNN
        df_knn = df.copy()
        knn_features = ["Sex", "FamilySize", "IsAlone"]  # можно добавить Title_*, Embarked_* для KNN
        for col in df.columns:
            if col.startswith("Title_") or col.startswith("Embarked_"):
                knn_features.append(col)
        knn_features.append("Age")

        knn_imputer = KNNImputer(n_neighbors=7)
        knn_arr = knn_imputer.fit_transform(df_knn[knn_features])
        df_knn["Age"] = knn_arr[:, knn_features.index("Age")]

        # -----------------------
        # 7. Сохраняем
        # -----------------------
        df_median.to_csv(processed_dir_path / f"epoch4_{df_name}_median.csv", index=False)
        df_knn.to_csv(processed_dir_path / f"epoch4_{df_name}_knn.csv", index=False)

    print("Epoch 4 processed data saved (median and KNN Age)")

if __name__ == "__main__":
    process_titanic_epoch4(
        train_path="data/raw/train.csv",
        test_path="data/raw/test.csv",
        processed_dir="data/processed"
    )