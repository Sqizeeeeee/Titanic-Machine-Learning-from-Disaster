import pandas as pd

def add_features_epoch4(df: pd.DataFrame,
                        add_age_squared: bool = True,
                        add_sex_age: bool = True,
                        add_embarked_pclass: bool = True,
                        add_is_senior: bool = True,
                        add_is_child: bool = True) -> pd.DataFrame:
    """
    Добавляет новые признаки для Epoch 4 к базовому DataFrame.
    Все преобразования делаются на копии df, исходные данные не меняются.
    """

    df = df.copy()

    # -----------------------
    # Age_squared
    # -----------------------
    if add_age_squared:
        df["Age_squared"] = df["Age"] ** 2

    # -----------------------
    # Sex * Age
    # -----------------------
    if add_sex_age:
        df["Sex_Age"] = df["Sex"] * df["Age"]

    # -----------------------
    # Embarked * Pclass
    # -----------------------
    if add_embarked_pclass:
        # предполагаем, что Embarked_* уже one-hot, Pclass в df
        for col in df.columns:
            if col.startswith("Embarked_") and "Pclass" in df.columns:
                df[f"{col}_x_Pclass"] = df[col] * df["Pclass"]

    # -----------------------
    # IsSenior (Age >= 60)
    # -----------------------
    if add_is_senior:
        df["IsSenior"] = (df["Age"] >= 60).astype(int)

    # -----------------------
    # IsChild (Age <= 15)
    # -----------------------
    if add_is_child:
        df["IsChild"] = (df["Age"] <= 15).astype(int)

    return df
