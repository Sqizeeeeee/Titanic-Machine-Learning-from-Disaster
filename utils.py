import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def load_processed_train(path: str):
    """
    Загружает train, добавляет новые признаки и разделяет на X и y
    Возвращает X_train, X_val, y_train, y_val как np.ndarray
    """
    df = pd.read_csv(path)
    X = df[["Age", "Sex", "Pclass", "Fare", "FamilySize"]].values
    y = df["Survived"].values

    # Разделяем на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Нормализация числовых признаков (Age, Fare, FamilySize)
    scaler = StandardScaler()
    X_train[:, [0, 3, 4]] = scaler.fit_transform(X_train[:, [0, 3, 4]])
    X_val[:, [0, 3, 4]] = scaler.transform(X_val[:, [0, 3, 4]])

    return X_train, X_val, y_train, y_val
    


def load_processed_test(path: str):
    """
    Загружает test, добавляет новые признаки и возвращает X и PassengerId
    """
    df = pd.read_csv(path)
    passenger_ids = df["PassengerId"].values
    X = df[["Age", "Sex", "Pclass", "Fare", "FamilySize"]].values

    # Нормализация числовых признаков (Age, Fare, FamilySize)
    scaler = StandardScaler()
    X[:, [0, 3, 4]] = scaler.fit_transform(X[:, [0, 3, 4]])

    return X, passenger_ids


def save_submission(
    passenger_ids: np.ndarray,
    predictions: np.ndarray,
    dir_path: str,
    filename: str = "submission.csv"
) -> None:
    os.makedirs(dir_path, exist_ok=True)

    full_path = os.path.join(dir_path, filename)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })

    submission.to_csv(full_path, index=False)

def count_disagreements(pred1: np.ndarray, pred2: np.ndarray) -> int:
    """
    Считает, сколько раз две модели предсказали разные значения
    """
    return np.sum(pred1 != pred2)

def count_changes(y_old: np.ndarray, y_new: np.ndarray):
    """Считает общее количество изменений и отдельно 0->1 и 1->0"""
    total = np.sum(y_old != y_new)
    zero_to_one = np.sum((y_old == 0) & (y_new == 1))
    one_to_zero = np.sum((y_old == 1) & (y_new == 0))
    return total, zero_to_one, one_to_zero


