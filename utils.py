import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split



def load_processed_train(path: str):

    df = pd.read_csv(path)
    X = df[["Age", "Sex", "Pclass"]].to_numpy()
    y = df["Survived"].to_numpy()


    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_val, y_train, y_val
    


def load_processed_test(path: str):
    df = pd.read_csv(path)
    X = df[["Age", "Sex", "Pclass"]].to_numpy()
    passenger_ids = df["PassengerId"].to_numpy()
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


def normalize_features(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Стандартизируем признаки: (X - mean) / std
    Возвращаем нормализованные X_train и X_val
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    return X_train_scaled, X_val_scaled
