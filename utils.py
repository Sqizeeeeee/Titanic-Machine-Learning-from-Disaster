import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_processed_train(path: str):
    """
    Загружает processed train csv и возвращает X_train, X_val, y_train, y_val
    """
    df = pd.read_csv(path)
    X = df.drop(columns="Survived")
    y = df["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_val, y_train, y_val

def load_processed_test(path: str):
    """
    Загружает processed test csv и возвращает X и passenger_ids
    """
    df = pd.read_csv(path)
    if "PassengerId" in df.columns:
        passenger_ids = df["PassengerId"].values
        X = df.drop(columns="PassengerId")
    else:
        # Если колонка PassengerId отсутствует
        passenger_ids = np.arange(len(df))
        X = df.copy()

    return X, passenger_ids

def save_submission(passenger_ids: np.ndarray, predictions: np.ndarray, dir_path: str, filename: str) -> None:
    import pandas as pd
    from pathlib import Path

    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Приводим к int
    passenger_ids = passenger_ids.astype(int)

    submission_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions.astype(int)
    })
    submission_path = dir_path / filename
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission: {submission_path}")


def count_disagreements(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Считает количество несовпадений предсказаний и истины
    """
    return int(np.sum(y_true != y_pred))

def count_changes(y_old: np.ndarray, y_new: np.ndarray):
    """
    Считает сколько предсказаний поменялось:
    - total: общее количество изменений
    - zero_to_one: количество изменений с 0 на 1
    - one_to_zero: количество изменений с 1 на 0
    """
    if len(y_old) != len(y_new):
        raise ValueError("Arrays must have the same length")
    total = np.sum(y_old != y_new)
    zero_to_one = np.sum((y_old == 0) & (y_new == 1))
    one_to_zero = np.sum((y_old == 1) & (y_new == 0))
    return int(total), int(zero_to_one), int(one_to_zero)
