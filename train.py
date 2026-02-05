import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import MyModel, SKLRWrapper
from utils import load_processed_train, load_processed_test, save_submission, count_disagreements

def synchronize_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Приводим train и test к одному набору колонок после one-hot encoding"""
    for c in train_df.columns:
        if c not in test_df.columns:
            test_df[c] = 0
    for c in test_df.columns:
        if c not in train_df.columns:
            train_df[c] = 0
    train_df = train_df[sorted(train_df.columns)]
    test_df = test_df[sorted(train_df.columns)]
    return train_df, test_df

def train_manual_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> MyModel:
    l2_candidates = [0.0, 0.01, 0.1]
    best_acc = 0
    best_model = None
    for l2 in l2_candidates:
        model = MyModel(lr=0.02, epochs=3000, l2_lambda=l2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = np.mean(y_pred == y_val)
        print(f"Manual Model: L2 Lambda={l2}, Validation accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
    print(f"\nBest Manual Model: L2={best_model.l2_lambda}, Validation accuracy={best_acc:.4f}")
    print(f"Weights: {best_model.w}")
    print(f"Bias: {best_model.b}")
    return best_model

def train_skl_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> SKLRWrapper:
    C_candidates = [0.1, 1.0, 10.0]
    best_acc = 0
    best_model = None
    for C in C_candidates:
        model = SKLRWrapper(C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = np.mean(y_pred == y_val)
        print(f"SKLearn Model: C={C}, Validation accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
    print(f"\nBest SKLearn Model: C={best_model.C}, Validation accuracy={best_acc:.4f}")
    print(f"Weights: {best_model.model.coef_}")
    print(f"Bias: {best_model.model.intercept_}")
    return best_model

def main():
    # 1. Загружаем обработанные данные Epoch 3
    X_train_df, X_val_df, y_train, y_val = load_processed_train("data/processed/epoch3_train.csv")
    X_test_df, passenger_ids = load_processed_test("data/processed/epoch3_test.csv")

    # 2. Синхронизируем колонки после one-hot
    X_train_df, X_test_df = synchronize_columns(X_train_df, X_test_df)
    X_train_df, X_val_df = synchronize_columns(X_train_df, X_val_df)

    # 3. Нормализуем
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)

    # 4. Обучаем модели
    best_manual_model = train_manual_model(X_train_scaled, y_train.to_numpy(), X_val_scaled, y_val.to_numpy())
    best_skl_model = train_skl_model(X_train_scaled, y_train.to_numpy(), X_val_scaled, y_val.to_numpy())

    # 5. Сравниваем предсказания
    manual_val_pred = best_manual_model.predict(X_val_scaled)
    skl_val_pred = best_skl_model.predict(X_val_scaled)
    diff = count_disagreements(manual_val_pred, skl_val_pred)
    print(f"\nNumber of different predictions between best models: {diff}")

    # 6. Сохраняем submission
    manual_preds = best_manual_model.predict(X_test_scaled)
    skl_preds = best_skl_model.predict(X_test_scaled)

    save_submission(passenger_ids, manual_preds, dir_path="submissions", filename="epoch3_submission_manual.csv")
    save_submission(passenger_ids, skl_preds, dir_path="submissions", filename="epoch3_submission_skl.csv")
    print("Submissions saved.")

if __name__ == "__main__":
    main()
