import numpy as np
from model import MyModel, SKLRWrapper
from utils import load_processed_train, load_processed_test, save_submission, count_disagreements, normalize_features





def run_sklearn_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """
    Обучает SKLRWrapper один раз, выводит веса и accuracy
    """
    skl_model = SKLRWrapper(epochs=1000)
    skl_model.fit(X_train, y_train)
    y_val_pred = skl_model.predict(X_val)
    acc = (y_val_pred == y_val).mean()
    print("SKLearn Wrapper Model:")
    print("Weights:", skl_model.w)
    print("Bias:", skl_model.b)
    print(f"Validation accuracy: {acc:.4f}\n")
    return skl_model


def save_predictions(model, X_test: np.ndarray, passenger_ids: np.ndarray, filename: str):
    """
    Делает предсказание и сохраняет submission
    """
    predictions = model.predict(X_test)
    save_submission(passenger_ids=passenger_ids, predictions=predictions, dir_path="submissions", filename=filename)
    print(f"Saved submission: submissions/{filename}\n")


def main() -> None:
    # 1. Загружаем train
    X_train, X_val, y_train, y_val = load_processed_train("data/processed/epoch1_train.csv")

    # 2. Нормализация
    X_train_scaled, X_val_scaled = normalize_features(X_train, X_val)

    # 3. Parameter tuning для ручной модели
    learning_rates = [0.01, 0.02, 0.05]
    epochs_list = [1000, 2000, 3000]

    best_acc = 0.0
    best_lr = None
    best_epochs = None
    best_w = None
    best_b = None

    for lr in learning_rates:
        for epochs in epochs_list:
            model = MyModel(lr=lr, epochs=epochs)
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            acc = (y_val_pred == y_val).mean()
            print(f"Manual Model: lr={lr}, epochs={epochs}, Validation accuracy={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
                best_epochs = epochs
                best_w = model.w.copy()
                best_b = model.b

    print("\nBest manual model:")
    print(f"lr={best_lr}, epochs={best_epochs}, Validation accuracy={best_acc:.4f}")
    print("Weights:", best_w)
    print("Bias:", best_b)

    # =====================================
    # 4. Обучение SKLRWrapper один раз
    # =====================================
    skl_model = run_sklearn_model(X_train_scaled, y_train, X_val_scaled, y_val)

    # =====================================
    # 5. Загружаем test
    # =====================================
    X_test, passenger_ids = load_processed_test("data/processed/epoch1_test.csv")
    # Нормализуем test по статистике train
    X_test_scaled = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)

    # =====================================
    # 6. Сохраняем submission для лучших моделей
    # =====================================
    best_manual_model = MyModel(lr=best_lr, epochs=best_epochs)
    best_manual_model.w = best_w
    best_manual_model.b = best_b

    save_predictions(best_manual_model, X_test_scaled, passenger_ids, "epoch1_submission_manual_best.csv")
    save_predictions(skl_model, X_test_scaled, passenger_ids, "epoch1_submission_skl.csv")


if __name__ == "__main__":
    main()