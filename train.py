import numpy as np
import pandas as pd
from model import MyModel, SKLRWrapper
from utils import load_processed_train, load_processed_test, save_submission, count_disagreements, count_changes



def train_manual(X_train, X_val, y_train, y_val, lr=0.02, epochs=3000, l2_lambda=0.01):
    model = MyModel(lr=lr, epochs=epochs, l2_lambda=l2_lambda)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    acc = np.mean(y_val_pred == y_val)
    print(f"\nManual Model: L2 Lambda={l2_lambda}, Validation accuracy={acc:.4f}")
    return model, y_val_pred, acc

def train_skl(X_train, X_val, y_train, y_val, C=0.1):
    skl_model = SKLRWrapper(C=C)
    skl_model.fit(X_train, y_train)
    y_val_pred = skl_model.predict(X_val)
    acc = np.mean(y_val_pred == y_val)
    print(f"\nSKLearn Model: C={C}, Validation accuracy={acc:.4f}")
    return skl_model, y_val_pred, acc

def main():
    # Загружаем эпоху 2
    X_train, X_val, y_train, y_val = load_processed_train("data/processed/epoch2_train.csv")
    X_test, passenger_ids = load_processed_test("data/processed/epoch2_test.csv")

    # Параметры для тюнинга
    l2_values = [0.0, 0.01, 0.1]
    C_values = [0.1, 1.0, 10.0]

    # ---------------- Manual Model ----------------
    best_manual_acc = 0.0
    for l2 in l2_values:
        model_manual, y_val_pred, acc = train_manual(X_train, X_val, y_train, y_val, l2_lambda=l2)
        if acc > best_manual_acc:
            best_manual_acc = acc
            best_manual_model = model_manual
            best_manual_y_val = y_val_pred
            best_l2 = l2

    print(f"\nBest Manual Model: L2={best_l2}, Validation accuracy={best_manual_acc:.4f}")

    # ---------------- SKLearn Model ----------------
    best_skl_acc = 0.0
    for C in C_values:
        skl_model, y_val_pred_skl, acc = train_skl(X_train, X_val, y_train, y_val, C=C)
        if acc > best_skl_acc:
            best_skl_acc = acc
            best_skl_model = skl_model
            best_skl_y_val = y_val_pred_skl
            best_C = C

    print(f"\nBest SKLearn Model: C={best_C}, Validation accuracy={best_skl_acc:.4f}")

    # ---------------- Сравнение предсказаний на validation ----------------
    disagreements = count_disagreements(best_manual_y_val, best_skl_y_val)
    print(f"\nNumber of different predictions between best models: {disagreements}")

    

    # ---------------- Сохраняем submission ----------------
    y_test_pred_manual = best_manual_model.predict(X_test)
    y_test_pred_skl = best_skl_model.predict(X_test)

    save_submission(passenger_ids, y_test_pred_manual, "submissions", "epoch2_submission_manual.csv")
    save_submission(passenger_ids, y_test_pred_skl, "submissions", "epoch2_submission_skl.csv")

if __name__ == "__main__":
    main()