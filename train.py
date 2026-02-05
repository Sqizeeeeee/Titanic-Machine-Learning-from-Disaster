import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import MyModel, SKLRWrapper
from utils import load_processed_train, load_processed_test, save_submission, count_disagreements




# 1. Load processed data
df = pd.read_csv("data/processed/epoch3_1_train.csv")

y = df["Survived"].to_numpy()
X = df.drop(columns=["Survived"]).to_numpy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# 3. Manual Logistic Regression Grid
l2_lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
epochs_list = [200, 500, 1000, 3000]

best_manual_acc = -1
best_manual_model = None
print("\n=== Manual Logistic Regression Grid Search ===")
for lr in learning_rates:
    for l2 in l2_lambdas:
        for ep in epochs_list:
            model = MyModel(lr=lr, epochs=ep, l2_lambda=l2)
            try:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_val_scaled)
                acc = np.mean(preds == y_val)
                print(f"lr={lr:.3f}, L2={l2}, epochs={ep}, val_acc={acc:.4f}")
                if acc > best_manual_acc:
                    best_manual_acc = acc
                    best_manual_model = model
            except Exception as e:
                print(f"lr={lr:.3f}, L2={l2}, epochs={ep} -> ERROR: {e}")

print(f"\nBest Manual Model: val_acc={best_manual_acc:.4f}")


# 4. SKLearn Logistic Regression
model_skl = SKLRWrapper(C=0.1, max_iter=5000)
model_skl.fit(X_train_scaled, y_train)
preds_skl = model_skl.predict(X_val_scaled)
acc_skl = np.mean(preds_skl == y_val)
print(f"\nSKLearn Model: C=0.1, Validation accuracy={acc_skl:.4f}")


# 5. Compare best models
best_manual_preds = best_manual_model.predict(X_val_scaled)
diff = np.sum(best_manual_preds != preds_skl)
print(f"\nNumber of different predictions between best manual model and SKLearn: {diff}")
