import os
import json
import pandas as pd
import time
from model import CatBoostModel
from utils import load_epoch_data, get_cat_features, train_val_split, accuracy_score


param_grid = {
    "iterations": [200, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05],
    "depth": [3, 4, 6],
    "l2_leaf_reg": [1, 3, 5, 7],
    "bagging_temp_grid": [0.0, 0.5, 1.0],
    "rsm": [0.7, 0.8, 1.0],
}


X, y = load_epoch_data("data/processed/epoch6_train.csv")
cat_features = get_cat_features(pd.DataFrame(X))
X_train, X_val, y_train, y_val = train_val_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = []
best_acc = 0.0
best_params = None

# -----------------------------
# Grid Search
# -----------------------------
total = (
    len(param_grid["iterations"]) *
    len(param_grid["learning_rate"]) *
    len(param_grid["depth"]) *
    len(param_grid["l2_leaf_reg"]) *
    len(param_grid["bagging_temp_grid"]) *
    len(param_grid["rsm"])
)
print(f"[grid] total combinations: {total}")


start_time = time.time()
count = 0

for it in param_grid["iterations"]:
    for lr in param_grid["learning_rate"]:
        for dp in param_grid["depth"]:
            for l2 in param_grid["l2_leaf_reg"]:
                for bt in param_grid["bagging_temp_grid"]:
                    for rsm in param_grid["rsm"]:
                        count += 1

                        elapsed = time.time() - start_time
                        avg_time = elapsed / count
                        eta = avg_time * (total - count)

                        print(
                            f"[grid] {count}/{total} ({count / total:.1%}) — "
                            f"it={it}, lr={lr}, depth={dp}, l2={l2}, "
                            f"bagging_temperature={bt}, rsm={rsm} | "
                            f"elapsed={elapsed/60:.1f}m, ETA={eta/60:.1f}m"
                        )

                        model = CatBoostModel(
                            iterations=it,
                            learning_rate=lr,
                            depth=dp,
                            l2_leaf_reg=l2,
                            bagging_temperature=bt,
                            rsm=rsm,
                            random_seed=42,
                            cat_features=cat_features
                        )

                        model.fit(X_train, y_train, X_val, y_val)
                        preds = model.predict(X_val)
                        acc = accuracy_score(y_val, preds)

                        results.append({
                            "iterations": it,
                            "learning_rate": lr,
                            "depth": dp,
                            "l2_leaf_reg": l2,
                            "bagging_temperature": bt,
                            "rsm": rsm,
                            "val_accuracy": acc
                        })

                        if acc > best_acc:
                            best_acc = acc
                            best_params = results[-1]

# -----------------------------
# Save results
# -----------------------------
os.makedirs("grid_results", exist_ok=True)
pd.DataFrame(results).to_csv("grid_results/catboost_grid.csv", index=False)

with open("grid_results/best_catboost_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

print(f"[grid] Best params: {best_params}")
print(f"[grid] Best val accuracy: {best_acc}")
