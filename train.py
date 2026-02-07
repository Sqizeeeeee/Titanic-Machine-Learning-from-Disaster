import numpy as np
import pandas as pd
import json
import os

from model import RandomForestClassifier
from utils import load_epoch_data, train_val_split, accuracy_score

def main():

    X, y = load_epoch_data("data/processed/epoch5_train.csv")
    X_train, X_val, y_train, y_val = train_val_split(
        X, y, test_size=0.2, random_state=42, stratify=True
    )

    param_grid = {
        "n_estimators": [25, 50, 100],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 3, 5],
        "max_features": [None, int(np.sqrt(X.shape[1]))]
    }

    best_acc = 0
    best_params = None

    for n in param_grid["n_estimators"]:
        for d in param_grid["max_depth"]:
            for leaf in param_grid["min_samples_leaf"]:
                for mf in param_grid["max_features"]:
                    rf = RandomForestClassifier(
                        n_estimators=n,
                        max_depth=d,
                        min_samples_leaf=leaf,
                        max_features=mf,
                        random_state=42,
                    )
                    rf.fit(X_train, y_train)
                    preds = rf.predict(X_val)
                    acc = accuracy_score(y_val, preds)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {"n_estimators": n, "max_depth": d,
                                       "min_samples_leaf": leaf, "max_features": mf}

    print("Best params:", best_params)
    print("Best val accuracy:", best_acc)

    os.makedirs("submissions", exist_ok=True)
    with open("submissions/best_params.json", "w") as f:
        json.dump(best_params, f)

    rf_final = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        random_state=42,
    )
    rf_final.fit(X, y)


    test_df = pd.read_csv("data/processed/epoch5_test.csv")
    X_test = test_df.values
    preds_test = rf_final.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": pd.read_csv("data/raw/test.csv")["PassengerId"],
        "Survived": preds_test
    })
    submission_path = "submissions/epoch5_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
