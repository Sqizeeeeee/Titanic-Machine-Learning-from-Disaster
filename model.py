from catboost import CatBoostClassifier
from typing import Optional, List


class CatBoostModel:

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 4,
        l2_leaf_reg: float = 3.0,
        bagging_temperature: float = 1.0,
        rsm: float = 1.0,
        random_seed: int = 42,
        cat_features: Optional[List[int]] = None,
        verbose: int = 100,
    ):
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "bagging_temperature": bagging_temperature,
            "rsm": rsm,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": random_seed,
            "verbose": verbose,
        }

        self.cat_features = cat_features
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            cat_features=self.cat_features,
            eval_set=(X_val, y_val) if X_val is not None else None,
            use_best_model=X_val is not None,
        )

    def predict(self, X):
        return self.model.predict(X).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
