import numpy as np
from sklearn.linear_model import LogisticRegression

class MyModel:
    w: np.ndarray | None
    b: float | None
    lr: float
    epochs: int
    l2_lambda: float


    def __init__(self, lr: float = 0.01, epochs: int = 1000, l2_lambda: float = 0.0) -> None:

        self.lr = lr
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.w = None
        self.b = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0


        for _ in range(self.epochs):
            z = X @ self.w + self.b  # type: np.ndarray

            y_prediction = self.sigmoid(z)

            d_w = (1 / n_samples) * (X.T @ (y_prediction - y)) + self.l2_lambda * self.w
            d_b = (1 / n_samples) * np.sum(y_prediction - y)

            self.w -= self.lr * d_w
            self.b -= self.lr * d_b

        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs: np.ndarray = self.predict_proba(X)
        return (probs >= threshold).astype(int)

# --------------------------------------------------------------------------------------------------


class SKLRWrapper:
    """
    Обёртка для sklearn LogisticRegression с L2 регуляризацией без FutureWarning
    """
    def __init__(self, C: float = 1.0, max_iter: int = 5000):
        # L2 регуляризация через l1_ratio=0
        self.model = LogisticRegression(
            l1_ratio=0,   # L2 регуляризация
            C=C,          # сила регуляризации
            max_iter=max_iter,
            solver='lbfgs'
        )
        self.w: np.ndarray | None = None
        self.b: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.w = self.model.coef_.flatten()
        self.b = self.model.intercept_[0]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
