import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR

class MyModel:
    w: np.ndarray | None
    b: float | None
    lr: float
    epochs: int


    def __init__(self, lr: float = 0.01, epochs: int = 1000) -> None:

        self.lr = lr
        self.epochs = epochs
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

            d_w = (1 / n_samples) * (X.T @ (y_prediction - y))
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

    def __init__(self, lr: float = 0.01, epochs: int = 1000) -> None:
        self.lr = lr
        self.epochs = epochs
        self.model: SklearnLR = SklearnLR(
            random_state=42,
            max_iter=epochs
        )
        self.w: np.ndarray | None = None
        self.b: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.w = self.model.coef_.flatten()
        self.b = self.model.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)












