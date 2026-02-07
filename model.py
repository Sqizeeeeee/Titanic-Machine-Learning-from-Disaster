import numpy as np
from typing import Optional, List
from collections import Counter
from sklearn.ensemble import RandomForestClassifier as SklearnRF


# ============================================================
# Tree node
# ============================================================

class Node:
    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[int] = None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None


# ============================================================
# Decision Tree (CART, Gini)
# ============================================================

class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root: Optional[Node] = None

    # -----------------------------
    # Public API
    # -----------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(x, self.root) for x in X])

    # -----------------------------
    # Internal logic
    # -----------------------------

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (
            num_labels == 1
            or n_samples < self.min_samples_leaf
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return Node(value=self._most_common_label(y))

        best_idx, best_thr = self._best_split(X, y, n_features)

        if best_idx is None:
            return Node(value=self._most_common_label(y))

        left_mask = X[:, best_idx] <= best_thr
        right_mask = X[:, best_idx] > best_thr

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_index=best_idx,
            threshold=best_thr,
            left=left,
            right=right,
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray, n_features: int):
        best_gini = 1.0
        split_idx, split_thr = None, None

        feature_indices = np.arange(n_features)
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])

            for i in range(len(thresholds) - 1):
                thr = (thresholds[i] + thresholds[i + 1]) / 2
                left_mask = X[:, feature_index] <= thr
                right_mask = X[:, feature_index] > thr

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                gini = self._gini_index(y[left_mask], y[right_mask])

                if gini < best_gini:
                    best_gini = gini
                    split_idx = feature_index
                    split_thr = thr

        return split_idx, split_thr

    @staticmethod
    def _gini_index(y_left: np.ndarray, y_right: np.ndarray) -> float:
        n = len(y_left) + len(y_right)
        return (
            len(y_left) / n * DecisionTreeClassifier._gini(y_left)
            + len(y_right) / n * DecisionTreeClassifier._gini(y_right)
        )

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    @staticmethod
    def _most_common_label(y: np.ndarray) -> int:
        return Counter(y).most_common(1)[0][0]

    def _predict_single(self, x: np.ndarray, node: Node) -> int:
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


# ============================================================
# Random Forest (manual)
# ============================================================

class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 3,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees: List[DecisionTreeClassifier] = []

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(self._majority_vote, axis=0, arr=predictions)

    @staticmethod
    def _majority_vote(x: np.ndarray) -> int:
        return Counter(x).most_common(1)[0][0]


# ============================================================
# Sklearn RF wrapper
# ============================================================

class SklearnRandomForestWrapper:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        self.model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
