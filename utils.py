import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as skl_accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os



def load_epoch_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y



def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    stratify_y = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )



def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return skl_accuracy_score(y_true, y_pred)


def roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """
    y_proba: probabilities for class 1
    """
    return roc_auc_score(y_true, y_proba)



def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
):
    """
    Saves confusion matrix plot to plots/
    """
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(np.unique(y_true)))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename))
    plt.close()
