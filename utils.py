import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

def load_epoch_data(path):

    df = pd.read_csv(path)
    if 'Survived' in df.columns:
        X = df.drop(columns=['Survived'])
        y = df['Survived'].values
        return X, y
    else:
        return df

def get_cat_features(df):

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    return cat_cols

def train_val_split(X, y, test_size=0.2, random_state=42, stratify=None):

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_val, y_train, y_val

def cross_val_predict(model_class, X, y, folds=5, cat_features=None, **model_params):

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    val_preds = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(cat_features=cat_features, **model_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        val_preds[val_idx] = model.predict(X_val)

    acc = accuracy_score(y, val_preds)
    return val_preds, acc
