import os
import pandas as pd
from model import CatBoostModel
from utils import get_cat_features


TRAIN_PATH = "data/processed/epoch6_train.csv"
TEST_PATH = "data/processed/epoch6_test.csv"
SUB_DIR = "submissions"

os.makedirs(SUB_DIR, exist_ok=True)


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["Survived"])
y_train = train_df["Survived"].values

X_test = test_df.copy()

cat_features = get_cat_features(X_train)

# cast categorical to str
for c in cat_features:
    X_train[c] = X_train[c].astype(str)
    X_test[c] = X_test[c].astype(str)


# Lite params

model = CatBoostModel(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=9,
    bagging_temperature=1.0,
    rsm=1.0,
    random_seed=42,
    cat_features=cat_features,
    verbose=100
)


print("Training final CatBoost on full data...")
model.fit(X_train, y_train)


test_proba = model.predict_proba(X_test)[:, 1]


raw_test = pd.read_csv("data/raw/test.csv")
passenger_ids = raw_test["PassengerId"].values


thresholds = [0.50]

for th in thresholds:
    preds = (test_proba >= th).astype(int)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds
    })

    fname = f"{SUB_DIR}/epoch6_1_submission_literules.csv"
    submission.to_csv(fname, index=False)

    print(f"Saved submission: {fname}")
