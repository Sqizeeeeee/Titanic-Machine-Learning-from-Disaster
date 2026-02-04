import pandas as pd
import os


os.makedirs("data/processed", exist_ok=True)


# Working with train file

train = pd.read_csv("data/raw/train.csv")

columns = ["Age", "Sex", "Pclass", "Survived"]
train_small = train[columns]

train_ready = train_small.copy()

train_ready['Sex'] = train_ready['Sex'].map({'male': 0, 'female': 1})

median_age = train_ready['Age'].median()
train_ready['Age'] = train_ready['Age'].fillna(median_age)

print(train_ready.head())

train_ready.to_csv("data/processed/epoch1_train.csv", index=False)




# Working with test file

test = pd.read_csv("data/raw/test.csv")

columns = ["Age", "Sex", "Pclass", "PassengerId"]
test_small = test[columns].copy()

test_small['Sex'] = test_small['Sex'].map({'male': 0, 'female': 1})

test_small['Age'] = test_small['Age'].fillna(median_age)

test_small.to_csv("data/processed/epoch1_test.csv", index=False)

