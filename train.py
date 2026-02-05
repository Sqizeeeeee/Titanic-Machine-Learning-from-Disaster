import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from model import MyModel, SKLRWrapper
from features import add_features_epoch4

def prepare_epoch4_submission(train_csv: str, test_csv: str, submission_path: str, desc: str):
    
    # 1. Загружаем данные
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    y_train = train_df["Survived"].to_numpy()
    X_train = train_df.drop(columns=["Survived", "PassengerId"])
    X_test = test_df.drop(columns=["PassengerId"], errors="ignore")

    
    # 2. Добавляем все признаки
    X_train = add_features_epoch4(X_train)
    X_test = add_features_epoch4(X_test)

    
    # 3. One-hot encoding для категориальных (Embarked, Title и др.)
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Согласуем колонки
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    
    # 4. Нормализация
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    # 5. Обучаем Manual модель
    manual_model = MyModel(lr=0.01, epochs=3000, l2_lambda=0.01)
    manual_model.fit(X_train_scaled, y_train)
    manual_preds = manual_model.predict(X_test_scaled)

    
    # 6. Обучаем SKLearn модель
    skl_model = SKLRWrapper(C=1.0, max_iter=5000)
    skl_model.fit(X_train_scaled, y_train)
    skl_preds = skl_model.predict(X_test_scaled)

    
    # 7. Сохраняем submission
    sub_manual = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": manual_preds
    })
    sub_skl = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": skl_preds
    })

    sub_manual.to_csv(f"submissions/epoch4_submission_manual_{desc}.csv", index=False)
    sub_skl.to_csv(f"submissions/epoch4_submission_skl_{desc}.csv", index=False)

    print(f"Epoch 4 submissions ({desc}) saved!")


if __name__ == "__main__":
    # Median age
    prepare_epoch4_submission(
        "data/processed/epoch4_train_median.csv",
        "data/processed/epoch4_test_median.csv",
        "submissions/epoch4_submission_median.csv",
        desc="median"
    )

    # KNN age
    prepare_epoch4_submission(
        "data/processed/epoch4_train_knn.csv",
        "data/processed/epoch4_test_knn.csv",
        "submissions/epoch4_submission_knn.csv",
        desc="knn"
    )
