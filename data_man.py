import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def fill_age_knn(df):
    # Для KNN берем числовые признаки + категориальные кодируем int
    tmp_df = df.copy()
    cat_cols = ['Sex', 'Title', 'Pclass']
    for col in cat_cols:
        tmp_df[col] = tmp_df[col].astype('category').cat.codes

    knn_cols = ['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
    imputer = KNNImputer(n_neighbors=5)
    tmp_df[knn_cols] = imputer.fit_transform(tmp_df[knn_cols])
    
    df['Age'] = tmp_df['Age']
    return df

def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

def create_age_group(age):
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teen'
    elif age <= 30:
        return 'YoungAdult'
    elif age <= 50:
        return 'Adult'
    else:
        return 'Senior'

def process_data_catboost(train_path, test_path, save_path_train, save_path_test):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    for df in [train_df, test_df]:
        # Title
        df['Title'] = df['Name'].apply(extract_title)
        
        # Family features
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(str)
        
        # Age_group
        df = fill_age_knn(df)
        df['Age_group'] = df['Age'].apply(lambda x: create_age_group(x) if not pd.isnull(x) else np.nan)
        # Fare_bin
        df['Fare_bin'] = pd.qcut(df['Fare'].fillna(df['Fare'].median()), 4, labels=False).astype(str)
        
        # Interaction features
        df['Title_Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
        df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
        df['IsAlone_AgeGroup'] = df['IsAlone'] + '_' + df['Age_group'].fillna('Unknown')
        
        # Fill missing Age with median for simplicity (CatBoost can handle missing too)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Columns to keep
    features = ['Pclass','Sex','Age','Fare','Embarked','Title','FamilySize','IsAlone',
                'Age_group','Fare_bin','Title_Pclass','Sex_Pclass','IsAlone_AgeGroup']
    
    train_df[features + ['Survived']].to_csv(save_path_train, index=False)
    test_df[features].to_csv(save_path_test, index=False)
    
    print(f"Processed data saved to {save_path_train} and {save_path_test}")

if __name__ == "__main__":
    process_data_catboost(
        train_path='data/raw/train.csv',
        test_path='data/raw/test.csv',
        save_path_train='data/processed/epoch6_train.csv',
        save_path_test='data/processed/epoch6_test.csv'
    )
