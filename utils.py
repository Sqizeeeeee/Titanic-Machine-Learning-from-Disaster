import pandas as pd
from typing import List, Optional, Tuple

def load_processed_data(
    path: str,
    return_ids: bool = False,
    id_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    
    df = pd.read_csv(path)
    
    ids = None
    if return_ids:
        if id_col is None:
            raise ValueError("Для возврата id необходимо указать 'id_col'")
        ids = df[id_col]
        df = df.drop(columns=[id_col])
    
    return (df, ids) if return_ids else df


def get_cat_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    
    exclude_cols = exclude_cols or []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Исключаем указанные колонки
    cat_cols = [c for c in cat_cols if c not in exclude_cols]
    return cat_cols


def cast_cat_to_str(df: pd.DataFrame, cat_features: List[str]) -> pd.DataFrame:
    
    for c in cat_features:
        df[c] = df[c].astype(str)
    return df
