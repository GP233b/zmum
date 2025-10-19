

# ===============================================
# preprocessing.py
# ===============================================
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def impute_missing_values(df: pd.DataFrame):
    simple_fill_df = df.copy()
    ffill_df = df.copy()
    mice_df = df.copy()

    for c in simple_fill_df.select_dtypes(include=[np.number]).columns:
        simple_fill_df[c] = simple_fill_df[c].fillna(simple_fill_df[c].median())

    for c in simple_fill_df.select_dtypes(include=['object']).columns:
        mode_val = simple_fill_df[c].mode().iloc[0] if not simple_fill_df[c].mode().empty else 'MISSING'
        simple_fill_df[c] = simple_fill_df[c].fillna(mode_val)

    ffill_df = ffill_df.fillna(method='ffill').fillna(method='bfill')

    mice_work = mice_df.copy()
    for c in mice_work.select_dtypes(include=['object']).columns:
        mice_work[c] = mice_work[c].astype('category').cat.codes.replace({-1: np.nan})

    num_data = mice_work.select_dtypes(include=[np.number])
    imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=20), max_iter=5)
    try:
        imputed = imp.fit_transform(num_data)
        mice_df[num_data.columns] = imputed
    except Exception as e:
        print('Błąd w IterativeImputer:', e)

    return simple_fill_df, ffill_df, mice_df

def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        if df[c].nunique() < 15:
            le = LabelEncoder()
            df[c + '_le'] = le.fit_transform(df[c].fillna('MISSING'))
        else:
            freq = df[c].value_counts(normalize=True)
            df[c + '_freqenc'] = df[c].map(freq).fillna(0)
    return df
