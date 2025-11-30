# ===============================================
# preprocessing.py
# ===============================================
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder
import os


def impute_simple(df: pd.DataFrame) -> pd.DataFrame:
    df_filled = df.copy()

    for c in df_filled.select_dtypes(include=[np.number]).columns:
        df_filled[c] = df_filled[c].fillna(df_filled[c].median())

    for c in df_filled.select_dtypes(include=['object']).columns:
        mode_val = df_filled[c].mode().iloc[0] if not df_filled[c].mode().empty else 'MISSING'
        df_filled[c] = df_filled[c].fillna(mode_val)

    return df_filled


def impute_ffill(df: pd.DataFrame) -> pd.DataFrame:
    df_filled = df.copy()
    df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    return df_filled


def impute_mice(df: pd.DataFrame) -> pd.DataFrame:
    df_filled = df.copy()

    for c in df_filled.select_dtypes(include=['object']).columns:
        df_filled[c] = df_filled[c].astype('category').cat.codes.replace({-1: np.nan})

    num_data = df_filled.select_dtypes(include=[np.number])
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=5)

    try:
        imputed = imp.fit_transform(num_data)
        df_filled[num_data.columns] = imputed
    except Exception as e:
        print('Błąd w IterativeImputer:', e)

    return df_filled


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


def impute_in_batches(df: pd.DataFrame, output_dir: str = 'datasets_out', method: str = "simple", batch_size: int = 100_000):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_rows = len(df)
    num_batches = (total_rows // batch_size) + int(total_rows % batch_size > 0)

    if method == "simple":
        impute_func = impute_simple
    elif method == "ffill":
        impute_func = impute_ffill
    elif method == "mice":
        impute_func = impute_mice
    else:
        raise ValueError("Nieznana metoda imputacji! Dozwolone: 'simple', 'ffill', 'mice'")

    output_path = os.path.join(output_dir, f"{method}_full.csv")

    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Start imputacji metodą: '{method}' w {num_batches} batch'ach po {batch_size} wierszy...")

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total_rows)
        batch = df.iloc[start:end].copy()

        print(f"Batch {i+1}/{num_batches} ({start}-{end})")

        imputed_df = impute_func(batch)

        if i == 0:
            imputed_df.to_csv(output_path, index=False)
        else:
            imputed_df.to_csv(output_path, mode='a', index=False, header=False)

        print(f"Zapisano batch {i+1}/{num_batches}")

    print(f"Wszystkie batch'e przetworzone. Wynik zapisany w: {output_path}")

    return pd.read_csv(output_path)