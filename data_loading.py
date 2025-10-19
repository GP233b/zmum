
# ===============================================
# data_loading.py
# ===============================================
import pandas as pd
import numpy as np
import random
import os

def load_and_prepare_data(filename: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Plik '{filename}' nie znaleziony. Pobierz go z Kaggle i umieść w tym samym katalogu.")

    df = pd.read_csv(filename)
    print(f"Wczytano dane: {df.shape}")

    total_missing = df.isna().sum().sum()
    if total_missing == 0:
        print("Brak braków danych — generuję 10% pustych komórek.")
        n_to_blank = int(0.10 * df.size)
        for _ in range(n_to_blank):
            i = random.randrange(df.shape[0])
            j = random.randrange(df.shape[1])
            df.iat[i, j] = np.nan

    print("Podsumowanie braków danych:")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    return df
