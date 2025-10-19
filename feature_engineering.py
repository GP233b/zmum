
# ===============================================
# feature_engineering.py
# ===============================================
import pandas as pd

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    possible_date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    for c in possible_date_cols:
        try:
            df[c + '_parsed'] = pd.to_datetime(df[c], errors='coerce')
            df['year'] = df[c + '_parsed'].dt.year
            df['month'] = df[c + '_parsed'].dt.month
            df['day'] = df[c + '_parsed'].dt.day
            df['dayofweek'] = df[c + '_parsed'].dt.dayofweek
            df['hour'] = df[c + '_parsed'].dt.hour
            break
        except Exception:
            continue

    text_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].str.len().max() > 20]
    for tc in text_cols:
        df[tc + '_char_len'] = df[tc].fillna('').str.len()
        df[tc + '_word_count'] = df[tc].fillna('').str.split().apply(len)

    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['lat_lon'] = df['latitude'].astype(str) + '_' + df['longitude'].astype(str)
        freq = df['lat_lon'].value_counts().to_dict()
        df['latlon_freq'] = df['lat_lon'].map(freq)
    return df