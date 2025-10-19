
# ===============================================
# analysis.py
# ===============================================
import pandas as pd

def run_pandas_analysis(df: pd.DataFrame):
    if 'state' in df.columns:
        print(df.groupby('state').size().sort_values(ascending=False).head(10))
    if 'year' in df.columns:
        print(df.groupby('year').size().head(10))
    if 'cause' in df.columns and 'area' in df.columns:
        print(df.groupby('cause')['area'].mean().sort_values(ascending=False).head(10))