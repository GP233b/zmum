
# ===============================================
# visualization.py
# ===============================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_advanced_plot(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    if 'year' in df.columns:
        df.groupby('year').size().plot(ax=axes[0, 0], title='Liczba pożarów per rok')
    if 'area' in df.columns:
        axes[0, 1].hist(np.log1p(df['area'].dropna()), bins=50)
        axes[0, 1].set_title('Histogram log(1+area)')
    if 'cause' in df.columns:
        df['cause'].value_counts().head(10).plot(kind='bar', ax=axes[1, 0], title='Top przyczyn')
    numcols = df.select_dtypes(include=[np.number]).columns
    if len(numcols) > 1:
        corr = df[numcols].corr()
        im = axes[1, 1].imshow(corr)
        axes[1, 1].set_title('Macierz korelacji')
        fig.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig('advanced_plot.png', dpi=150)


def generate_histograms(df: pd.DataFrame):
    for c in ['area', 'year', 'month', 'hour']:
        if c in df.columns:
            plt.figure()
            plt.hist(df[c].dropna(), bins=50)
            plt.title(f'Histogram {c}')
            plt.savefig(f'hist_{c}.png')
