# Struktura projektu: Wildfire Ignition Forecasting
# ===============================================
# Pliki:
#   ├── main.py                  -> główny plik uruchamiający cały pipeline
#   ├── data_loading.py          -> wczytanie danych, identyfikacja braków, generacja pustych komórek
#   ├── feature_engineering.py   -> ekstrakcja cech, analiza timestamp, teksty, łączenie kolumn
#   ├── preprocessing.py         -> imputacja (3 metody), kodowanie kategorii
#   ├── analysis.py              -> analizy Pandas, grupowania, statystyki
#   ├── visualization.py         -> zaawansowany wykres, histogramy
#   ├── utils.py                 -> funkcje pomocnicze

# ===============================================
# main.py
# ===============================================
from data_loading import load_and_prepare_data
from feature_engineering import process_features
from preprocessing import impute_missing_values, encode_categories
from analysis import run_pandas_analysis
from visualization import generate_advanced_plot, generate_histograms

if __name__ == "__main__":
    print("--- WILDFIRE IGNITION FORECASTING PIPELINE ---")

    # 1. Wczytanie danych
    print('Wczytanie danych')
    df = load_and_prepare_data('Wildfire_Dataset.csv')

    # 2. Feature engineering
    print('Feature engineering')
    df = process_features(df)

    # 3. Imputacja braków i kodowanie kategorii
    print('Imputacja braków i kodowanie kategorii')
    df_simple, df_ffill, df_mice = impute_missing_values(df)
    df_encoded = encode_categories(df_mice)

    # 4. Analizy Pandas
    print('Analizy Pandas')
    run_pandas_analysis(df_encoded)

    # 5. Wizualizacje
    print('Wizualizacje')
    generate_advanced_plot(df_encoded)
    generate_histograms(df_encoded)

    print("--- KONIEC PIPELINE ---")


