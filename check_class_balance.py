"""
check_class_balance.py
Szybki skrypt do sprawdzenia niezbalansowania klas w zbiorze danych
"""
import pandas as pd
import numpy as np
from collections import Counter

def check_balance(df, target_col='Wildfire'):
    """
    Sprawdza i wyświetla informacje o niezbalansowaniu klas
    """
    print("\n" + "="*70)
    print("ANALIZA NIEZBALANSOWANIA KLAS")
    print("="*70)
    
    # Pobierz wartości targetowe
    y = df[target_col].values
    
    # Policz klasy
    counts = Counter(y)
    unique_classes = sorted(counts.keys())
    
    print(f"\nKolumna celu: {target_col}")
    print(f"Liczba próbek: {len(y)}")
    print(f"Liczba unikalnych klas: {len(unique_classes)}")
    
    print("\n--- ROZKŁAD KLAS ---")
    total = len(y)
    for cls in unique_classes:
        count = counts[cls]
        percentage = (count / total) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        print(f"Klasa {cls}: {count:6d} próbek ({percentage:6.2f}%) {bar}")
    
    # Oblicz niebalans
    min_class = min(unique_classes, key=lambda c: counts[c])
    max_class = max(unique_classes, key=lambda c: counts[c])
    imbalance_ratio = counts[max_class] / counts[min_class]
    
    print(f"\n--- METRYKA NIEZBALANSOWANIA ---")
    print(f"Klasa większościowa: {max_class} ({counts[max_class]} próbek)")
    print(f"Klasa mniejszościowa: {min_class} ({counts[min_class]} próbek)")
    print(f"Stosunek niezbalansowania: {imbalance_ratio:.2f}x")


    print("\n" + "="*70 + "\n")
    
    return counts, imbalance_ratio

if __name__ == "__main__":
    # Wczytaj dane
    try:
        df = pd.read_csv('datasets/Wildfire_Dataset.csv')
        check_balance(df)
    except FileNotFoundError:
        print("Nie znaleziono pliku datasets/Wildfire_Dataset.csv")
        print("Uruchom główny pipeline najpierw: python main.py")
