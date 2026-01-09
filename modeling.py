# ===============================================
# modeling.py
# Data split, manual k-fold CV, model training and evaluation
# ===============================================
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from metrics import classification_report_manual, accuracy_score_manual, confusion_matrix_manual, precision_recall_f1_manual


def get_class_balance(y: np.ndarray) -> Dict[int, int]:
    """
    Zwraca liczbę próbek dla każdej klasy.
    """
    unique, counts = np.unique(y, return_counts=True)
    return {cls: count for cls, count in zip(unique, counts)}

def balance_training_data_by_batches(X_train: np.ndarray, y_train: np.ndarray, 
                                     batch_idx: int, total_batches: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balansuuje dane treningowe poprzez stratified batch sampling.
    
    Args:
        X_train: Macierz cech treningowych
        y_train: Wektor etykiet treningowych
        batch_idx: Numer bieżącego batcha (0-indexed)
        total_batches: Całkowita liczba batchów
    
    Returns:
        Tuple: (X_balanced, y_balanced) - dane dla tego batcha
    """
    unique_classes = np.unique(y_train)
    if len(unique_classes) != 2:
        raise ValueError(f"Funkcja obsługuje tylko klasyfikację binarną, znaleziono {len(unique_classes)} klas")
    
    class_counts = get_class_balance(y_train)
    
    # Znajdź klasę większościową i mniejszościową
    minority_class = min(unique_classes, key=lambda c: class_counts[c])
    majority_class = max(unique_classes, key=lambda c: class_counts[c])
    
    # Pobierz indeksy dla obu klas
    minority_idx = np.where(y_train == minority_class)[0]
    majority_idx = np.where(y_train == majority_class)[0]
    
    # Podziel klasę większościową na batche
    majority_split = np.array_split(majority_idx, total_batches)
    batch_majority_idx = majority_split[batch_idx % total_batches]
    
    # Połącz: wszystkie mniejszościowe + jeden batch większościowych
    balanced_idx = np.concatenate([minority_idx, batch_majority_idx])
    np.random.shuffle(balanced_idx)
    
    X_balanced = X_train[balanced_idx]
    y_balanced = y_train[balanced_idx]
    
    return X_balanced, y_balanced


def train_and_evaluate_with_batch_balancing(models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                                           X_test: np.ndarray, y_test: np.ndarray, out_dir: str = '.', 
                                           n_batch_iterations: int = 3) -> pd.DataFrame:
    """
    Trenuje modele na WSZYSTKICH danych klasy większościowej, podzielonych na batche.
    
    Args:
        models: Słownik modeli do trenowania
        X_train: Dane treningowe
        y_train: Etykiety treningowe
        X_test: Dane testowe (niezmieniane)
        y_test: Etykiety testowe (niezmieniane)
        out_dir: Katalog dla wyników
        n_batch_iterations: Ile razy przejść przez wszystkie batche
    
    Returns:
        pd.DataFrame: Wyniki agregowane
    """
    os.makedirs(out_dir, exist_ok=True)
    
    class_counts = get_class_balance(y_train)
    majority_class_count = max(class_counts.values())
    minority_class_count = min(class_counts.values())
    
    # Liczba batchów = liczba batchów klasy większościowej aby zrównoważyć z mniejszościową
    n_batches = max(1, int(np.ceil(majority_class_count / minority_class_count))) // 2
    total_batch_count = n_batches * n_batch_iterations
    
    print(f"\n=== TRENING Z BATCH BALANSOWANIEM (Wszystkie dane, podzielone na batche) ===")
    print(f"Rozkład klas w zbiorze treningowym: {class_counts}")
    print(f"Niebalans: {majority_class_count / minority_class_count:.2f}x")
    print(f"Liczba batchów na iterację: {n_batches}")
    print(f"Liczba iteracji przez batche: {n_batch_iterations}")
    print(f"Całkowita liczba batchów do trenowania: {total_batch_count}\n")
    
    all_results = []
    model_predictions = {name: [] for name in models.keys()}
    model_true = None
    
    batch_counter = 0
    for batch_iteration in range(n_batch_iterations):
        print(f"--- Iteracja {batch_iteration + 1}/{n_batch_iterations} (przez wszystkie batche) ---")
        
        for batch_idx in range(n_batches):
            batch_counter += 1
            print(f"  Batch {batch_idx + 1}/{n_batches}")
            
            X_batch, y_batch = balance_training_data_by_batches(X_train, y_train, batch_idx, n_batches)
            batch_counts = get_class_balance(y_batch)
            print(f"    Rozkład w batchu: {batch_counts}")
            
            for name, clf in models.items():
                clf_iter = clone(clf)
                clf_iter.fit(X_batch, y_batch)
                
                preds = clf_iter.predict(X_test)
                model_predictions[name].append(preds)
                
                if model_true is None:
                    model_true = y_test
                
                report = classification_report_manual(y_test, preds, average='weighted')
                all_results.append({
                    'batch_iteration': batch_iteration + 1,
                    'batch_num': batch_idx + 1,
                    'total_batch': batch_counter,
                    'model': name,
                    'accuracy': report['accuracy'],
                    'precision': report['precision'],
                    'recall': report['recall'],
                    'f1': report['f1'],
                })
    
    results_df = pd.DataFrame(all_results)
    
    # Agreguj wyniki po modelach
    aggregated = results_df.groupby('model')[['accuracy', 'precision', 'recall', 'f1']].agg(['mean'])
    print(f"\n=== WYNIKI ZAGREGOWANE (ze wszystkich {total_batch_count} batchów) ===")
    print(aggregated)
    
    # Zapisz szczegółowe wyniki
    results_df.to_csv(os.path.join(out_dir, 'batch_balancing_detailed.csv'), index=False)
    aggregated.to_csv(os.path.join(out_dir, 'batch_balancing_aggregated.csv'))
    
    # Utwórz ensemble
    print(f"\n=== ENSEMBLE Z WSZYSTKICH BATCHÓW ===")
    final_results = []
    for name in models.keys():
        if model_predictions[name]:
            ensemble_preds = np.round(np.mean(model_predictions[name], axis=0)).astype(int)
            report = classification_report_manual(model_true, ensemble_preds, average='weighted')
            final_results.append({
                'model': f'{name}_ensemble',
                'accuracy': report['accuracy'],
                'precision': report['precision'],
                'recall': report['recall'],
                'f1': report['f1'],
            })
            print(f"{name}_ensemble (z {total_batch_count} batchów) - Accuracy: {report['accuracy']:.4f}, F1: {report['f1']:.4f}")
    
    final_results_df = pd.DataFrame(final_results)
    final_results_df.to_csv(os.path.join(out_dir, 'batch_balancing_ensemble.csv'), index=False)
    
    for name in models.keys():
        if model_predictions[name]:
            # Agreguj predykcje (głosowanie większościowe - bardziej stabilne niż średnia)
            predictions_array = np.array(model_predictions[name])  # shape: (n_batches, n_samples)
            # Głosowanie: ile razy każda próbka została zaklasyfikowana jako 1
            ensemble_preds = (np.sum(predictions_array, axis=0) > len(model_predictions[name]) / 2).astype(int)
            
            # Utwórz macierz pomyłek
            cm = confusion_matrix_manual(model_true, ensemble_preds)
            cm_path = os.path.join(out_dir, f'confusion_batch_{name}_ensemble.csv')
            pd.DataFrame(cm).to_csv(cm_path, index=False)
            
            # Narysuj confusion matrix
            try:
                labels = np.unique(np.concatenate([model_true, ensemble_preds]))
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix: {name} (ensemble z {total_batch_count} batchów)')
                plt.tight_layout()
                png_path = os.path.join(out_dir, f'confusion_batch_{name}_ensemble.png')
                plt.savefig(png_path, dpi=150)
                plt.close()
            except Exception as e:
                print(f"  {name}: Błąd przy rysowaniu macierzy - {e}")
    
    print("\n" + "="*70)
    return results_df


def manual_train_test_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42,
                            stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(random_state)
    n = len(df)
    indices = np.arange(n)

    if stratify and df[target_col].nunique() > 1:
        train_idx = []
        test_idx = []
        for cls, grp in df.groupby(target_col):
            grp_idx = grp.index.values
            rng.shuffle(grp_idx)
            n_test = max(1, int(len(grp_idx) * test_size))
            test_idx.extend(grp_idx[:n_test].tolist())
            train_idx.extend(grp_idx[n_test:].tolist())
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)
        return train_df, test_df
    else:
        rng.shuffle(indices)
        split = int((1 - test_size) * n)
        train_idx = indices[:split]
        test_idx = indices[split:]
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def manual_kfold_cv(X: np.ndarray, y: np.ndarray, model, k: int = 5, random_state: int = 42) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        clf = clone(model)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[val_idx])
        acc = accuracy_score_manual(y[val_idx], preds)
        metrics = precision_recall_f1_manual(y[val_idx], preds)
        fold_results.append({'fold': fold, 'accuracy': acc, **metrics})
    avg_accuracy = float(np.mean([r['accuracy'] for r in fold_results]))
    avg_precision = float(np.mean([r['precision'] for r in fold_results]))
    avg_recall = float(np.mean([r['recall'] for r in fold_results]))
    avg_f1 = float(np.mean([r['f1'] for r in fold_results]))
    return {
        'folds': fold_results,
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
    }


def build_default_models(random_state: int = 42) -> Dict[str, Any]:
    models = {
        'logistic': LogisticRegression(
            max_iter=50,
            n_jobs=-1,
            random_state=random_state
        ),

        'rf': RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            n_jobs=-1,
            random_state=random_state
        ),

        'gboost': GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=random_state
        )
    }

    if XGBClassifier is not None:
        models['xgb'] = XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            n_jobs=-1,
            random_state=random_state,
            eval_metric='logloss'
        )

    if LGBMClassifier is not None:
        models['lgbm'] = LGBMClassifier(
            n_estimators=50,
            max_depth=-1,
            learning_rate=0.1,
            num_leaves=31,
            n_jobs=-1,
            random_state=random_state
        )

    return models


def build_ensembles(models: Dict[str, Any]) -> Dict[str, Any]:
    add_models = models.copy()
    add_models['ridge'] = RidgeClassifier()
    add_models['nb'] = GaussianNB()

    estimators = [(k, v) for k, v in add_models.items() if k in ('rf', 'nb')]

    ensembles = {}
    if len(estimators) >= 2:
        ensembles['voting'] = VotingClassifier(estimators=estimators, voting='soft')
        final = LogisticRegression(max_iter=1000)
        ensembles['stacking'] = StackingClassifier(estimators=estimators, final_estimator=final, passthrough=False)
    return ensembles
