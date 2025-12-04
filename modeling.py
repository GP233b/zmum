# ===============================================
# modeling.py
# Data split, manual k-fold CV, model training and evaluation
# ===============================================
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from metrics import classification_report_manual, accuracy_score_manual, confusion_matrix_manual, precision_recall_f1_manual


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
        'logistic': LogisticRegression(max_iter=1000, random_state=random_state),
        'rf': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'gboost': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        # 'knn': KNeighborsClassifier(n_neighbors=5),
    }
    if XGBClassifier is not None:
        models['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    if LGBMClassifier is not None:
        models['lgbm'] = LGBMClassifier(random_state=random_state)
    return models


def train_and_evaluate(models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray, out_dir: str = '.') -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for name, clf in models.items():
        print(f"Trenuję model: {name}")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        report = classification_report_manual(y_test, preds, average='weighted')
        cm = confusion_matrix_manual(y_test, preds)
        cm_path = os.path.join(out_dir, f'confusion_{name}.csv')
        pd.DataFrame(cm).to_csv(cm_path, index=False)
        try:
            labels = np.unique(np.concatenate([y_test, preds]))
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix: {name}')
            plt.tight_layout()
            png_path = os.path.join(out_dir, f'confusion_{name}.png')
            plt.savefig(png_path, dpi=150)
            plt.close()
        except Exception:
            pass
        results.append({
            'model': name,
            'accuracy': report['accuracy'],
            'precision': report['precision'],
            'recall': report['recall'],
            'f1': report['f1'],
            'cm_path': cm_path,
        })
    return pd.DataFrame(results)


def build_ensembles(models: Dict[str, Any]) -> Dict[str, Any]:
    estimators = [(k, v) for k, v in models.items() if k in ('logistic', 'rf', 'gboost', 'xgb', 'lgbm')]
    ensembles = {}
    if len(estimators) >= 2:
        ensembles['voting'] = VotingClassifier(estimators=estimators, voting='soft')
        final = LogisticRegression(max_iter=1000)
        ensembles['stacking'] = StackingClassifier(estimators=estimators, final_estimator=final, passthrough=False)
    return ensembles
