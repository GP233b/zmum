# ===============================================
# metrics.py
# Manual implementations of common classification metrics
# ===============================================
from typing import List, Dict, Any
import numpy as np


def confusion_matrix_manual(y_true: List, y_pred: List, labels: List = None) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    label_to_index = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = label_to_index.get(t)
        j = label_to_index.get(p)
        if i is None or j is None:
            continue
        mat[i, j] += 1
    return mat


def accuracy_score_manual(y_true: List, y_pred: List) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).sum()) / max(len(y_true), 1)


def precision_recall_f1_manual(y_true: List, y_pred: List, average: str = 'macro') -> Dict[str, Any]:
    # Works for binary and multiclass. Returns per-class and averaged metrics.
    cm = confusion_matrix_manual(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp

    with np.errstate(divide='ignore', invalid='ignore'):
        precision_per_class = np.where(tp + fp == 0, 0.0, tp / (tp + fp))
        recall_per_class = np.where(tp + fn == 0, 0.0, tp / (tp + fn))
        f1_per_class = np.where(precision_per_class + recall_per_class == 0, 0.0,
                                2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class))

    results = {
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
    }

    if average == 'macro':
        results['precision'] = float(np.nanmean(precision_per_class))
        results['recall'] = float(np.nanmean(recall_per_class))
        results['f1'] = float(np.nanmean(f1_per_class))
    elif average == 'micro':
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
    else:
        raise ValueError("Unsupported average. Use 'macro' or 'micro'.")

    return results


def classification_report_manual(y_true: List, y_pred: List, labels: List = None, average: str = 'macro') -> Dict[str, Any]:
    cm = confusion_matrix_manual(y_true, y_pred, labels)
    metrics = precision_recall_f1_manual(y_true, y_pred, average=average)
    acc = accuracy_score_manual(y_true, y_pred)
    return {
        'confusion_matrix': cm,
        'accuracy': acc,
        **metrics,
    }
