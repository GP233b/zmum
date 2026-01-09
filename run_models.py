import os
import numpy as np
import pandas as pd

from preprocessing import impute_simple, encode_categories
from modeling import manual_train_test_split, build_default_models, build_ensembles, manual_kfold_cv, train_and_evaluate_with_batch_balancing

    
def run_models_from_df(df: pd.DataFrame, target = 'Wildfire_le', out_dir: str = 'model_outputs'):
    os.makedirs(out_dir, exist_ok=True)

    print(f'Używam kolumny celu: {target}')

    print('Przygotowanie X,y')
    X, y, feature_cols = prepare_X_y(df, target)
    print(f'Features: {len(feature_cols)} kolumn')

    print('Podział na train/test (stratyfikowany)')
    df_train, df_test = manual_train_test_split(pd.concat([pd.DataFrame(X, columns=feature_cols), pd.Series(y, name=target)], axis=1), target,
                                                test_size=0.2, random_state=42)

    X_train = df_train[feature_cols].values
    y_train = df_train[target].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target].values

    print('Buduję modele domyślne...')
    models = build_default_models()

    # NOWE: Trening z batch balansowaniem (Wszystkie dane, podzielone na batche)
    print('\n' + '='*70)
    print('TRENING Z BATCH BALANSOWANIEM (WSZYSTKIE DANE)')
    print('='*70)
    batch_results_df = train_and_evaluate_with_batch_balancing(
        models, X_train, y_train, X_test, y_test, 
        out_dir=out_dir, 
        n_batch_iterations=1
    )
    print('='*70 + '\n')

    results_df = batch_results_df

    print('Trenowanie i ewaluacja modeli (bez CV)')
    # results_df = train_and_evaluate(models, X_train, y_train, X_test, y_test, out_dir=out_dir)
    results_df.to_csv(os.path.join(out_dir, 'model_results_simple_split.csv'), index=False)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('seaborn')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        melt = results_df.melt(id_vars=['model'], value_vars=metrics_to_plot, var_name='metric', value_name='value')
        sns.barplot(data=melt, x='model', y='value', hue='metric', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.title('Model metrics (simple train/test split)')
        plt.tight_layout()
        cmp_path = os.path.join(out_dir, 'metrics_comparison_simple_split.png')
        plt.savefig(cmp_path, dpi=150)
        plt.close()
    except Exception:
        pass

    chosen = 'rf' if 'rf' in models else list(models.keys())[0]
    print(f'Uruchamiam ręczną walidację krzyżową dla modelu: {chosen}')
    cv_res = manual_kfold_cv(X, y, models[chosen], k=5, random_state=42)
    cv_summary = {
        'model': chosen,
        'avg_accuracy': cv_res['avg_accuracy'],
        'avg_precision': cv_res['avg_precision'],
        'avg_recall': cv_res['avg_recall'],
        'avg_f1': cv_res['avg_f1'],
    }
    pd.DataFrame([cv_summary]).to_csv(os.path.join(out_dir, 'cv_summary.csv'), index=False)

    ensembles = build_ensembles(models)
    if ensembles:
        print('Trenowanie ensemble (voting + stacking) i ewaluacja')
        ens_df = train_and_evaluate_with_batch_balancing(ensembles, X_train, y_train, X_test, y_test, out_dir=out_dir)
        ens_df.to_csv(os.path.join(out_dir, 'ensemble_results.csv'), index=False)
        combined = pd.concat([results_df, ens_df], ignore_index=True)
        combined.to_csv(os.path.join(out_dir, 'all_model_results.csv'), index=False)
    else:
        print('Brak wystarczających klasyfikatorów do ensemble (xgb/lgbm/sklearn).')

    print('Zakończono. Wyniki zapisane w:', out_dir)


def prepare_X_y(df: pd.DataFrame, target_col: str):
    df2 = df.copy()
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    X = df2[numeric_cols].values
    y = df2[target_col].values
    return X, y, numeric_cols