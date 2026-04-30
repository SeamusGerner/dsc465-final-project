"""
Evaluation, simulation, and plot generation.

Run:
    python src/evaluate.py
"""

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from basic_strategy import basic_strategy_from_features


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def eval_classification(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'per_class': report,
        'confusion_matrix': cm.tolist(),
    }, y_pred


def plot_confusion_matrix(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------

def simulate_strategies(rf_model, df_test: pd.DataFrame, label_encoder, feature_cols, n_sample=100_000):
    """
    Compare RF model vs basic strategy on the test set.

    For each hand, we compare:
      - basic strategy action
      - RF model's predicted action
    Both are scored using the mean payout for that (state bucket, action) pair
    observed in the test set itself (the best approximation without a live sim engine).

    Also includes the actual actions taken and their payouts as a third baseline.
    """
    rng = np.random.default_rng(42)

    if len(df_test) > n_sample:
        idx = rng.choice(len(df_test), n_sample, replace=False)
        df_sim = df_test.iloc[idx].reset_index(drop=True)
    else:
        df_sim = df_test.reset_index(drop=True)

    X_sim = df_sim[feature_cols].values

    # RF predictions
    rf_pred_encoded = rf_model.predict(X_sim)
    rf_actions = label_encoder.inverse_transform(rf_pred_encoded)

    # Basic strategy actions
    bs_actions = df_sim.apply(basic_strategy_from_features, axis=1).values

    # Build an EV lookup: (state_bucket, action) -> mean payout
    # Bucket: player_total_bin × dealer_upcard × soft × pair
    df_sim = df_sim.copy()
    df_sim['_pt_bin'] = pd.cut(df_sim['player_total'],
                                bins=[0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22],
                                labels=['<=8','9','10','11','12','13','14','15','16','17','18','19','20','21'],
                                right=True)
    df_sim['_bs_action'] = bs_actions
    df_sim['_rf_action'] = rf_actions

    # Compute per-row EV lookup from test set
    ev_lookup = (
        df_sim.groupby(['_pt_bin', 'dealer_upcard_val', 'is_soft', 'is_pair', '_bs_action'])['payout']
        .mean()
        .to_dict()
    )

    def lookup_ev(row, action_col):
        key = (row['_pt_bin'], row['dealer_upcard_val'], row['is_soft'], row['is_pair'], row[action_col])
        if key in ev_lookup:
            return ev_lookup[key]
        return row['payout']  # fallback: observed payout

    # Actual payout (what happened with the action actually taken)
    actual_payouts = df_sim['payout'].values

    # RF payouts
    rf_payouts = np.array([lookup_ev(df_sim.iloc[i], '_rf_action') for i in range(len(df_sim))])

    # Basic strategy payouts
    bs_payouts = np.array([lookup_ev(df_sim.iloc[i], '_bs_action') for i in range(len(df_sim))])

    results = {
        'n_hands': len(df_sim),
        'actual': {
            'ev_per_hand': float(actual_payouts.mean()),
            'win_rate': float((actual_payouts > 0).mean()),
            'cumulative_payouts': actual_payouts.cumsum().tolist(),
        },
        'basic_strategy': {
            'ev_per_hand': float(bs_payouts.mean()),
            'win_rate': float((bs_payouts > 0).mean()),
            'cumulative_payouts': bs_payouts.cumsum().tolist(),
        },
        'rf_model': {
            'ev_per_hand': float(rf_payouts.mean()),
            'win_rate': float((rf_payouts > 0).mean()),
            'cumulative_payouts': rf_payouts.cumsum().tolist(),
        },
    }
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ev_curve(sim_results: dict, path):
    n = sim_results['n_hands']
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, key, color in [
        ('RF Model', 'rf_model', 'steelblue'),
        ('Basic Strategy', 'basic_strategy', 'darkorange'),
        ('Actual (dataset)', 'actual', 'gray'),
    ]:
        cum = np.array(sim_results[key]['cumulative_payouts'])
        ax.plot(x, cum / x, label=label, color=color, linewidth=1.5)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Hands played')
    ax.set_ylabel('Cumulative EV per hand')
    ax.set_title('EV per Hand — RF Model vs Basic Strategy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_strategy_comparison(sim_results: dict, path):
    labels = ['RF Model', 'Basic Strategy', 'Actual (dataset)']
    keys = ['rf_model', 'basic_strategy', 'actual']
    evs = [sim_results[k]['ev_per_hand'] for k in keys]
    wins = [sim_results[k]['win_rate'] * 100 for k in keys]

    x = np.arange(len(labels))
    width = 0.35
    colors_ev = ['steelblue', 'darkorange', 'gray']
    colors_wr = ['cornflowerblue', 'sandybrown', 'lightgray']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    bars1 = ax1.bar(x, evs, width, color=colors_ev)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.set_ylabel('EV per hand')
    ax1.set_title('Expected Value per Hand')
    ax1.axhline(0, color='black', linewidth=0.8)
    for bar, v in zip(bars1, evs):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f'{v:.4f}',
                 ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

    bars2 = ax2.bar(x, wins, width, color=colors_wr)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_ylabel('Win rate (%)')
    ax2.set_title('Win Rate')
    for bar, v in zip(bars2, wins):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f'{v:.1f}%',
                 ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_feature_importance(rf_model, feature_cols, path, top_n=10):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_cols[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(top_n), vals[::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel('Feature importance (Gini)')
    ax.set_title('Top 10 Feature Importances — Random Forest')
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f'  {v:.4f}',
                va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    models_dir = ROOT / 'models'
    reports_dir = ROOT / 'reports'
    reports_dir.mkdir(exist_ok=True)

    for p in [models_dir / 'rf_model.joblib', models_dir / 'label_encoder.joblib',
              models_dir / 'X_test.npy', models_dir / 'y_test.npy']:
        if not p.exists():
            print(f"Missing {p}. Run `python src/train.py` first.")
            sys.exit(1)

    rf = joblib.load(models_dir / 'rf_model.joblib')
    lr = joblib.load(models_dir / 'lr_baseline.joblib')
    le = joblib.load(models_dir / 'label_encoder.joblib')
    feature_cols = joblib.load(models_dir / 'feature_cols.joblib')
    X_test = np.load(models_dir / 'X_test.npy')
    y_test = np.load(models_dir / 'y_test.npy')
    df_test = pd.read_parquet(models_dir / 'test_set.parquet')

    print("=== Random Forest ===")
    rf_metrics, rf_pred = eval_classification(rf, X_test, y_test, le)
    print(f"Accuracy: {rf_metrics['accuracy']:.4f}  |  Macro-F1: {rf_metrics['macro_f1']:.4f}")

    print("\n=== Logistic Regression (baseline) ===")
    lr_metrics, lr_pred = eval_classification(lr, X_test, y_test, le)
    print(f"Accuracy: {lr_metrics['accuracy']:.4f}  |  Macro-F1: {lr_metrics['macro_f1']:.4f}")

    # Confusion matrices
    cm_rf = confusion_matrix(y_test, rf_pred)
    plot_confusion_matrix(cm_rf, le.classes_, 'Random Forest Confusion Matrix',
                          reports_dir / 'confusion_matrix.png')

    # Feature importance
    plot_feature_importance(rf, feature_cols, reports_dir / 'feature_importance.png')

    # Strategy simulation
    print("\nRunning strategy simulation (100K hands) ...")
    sim = simulate_strategies(rf, df_test, le, feature_cols, n_sample=100_000)
    print(f"\nEV per hand — RF: {sim['rf_model']['ev_per_hand']:.4f} | "
          f"Basic strategy: {sim['basic_strategy']['ev_per_hand']:.4f} | "
          f"Actual: {sim['actual']['ev_per_hand']:.4f}")
    print(f"Win rate   — RF: {sim['rf_model']['win_rate']:.3f}  | "
          f"Basic strategy: {sim['basic_strategy']['win_rate']:.3f}  | "
          f"Actual: {sim['actual']['win_rate']:.3f}")

    plot_ev_curve(sim, reports_dir / 'ev_curve.png')
    plot_strategy_comparison(sim, reports_dir / 'model_vs_basic_strategy.png')

    # Save metrics.json
    metrics = {
        'random_forest': {
            'accuracy': rf_metrics['accuracy'],
            'macro_f1': rf_metrics['macro_f1'],
            'per_class': rf_metrics['per_class'],
        },
        'logistic_regression': {
            'accuracy': lr_metrics['accuracy'],
            'macro_f1': lr_metrics['macro_f1'],
        },
        'simulation': {
            'n_hands': sim['n_hands'],
            'rf_model': {
                'ev_per_hand': sim['rf_model']['ev_per_hand'],
                'win_rate': sim['rf_model']['win_rate'],
            },
            'basic_strategy': {
                'ev_per_hand': sim['basic_strategy']['ev_per_hand'],
                'win_rate': sim['basic_strategy']['win_rate'],
            },
        },
    }
    metrics_path = reports_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {metrics_path}")
    print("\nEvaluation complete.")


if __name__ == '__main__':
    main()
