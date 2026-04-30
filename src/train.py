"""
Training pipeline for the blackjack optimal-action ML model.

Tailored for the Kaggle dataset `blkjckhands.csv` (mojocolors, 900K hands).

The dataset does NOT directly record the player's action or a game ID. After
inspecting all 900K rows, we found:
  - plwinamt ∈ {0, 10, 20, 25} and dlwinamt ∈ {0, 10}, so net payouts are
    {-1, 0, +1, +1.5} in bet units. NO doubles (would be ±2). NO splits.
  - The simulator's strategy only used Hit and Stand.

So:
  - game_id is inferred as row_index // 6 (PlayerNo cycles Player1..Player6).
  - First action is a 2-class inference: card3 == 0 → Stand, else Hit.
  - Net payout: -1 (Loss), 0 (Push), +1 (Win), +1.5 (Blackjack).
  - The model is trained as a 2-class problem (H vs S). Basic strategy is
    still computed at full granularity (H/S/D/P) and shown in evaluation;
    where it says D, we collapse to H for direct comparison.

Run:
    python src/train.py --data data/blkjckhands.csv --sample 200000
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / 'src'))
from features import build_features, FEATURE_COLS


# ---------------------------------------------------------------------------
# Hardcoded schema (verified against blkjckhands.csv)
# ---------------------------------------------------------------------------

SCHEMA = {
    'player_position': 'PlayerNo',
    'player_card_cols': ['card1', 'card2', 'card3', 'card4', 'card5'],
    'dealer_card_cols': ['dealcard1', 'dealcard2', 'dealcard3', 'dealcard4', 'dealcard5'],
    'dealer_upcard': 'dealcard1',
    'sumofcards': 'sumofcards',
    'sumofdeal': 'sumofdeal',
    'blkjck': 'blkjck',
    'winloss': 'winloss',
    'plwinamt': 'plwinamt',
    'dlwinamt': 'dlwinamt',
    'ply2cardsum': 'ply2cardsum',
    'plybustbeat': 'plybustbeat',
    'dlbustbeat': 'dlbustbeat',
}

# Schema dict expected by features.build_features()
FEATURES_SCHEMA = {
    'game_id': 'game_id',                # injected by add_game_id()
    'position': SCHEMA['player_position'],
    'player_cards': SCHEMA['player_card_cols'],
    'dealer_upcard': SCHEMA['dealer_upcard'],
}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def add_game_id(df: pd.DataFrame) -> pd.DataFrame:
    """Each block of 6 consecutive rows = one game (Player1..Player6)."""
    df = df.reset_index(drop=True).copy()
    df['game_id'] = df.index // 6
    return df


def _count_nonzero_cards(row, cols):
    """Count non-zero entries among the listed card columns."""
    n = 0
    for c in cols:
        v = row[c]
        if pd.isna(v):
            continue
        try:
            if int(v) != 0:
                n += 1
        except (ValueError, TypeError):
            if str(v).strip():
                n += 1
    return n


def infer_first_action(row) -> str:
    """
    Infer the player's FIRST action.

    Returns: 'H' (Hit) or 'S' (Stand).
    The simulator that produced blkjckhands.csv never doubled or split
    (verified: max |payout| is 1.5x bet = blackjack), so this is 2-class.
    """
    n_post = _count_nonzero_cards(row, ['card3', 'card4', 'card5'])
    return 'S' if n_post == 0 else 'H'


def compute_net_payout(row, action: str = None) -> float:
    """
    Net payout in BET UNITS (1 unit = 1 standard bet of 10 chips).

    Verified mapping for blkjckhands.csv:
      plwinamt=0,  dlwinamt=10 → Loss      → -1
      plwinamt=10, dlwinamt=0  → Push      →  0
      plwinamt=20, dlwinamt=0  → Win       → +1
      plwinamt=25, dlwinamt=0  → Blackjack → +1.5
    """
    plwin = float(row['plwinamt'])
    dlwin = float(row['dlwinamt'])
    if plwin == 25:
        return 1.5
    if plwin == 20:
        return 1.0
    if plwin == 10:
        return 0.0
    if dlwin == 10:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# EV-maximizing label generation (Approach A)
# ---------------------------------------------------------------------------

def create_ev_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each discretized game-state bucket, find the action with the highest
    mean payout. Label each row with that optimal action.

    Bucket dimensions: player_total_bin × dealer_upcard × soft × pair × tc_bin
    """
    df = df.copy()

    total_bins = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]
    total_labels = ['<=8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                    '18', '19', '20', '21']
    df['_pt_bin'] = pd.cut(df['player_total'], bins=total_bins,
                           labels=total_labels, right=True)

    tc_bins = [-np.inf, -2, -1, 0, 1, 2, np.inf]
    tc_labels = ['<=-2', '-1', '0', '+1', '+2', '>=+2']
    df['_tc_bin'] = pd.cut(df['true_count'], bins=tc_bins, labels=tc_labels)

    group_cols = ['_pt_bin', 'dealer_upcard_val', 'is_soft', 'is_pair', '_tc_bin']
    df = df.dropna(subset=['inferred_action', 'payout'])

    print(f"Inferred action distribution:\n{df['inferred_action'].value_counts()}\n")

    ev = (
        df.groupby(group_cols + ['inferred_action'], observed=True)['payout']
        .mean()
        .reset_index()
        .rename(columns={'payout': 'mean_ev'})
    )

    best = ev.loc[ev.groupby(group_cols, observed=True)['mean_ev'].idxmax()].copy()
    best['_bucket'] = best[group_cols].apply(lambda r: tuple(r), axis=1)
    bucket_to_action = dict(zip(best['_bucket'], best['inferred_action']))

    df['_bucket'] = df[group_cols].apply(lambda r: tuple(r), axis=1)
    df['optimal_action'] = df['_bucket'].map(bucket_to_action)

    n_unlabeled = df['optimal_action'].isna().sum()
    if n_unlabeled > 0:
        print(f"Warning: {n_unlabeled} rows ({n_unlabeled/len(df):.1%}) "
              f"have no EV label (sparse buckets). Dropping.")
    df = df.dropna(subset=['optimal_action'])

    print(f"Optimal action distribution:\n{df['optimal_action'].value_counts()}\n")
    print(f"Unique buckets: {len(bucket_to_action)}")

    return df


# ---------------------------------------------------------------------------
# Train / test split (by game_id to prevent leakage)
# ---------------------------------------------------------------------------

def game_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split by game_id so no game's rows appear in both train and test."""
    game_ids = df['game_id'].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    split = int(len(game_ids) * (1 - test_size))
    train_ids = set(game_ids[:split])
    test_ids = set(game_ids[split:])
    train = df[df['game_id'].isin(train_ids)]
    test = df[df['game_id'].isin(test_ids)]
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_logistic(X_train, y_train, seed=42):
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=seed,
        solver='lbfgs',
        C=1.0,
    )
    lr.fit(X_train, y_train)
    return lr


def tune_random_forest(X_train, y_train, seed=42):
    """Grid search over key RF hyperparameters using 5-fold CV."""
    from sklearn.model_selection import GridSearchCV

    base = RandomForestClassifier(
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1,
    )
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [5, 10, 20],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(base, param_grid, cv=cv, scoring='f1_macro',
                     n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"Best RF params: {gs.best_params_}  "
          f"(CV macro-F1 = {gs.best_score_:.4f})")
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/blkjckhands.csv')
    parser.add_argument('--sample', type=int, default=200000,
                        help='Row sample size (use 0 for full dataset)')
    parser.add_argument('--full-tune', action='store_true',
                        help='Run full grid search (slow)')
    args = parser.parse_args()

    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        sys.exit(1)

    print(f"Loading data from {data_path} ...")
    df_raw = pd.read_csv(data_path, index_col=0)  # first col is unnamed index
    print(f"Raw shape: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}")

    # Inject game_id
    df_raw = add_game_id(df_raw)
    print(f"Games inferred: {df_raw['game_id'].nunique():,}")

    # Optional stratified subsample by inferred action
    if args.sample and args.sample < len(df_raw):
        # Compute action quickly for stratification
        df_raw['_inf_act'] = df_raw.apply(infer_first_action, axis=1)
        df_raw = (
            df_raw.groupby('_inf_act', group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), int(args.sample * len(g) / len(df_raw))),
                random_state=42
            ))
        ).reset_index(drop=True)
        print(f"Subsampled to {len(df_raw):,} rows (stratified by inferred action)")
        # Re-add game_id since we resampled (preserve original game grouping)
        # Note: this is stratified sampling — game_id is no longer contiguous,
        # but the original game_id column is preserved from before, so split-by-game still works.

    # Infer action (idempotent — recompute on final df)
    print("Inferring first action per row ...")
    df_raw['inferred_action'] = df_raw.apply(infer_first_action, axis=1)
    print(df_raw['inferred_action'].value_counts())

    # Feature engineering
    print("\nBuilding features ...")
    df_feat, feature_cols = build_features(df_raw, FEATURES_SCHEMA)

    # Compute payout
    print("Computing payouts ...")
    df_feat['payout'] = df_feat.apply(
        lambda r: compute_net_payout(r, r['inferred_action']), axis=1
    )

    # EV labels
    print("Creating EV-optimal labels ...")
    df_labeled = create_ev_labels(df_feat)

    # Train/test split by game_id
    df_train, df_test = game_split(df_labeled)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['optimal_action'])
    y_test = le.transform(df_test['optimal_action'])
    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values

    print(f"\nClasses: {le.classes_}")

    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    joblib.dump(le, models_dir / 'label_encoder.joblib')
    np.save(models_dir / 'X_test.npy', X_test)
    np.save(models_dir / 'y_test.npy', y_test)
    # Drop helper columns that aren't parquet-friendly (tuples, categoricals from cut)
    test_save = df_test.drop(columns=['_bucket', '_pt_bin', '_tc_bin'], errors='ignore').copy()
    test_save.to_parquet(models_dir / 'test_set.parquet', index=False)

    # Baseline LR
    print("\n--- Training Logistic Regression baseline ---")
    lr = train_logistic(X_train, y_train)
    lr_score = cross_val_score(lr, X_train, y_train, cv=3, scoring='f1_macro').mean()
    print(f"LR baseline 3-fold macro-F1: {lr_score:.4f}")
    joblib.dump(lr, models_dir / 'lr_baseline.joblib')

    # Main RF
    print("\n--- Training Random Forest (main model) ---")
    if args.full_tune:
        rf = tune_random_forest(X_train, y_train)
    else:
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

    joblib.dump(rf, models_dir / 'rf_model.joblib')
    joblib.dump(feature_cols, models_dir / 'feature_cols.joblib')
    print(f"RF saved to models/rf_model.joblib")

    rf_acc = (rf.predict(X_test) == y_test).mean()
    lr_acc = (lr.predict(X_test) == y_test).mean()
    print(f"\nTest accuracy — RF: {rf_acc:.4f} | LR: {lr_acc:.4f}")
    print("\nTraining complete. Run `python src/evaluate.py` for full metrics + plots.")


if __name__ == '__main__':
    main()
