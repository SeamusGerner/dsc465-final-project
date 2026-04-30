"""
Training pipeline for the blackjack optimal-action ML model.

Run:
    python src/train.py --data data/blackjack_simulator.csv --sample 200000
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
from basic_strategy import normalize_action, ACTION_MAP


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

def detect_schema(df: pd.DataFrame) -> dict:
    """
    Auto-detect column roles from the dataframe.
    Prints a summary and returns a schema dict.
    """
    cols = set(df.columns.str.lower())
    col_map = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols:
                return col_map[c]
        return None

    schema = {}

    # Game ID
    schema['game_id'] = find(['game_id', 'gameid', 'game', 'hand_id', 'handid', 'session_id'])

    # Action
    schema['action'] = find(['action', 'player_action', 'move', 'decision', 'play'])

    # Outcome / payout — try numeric payout first, then win flags
    schema['payout'] = find(['payout', 'net_payout', 'result', 'outcome', 'profit', 'net', 'winnings'])
    schema['win_flag'] = find(['win', 'player_win', 'won', 'player_won', 'winner'])
    schema['bust_flag'] = find(['bust', 'player_bust', 'player_busted', 'busted'])
    schema['blackjack_flag'] = find(['blackjack', 'natural', 'player_blackjack', 'bj'])
    schema['push_flag'] = find(['push', 'tie', 'tied'])

    # Player position within game
    schema['position'] = find(['position', 'player_position', 'seat', 'player_id', 'player_num', 'player'])

    # Dealer upcard
    schema['dealer_upcard'] = find([
        'dealer_upcard', 'dealer_card', 'dealer_card_1', 'dealer_up',
        'upcard', 'dealer_face', 'dealer_showing'
    ])

    # Player cards — try to find all player card columns
    player_card_cols = []
    for i in range(1, 10):
        c = find([f'player_card_{i}', f'card_{i}', f'p_card_{i}', f'player_{i}',
                  f'card{i}', f'pcard{i}'])
        if c:
            player_card_cols.append(c)
        elif i > 2:
            break  # stop looking after a gap

    # Fallback: look for any column with "card" in name
    if len(player_card_cols) < 2:
        player_card_cols = [c for c in df.columns if 'card' in c.lower() and 'dealer' not in c.lower()]

    schema['player_cards'] = player_card_cols

    # Dealer total
    schema['dealer_total'] = find(['dealer_total', 'dealer_sum', 'dealer_hand'])
    # Player total (if pre-computed in dataset)
    schema['player_total_raw'] = find(['player_total', 'player_sum', 'hand_total', 'total'])

    print("\n=== Detected Schema ===")
    for k, v in schema.items():
        print(f"  {k:20s} -> {v}")
    print()

    # Validate required fields
    missing = []
    for req in ['game_id', 'action', 'dealer_upcard']:
        if not schema[req]:
            missing.append(req)
    if len(schema['player_cards']) < 2:
        missing.append('player_cards (need at least 2)')

    if missing:
        print("SCHEMA DETECTION FAILED. Could not find columns for:", missing)
        print("\nActual columns in dataset:")
        print(df.columns.tolist())
        print("\nPlease update detect_schema() with the correct column names.")
        sys.exit(1)

    return schema


# ---------------------------------------------------------------------------
# Payout construction
# ---------------------------------------------------------------------------

def build_payout(df: pd.DataFrame, schema: dict) -> pd.Series:
    """
    Construct a numeric payout column.
    Win = +1, Blackjack win = +1.5, Lose = -1, Push = 0.
    """
    if schema['payout'] and df[schema['payout']].dtype in (np.float64, np.int64, float, int):
        raw = df[schema['payout']]
        # Check if already in [-2, 2] range — assume it's already a payout
        if raw.abs().max() <= 2.5:
            return raw.astype(float)
        # Otherwise might be in bet units — normalize
        return raw.astype(float)

    # Construct from flags
    payout = pd.Series(-1.0, index=df.index)

    if schema['win_flag']:
        win = df[schema['win_flag']].astype(bool)
        payout[win] = 1.0

    if schema['blackjack_flag']:
        bj = df[schema['blackjack_flag']].astype(bool)
        payout[bj] = 1.5

    if schema['push_flag']:
        push = df[schema['push_flag']].astype(bool)
        payout[push] = 0.0

    if schema['bust_flag']:
        bust = df[schema['bust_flag']].astype(bool)
        payout[bust] = -1.0

    return payout


# ---------------------------------------------------------------------------
# EV-maximizing label generation (Approach A)
# ---------------------------------------------------------------------------

def create_ev_labels(df: pd.DataFrame, schema: dict) -> pd.Series:
    """
    For each discretized game-state bucket, find the action with the highest
    mean payout. Label each row with that optimal action.

    Bucket dimensions: player_total_bin × dealer_upcard × soft × pair × tc_bin
    """
    df = df.copy()

    # Discretize player total
    total_bins = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]
    total_labels = ['<=8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    df['_pt_bin'] = pd.cut(df['player_total'], bins=total_bins, labels=total_labels, right=True)

    # Discretize true count
    tc_bins = [-np.inf, -2, -1, 0, 1, 2, np.inf]
    tc_labels = ['<=-2', '-1', '0', '+1', '+2', '>=+2']
    df['_tc_bin'] = pd.cut(df['true_count'], bins=tc_bins, labels=tc_labels)

    group_cols = ['_pt_bin', 'dealer_upcard_val', 'is_soft', 'is_pair', '_tc_bin']
    action_col = '_action_norm'

    df[action_col] = df[schema['action']].apply(normalize_action)
    df = df.dropna(subset=[action_col, 'payout'])

    print(f"Action distribution (normalized):\n{df[action_col].value_counts()}\n")

    # EV per bucket × action
    ev = (
        df.groupby(group_cols + [action_col])['payout']
        .mean()
        .reset_index()
        .rename(columns={'payout': 'mean_ev'})
    )

    # Best action per bucket
    best = ev.loc[ev.groupby(group_cols)['mean_ev'].idxmax()].copy()
    best['_bucket'] = best[group_cols].apply(lambda r: tuple(r), axis=1)
    bucket_to_action = dict(zip(best['_bucket'], best[action_col]))

    df['_bucket'] = df[group_cols].apply(lambda r: tuple(r), axis=1)
    df['optimal_action'] = df['_bucket'].map(bucket_to_action)

    n_unlabeled = df['optimal_action'].isna().sum()
    if n_unlabeled > 0:
        print(f"Warning: {n_unlabeled} rows ({n_unlabeled/len(df):.1%}) have no EV label "
              f"(sparse buckets). Dropping them.")
    df = df.dropna(subset=['optimal_action'])

    print(f"Optimal action distribution:\n{df['optimal_action'].value_counts()}\n")
    print(f"Unique buckets: {len(bucket_to_action)}")

    return df


# ---------------------------------------------------------------------------
# Train / test split (by game_id to prevent leakage)
# ---------------------------------------------------------------------------

def game_split(df: pd.DataFrame, schema: dict, test_size: float = 0.2, seed: int = 42):
    """Split by game_id so no game's rows appear in both train and test."""
    game_ids = df[schema['game_id']].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    split = int(len(game_ids) * (1 - test_size))
    train_ids = set(game_ids[:split])
    test_ids = set(game_ids[split:])
    train = df[df[schema['game_id']].isin(train_ids)]
    test = df[df[schema['game_id']].isin(test_ids)]
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
        multi_class='multinomial',
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
    gs = GridSearchCV(base, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"Best RF params: {gs.best_params_}  (CV macro-F1 = {gs.best_score_:.4f})")
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/blackjack_simulator.csv')
    parser.add_argument('--sample', type=int, default=200000,
                        help='Row sample size (use 0 for full dataset)')
    parser.add_argument('--full-tune', action='store_true',
                        help='Run full grid search (slow)')
    args = parser.parse_args()

    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        print("Place the Kaggle CSV at data/blackjack_simulator.csv and re-run.")
        sys.exit(1)

    print(f"Loading data from {data_path} ...")
    df_raw = pd.read_csv(data_path)
    print(f"Raw shape: {df_raw.shape}")
    print(f"\nFirst 3 rows:\n{df_raw.head(3)}")
    print(f"\nDtypes:\n{df_raw.dtypes}")
    print(f"\nDescribe:\n{df_raw.describe()}")

    schema = detect_schema(df_raw)

    # Stratified subsample if requested
    if args.sample and args.sample < len(df_raw):
        action_col = schema['action']
        df_raw = (
            df_raw.groupby(action_col, group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), int(args.sample * len(g) / len(df_raw))),
                random_state=42
            ))
        ).reset_index(drop=True)
        print(f"Subsampled to {len(df_raw):,} rows (stratified by action)")

    # Feature engineering
    print("Building features ...")
    df_feat, feature_cols = build_features(df_raw, schema)
    df_feat['payout'] = build_payout(df_feat, schema)

    # EV labels
    print("Creating EV-optimal labels ...")
    df_labeled = create_ev_labels(df_feat, schema)

    # Train/test split
    df_train, df_test = game_split(df_labeled, schema)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['optimal_action'])
    y_test = le.transform(df_test['optimal_action'])
    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values

    print(f"\nClasses: {le.classes_}")

    # Save test set and label encoder for evaluation
    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    joblib.dump(le, models_dir / 'label_encoder.joblib')
    np.save(models_dir / 'X_test.npy', X_test)
    np.save(models_dir / 'y_test.npy', y_test)

    # Also save a small df for simulation (with all features + payout)
    df_test.to_parquet(models_dir / 'test_set.parquet', index=False)

    # Baseline: Logistic Regression
    print("\n--- Training Logistic Regression baseline ---")
    lr = train_logistic(X_train, y_train)
    lr_score = cross_val_score(lr, X_train, y_train, cv=3, scoring='f1_macro').mean()
    print(f"LR baseline 3-fold macro-F1: {lr_score:.4f}")
    joblib.dump(lr, models_dir / 'lr_baseline.joblib')

    # Main: Random Forest
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
    print(f"RF saved to models/rf_model.joblib")

    # Save feature names alongside model
    joblib.dump(feature_cols, models_dir / 'feature_cols.joblib')

    # Quick test-set accuracy
    rf_acc = (rf.predict(X_test) == y_test).mean()
    lr_acc = (lr.predict(X_test) == y_test).mean()
    print(f"\nTest accuracy — RF: {rf_acc:.4f} | LR: {lr_acc:.4f}")
    print("\nTraining complete. Run `python src/evaluate.py` to generate full metrics and plots.")


if __name__ == '__main__':
    main()
