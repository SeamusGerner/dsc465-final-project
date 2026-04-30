"""Feature engineering for blackjack ML model."""

import numpy as np
import pandas as pd
from typing import List, Union

CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11,
    '1': 1,  # some datasets encode ace as 1
}

CARD_RANKS = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    '1': 14,
}

DECKS_IN_SHOE = 6
CARDS_PER_SHOE = 52 * DECKS_IN_SHOE  # 312
HIGH_CARDS_PER_SHOE = 5 * 4 * DECKS_IN_SHOE   # 10,J,Q,K,A = 120
LOW_CARDS_PER_SHOE = 5 * 4 * DECKS_IN_SHOE    # 2-6 = 120


def card_to_value(card) -> int:
    """Convert a card representation to its blackjack point value (ace = 11)."""
    if pd.isna(card):
        return 0
    s = str(card).strip().upper()
    if s in CARD_VALUES:
        return CARD_VALUES[s]
    try:
        v = int(s)
        return min(v, 10)  # cap at 10; ace encoded as 1 stays 1
    except ValueError:
        return 0


def card_to_rank(card) -> int:
    """Convert card to rank integer (used for pair detection)."""
    if pd.isna(card):
        return 0
    s = str(card).strip().upper()
    if s in CARD_RANKS:
        return CARD_RANKS[s]
    try:
        v = int(s)
        return v if v != 1 else 14  # ace as 14
    except ValueError:
        return 0


def hand_total(cards: List) -> int:
    """
    Compute optimal blackjack hand total.
    Counts aces as 11 until that would bust, then counts as 1.
    """
    values = [card_to_value(c) for c in cards if not (pd.isna(c) if not isinstance(c, str) else False)]
    values = [v for v in values if v > 0]
    total = sum(values)
    aces = sum(1 for v in values if v == 11)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def is_soft(cards: List) -> bool:
    """True if the hand contains an ace currently counted as 11."""
    values = [card_to_value(c) for c in cards if not (pd.isna(c) if not isinstance(c, str) else False)]
    values = [v for v in values if v > 0]
    total = sum(values)
    aces = sum(1 for v in values if v == 11)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return aces > 0


def is_pair(cards: List) -> bool:
    """True if the first two cards share the same rank (eligible to split)."""
    valid = [c for c in cards if not (pd.isna(c) if not isinstance(c, str) else False)]
    if len(valid) < 2:
        return False
    return card_to_rank(valid[0]) == card_to_rank(valid[1])


def pair_value(cards: List) -> int:
    """Return the value of the paired card (0 if not a pair)."""
    if not is_pair(cards):
        return 0
    return card_to_value(cards[0])


def hilo_value(card) -> int:
    """Hi-Lo count contribution of a single card: +1 low, 0 neutral, -1 high."""
    v = card_to_value(card)
    if v == 0:
        return 0
    if 2 <= v <= 6:
        return 1
    if v >= 10:  # 10,J,Q,K,A (11 maps to 10+ bucket)
        return -1
    return 0  # 7,8,9


def true_count(running_count: float, decks_remaining: float) -> float:
    """True count = running count / decks remaining."""
    if decks_remaining <= 0:
        return 0.0
    return running_count / decks_remaining


def frac_high_remaining(high_cards_dealt: int, total_dealt: int) -> float:
    """Fraction of high cards (10+) remaining in shoe."""
    remaining = CARDS_PER_SHOE - total_dealt
    high_remaining = HIGH_CARDS_PER_SHOE - high_cards_dealt
    if remaining <= 0:
        return HIGH_CARDS_PER_SHOE / CARDS_PER_SHOE
    return max(0.0, high_remaining / remaining)


def frac_low_remaining(low_cards_dealt: int, total_dealt: int) -> float:
    """Fraction of low cards (2-6) remaining in shoe."""
    remaining = CARDS_PER_SHOE - total_dealt
    low_remaining = LOW_CARDS_PER_SHOE - low_cards_dealt
    if remaining <= 0:
        return LOW_CARDS_PER_SHOE / CARDS_PER_SHOE
    return max(0.0, low_remaining / remaining)


def decks_remaining(total_dealt: int) -> float:
    """Estimate decks remaining from total cards dealt."""
    return max(0.5, (CARDS_PER_SHOE - total_dealt) / 52.0)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Construct all model features from the raw dataframe.

    schema keys used:
        player_cards  : list of column names for player's cards (in order)
        dealer_upcard : column name for dealer's visible card
        game_id       : column name for game identifier
        position      : column name for player position within game (optional)

    Returns a dataframe with columns:
        player_total, is_soft, is_pair, pair_val, n_cards,
        dealer_upcard, running_count, true_count,
        frac_high, frac_low, cards_dealt
    """
    player_card_cols = schema['player_cards']
    dealer_col = schema['dealer_upcard']
    game_col = schema['game_id']
    pos_col = schema.get('position')

    # Sort for sequential processing within each game/shoe
    sort_cols = [game_col] + ([pos_col] if pos_col else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- Player hand features ---
    player_cards = df[player_card_cols].values.tolist()

    df['player_total'] = [hand_total(row) for row in player_cards]
    df['is_soft'] = [int(is_soft(row)) for row in player_cards]
    df['is_pair'] = [int(is_pair(row)) for row in player_cards]
    df['pair_val'] = [pair_value(row) for row in player_cards]
    df['n_cards'] = [sum(1 for c in row if not (pd.isna(c) if not isinstance(c, str) else False) and str(c).strip() != '') for row in player_cards]

    # --- Dealer features ---
    df['dealer_upcard_val'] = df[dealer_col].apply(card_to_value)

    # --- Card counting features (computed per shoe/game) ---
    # For each row, all_cards contains the player's cards AND dealer upcard visible to the table
    all_card_cols = player_card_cols + [dealer_col]

    running_counts = []
    true_counts = []
    fracs_high = []
    fracs_low = []
    cards_dealt_list = []

    cumulative_rc = {}       # game_id -> running count so far
    cumulative_high = {}     # game_id -> high cards dealt so far
    cumulative_low = {}      # game_id -> low cards dealt so far
    cumulative_total = {}    # game_id -> total cards dealt so far

    for _, row in df.iterrows():
        gid = row[game_col]
        if gid not in cumulative_rc:
            cumulative_rc[gid] = 0
            cumulative_high[gid] = 0
            cumulative_low[gid] = 0
            cumulative_total[gid] = 0

        # Snapshot BEFORE this hand's cards are added (state player sees at decision time)
        rc = cumulative_rc[gid]
        total_d = cumulative_total[gid]
        dr = decks_remaining(total_d)
        tc = true_count(rc, dr)
        fh = frac_high_remaining(cumulative_high[gid], total_d)
        fl = frac_low_remaining(cumulative_low[gid], total_d)

        running_counts.append(rc)
        true_counts.append(tc)
        fracs_high.append(fh)
        fracs_low.append(fl)
        cards_dealt_list.append(total_d)

        # Update cumulative counts with this hand's visible cards
        for col in all_card_cols:
            c = row[col]
            if pd.isna(c) if not isinstance(c, str) else not str(c).strip():
                continue
            v = card_to_value(c)
            if v == 0:
                continue
            cumulative_rc[gid] += hilo_value(c)
            if v >= 10:
                cumulative_high[gid] += 1
            elif v <= 6:
                cumulative_low[gid] += 1
            cumulative_total[gid] += 1

    df['running_count'] = running_counts
    df['true_count'] = true_counts
    df['frac_high'] = fracs_high
    df['frac_low'] = fracs_low
    df['cards_dealt'] = cards_dealt_list

    feature_cols = [
        'player_total', 'is_soft', 'is_pair', 'pair_val', 'n_cards',
        'dealer_upcard_val', 'running_count', 'true_count',
        'frac_high', 'frac_low', 'cards_dealt',
    ]
    return df, feature_cols


def build_features_single(
    player_cards: List[str],
    dealer_upcard: str,
    running_count_val: float = 0.0,
    decks_remaining_val: float = 6.0,
) -> dict:
    """
    Compute features for a single hand (used by the deployment app).
    Does not need a full dataframe — all card counting inputs provided directly.
    """
    total = hand_total(player_cards)
    soft = int(is_soft(player_cards))
    pair = int(is_pair(player_cards))
    pval = pair_value(player_cards)
    n = len([c for c in player_cards if str(c).strip()])
    duv = card_to_value(dealer_upcard)

    dr = max(0.5, decks_remaining_val)
    tc = true_count(running_count_val, dr)

    total_dealt_est = int((DECKS_IN_SHOE - dr) * 52)
    hi_dealt_est = max(0, int(HIGH_CARDS_PER_SHOE - (HIGH_CARDS_PER_SHOE / CARDS_PER_SHOE) * (CARDS_PER_SHOE - total_dealt_est) - running_count_val * 0.5))
    lo_dealt_est = max(0, int(LOW_CARDS_PER_SHOE - (LOW_CARDS_PER_SHOE / CARDS_PER_SHOE) * (CARDS_PER_SHOE - total_dealt_est) + running_count_val * 0.5))
    fh = frac_high_remaining(hi_dealt_est, total_dealt_est)
    fl = frac_low_remaining(lo_dealt_est, total_dealt_est)

    return {
        'player_total': total,
        'is_soft': soft,
        'is_pair': pair,
        'pair_val': pval,
        'n_cards': n,
        'dealer_upcard_val': duv,
        'running_count': running_count_val,
        'true_count': tc,
        'frac_high': fh,
        'frac_low': fl,
        'cards_dealt': total_dealt_est,
    }


FEATURE_COLS = [
    'player_total', 'is_soft', 'is_pair', 'pair_val', 'n_cards',
    'dealer_upcard_val', 'running_count', 'true_count',
    'frac_high', 'frac_low', 'cards_dealt',
]
