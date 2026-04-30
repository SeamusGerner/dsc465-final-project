"""
Basic strategy implemented as pure logic (not a copied table).

Actions: 'H' = Hit, 'S' = Stand, 'D' = Double (else Hit), 'P' = Split
'DS' = Double (else Stand) — used for soft hands where standing beats hitting

All rules derived from standard 6-deck, dealer-hits-soft-17, DAS allowed strategy.
"""

ACTION_HIT = 'H'
ACTION_STAND = 'S'
ACTION_DOUBLE = 'D'
ACTION_SPLIT = 'P'


def _normalize_upcard(dealer_upcard: int) -> int:
    """Treat ace (11) as 1 for lookup purposes."""
    if dealer_upcard == 11:
        return 1
    return dealer_upcard


def _pair_action(pair_card_val: int, dealer_up: int) -> str:
    """Return action for pairs."""
    d = _normalize_upcard(dealer_up)
    v = min(pair_card_val, 10)  # face cards all equal 10

    if v == 11 or v == 1:   # Aces
        return ACTION_SPLIT
    if v == 10:             # 10s — always stand
        return ACTION_STAND
    if v == 9:
        # Split unless dealer shows 7, 10, or Ace
        return ACTION_SPLIT if d not in (7, 10, 1) else ACTION_STAND
    if v == 8:
        return ACTION_SPLIT
    if v == 7:
        return ACTION_SPLIT if d <= 7 else ACTION_HIT
    if v == 6:
        return ACTION_SPLIT if d <= 6 else ACTION_HIT
    if v == 5:
        # Never split 5s — treat as hard 10
        return ACTION_DOUBLE if d <= 9 else ACTION_HIT
    if v == 4:
        # Split only vs dealer 5 or 6
        return ACTION_SPLIT if d in (5, 6) else ACTION_HIT
    if v in (2, 3):
        return ACTION_SPLIT if d <= 7 else ACTION_HIT

    return ACTION_HIT


def _soft_action(total: int, dealer_up: int) -> str:
    """Return action for soft hands (ace counted as 11). total includes the ace as 11."""
    d = _normalize_upcard(dealer_up)

    if total >= 20:   # Soft 20 (A+9)
        return ACTION_STAND
    if total == 19:   # Soft 19 (A+8)
        return ACTION_DOUBLE if d == 6 else ACTION_STAND
    if total == 18:   # Soft 18 (A+7)
        if d in (2, 3, 4, 5, 6):
            return ACTION_DOUBLE
        if d in (7, 8):
            return ACTION_STAND
        return ACTION_HIT
    if total == 17:   # Soft 17 (A+6)
        return ACTION_DOUBLE if d in (3, 4, 5, 6) else ACTION_HIT
    if total == 16:   # Soft 16 (A+5)
        return ACTION_DOUBLE if d in (4, 5, 6) else ACTION_HIT
    if total == 15:   # Soft 15 (A+4)
        return ACTION_DOUBLE if d in (4, 5, 6) else ACTION_HIT
    if total == 14:   # Soft 14 (A+3)
        return ACTION_DOUBLE if d in (5, 6) else ACTION_HIT
    if total == 13:   # Soft 13 (A+2)
        return ACTION_DOUBLE if d in (5, 6) else ACTION_HIT

    return ACTION_HIT


def _hard_action(total: int, dealer_up: int) -> str:
    """Return action for hard hands."""
    d = _normalize_upcard(dealer_up)

    if total >= 17:
        return ACTION_STAND
    if total == 16:
        return ACTION_STAND if d <= 6 else ACTION_HIT
    if total == 15:
        return ACTION_STAND if d <= 6 else ACTION_HIT
    if total == 14:
        return ACTION_STAND if d <= 6 else ACTION_HIT
    if total == 13:
        return ACTION_STAND if d <= 6 else ACTION_HIT
    if total == 12:
        return ACTION_STAND if d in (4, 5, 6) else ACTION_HIT
    if total == 11:
        # Double vs dealer 2-10; hit vs ace
        return ACTION_DOUBLE if d != 1 else ACTION_HIT
    if total == 10:
        return ACTION_DOUBLE if d <= 9 else ACTION_HIT
    if total == 9:
        return ACTION_DOUBLE if d in (3, 4, 5, 6) else ACTION_HIT

    return ACTION_HIT  # hard 8 or less


def basic_strategy(
    player_total: int,
    soft: bool,
    pair: bool,
    pair_card_val: int,
    dealer_upcard: int,
) -> str:
    """
    Return basic strategy action.

    Parameters
    ----------
    player_total  : best hand total (int)
    soft          : True if hand is soft (ace counted as 11)
    pair          : True if first two cards are a pair
    pair_card_val : value of the paired card (0 if not a pair)
    dealer_upcard : dealer's visible card value (1-11)

    Returns
    -------
    One of 'H', 'S', 'D', 'P'
    """
    if pair:
        return _pair_action(pair_card_val, dealer_upcard)
    if soft:
        return _soft_action(player_total, dealer_upcard)
    return _hard_action(player_total, dealer_upcard)


def basic_strategy_from_features(row: dict) -> str:
    """Convenience wrapper accepting a feature dict."""
    return basic_strategy(
        player_total=int(row['player_total']),
        soft=bool(row['is_soft']),
        pair=bool(row['is_pair']),
        pair_card_val=int(row['pair_val']),
        dealer_upcard=int(row['dealer_upcard_val']),
    )


# Action label normalizer — unify whatever strings the dataset uses
ACTION_MAP = {
    'hit': 'H', 'h': 'H',
    'stand': 'S', 's': 'S', 'stay': 'S',
    'double': 'D', 'd': 'D', 'double down': 'D', 'doubledown': 'D',
    'split': 'P', 'p': 'P',
    'H': 'H', 'S': 'S', 'D': 'D', 'P': 'P',
}


def normalize_action(raw) -> str:
    """Normalize a raw action string to H/S/D/P."""
    if pd.isna(raw) if not isinstance(raw, str) else False:
        return None
    key = str(raw).strip().lower()
    return ACTION_MAP.get(key, ACTION_MAP.get(str(raw).strip(), None))


import pandas as pd  # noqa: E402 (placed here to avoid circular at module level)
