"""
Gradio interface for the Blackjack Optimal Action Predictor.
Deploy to Hugging Face Spaces: place this file and rf_model.joblib in the same directory.
"""

import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Support both local dev (model in ../models/) and HF Spaces (model alongside app.py)
_here = Path(__file__).resolve().parent
_model_candidates = [
    _here / 'rf_model.joblib',
    _here.parent / 'models' / 'rf_model.joblib',
]
_label_candidates = [
    _here / 'label_encoder.joblib',
    _here.parent / 'models' / 'label_encoder.joblib',
]

RF_PATH = next((p for p in _model_candidates if p.exists()), None)
LE_PATH = next((p for p in _label_candidates if p.exists()), None)

if RF_PATH is None:
    raise FileNotFoundError(
        "rf_model.joblib not found. Run `python src/train.py` first, "
        "then copy models/rf_model.joblib and models/label_encoder.joblib "
        "into the deployment/ folder."
    )

RF_MODEL = joblib.load(RF_PATH)
LABEL_ENCODER = joblib.load(LE_PATH)
CLASSES = LABEL_ENCODER.classes_

sys.path.insert(0, str(_here.parent / 'src'))
from features import build_features_single, FEATURE_COLS
from basic_strategy import basic_strategy, normalize_action

CARD_OPTIONS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

ACTION_LABELS = {
    'H': 'Hit',
    'S': 'Stand',
    'D': 'Double Down',
    'P': 'Split',
}

ACTION_COLORS = {
    'H': '#e74c3c',   # red
    'S': '#27ae60',   # green
    'D': '#2980b9',   # blue
    'P': '#8e44ad',   # purple
}


def card_to_display(card: str) -> str:
    suits = {'A': '♠', 'K': '♦', 'Q': '♥', 'J': '♣', '10': '♠', '9': '♦', '8': '♥',
             '7': '♣', '6': '♠', '5': '♦', '4': '♥', '3': '♣', '2': '♠'}
    return f"{card}{suits.get(card, '')}"


def predict(card1, card2, dealer_card, running_count, decks_remaining):
    """Core prediction function called by Gradio."""
    if not all([card1, card2, dealer_card]):
        return "Please select all cards.", "", "", ""

    # Feature vector
    feats = build_features_single(
        player_cards=[card1, card2],
        dealer_upcard=dealer_card,
        running_count_val=float(running_count),
        decks_remaining_val=float(decks_remaining),
    )
    X = np.array([[feats[col] for col in FEATURE_COLS]])

    # RF prediction
    probs = RF_MODEL.predict_proba(X)[0]
    pred_idx = np.argmax(probs)
    pred_action = CLASSES[pred_idx]
    pred_label = ACTION_LABELS.get(pred_action, pred_action)

    # Basic strategy
    from features import card_to_value, is_soft as _is_soft, is_pair as _is_pair, pair_value as _pair_value
    bs_action = basic_strategy(
        player_total=feats['player_total'],
        soft=bool(feats['is_soft']),
        pair=bool(feats['is_pair']),
        pair_card_val=feats['pair_val'],
        dealer_upcard=feats['dealer_upcard_val'],
    )
    bs_label = ACTION_LABELS.get(bs_action, bs_action)

    # Build probability table HTML
    prob_rows = ""
    for cls, prob in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
        bar_width = int(prob * 200)
        color = ACTION_COLORS.get(cls, '#888')
        prob_rows += (
            f"<tr>"
            f"<td style='padding:4px 8px'>{ACTION_LABELS.get(cls, cls)}</td>"
            f"<td style='padding:4px 8px'>{prob:.1%}</td>"
            f"<td style='padding:4px 2px'>"
            f"<div style='background:{color};width:{bar_width}px;height:16px;border-radius:3px'></div>"
            f"</td>"
            f"</tr>"
        )
    prob_html = f"<table style='font-size:14px;border-collapse:collapse'>{prob_rows}</table>"

    # Feature summary
    hand_desc = "Soft" if feats['is_soft'] else ("Pair" if feats['is_pair'] else "Hard")
    feat_html = (
        f"<small>"
        f"Hand: {hand_desc} {feats['player_total']} &nbsp;|&nbsp; "
        f"Dealer: {feats['dealer_upcard_val']} &nbsp;|&nbsp; "
        f"True count: {feats['true_count']:.2f} &nbsp;|&nbsp; "
        f"Hi%: {feats['frac_high']:.1%} &nbsp;|&nbsp; "
        f"Lo%: {feats['frac_low']:.1%}"
        f"</small>"
    )

    # Recommendation box
    color = ACTION_COLORS.get(pred_action, '#888')
    rec_html = (
        f"<div style='background:{color};color:white;padding:20px;border-radius:8px;"
        f"font-size:28px;font-weight:bold;text-align:center'>"
        f"{pred_label}"
        f"</div>"
    )

    agree = " ✓ Agrees with basic strategy" if pred_action == bs_action else f" (Basic strategy: {bs_label})"
    bs_html = f"<div style='padding:8px;font-size:14px'>Basic strategy: <b>{bs_label}</b>{agree}</div>"

    return rec_html, prob_html, bs_html, feat_html


with gr.Blocks(title="Blackjack Optimal Action Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🃏 Blackjack Optimal Action Predictor
    **DSC 465 Final Project** — Sam Herdlick, Seamus Gerner, Garrett Jones

    Enter your hand and game state to get an ML-powered action recommendation.
    The model was trained on 900K hands using EV-maximizing labels and incorporates
    card counting via the Hi-Lo system.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Your Hand")
            card1 = gr.Dropdown(choices=CARD_OPTIONS, label="Card 1", value='A')
            card2 = gr.Dropdown(choices=CARD_OPTIONS, label="Card 2", value='6')

            gr.Markdown("### Dealer")
            dealer_card = gr.Dropdown(choices=CARD_OPTIONS, label="Dealer Upcard", value='7')

            gr.Markdown("### Shoe State")
            running_count = gr.Number(label="Running Count (Hi-Lo)", value=0, step=1)
            decks_remaining = gr.Number(label="Decks Remaining", value=6, minimum=0.5, maximum=8, step=0.5)

            predict_btn = gr.Button("Get Recommendation", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Recommendation")
            rec_out = gr.HTML()
            bs_out = gr.HTML()
            gr.Markdown("### Class Probabilities")
            prob_out = gr.HTML()
            gr.Markdown("### Hand Details")
            feat_out = gr.HTML()

    predict_btn.click(
        fn=predict,
        inputs=[card1, card2, dealer_card, running_count, decks_remaining],
        outputs=[rec_out, prob_out, bs_out, feat_out],
    )

    # Run on load with defaults
    demo.load(
        fn=predict,
        inputs=[card1, card2, dealer_card, running_count, decks_remaining],
        outputs=[rec_out, prob_out, bs_out, feat_out],
    )

    gr.Markdown("""
    ---
    **Action guide:** Hit = take another card | Stand = keep current hand |
    Double Down = double bet, take exactly one more card | Split = split pair into two hands

    *Model: Random Forest trained on EV-maximizing labels. Card counting features use Hi-Lo system.*
    """)

if __name__ == '__main__':
    demo.launch()
