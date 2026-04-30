---
title: Blackjack Optimal Action Predictor
emoji: 🃏
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Blackjack Optimal Action Predictor

**DSC 465 Final Project** — Sam Herdlick, Seamus Gerner, Garrett Jones

A Random Forest model that recommends the optimal blackjack action (Hit / Stand / Double Down / Split) given your current hand and shoe state, incorporating Hi-Lo card counting features.

## How it works

1. Enter your two cards and the dealer's upcard
2. Optionally enter your running count and estimated decks remaining
3. The model returns a recommended action with confidence probabilities
4. Basic strategy recommendation is shown for comparison

## Model

- **Architecture:** Random Forest (300 trees, max_depth=20)
- **Training data:** 900K blackjack hands (Kaggle), subsampled with stratification
- **Labels:** EV-maximizing action per game-state bucket (Approach A)
- **Key features:** player total, soft/pair flags, dealer upcard, true count, high/low card fractions
- **Train/test split:** by game_id to prevent leakage

## Card counting

The Hi-Lo system assigns:
- Cards 2–6: +1 (low — good for player)
- Cards 7–9: 0 (neutral)
- Cards 10/J/Q/K/A: −1 (high — bad for player)

True count = Running count ÷ Decks remaining. A high positive count favors the player.
