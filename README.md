---
title: Blackjack Optimal Action Predictor
emoji: 🃏
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: deployment/app.py
pinned: false
---

# All In on AI: Predicting Optimal Blackjack Plays

**DSC 465 Final Project** — Sam Herdlick, Seamus Gerner, Garrett Jones

A machine learning system that recommends the optimal blackjack action given full game state, including card counting features derived from the Hi-Lo system.

---

## Motivation

Traditional basic strategy uses static probability tables and ignores deck composition. This model incorporates *full game state* — including remaining deck composition — to recommend actions that maximize expected value (EV) beyond static tables.

---

## Mathematical Formulation

### Random Forest Classifier

Given training set $D = \{(x_i, y_i)\}_{i=1}^n$ where $x_i \in \mathbb{R}^d$ is the game-state feature vector and $y_i \in \{\text{Hit, Stand, Double, Split}\}$:

1. For $b = 1, \dots, B$ (number of trees):
   - Draw bootstrap sample $D_b$ from $D$
   - Grow decision tree $T_b$ on $D_b$, at each split selecting from a random subset of $\sqrt{d}$ features
   - Split criterion: Gini impurity $G = 1 - \sum_{k=1}^K p_k^2$ where $p_k$ is the proportion of class $k$ at the node

2. Prediction: $\hat{y}(x) = \text{mode}\{T_1(x), T_2(x), \dots, T_B(x)\}$

### Evaluation Objective

Maximize expected value per hand:

$$EV = \sum_{a \in A} P(a \mid s) \cdot \mathbb{E}[\text{payout} \mid s, a]$$

where $s$ is game state and $a$ is action.

---

## Dataset

**Source:** [900000 Hands of Blackjack Results](https://www.kaggle.com/datasets/mojocolors/900000-hands-of-blackjack-results) (Kaggle)
**File:** `data/blkjckhands.csv` (~59 MB, tracked in Git LFS)

- 900,000 observations, 150,000 unique games, 6 players per game (`Player1`..`Player6`)
- 20 columns: player cards (`card1`..`card5`), dealer cards (`dealcard1`..`dealcard5`),
  hand totals (`sumofcards`, `sumofdeal`), outcome flags (`winloss`, `blkjck`,
  `plybustbeat`, `dlbustbeat`), and payouts (`plwinamt`, `dlwinamt`)
- Cards are encoded `1`–`10` where **`1` = Ace** and **`10` = any of {10, J, Q, K}**

---

## Features

| Feature | Description |
|---|---|
| `player_total` | Hand total (hard value) |
| `is_soft` | Ace counted as 11 |
| `is_pair` | First two cards same rank |
| `pair_val` | Value of paired card |
| `n_cards` | Cards in hand |
| `dealer_upcard_val` | Dealer's visible card (1–10) |
| `running_count` | Hi-Lo running count |
| `true_count` | Running count ÷ decks remaining |
| `frac_high` | Fraction of 10+ cards remaining |
| `frac_low` | Fraction of 2–6 cards remaining |
| `cards_dealt` | Total cards dealt from shoe |

### Hi-Lo Count System
- Cards 2–6: **+1**
- Cards 7–9: **0**
- Cards 10/J/Q/K/A: **−1**

---

## Target Labels (Approach A — EV Maximizing)

The dataset records *outcomes* but not the player's chosen action or a game ID,
so we infer both:

- **`game_id`**: every 6 consecutive rows = one game (Player1..Player6 cycle)
- **First action**: from card sequence + payouts
  - `S` (Stand) when there's no `card3`
  - `D` (Double) when there are exactly 3 cards AND payout shows 2× stake
  - `H` (Hit) otherwise
  - **Split is dropped** — splits aren't recoverable from the row structure

EV labeling pipeline:
1. Bucket rows by `(player_total_bin, dealer_upcard, soft, pair, true_count_bin)`
2. Compute mean payout per action within each bucket
3. Label each row with the highest-EV action in its bucket
4. Train classifier to predict that label from continuous features

---

## Models

| Model | Role |
|---|---|
| `LogisticRegression` | Baseline with class balancing |
| `RandomForestClassifier` | Main model, tuned via 5-fold CV |

Train/test split is by `game_id` to prevent within-game correlation leakage.

---

## Project Structure

```
blackjack-ml/
├── data/
│   └── blackjack_simulator.csv          # user provides
├── notebooks/
│   └── 01_eda.ipynb                     # exploratory data analysis
├── src/
│   ├── features.py                      # feature engineering
│   ├── train.py                         # training pipeline
│   ├── evaluate.py                      # evaluation + simulation
│   └── basic_strategy.py                # basic strategy logic
├── models/
│   └── rf_model.joblib                  # serialized final model
├── reports/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── model_vs_basic_strategy.png
│   ├── ev_curve.png
│   └── metrics.json
├── deployment/
│   ├── app.py                           # Gradio interface
│   ├── requirements.txt
│   └── README.md                        # HF Space readme
└── requirements.txt
```

---

## Setup & Usage

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Dataset is already in the repo at data/blkjckhands.csv (Git LFS)
# If cloning fresh, run: git lfs pull

# 3. Explore data
jupyter notebook notebooks/01_eda.ipynb

# 4. Train (uses 200K sample by default; --sample 0 for full dataset)
python src/train.py --data data/blkjckhands.csv --sample 200000

# 5. Evaluate & generate plots
python src/evaluate.py

# 6. Test deployment locally
python deployment/app.py
```

---

## Deploying to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
   - SDK: **Gradio**
2. Copy these files into the Space repo:
   ```
   deployment/app.py       → app.py
   deployment/requirements.txt → requirements.txt
   deployment/README.md    → README.md
   models/rf_model.joblib  → rf_model.joblib
   models/label_encoder.joblib → label_encoder.joblib
   src/features.py         → src/features.py     (imported by app.py)
   src/basic_strategy.py   → src/basic_strategy.py
   ```
3. Push to the Space:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/blackjack-predictor
   # copy files in
   git add .
   git commit -m "initial deploy"
   git push
   ```
4. The Space will build automatically (~2 min).

---

## Results

See `reports/metrics.json` after training for final numbers.
Key metrics: accuracy, macro-F1, simulated EV per hand vs basic strategy.

---

## Assumptions & Limitations

- **No explicit action column.** The Kaggle dataset records what cards/outcomes
  occurred but not what action the simulator chose. We infer the player's first
  action (Hit / Stand / Double) from `card3`/`card4`/`card5` + `plwinamt`/`dlwinamt`.
  Split is unrecoverable and excluded.
- **No game_id column.** Inferred as `row_index // 6`. Verified by the fact that
  `PlayerNo` cycles `Player1`..`Player6` deterministically.
- **Card encoding.** `1` = Ace (treated as 11 by default, downgraded to 1 to
  avoid bust). `10` collapses {10, J, Q, K} — pair detection treats them all
  as identical 10s, which is correct for basic strategy.
- **Card counting features** assume a 6-deck shoe and that rows within each game
  are ordered by player position.
- **The simulation** uses within-dataset EV estimates as a proxy for true game EV,
  since we cannot observe counterfactual outcomes.
- **Class imbalance** (Hit/Stand dominate over Double) is handled with `class_weight="balanced"`.
