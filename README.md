# Credit Card Fraud Analytics: Risk Monitoring and Alert Triage

Most fraud detection projects end the same way: train a model, save a confusion matrix, move on. After going through a few of them, I kept running into the same gap. A model score sitting in a CSV does nothing for an investigator who needs to know which transaction to open first, how urgent it is, and what actually triggered the flag.

So I built the layer between the model and the analyst. The goal was to simulate how a first-line fraud monitoring workflow actually operates, not just produce predictions.

---

## Dataset

I used the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Key columns:

- `Time` — seconds elapsed since the first transaction
- `V1` through `V28` — anonymized PCA-transformed features
- `Amount` — transaction amount
- `Class` — fraud label (1 = fraud, 0 = not fraud)

Fraud makes up about 0.17% of the dataset. Worth stating upfront because it shapes almost every modeling decision downstream, especially around which metrics actually matter.

---

## What the pipeline does

1. Trains a Random Forest classifier on engineered features
2. Runs a rules-based screening baseline as a comparison point
3. Sweeps candidate alert thresholds on a validation set and selects one based on a precision floor plus F1 tradeoff
4. Assigns risk bands and generates plain-English alert reasons for flagged transactions
5. Builds a prioritized investigator queue sorted by fraud probability and transaction size
6. Runs SQL monitoring queries against the scored output to produce summary tables

---

## Feature engineering

The `V1`-`V28` columns are the backbone, but I layered in a few interpretable signals on top:

- Log-transformed amount (the raw distribution is heavily right-skewed)
- High-amount flags at the 95th and 99th percentiles
- Micro-transaction flag at the 5th percentile, since small test charges before a larger fraud are a known pattern
- Off-hours proxy derived from `Time`, covering roughly midnight to 5am
- A count of how many `V` columns simultaneously fall in their extreme tails (`extreme_v_count`)
- Average absolute deviation across all `V` columns as a composite anomaly signal

One thing worth calling out: all percentile cutoffs are fit on the training set only, then applied to validation and test. Fitting on the full dataset would let future information leak into feature construction, and the results would be misleading.

---

## Why a rules-based baseline

Fraud teams almost always have some form of rule-based screening already running before a model ever enters the picture. A model needs to justify itself against whatever process it's replacing, not just against a blank slate.

The baseline uses combinations of:
- Very high transaction amounts
- Off-hours activity paired with anomaly signals
- Micro-transaction anomaly combinations
- High-amount plus multi-signal anomalies

Putting the two side by side is probably the most practically useful part of the project. The numbers tell a clear story (see results below).

---

## Threshold selection

Setting the alert threshold is a real operational decision. Lower thresholds catch more fraud but push more false positives into the analyst queue. Higher thresholds keep the queue clean but leave more fraud undetected. Picking a number without making that tradeoff explicit felt like cheating.

Split structure:
- **Training** — fit the model and feature cutoffs
- **Validation** — sweep thresholds from 0.30 to 0.90, pick the operating point
- **Test** — locked away until final reporting

For selection, I required a minimum precision of 0.40 first, then picked the best F1 among thresholds clearing that bar. A queue running below 40% real fraud isn't a useful tool for an analyst. Each threshold also gets a label:

| Mode | Threshold |
|------|-----------|
| Aggressive | ≤ 0.30 |
| Balanced | 0.31 – 0.50 |
| High Precision | > 0.50 |

---

## Risk banding and alert triage

After scoring, every transaction gets assigned a risk band:

| Band | Probability |
|------|-------------|
| Low | 0.00 – 0.30 |
| Medium | 0.30 – 0.70 |
| High | 0.70 – 0.90 |
| Critical | 0.90 – 1.00 |

Every flagged transaction also gets a plain-English alert reason, a review priority (Immediate / Same Day / Queue / Monitor), and a recommended action.

A raw list of probabilities doesn't help an investigator work efficiently. Knowing *why* a transaction was flagged, whether it's an unusual amount, off-hours activity, or a cluster of anomaly signals, lets a reviewer make a judgment call in seconds instead of minutes.

---

## SQL monitoring layer

The scored output gets loaded into SQLite and a set of monitoring queries generates summary tables: fraud rate by risk band, alerts by review priority, top alert reason combinations, hourly volume, and average amount by alert status.

Worth being clear: the SQL runs against the project's own scored output, not a live upstream database. It's meant to demonstrate what downstream reporting logic looks like in practice, not to simulate a production data environment.

---

## Results

The validation sweep selected a **Balanced threshold of 0.50**. On the locked test set:

| Metric | Value |
|--------|-------|
| Alerts generated | 94 |
| Frauds caught | 82 |
| Precision | 87.23% |
| Recall | 82.83% |

The rules baseline generated 1,513 alerts and caught 42 frauds at 2.78% precision. The model generated roughly 13x fewer alerts while catching more than twice as many actual fraud cases. Without the side-by-side, the model numbers look fine in isolation. Against the baseline, the improvement becomes concrete.

Risk banding held up meaningfully:
- Critical band observed fraud rate: **92.5%**
- High band observed fraud rate: **69.23%**

The bands are doing real segmentation work, not just sorting the output cosmetically.

---

## Limitations

A few things worth being direct about:

- No SAR filing, sanctions screening, case management, or BSA compliance layer. Calling it a production AML system would be a stretch.
- The dataset is public and anonymized. The `V` features have no interpretable meaning, which caps how specific the alert reasons can actually be.
- The SQL layer is self-contained. It runs against scored output I generated, not an upstream warehouse or live monitoring table.

Throughout the project I tried to frame elevated scores and anomaly flags as reasons to investigate, not conclusions. In a real environment, an analyst makes the final call after reviewing context. The model's job is to get the right transactions in front of them.

---

## Forensic accounting connection

A forensic accounting mindset influenced how the triage layer is designed. In forensic work, unusual patterns are treated as red flags requiring corroboration, not automatic findings. Carrying that framing into the alert logic meant building a queue meant to support human judgment rather than replace it.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download creditcard.csv from Kaggle and place it at data/raw/creditcard.csv

python src/train_fraud_model.py
```

---

## Outputs

**Processed data** (`data/processed/`)
- `scored_test_transactions.csv`
- `high_risk_alert_queue.csv`
- `top_100_alerts_for_review.csv`

**Reporting** (`outputs/`)
- Threshold analysis for validation and test splits
- Risk band summary
- Rules vs. model comparison
- Feature importances
- SQL monitoring summary
- Model and methodology summaries
- SQLite database of scored transactions
- Dashboard-ready CSVs for queue, alert reasons, and hourly breakdowns

**Charts** (`outputs/charts/`)
- Confusion matrix
- Precision-recall curve
- Threshold tradeoff charts
- Fraud rate by risk band

---

## Stack

Python (pandas, NumPy, scikit-learn, matplotlib), SQLite

---

## Repo structure

```
credit-card-fraud-analytics/
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
├── outputs/
│   └── charts/
├── src/
│   └── train_fraud_model.py
├── requirements.txt
└── README.md
```