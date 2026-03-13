# Credit Card Fraud Risk Monitoring and Alert Triage

## The Problem

Most fraud detection projects stop way too early. You build a classifier, report the accuracy, save a confusion matrix, and call it a day.

Real financial crime teams operate differently. A raw model score provides little value to an investigator. An investigator needs more than a decimal point; they need to know where to set the alert threshold, how to prioritize a massive queue, and why a specific transaction was flagged in the first place.

This project builds the second half of the process. It is the operational layer sitting between a trained model and a functional monitoring workflow.

---

## Dataset

This project uses the public [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`), which contains anonymized card transaction records with a binary fraud label.

Relevant columns:
- `Time` — seconds elapsed since the first transaction in the dataset
- `V1` to `V28` — anonymized PCA-transformed features
- `Amount` — transaction amount
- `Class` — fraud label (1 = fraud, 0 = not fraud)

Because the data is anonymized and public, this is a simulation rather than a direct representation of a live bank fraud environment.

---

## Scope

This project should be read as a **financial crime alert-operations simulation**, not a production fraud or AML system. A few honest limits worth noting upfront:

- The SQL layer runs against a SQLite table built from the project's own scored output. In a real environment, analysts would query upstream warehouse or monitoring tables they don't directly control
- The dataset is public and anonymized, not proprietary banking data
- The workflow is adjacent to AML and compliance analytics, but it doesn't simulate SAR filing, BSA program metrics, sanctions screening, or case management systems

The SQL component is included to demonstrate the reporting logic and monitoring summaries an analyst would generate downstream from a scored transaction table — not to simulate live production querying.

---

## What It Does

The script processes the transaction dataset and simulates a first-line monitoring workflow end to end:

1. **Scores transactions** using a Random Forest classifier trained on engineered features
2. **Runs a rules-based baseline** to compare against because, in practice, a model has to prove itself against the simpler thing that already exists
3. **Sweeps candidate alert thresholds** and selects one based on a precision floor + F1 tradeoff, rather than just maximizing recall
4. **Assigns risk bands** (Low / Medium / High / Critical) and generates plain-English alert reasons for each flagged transaction
5. **Builds an investigator-ready alert queue** sorted by fraud probability and transaction size
6. **Runs SQL monitoring queries** against the scored data to produce the kind of summary tables a reporting or compliance team would actually want

---

## Methods

### Feature Engineering

The raw features are augmented with interpretable signals the model and alert reasons can both use:

- Log-transformed transaction amount (raw amount is heavily right-skewed)
- High-amount and very-high-amount flags (top 5% and top 1%)
- Micro-transaction flag (bottom 5%) — micro + anomaly signals is a classic card-probing pattern
- Off-hours transaction proxy derived from the `Time` column
- Count of how many V-columns are simultaneously in their extreme tail (`extreme_v_count`)
- Composite PCA anomaly score (average absolute deviation across all V-columns)

### Model-Based Scoring

A Random Forest classifier generates fraud probabilities for each transaction. Because fraud is a rare-event problem (~0.17% of transactions), the project focuses on precision, recall, F1, and PR AUC rather than accuracy.

### Rules-Based Baseline

A simple screening baseline is built from combinations of:
- Very high transaction amount
- Off-hours activity combined with anomaly signals
- Micro-transaction anomaly combinations
- Large-amount plus anomaly combinations

This allows it to be possible to compare a control-style rules approach directly against the model-based workflow.

### Threshold Analysis

Rather than treating the classification threshold as a fixed tuning detail, the project evaluates multiple candidates (0.30 through 0.90) and records alerts generated, frauds caught, false positives, false negatives, precision, recall, and F1 at each level. Each threshold is also labeled with an operating mode:

| Mode | Threshold Range |
|------|----------------|
| Aggressive | ≤ 0.30 |
| Balanced | 0.31 – 0.50 |
| High Precision | > 0.50 |

The final threshold is selected by requiring a minimum precision floor of 0.40, then choosing the best F1 among thresholds that clear it, avoiding chasing recall at the cost of flooding analysts with false positives.

### Risk Banding and Triage

Transactions are grouped into four bands based on fraud probability:

| Band | Probability Range |
|------|------------------|
| Low | 0 – 0.30 |
| Medium | 0.30 – 0.70 |
| High | 0.70 – 0.90 |
| Critical | 0.90 – 1.00 |

High-risk transactions are then assigned alert reasons, review priorities (Immediate / Same Day / Queue / Monitor), and recommended actions. This is so an analyst opening the queue knows what to do with each row without digging into raw scores.

### SQL Monitoring Layer

The scored transaction file is loaded into SQLite and a set of monitoring queries generate downstream summaries: fraud rate by risk band, alerts by review priority, top alert reason combinations, alert volume by hour, and average transaction amount by alert status.

---

## Key Results

The final workflow selected a Balanced operating threshold of **0.50**, producing:

| Metric | Value |
|--------|-------|
| Alerts generated | 119 |
| Frauds caught | 98 |
| Precision | 82.35% |
| Recall | 79.67% |

For comparison, the rules-based baseline generated 1,513 alerts but caught only 42 frauds at 2.78% precision. The model produced a significantly cleaner and more efficient investigator queue.

Risk segmentation also held up well:
- **Critical band** observed fraud rate: 92.5%
- **High band** observed fraud rate: 69.23%

The risk bands meaningfully separated higher-risk transactions from the broader population rather than just being cosmetic bucketing.

---

## Why the Operational Layer Matters

The threshold choice is a real decision, not a tuning detail. A lower threshold catches more fraud but sends more false positives to analysts. A higher threshold keeps the queue manageable but misses cases. The project makes that tradeoff explicit and documents the selection logic rather than hiding it inside the code.

The rules-versus-model comparison exists for the same reason. Financial crime teams almost always have an existing rules engine. A new model needs to justify itself against that baseline.

The alert reasons and triage fields exist because a queue of raw fraud probabilities isn't actually useful to an investigator. They need to know what triggered the flag, such as the amount, off-hours activity, anomaly pattern, or some combination, without reading model internals.

---

## Forensic Accounting Connection

The project is informed by a forensic accounting mindset. In forensic work, unusual patterns are not treated as automatic proof, but are red flags that require further review, corroboration, and escalation.

Likewise, the same logic applies here. Elevated fraud scores, unusual amount patterns, off-hours activity, and multiple anomaly indicators are framed as evidence for review, not final conclusions. This is why the project emphasizes alert operations, triage, and reporting instead of just prediction.

---

## Relevance to Financial Crime Analytics

The project is most relevant to roles involving fraud analytics, financial crime monitoring, risk reporting, alert operations, first-line controls analytics, and investigation support workflows.

It is not a substitute for AML regulatory experience, but it does demonstrate a transferable foundation in alert generation logic, threshold tradeoffs, queue design, risk segmentation, monitoring summaries, and structured anomaly interpretation.

---

## Outputs

**Processed Data**
- `data/processed/scored_test_transactions.csv` — transaction-level scored dataset with fraud probabilities and operational fields
- `data/processed/high_risk_alert_queue.csv` — investigator-facing queue of flagged transactions with alert reasons, priorities, and recommended actions
- `data/processed/top_100_alerts_for_review.csv` — top 100 alerts sorted by fraud probability for quick manual review

**Reporting Outputs**
- `outputs/threshold_analysis.csv` — full sweep of candidate thresholds with monitoring mode labels
- `outputs/risk_band_summary.csv` — fraud rate, average probability, and average amount by risk band
- `outputs/rule_vs_model_comparison.csv` — side-by-side comparison of rules-based screening vs. model-based alerting
- `outputs/sql_monitoring_summary.txt` — SQL-generated monitoring summaries from the scored transaction table
- `outputs/model_summary.txt` — core model performance summary
- `outputs/risk_reporting_summary.txt` — narrative alert-operations summary with control framing

**Charts**
- `outputs/charts/confusion_matrix.png`
- `outputs/charts/precision_recall_curve.png`
- `outputs/charts/threshold_tradeoff.png`
- `outputs/charts/risk_band_fraud_rate.png`

---

## Setup

### 1. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`.

### 4. Run the pipeline
```bash
python src/train_fraud_model.py
```

---

## Tech Stack

- **Python**: pandas, NumPy, scikit-learn, matplotlib
- **SQL / SQLite**: downstream monitoring queries and reporting summaries

---

## Repository Structure

```
credit-card-fraud-risk-monitoring/
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       ├── scored_test_transactions.csv
│       ├── high_risk_alert_queue.csv
│       └── top_100_alerts_for_review.csv
├── outputs/
│   ├── charts/
│   │   ├── confusion_matrix.png
│   │   ├── precision_recall_curve.png
│   │   ├── threshold_tradeoff.png
│   │   └── risk_band_fraud_rate.png
│   ├── threshold_analysis.csv
│   ├── risk_band_summary.csv
│   ├── rule_vs_model_comparison.csv
│   ├── sql_monitoring_summary.txt
│   ├── model_summary.txt
│   ├── risk_reporting_summary.txt
│   └── fraud_monitoring.db
├── src/
│   └── train_fraud_model.py
├── requirements.txt
└── README.md
```