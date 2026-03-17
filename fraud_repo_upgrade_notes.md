# Credit Card Fraud Risk Monitoring — Upgrade Notes

## What changed

### 1. Methodology is now harder to attack
- Split the data into **train / validation / test** instead of only train / test.
- Fit amount cutoffs and PCA tail cutoffs on the **training split only**.
- Choose the alert threshold on the **validation split**.
- Report final performance on the **locked test split**.

That prevents two common credibility problems:
- leaking information from the test set into engineered-feature cutoffs
- choosing the operating threshold on the same split used for final reporting

### 2. Added dashboard-friendly exports
The improved script writes extra CSVs designed for Tableau or Power BI:
- `outputs/dashboard_queue_summary.csv`
- `outputs/dashboard_alert_reason_summary.csv`
- `outputs/dashboard_hourly_summary.csv`
- `outputs/dashboard_threshold_validation.csv`
- `outputs/dashboard_threshold_test.csv`

### 3. Added feature importance output
- `outputs/feature_importance.csv`

### 4. Added methodology narrative output
- `outputs/methodology_summary.txt`

This file explains why the revised workflow is more trustworthy.

## Recommended repository structure after upgrade

```text
credit-card-fraud-risk-monitoring/
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
├── outputs/
│   ├── charts/
│   ├── dashboard_*.csv
│   ├── feature_importance.csv
│   ├── methodology_summary.txt
│   └── ...
├── src/
│   ├── train_fraud_model.py
│   └── train_fraud_model_v2.py
├── requirements.txt
└── README.md
```

## Best next visual layer for Mac
Use **Tableau first**.

Why:
- easier Mac-native path than Power BI Desktop
- fast way to turn your outputs into a recruiter-friendly artifact
- still consistent with the project's operational story

## Tableau dashboard tabs

### 1. Operations overview
Show:
- alerts generated
- frauds caught
- precision
- recall
- selected operating threshold
- rules baseline vs model

### 2. Threshold tuning
Use `dashboard_threshold_validation.csv` and `dashboard_threshold_test.csv`.

Show:
- threshold
- precision
- recall
- F1
- alerts generated
- false positives

### 3. Investigator queue
Use `data/processed/high_risk_alert_queue.csv`.

Show:
- fraud probability
- review priority
- alert reason
- amount
- recommended action

### 4. Alert reason and workload mix
Use:
- `dashboard_queue_summary.csv`
- `dashboard_alert_reason_summary.csv`
- `dashboard_hourly_summary.csv`

## README lines worth updating

Add language like this:

> To reduce leakage and make the project more credible, engineered-feature cutoffs are fit on the training split only, the operational threshold is selected on a validation split, and final performance is reported on a locked test split.

And in the outputs section, add:
- `outputs/methodology_summary.txt`
- `outputs/feature_importance.csv`
- dashboard export CSVs for Tableau / Power BI

## Resume bullet after the upgrade

Built a fraud monitoring workflow that scored transactions, selected an operational alert threshold on a validation split, and surfaced investigator-ready alert queues, risk-band reporting, and dashboard-ready workload summaries for triage and threshold tuning.
