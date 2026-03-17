import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# paths - adjust if running from a different directory
DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
CHARTS_DIR = OUTPUTS_DIR / "charts"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find the dataset at {DATA_PATH}. "
            "Download creditcard.csv from Kaggle and drop it in data/raw/"
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows. Fraud rate: {df['Class'].mean():.4%}")
    return df


def make_splits(df):
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # 60/20/20 split, stratified so fraud cases are spread evenly across splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return (
        X_train.reset_index(drop=True), y_train.reset_index(drop=True),
        X_val.reset_index(drop=True),   y_val.reset_index(drop=True),
        X_test.reset_index(drop=True),  y_test.reset_index(drop=True),
    )


def get_feature_cutoffs(X_train):
    # fit thresholds on training data only - don't want test info leaking in here
    v_cols = [c for c in X_train.columns if c.startswith("V")]
    v_thresholds = {col: float(X_train[col].abs().quantile(0.99)) for col in v_cols}

    return {
        "amount_high":      float(X_train["Amount"].quantile(0.95)),
        "amount_very_high": float(X_train["Amount"].quantile(0.99)),
        "amount_micro":     float(X_train["Amount"].quantile(0.05)),
        "v_thresholds":     v_thresholds,
    }


def add_features(X, cutoffs):
    df = X.copy()

    # log transform helps with the skewed amount distribution
    df["log_amount"] = np.log1p(df["Amount"])
    df["is_high_amount"]      = (df["Amount"] > cutoffs["amount_high"]).astype(int)
    df["is_very_high_amount"] = (df["Amount"] > cutoffs["amount_very_high"]).astype(int)
    df["is_micro_amount"]     = (df["Amount"] < cutoffs["amount_micro"]).astype(int)

    if "Time" in df.columns:
        df["hour_proxy"] = (df["Time"] // 3600) % 24
        df["is_night_proxy"] = df["hour_proxy"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    else:
        df["hour_proxy"] = 0
        df["is_night_proxy"] = 0

    v_cols = [c for c in df.columns if c.startswith("V") and len(c) > 1]
    extreme_flags = []
    for col in v_cols:
        if col not in cutoffs["v_thresholds"]:
            continue
        flag = f"{col}_extreme"
        df[flag] = (df[col].abs() > cutoffs["v_thresholds"][col]).astype(int)
        extreme_flags.append(flag)

    if extreme_flags:
        df["extreme_v_count"]  = df[extreme_flags].sum(axis=1)
        df["pca_anomaly_score"] = df[v_cols].abs().mean(axis=1)
    else:
        df["extreme_v_count"]  = 0
        df["pca_anomaly_score"] = 0.0

    return df


def train_model(X_train, y_train):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, X_train.columns.tolist())
    ])

    # class_weight="balanced" is important here since fraud is ~0.17% of data
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", clf)])
    pipeline.fit(X_train, y_train)
    return pipeline


def pick_threshold(y_true, probs):
    # try a range of thresholds and pick the one with best F1, but only if precision >= 0.4
    # (too many false positives isn't useful for an analyst queue)
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []
    total_frauds = int(y_true.sum())

    for t in thresholds:
        preds = (probs >= t).astype(int)
        n_alerts   = int(preds.sum())
        n_caught   = int(((preds == 1) & (y_true.values == 1)).sum())
        n_fp       = int(((preds == 1) & (y_true.values == 0)).sum())
        n_fn       = int(((preds == 0) & (y_true.values == 1)).sum())
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)

        rows.append({
            "threshold":         t,
            "alerts_generated":  n_alerts,
            "alert_rate":        round(n_alerts / len(y_true), 6),
            "frauds_caught":     n_caught,
            "fraud_capture_rate": round(n_caught / total_frauds if total_frauds else 0, 6),
            "false_positives":   n_fp,
            "false_negatives":   n_fn,
            "precision":         round(prec, 6),
            "recall":            round(rec, 6),
            "f1":                round(f1, 6),
        })

    df = pd.DataFrame(rows)

    good = df[df["precision"] >= 0.40]
    if not good.empty:
        best_t = float(good.sort_values(["f1", "precision"], ascending=False).iloc[0]["threshold"])
    else:
        # fallback if nothing clears 0.4 precision - just take best F1
        best_t = float(df.sort_values("f1", ascending=False).iloc[0]["threshold"])

    return best_t, df


def get_monitoring_mode(threshold):
    if threshold <= 0.30:
        return "Aggressive"
    if threshold <= 0.50:
        return "Balanced"
    return "High Precision"


def get_risk_band(prob):
    if prob >= 0.90:
        return "Critical"
    if prob >= 0.70:
        return "High"
    if prob >= 0.30:
        return "Medium"
    return "Low"


def get_alert_reason(row):
    reasons = []

    if row["fraud_probability"] >= 0.90:
        reasons.append("Critical model score")
    elif row["fraud_probability"] >= 0.70:
        reasons.append("Above high-risk threshold")

    if row.get("is_very_high_amount", 0):
        reasons.append("Very high transaction amount")
    elif row.get("is_high_amount", 0):
        reasons.append("High transaction amount")

    if row.get("is_micro_amount", 0) and row["fraud_probability"] >= 0.70:
        reasons.append("Micro-transaction anomaly")

    if row.get("is_night_proxy", 0) and row["fraud_probability"] >= 0.70:
        reasons.append("Off-hours activity")

    if row.get("extreme_v_count", 0) >= 3:
        reasons.append("Multiple PCA anomaly signals")
    elif row.get("extreme_v_count", 0) >= 1 and row["fraud_probability"] >= 0.70:
        reasons.append("PCA anomaly present")

    return " | ".join(reasons) if reasons else "Routine flag"


def build_alert_queue(X_test, y_test, probs, threshold):
    df = X_test.copy()
    df["actual_fraud"]       = y_test.values
    df["fraud_probability"]  = probs
    df["alert_flag"]         = (probs >= threshold).astype(int)
    df["risk_band"]          = df["fraud_probability"].apply(get_risk_band)
    df["alert_reason"]       = df.apply(get_alert_reason, axis=1)

    # priority logic for the analyst queue
    priority_map = {"Critical": "Immediate", "High": "Same Day", "Medium": "Queue", "Low": "Monitor"}
    action_map = {
        "Critical": "Escalate for immediate manual review",
        "High":     "Review account activity and transaction context",
        "Medium":   "Queue for analyst review if capacity allows",
        "Low":      "No immediate action; keep monitoring",
    }
    df["review_priority"]    = df["risk_band"].map(priority_map)
    df["recommended_action"] = df["risk_band"].map(action_map)

    alerts = df[df["alert_flag"] == 1].copy()
    alerts = alerts.sort_values(["fraud_probability", "Amount"], ascending=[False, False])

    # reorder columns so the most useful ones are first
    front_cols = [
        "fraud_probability", "risk_band", "review_priority", "alert_reason",
        "recommended_action", "actual_fraud", "Amount",
        "log_amount", "is_high_amount", "is_very_high_amount",
        "is_micro_amount", "is_night_proxy", "extreme_v_count",
        "pca_anomaly_score", "Time", "hour_proxy",
    ]
    col_order = [c for c in front_cols if c in alerts.columns] + \
                [c for c in alerts.columns if c not in front_cols]
    alerts = alerts[col_order]

    return df, alerts


def rules_baseline(X_test):
    # simple rules to compare against - roughly what a manual analyst might use
    df = X_test.copy()
    df["rule_high_amount"]       = (df["is_very_high_amount"] == 1).astype(int)
    df["rule_off_hours_anomaly"] = ((df["is_night_proxy"] == 1) & (df["extreme_v_count"] >= 2)).astype(int)
    df["rule_micro_anomaly"]     = ((df["is_micro_amount"] == 1) & (df["extreme_v_count"] >= 3)).astype(int)
    df["rule_big_anomaly_combo"] = ((df["is_high_amount"] == 1) & (df["extreme_v_count"] >= 3)).astype(int)
    df["rules_alert"] = (
        df[["rule_high_amount", "rule_off_hours_anomaly",
            "rule_micro_anomaly", "rule_big_anomaly_combo"]].max(axis=1)
    )
    return df["rules_alert"].values


def score_preds(y_true, preds, label):
    n_alerts  = int(preds.sum())
    n_caught  = int(((preds == 1) & (y_true.values == 1)).sum())
    n_fp      = int(((preds == 1) & (y_true.values == 0)).sum())
    n_fn      = int(((preds == 0) & (y_true.values == 1)).sum())

    return {
        "method":           label,
        "alerts_generated": n_alerts,
        "frauds_caught":    n_caught,
        "false_positives":  n_fp,
        "false_negatives":  n_fn,
        "precision":        round(precision_score(y_true, preds, zero_division=0), 6),
        "recall":           round(recall_score(y_true, preds, zero_division=0), 6),
        "f1":               round(f1_score(y_true, preds, zero_division=0), 6),
        "alert_rate":       round(n_alerts / len(y_true), 6),
    }


# --- chart helpers ---

def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Fraud", "Fraud"])
    ax.set_yticklabels(["Not Fraud", "Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()




def plot_pr_curve(y_true, probs, path):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve  (AP = {ap:.4f})")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_threshold_tradeoff(threshold_df, path, label):
    plt.figure(figsize=(7, 4))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"],    label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"],        label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Tradeoff ({label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_risk_bands(risk_summary, path):
    plt.figure(figsize=(7, 4))
    plt.bar(risk_summary["risk_band"], risk_summary["fraud_rate"])
    plt.xlabel("Risk Band")
    plt.ylabel("Observed Fraud Rate")
    plt.title("Fraud Rate by Risk Band")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_sql_summary(scored_df, db_path, out_path):
    conn = sqlite3.connect(db_path)
    scored_df.to_sql("scored_transactions", conn, if_exists="replace", index=False)

    # TODO: could pull this into a separate .sql file at some point
    queries = {
        "Fraud rate by risk band": """
            SELECT risk_band,
                   COUNT(*) AS transactions,
                   SUM(actual_fraud) AS actual_frauds,
                   ROUND(AVG(actual_fraud), 6) AS fraud_rate,
                   ROUND(AVG(fraud_probability), 6) AS avg_probability
            FROM scored_transactions
            GROUP BY risk_band
            ORDER BY CASE risk_band
                WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2
                WHEN 'High' THEN 3 WHEN 'Critical' THEN 4 ELSE 5 END;
        """,
        "Alerts by review priority": """
            SELECT review_priority,
                   COUNT(*) AS transactions,
                   SUM(actual_fraud) AS actual_frauds,
                   ROUND(AVG(actual_fraud), 6) AS fraud_rate
            FROM scored_transactions
            WHERE alert_flag = 1
            GROUP BY review_priority
            ORDER BY transactions DESC;
        """,
        "Top alert reasons": """
            SELECT alert_reason,
                   COUNT(*) AS alerts,
                   SUM(actual_fraud) AS actual_frauds,
                   ROUND(AVG(actual_fraud), 6) AS fraud_rate
            FROM scored_transactions
            WHERE alert_flag = 1
            GROUP BY alert_reason
            ORDER BY alerts DESC
            LIMIT 10;
        """,
        "Hourly breakdown": """
            SELECT hour_proxy,
                   COUNT(*) AS transactions,
                   SUM(alert_flag) AS alerts,
                   SUM(actual_fraud) AS actual_frauds,
                   ROUND(AVG(actual_fraud), 6) AS fraud_rate
            FROM scored_transactions
            GROUP BY hour_proxy
            ORDER BY hour_proxy;
        """,
        "Amount by alert status": """
            SELECT alert_flag,
                   COUNT(*) AS transactions,
                   ROUND(AVG(Amount), 2) AS avg_amount,
                   ROUND(AVG(fraud_probability), 6) AS avg_probability,
                   ROUND(AVG(actual_fraud), 6) AS fraud_rate
            FROM scored_transactions
            GROUP BY alert_flag;
        """,
    }

    with open(out_path, "w") as f:
        f.write("SQL Monitoring Summary\n======================\n\n")
        for title, query in queries.items():
            result = pd.read_sql_query(query, conn)
            f.write(f"{title}\n{'-' * len(title)}\n")
            f.write(result.to_string(index=False))
            f.write("\n\n")

    conn.close()


def main():
    print("Loading data...")
    raw_df = load_data()

    print("Splitting into train / val / test...")
    X_train, y_train, X_val, y_val, X_test, y_test = make_splits(raw_df)

    print("Fitting feature cutoffs on training data...")
    cutoffs = get_feature_cutoffs(X_train)

    print("Engineering features...")
    X_train = add_features(X_train, cutoffs)
    X_val   = add_features(X_val, cutoffs)
    X_test  = add_features(X_test, cutoffs)

    print("Training model...")
    model = train_model(X_train, y_train)

    # use validation set to pick threshold, then freeze it for test
    print("Picking threshold on validation set...")
    val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold, val_threshold_df = pick_threshold(y_val, val_probs)
    mode = get_monitoring_mode(best_threshold)
    print(f"  Selected threshold: {best_threshold:.2f}  ({mode})")

    print("Evaluating on test set (locked)...")
    test_probs = model.predict_proba(X_test)[:, 1]
    _, test_threshold_df = pick_threshold(y_test, test_probs)

    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc  = average_precision_score(y_test, test_probs)
    print(f"  Test ROC AUC: {roc_auc:.4f}")
    print(f"  Test PR AUC:  {pr_auc:.4f}")

    test_preds = (test_probs >= best_threshold).astype(int)
    report = classification_report(y_test, test_preds, digits=4)
    print("\n=== TEST PERFORMANCE ===")
    print(report)

    # build alert queue
    all_scored, alert_queue = build_alert_queue(X_test, y_test, test_probs, best_threshold)

    risk_summary = (
        all_scored.groupby("risk_band", observed=False)
        .agg(
            transactions=("fraud_probability", "size"),
            avg_probability=("fraud_probability", "mean"),
            actual_frauds=("actual_fraud", "sum"),
            fraud_rate=("actual_fraud", "mean"),
            avg_amount=("Amount", "mean"),
        )
        .reset_index()
    )

    # compare model vs simple rules
    rules_preds = rules_baseline(X_test)
    comparison = pd.DataFrame([
        score_preds(y_test, rules_preds, "Rules baseline"),
        score_preds(y_test, test_preds, f"Model (threshold={best_threshold:.2f})"),
    ])

    # save everything
    print("\nSaving outputs...")

    # feature importances
    feat_imp = pd.DataFrame({
        "feature":    X_train.columns.tolist(),
        "importance": model.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    val_threshold_df.to_csv(OUTPUTS_DIR / "threshold_analysis_validation.csv", index=False)
    test_threshold_df.to_csv(OUTPUTS_DIR / "threshold_analysis_test.csv", index=False)
    risk_summary.to_csv(OUTPUTS_DIR / "risk_band_summary.csv", index=False)
    comparison.to_csv(OUTPUTS_DIR / "rules_vs_model.csv", index=False)
    all_scored.to_csv(PROCESSED_DIR / "scored_test_transactions.csv", index=False)
    alert_queue.to_csv(PROCESSED_DIR / "high_risk_alert_queue.csv", index=False)
    alert_queue.head(100).to_csv(PROCESSED_DIR / "top_100_alerts.csv", index=False)

    # dashboard exports
    queue_summary = (
        all_scored[all_scored["alert_flag"] == 1]
        .groupby(["review_priority", "risk_band"], observed=False)
        .agg(alerts=("alert_flag", "size"), actual_frauds=("actual_fraud", "sum"),
             avg_amount=("Amount", "mean"), avg_probability=("fraud_probability", "mean"))
        .reset_index()
        .sort_values(["review_priority", "risk_band"])
    )
    reason_summary = (
        all_scored[all_scored["alert_flag"] == 1]
        .groupby("alert_reason")
        .agg(alerts=("alert_flag", "size"), actual_frauds=("actual_fraud", "sum"),
             fraud_rate=("actual_fraud", "mean"), avg_probability=("fraud_probability", "mean"))
        .reset_index()
        .sort_values("alerts", ascending=False)
    )
    hourly_summary = (
        all_scored.groupby("hour_proxy")
        .agg(transactions=("actual_fraud", "size"), alerts=("alert_flag", "sum"),
             actual_frauds=("actual_fraud", "sum"), fraud_rate=("actual_fraud", "mean"))
        .reset_index()
    )
    queue_summary.to_csv(OUTPUTS_DIR / "dashboard_queue_summary.csv", index=False)
    reason_summary.to_csv(OUTPUTS_DIR / "dashboard_alert_reasons.csv", index=False)
    hourly_summary.to_csv(OUTPUTS_DIR / "dashboard_hourly.csv", index=False)
    val_threshold_df.to_csv(OUTPUTS_DIR / "dashboard_threshold_val.csv", index=False)
    test_threshold_df.to_csv(OUTPUTS_DIR / "dashboard_threshold_test.csv", index=False)

    # charts
    cm = confusion_matrix(y_test, test_preds)
    plot_confusion_matrix(cm, CHARTS_DIR / "confusion_matrix.png")
    plot_pr_curve(y_test, test_probs, CHARTS_DIR / "precision_recall_curve.png")
    plot_threshold_tradeoff(val_threshold_df,  CHARTS_DIR / "threshold_tradeoff_val.png",  "Validation")
    plot_threshold_tradeoff(test_threshold_df, CHARTS_DIR / "threshold_tradeoff_test.png", "Test")
    plot_risk_bands(risk_summary, CHARTS_DIR / "risk_band_fraud_rate.png")

    # sql summary
    save_sql_summary(
        all_scored,
        OUTPUTS_DIR / "fraud_monitoring.db",
        OUTPUTS_DIR / "sql_monitoring_summary.txt",
    )

    # model summary text file
    with open(OUTPUTS_DIR / "model_summary.txt", "w") as f:
        f.write("Credit Card Fraud Detection - Results\n")
        f.write("=====================================\n\n")
        f.write(f"Total rows: {len(raw_df):,}\n")
        f.write(f"Train: {len(X_train):,} rows (fraud rate: {y_train.mean():.4%})\n")
        f.write(f"Val:   {len(X_val):,} rows (fraud rate: {y_val.mean():.4%})\n")
        f.write(f"Test:  {len(X_test):,} rows (fraud rate: {y_test.mean():.4%})\n\n")
        f.write(f"Threshold: {best_threshold:.2f} ({mode}) - selected on validation set\n")
        f.write(f"Test ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Test PR AUC:  {pr_auc:.4f}\n\n")
        f.write("Classification Report\n--------------------\n")
        f.write(report)

    print("\nDone. Output files saved to outputs/ and data/processed/")


if __name__ == "__main__":
    main()