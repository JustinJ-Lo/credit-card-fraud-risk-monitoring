from __future__ import annotations
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier


# Keeping all paths relative to the project root 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    # Fail faster
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at: {path}\n"
            "Download the Kaggle credit card fraud dataset and save it as data/raw/creditcard.csv"
        )

    df = pd.read_csv(path)

    # Validate the two columns we hard-depend on before 
    required_cols = {"Amount", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Amount is heavily right-skewed. log-transform $5 -> $50
    # jump gets treated similarly to a $500 -> $5000 one
    df["log_amount"] = np.log1p(df["Amount"])
    df["is_high_amount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
    df["is_very_high_amount"] = (df["Amount"] > df["Amount"].quantile(0.99)).astype(int)
    df["is_micro_amount"] = (df["Amount"] < df["Amount"].quantile(0.05)).astype(int)

    # Time is seconds elapsed since the first transaction in the dataset
    # Converting to a rough hour-of-day isn't exact, but night-time transactions
    # genuinely have a different fraud profile so it's worth including.
    if "Time" in df.columns:
        df["hour_proxy"] = (df["Time"] // 3600) % 24
        df["is_night_proxy"] = df["hour_proxy"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    else:
        df["hour_proxy"] = 0
        df["is_night_proxy"] = 0

    # columns are already PCA-transformed by Kaggle 
    # we flag how many are simultaneously in their extreme tail
    # one unusual V is noise, five unusual Vs at once is a signal
    v_cols = [c for c in df.columns if c.startswith("V")]
    if v_cols:
        extreme_flags = []
        for col in v_cols:
            threshold = df[col].abs().quantile(0.99)
            flag_col = f"{col}_extreme"
            df[flag_col] = (df[col].abs() > threshold).astype(int)
            extreme_flags.append(flag_col)

        df["extreme_v_count"] = df[extreme_flags].sum(axis=1)

        # Cheap composite anomaly score — average absolute deviation across all
        # PCA components. Doesn't require fitting anything separately
        df["pca_anomaly_score"] = df[v_cols].abs().mean(axis=1)
    else:
        df["extreme_v_count"] = 0
        df["pca_anomaly_score"] = 0.0

    return df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    numeric_features = X_train.columns.tolist()

    # Median imputation is more robust than mean when there are outliers,
    # which is guaranteed in transaction data
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )

    # class_weight="balanced" handles the class imbalance. fraud is ~0.17%
    # of all transactions, so without this the model just predicts "not fraud"
    # on everything and looks good on accuracy while catching nothing.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Fraud", "Fraud"])
    ax.set_yticklabels(["Non-Fraud", "Fraud"])

    # write counts directly on cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_precision_recall_curve(y_true: pd.Series, y_score: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # PR curve is more honest than ROC when classes are this imbalanced
    # a model can hit 0.99 ROC-AUC and still flood analysts with false positives
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP = {ap:.4f})")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_threshold_tradeoff_chart(threshold_df: pd.DataFrame, path: Path) -> None:
    # tool for explaining the threshold to someone
    # who didn't run the numbers themselves. You can see where precision
    # falls off and where recall starts to plateau.
    plt.figure(figsize=(7, 4))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    plt.xlabel("Alert Threshold")
    plt.ylabel("Metric")
    plt.title("Threshold Tradeoff: Fraud Capture vs Queue Quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_risk_band_chart(risk_band_summary: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(risk_band_summary["risk_band"], risk_band_summary["fraud_rate"])
    plt.xlabel("Risk Band")
    plt.ylabel("Observed Fraud Rate")
    plt.title("Observed Fraud Rate by Risk Band")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def threshold_analysis(
    y_true: pd.Series,
    fraud_probs: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    # Sweep across candidate thresholds and record the full picture at each one.
    # goal is to give a human the numbers they need to make a real decision
    # about alert volume vs. fraud capture, not to auto-tune silently.
    rows = []
    total_frauds = int(y_true.sum())

    for threshold in thresholds:
        y_pred = (fraud_probs >= threshold).astype(int)

        alerts_generated = int(y_pred.sum())
        actual_frauds_caught = int(((y_pred == 1) & (y_true.values == 1)).sum())

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        false_positives = int(((y_pred == 1) & (y_true.values == 0)).sum())
        false_negatives = int(((y_pred == 0) & (y_true.values == 1)).sum())

        alert_rate = alerts_generated / len(y_true)
        fraud_capture_rate = actual_frauds_caught / total_frauds if total_frauds > 0 else 0.0

        rows.append(
            {
                "threshold": threshold,
                "alerts_generated": alerts_generated,
                "alert_rate": round(alert_rate, 6),
                "actual_frauds_caught": actual_frauds_caught,
                "fraud_capture_rate": round(fraud_capture_rate, 6),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )

    return pd.DataFrame(rows)


def assign_monitoring_mode(threshold: float) -> str:
    # Labels each threshold row with a English operating mode.
    # Useful for explaining tradeoffs in a presentation without leading with numbers.
    if threshold <= 0.30:
        return "Aggressive"
    if threshold <= 0.50:
        return "Balanced"
    return "High Precision"


def choose_operational_threshold(threshold_df: pd.DataFrame) -> float:
    """
    Pick a threshold that's actually usable in production.

    Logic:
    - Require precision >= 0.40 so the alert queue isn't drowning analysts
      in false positives before they even start reviewing.
    - Among thresholds that clear that bar, take the one with the best F1
      so we're balancing fraud capture and queue quality together, not just
      chasing the highest recall.
    - If nothing clears the precision floor, fall back to best F1 overall —
      at least that's a balanced choice under bad conditions.
    """
    candidates = threshold_df[threshold_df["precision"] >= 0.40].copy()

    if not candidates.empty:
        candidates = candidates.sort_values(
            ["f1", "precision", "recall", "threshold"],
            ascending=[False, False, False, True],
        )
        return float(candidates.iloc[0]["threshold"])

    # take the least-bad option by F1.
    best_f1_row = threshold_df.sort_values(
        ["f1", "precision", "recall"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best_f1_row["threshold"])


def assign_risk_bands(alerts: pd.DataFrame) -> pd.DataFrame:
    alerts = alerts.copy()

    # Four bands gives enough granularity to route work without overwhelming
    # analysts with subcategories. Critical to the top of the queue immediately
    alerts["risk_band"] = pd.cut(
        alerts["fraud_probability"],
        bins=[-0.01, 0.30, 0.70, 0.90, 1.00],
        labels=["Low", "Medium", "High", "Critical"],
    )

    return alerts


def summarize_risk_bands(alerts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        alerts.groupby("risk_band", observed=False)
        .agg(
            transactions=("fraud_probability", "size"),
            avg_fraud_probability=("fraud_probability", "mean"),
            actual_frauds=("actual_fraud", "sum"),
            fraud_rate=("actual_fraud", "mean"),
            avg_amount=("Amount", "mean"),
        )
        .reset_index()
    )

    summary["avg_fraud_probability"] = summary["avg_fraud_probability"].round(6)
    summary["fraud_rate"] = summary["fraud_rate"].round(6)
    summary["avg_amount"] = summary["avg_amount"].round(2)

    return summary


def generate_alert_reason(row: pd.Series) -> str:
    # Build a plain-English explanation of why this transaction was flagged.
    # Analysts should be able to scan this column without touching the raw scores.
    reasons = []

    if row["fraud_probability"] >= 0.90:
        reasons.append("Critical model score")
    elif row["fraud_probability"] >= 0.70:
        reasons.append("Above high-risk monitoring threshold")

    if row.get("is_very_high_amount", 0) == 1:
        reasons.append("Very high transaction amount")
    elif row.get("is_high_amount", 0) == 1:
        reasons.append("High transaction amount")

    # Micro-transactions paired with a high fraud score are a classic card-probing
    # pattern. criminals test stolen cards with small amounts before going big.
    if row.get("is_micro_amount", 0) == 1 and row["fraud_probability"] >= 0.70:
        reasons.append("Micro-transaction anomaly")

    if row.get("is_night_proxy", 0) == 1 and row["fraud_probability"] >= 0.70:
        reasons.append("Elevated risk during off-hours proxy")

    if row.get("extreme_v_count", 0) >= 3:
        reasons.append("Multiple extreme PCA anomaly signals")
    elif row.get("extreme_v_count", 0) >= 1 and row["fraud_probability"] >= 0.70:
        reasons.append("PCA anomaly pattern present")

    if not reasons:
        reasons.append("Routine monitoring flag")

    return " | ".join(reasons)


def assign_review_priority(row: pd.Series) -> str:
    # Maps risk bands to actionable SLAs. keeps routing logic in one place
    # so alert queue output doesn't need a separate lookup table.
    if row["risk_band"] == "Critical":
        return "Immediate"
    if row["risk_band"] == "High":
        return "Same Day"
    if row["risk_band"] == "Medium":
        return "Queue"
    return "Monitor"


def assign_recommended_action(row: pd.Series) -> str:
    if row["risk_band"] == "Critical":
        return "Escalate for immediate manual review"
    if row["risk_band"] == "High":
        return "Review account activity and transaction context"
    if row["risk_band"] == "Medium":
        return "Queue for analyst review if capacity permits"
    return "No immediate action; retain for monitoring"


def score_rules_baseline(df: pd.DataFrame) -> pd.DataFrame:
    # Simple rules engine to use as a comparison baseline.
    # the model needs to justify itself against it, not just against nothing.
    baseline = df.copy()

    baseline["rule_high_amount"] = (baseline["is_very_high_amount"] == 1).astype(int)

    baseline["rule_off_hours_anomaly"] = (
        (baseline["is_night_proxy"] == 1) & (baseline["extreme_v_count"] >= 2)
    ).astype(int)

    # Micro + multiple anomaly signals is classic probe pattern
    baseline["rule_micro_anomaly"] = (
        (baseline["is_micro_amount"] == 1) & (baseline["extreme_v_count"] >= 3)
    ).astype(int)

    baseline["rule_large_anomaly_combo"] = (
        (baseline["is_high_amount"] == 1) & (baseline["extreme_v_count"] >= 3)
    ).astype(int)

    baseline["rules_alert_flag"] = (
        (baseline["rule_high_amount"] == 1)
        | (baseline["rule_off_hours_anomaly"] == 1)
        | (baseline["rule_micro_anomaly"] == 1)
        | (baseline["rule_large_anomaly_combo"] == 1)
    ).astype(int)

    return baseline


def evaluate_binary_predictions(y_true: pd.Series, y_pred: np.ndarray, label: str) -> dict:
    alerts_generated = int(y_pred.sum())
    actual_frauds_caught = int(((y_pred == 1) & (y_true.values == 1)).sum())
    false_positives = int(((y_pred == 1) & (y_true.values == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_true.values == 1)).sum())

    return {
        "method": label,
        "alerts_generated": alerts_generated,
        "actual_frauds_caught": actual_frauds_caught,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 6),
        "alert_rate": round(alerts_generated / len(y_true), 6),
    }


def build_alert_queue(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fraud_probs: np.ndarray,
    chosen_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alerts = X_test.copy()
    alerts["actual_fraud"] = y_test.values
    alerts["fraud_probability"] = fraud_probs
    alerts["alert_flag"] = (alerts["fraud_probability"] >= chosen_threshold).astype(int)

    alerts = assign_risk_bands(alerts)
    alerts["alert_reason"] = alerts.apply(generate_alert_reason, axis=1)
    alerts["review_priority"] = alerts.apply(assign_review_priority, axis=1)
    alerts["recommended_action"] = alerts.apply(assign_recommended_action, axis=1)

    high_risk_alerts = alerts[alerts["alert_flag"] == 1].copy()

    # sort by probability first, then amount. if two transactions score the same,
    # the higher-value one gets reviewed first.
    high_risk_alerts = high_risk_alerts.sort_values(
        ["fraud_probability", "Amount"],
        ascending=[False, False],
    )

    # Put the columns an analyst actually cares about up front 
    preferred_cols = [
        "fraud_probability",
        "risk_band",
        "alert_flag",
        "review_priority",
        "alert_reason",
        "recommended_action",
        "actual_fraud",
        "Amount",
        "log_amount",
        "is_high_amount",
        "is_very_high_amount",
        "is_micro_amount",
        "is_night_proxy",
        "extreme_v_count",
        "pca_anomaly_score",
        "Time",
        "hour_proxy",
    ]

    ordered_cols = [c for c in preferred_cols if c in high_risk_alerts.columns] + [
        c for c in high_risk_alerts.columns if c not in preferred_cols
    ]

    high_risk_alerts = high_risk_alerts[ordered_cols]

    return alerts, high_risk_alerts


def write_sql_monitoring_summary(scored_df: pd.DataFrame, db_path: Path, output_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    scored_df.to_sql("scored_transactions", conn, if_exists="replace", index=False)

    queries = {
        "Fraud rate by risk band": """
            SELECT
                risk_band,
                COUNT(*) AS transactions,
                SUM(actual_fraud) AS actual_frauds,
                ROUND(AVG(actual_fraud), 6) AS fraud_rate,
                ROUND(AVG(fraud_probability), 6) AS avg_fraud_probability
            FROM scored_transactions
            GROUP BY risk_band
            ORDER BY
                CASE risk_band
                    WHEN 'Low' THEN 1
                    WHEN 'Medium' THEN 2
                    WHEN 'High' THEN 3
                    WHEN 'Critical' THEN 4
                    ELSE 5
                END;
        """,
        "Alerts by review priority": """
            SELECT
                review_priority,
                COUNT(*) AS transactions,
                SUM(actual_fraud) AS actual_frauds,
                ROUND(AVG(actual_fraud), 6) AS fraud_rate
            FROM scored_transactions
            WHERE alert_flag = 1
            GROUP BY review_priority
            ORDER BY transactions DESC;
        """,
        "Top alert reasons": """
            SELECT
                alert_reason,
                COUNT(*) AS alerts,
                SUM(actual_fraud) AS actual_frauds,
                ROUND(AVG(actual_fraud), 6) AS fraud_rate
            FROM scored_transactions
            WHERE alert_flag = 1
            GROUP BY alert_reason
            ORDER BY alerts DESC
            LIMIT 10;
        """,
        "Alert volume by hour proxy": """
            SELECT
                hour_proxy,
                COUNT(*) AS transactions,
                SUM(alert_flag) AS alerts,
                SUM(actual_fraud) AS actual_frauds,
                ROUND(AVG(actual_fraud), 6) AS fraud_rate
            FROM scored_transactions
            GROUP BY hour_proxy
            ORDER BY hour_proxy;
        """,
        "Average transaction amount by alert status": """
            SELECT
                alert_flag,
                COUNT(*) AS transactions,
                ROUND(AVG(Amount), 2) AS avg_amount,
                ROUND(AVG(fraud_probability), 6) AS avg_fraud_probability,
                ROUND(AVG(actual_fraud), 6) AS fraud_rate
            FROM scored_transactions
            GROUP BY alert_flag
            ORDER BY alert_flag;
        """,
    }

    with open(output_path, "w") as f:
        f.write("SQL Monitoring Summary\n")
        f.write("======================\n\n")
        f.write(
            "This file shows example SQL-based monitoring outputs generated from the scored "
            "transaction table, simulating how an analytics team might summarize risk signals "
            "for first-line review and reporting.\n\n"
        )
        f.write(
            "Implementation note: the SQL queries here run against a SQLite table created "
            "from this project's own scored output. In a real financial-crime environment, "
            "analysts would more typically query upstream warehouse or transaction-monitoring "
            "tables they don't directly control. SQLite is used to simulate the downstream "
            "reporting logic, aggregation structure, and monitoring summaries that workflow would require.\n\n"
        )

        for title, query in queries.items():
            f.write(f"{title}\n")
            f.write("-" * len(title) + "\n")
            result_df = pd.read_sql_query(query, conn)
            f.write(result_df.to_string(index=False))
            f.write("\n\n")

    conn.close()


def write_model_summary(
    path: Path,
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    roc: float,
    ap: float,
    report_text: str,
) -> None:
    with open(path, "w") as f:
        f.write("Credit Card Fraud Risk Monitoring - Model Summary\n")
        f.write("=================================================\n")
        f.write(f"Rows in full dataset: {len(df):,}\n")
        f.write(f"Train rows: {len(X_train):,}\n")
        f.write(f"Test rows: {len(X_test):,}\n")
        f.write(f"Fraud rate in full dataset: {y.mean():.4%}\n")
        f.write(f"ROC AUC: {roc:.4f}\n")
        f.write(f"Average Precision (PR AUC): {ap:.4f}\n\n")
        f.write("Classification Report at Default Threshold 0.50\n")
        f.write("-----------------------------------------------\n")
        f.write(report_text)
        f.write("\n")


def write_risk_reporting_summary(
    path: Path,
    threshold_df: pd.DataFrame,
    risk_band_summary: pd.DataFrame,
    chosen_threshold: float,
    high_risk_alerts: pd.DataFrame,
    rules_vs_model_df: pd.DataFrame,
) -> None:
    selected_row = threshold_df[threshold_df["threshold"] == chosen_threshold].iloc[0]

    high_risk_precision = (
        high_risk_alerts["actual_fraud"].mean() if len(high_risk_alerts) > 0 else 0.0
    )

    critical_band = risk_band_summary[risk_band_summary["risk_band"] == "Critical"]
    high_band = risk_band_summary[risk_band_summary["risk_band"] == "High"]

    critical_rate = float(critical_band["fraud_rate"].iloc[0]) if not critical_band.empty else 0.0
    high_rate = float(high_band["fraud_rate"].iloc[0]) if not high_band.empty else 0.0

    with open(path, "w") as f:
        f.write("Financial Crimes Risk Monitoring - Alert Operations Summary\n")
        f.write("===========================================================\n\n")

        f.write("Control objective\n")
        f.write("-----------------\n")
        f.write(
            "Flag potentially fraudulent credit card transactions for review while balancing "
            "fraud capture against false positive alert volume.\n\n"
        )

        f.write("Operational threshold selection\n")
        f.write("-------------------------------\n")
        f.write(f"Chosen alert threshold: {chosen_threshold:.2f}\n")
        f.write(f"Monitoring mode: {assign_monitoring_mode(chosen_threshold)}\n")
        f.write(f"Alerts generated: {int(selected_row['alerts_generated']):,}\n")
        f.write(f"Frauds caught in alert queue: {int(selected_row['actual_frauds_caught']):,}\n")
        f.write(f"Precision: {selected_row['precision']:.4f}\n")
        f.write(f"Recall: {selected_row['recall']:.4f}\n")
        f.write(f"False positives: {int(selected_row['false_positives']):,}\n")
        f.write(f"False negatives: {int(selected_row['false_negatives']):,}\n\n")
        f.write(
            "The threshold was selected by requiring a minimum precision floor of 0.40, "
            "then choosing the best F1 among thresholds that cleared it. This avoids "
            "chasing recall at the cost of drowning analysts in false positives.\n\n"
        )

        f.write("Risk band interpretation\n")
        f.write("------------------------\n")
        f.write(
            f"Observed fraud rate in High band: {high_rate:.2%}\n"
            f"Observed fraud rate in Critical band: {critical_rate:.2%}\n"
        )
        f.write(
            "Risk bands convert raw probabilities into operational triage categories, "
            "supporting queue prioritization and review routing.\n\n"
        )

        f.write("Rules baseline versus model\n")
        f.write("---------------------------\n")
        f.write(rules_vs_model_df.to_string(index=False))
        f.write("\n\n")
        f.write(
            "The comparison matters because financial crime teams almost always have an existing "
            "rules engine. The model needs to justify itself against that baselinenot just against "
            "doing nothing. The table above shows how both approaches compare on fraud capture, "
            "queue volume, and false positive burden.\n\n"
        )

        f.write("SQL implementation note\n")
        f.write("-----------------------\n")
        f.write(
            "The SQL monitoring layer in this project runs against a SQLite table created from "
            "the scored transaction output. In a real production environment, analysts would more "
            "likely query upstream transaction-monitoring or warehouse tables they don't directly "
            "control. The SQL layer here is used to simulate the downstream monitoring summaries "
            "and reporting logic that workflow would require.\n\n"
        )

        f.write("Forensic accounting tie-in\n")
        f.write("--------------------------\n")
        f.write(
            "This workflow mirrors a forensic mindset to identify unusual patterns, surface transactions "
            "with multiple red flags, and translate those signals into a reviewable case queue. Similar to "
            "forensic accounting, the goal is not to treat anomalies as proof, but as evidence requiring "
            "structured investigation and documented escalation.\n\n"
        )

        f.write("Recommendation\n")
        f.write("--------------\n")
        f.write(
            f"At the selected threshold of {chosen_threshold:.2f} ({assign_monitoring_mode(chosen_threshold)} mode), "
            f"the model generated {len(high_risk_alerts):,} high-risk alerts with an observed fraud hit rate of "
            f"{high_risk_precision:.2%}. This supports an alert operations workflow where model scores, "
            "rule-based reasoning, and reporting outputs are used together to prioritize review. "
            "Teams prioritizing investigator efficiency can stay at this threshold; teams prioritizing "
            "maximum fraud capture should consider lowering it and accepting a higher false positive burden.\n"
        )


def main() -> None:
    print("Loading dataset")
    df = load_data(RAW_DATA_PATH)

    print("Engineering features")
    df = engineer_features(df)

    print("Preparing train/test split")
    X, y = prepare_xy(df)

    # Stratify on y so both splits see the same (tiny) fraud rate.
    # Without this you can easily get a test set with different class balance by chance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("Training model")
    pipeline = train_model(X_train, y_train)

    print("Scoring test set")
    fraud_probs = pipeline.predict_proba(X_test)[:, 1]

    # Report performance at the default 0.50 threshold for baseline reference.
    # The operational threshold we actually use gets chosen separately below.
    default_threshold = 0.50
    y_pred_default = (fraud_probs >= default_threshold).astype(int)

    print("\n=== MODEL PERFORMANCE @ 0.50 THRESHOLD ===")
    report_text = classification_report(y_test, y_pred_default, digits=4)
    print(report_text)

    roc = roc_auc_score(y_test, fraud_probs)
    ap = average_precision_score(y_test, fraud_probs)
    print(f"ROC AUC: {roc:.4f}")
    print(f"Average Precision (PR AUC): {ap:.4f}")

    cm = confusion_matrix(y_test, y_pred_default)
    save_confusion_matrix(cm, CHARTS_DIR / "confusion_matrix.png")
    save_precision_recall_curve(y_test, fraud_probs, CHARTS_DIR / "precision_recall_curve.png")

    # Step 1: sweep thresholds, label each with a monitoring mode, then pick one
    print("\nRunning threshold analysis")
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    threshold_df = threshold_analysis(y_test, fraud_probs, thresholds)
    threshold_df["monitoring_mode"] = threshold_df["threshold"].apply(assign_monitoring_mode)

    chosen_threshold = choose_operational_threshold(threshold_df)
    print(f"Chosen operational threshold: {chosen_threshold:.2f} ({assign_monitoring_mode(chosen_threshold)} mode)")

    threshold_df.to_csv(OUTPUTS_DIR / "threshold_analysis.csv", index=False)
    save_threshold_tradeoff_chart(threshold_df, CHARTS_DIR / "threshold_tradeoff.png")

    # Step 2: build the full alert queue at the chosen threshold
    print("Building alert operations outputs")
    all_scored, high_risk_alerts = build_alert_queue(
        X_test=X_test,
        y_test=y_test,
        fraud_probs=fraud_probs,
        chosen_threshold=chosen_threshold,
    )

    all_scored.to_csv(PROCESSED_DIR / "scored_test_transactions.csv", index=False)
    high_risk_alerts.to_csv(PROCESSED_DIR / "high_risk_alert_queue.csv", index=False)

    # Separate top-100 file so an analyst can open one small CSV for a quick
    # look without wading through the full alert queue.
    high_risk_alerts.head(100).to_csv(PROCESSED_DIR / "top_100_alerts_for_review.csv", index=False)

    risk_band_summary = summarize_risk_bands(all_scored)
    risk_band_summary.to_csv(OUTPUTS_DIR / "risk_band_summary.csv", index=False)
    save_risk_band_chart(risk_band_summary, CHARTS_DIR / "risk_band_fraud_rate.png")

    # Step 3: evaluate rules baseline and compare to model
    print("Evaluating rules-based baseline")
    rules_df = score_rules_baseline(X_test.copy())
    rules_pred = rules_df["rules_alert_flag"].values
    model_operational_pred = (fraud_probs >= chosen_threshold).astype(int)

    rules_vs_model_df = pd.DataFrame(
        [
            evaluate_binary_predictions(y_test, rules_pred, "Rules-based baseline"),
            evaluate_binary_predictions(y_test, model_operational_pred, f"Model @ threshold {chosen_threshold:.2f}"),
        ]
    )
    rules_vs_model_df.to_csv(OUTPUTS_DIR / "rule_vs_model_comparison.csv", index=False)

    # Step 4: write the text summaries
    write_model_summary(
        OUTPUTS_DIR / "model_summary.txt",
        df,
        X_train,
        X_test,
        y,
        roc,
        ap,
        report_text,
    )

    write_risk_reporting_summary(
        OUTPUTS_DIR / "risk_reporting_summary.txt",
        threshold_df,
        risk_band_summary,
        chosen_threshold,
        high_risk_alerts,
        rules_vs_model_df,
    )

    # Step 5: run the SQL monitoring queries and write the output file
    print("Writing SQL monitoring summary...")
    db_path = OUTPUTS_DIR / "fraud_monitoring.db"
    write_sql_monitoring_summary(
        scored_df=all_scored,
        db_path=db_path,
        output_path=OUTPUTS_DIR / "sql_monitoring_summary.txt",
    )

    print("\nSaved files:")
    print(f"- {PROCESSED_DIR / 'scored_test_transactions.csv'}")
    print(f"- {PROCESSED_DIR / 'high_risk_alert_queue.csv'}")
    print(f"- {PROCESSED_DIR / 'top_100_alerts_for_review.csv'}")
    print(f"- {OUTPUTS_DIR / 'threshold_analysis.csv'}")
    print(f"- {OUTPUTS_DIR / 'risk_band_summary.csv'}")
    print(f"- {OUTPUTS_DIR / 'rule_vs_model_comparison.csv'}")
    print(f"- {OUTPUTS_DIR / 'sql_monitoring_summary.txt'}")
    print(f"- {OUTPUTS_DIR / 'fraud_monitoring.db'}")
    print(f"- {OUTPUTS_DIR / 'model_summary.txt'}")
    print(f"- {OUTPUTS_DIR / 'risk_reporting_summary.txt'}")
    print(f"- {CHARTS_DIR / 'confusion_matrix.png'}")
    print(f"- {CHARTS_DIR / 'precision_recall_curve.png'}")
    print(f"- {CHARTS_DIR / 'threshold_tradeoff.png'}")
    print(f"- {CHARTS_DIR / 'risk_band_fraud_rate.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()