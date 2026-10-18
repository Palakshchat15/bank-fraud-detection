"""
Bank transaction fraud classification.

Reads warehouse.mart_transaction_features (the ML-ready table built by dbt),
splits it BY TIME (train: Jan-Jun 2025, test: Jul-Aug 2025 - the model is
always evaluated on months it has never seen, as it would be in production),
and compares TWO approaches to handling severe class imbalance:

  (a) Class-weighted XGBoost: scale_pos_weight set from the actual observed
      train-set class ratio (not a fixed/assumed constant).
  (b) SMOTE-resampled XGBoost: SMOTE oversampling applied to the TRAINING
      SET ONLY (never touches the test set, to avoid leaking synthetic
      neighbors of test-set fraud into training), then a plain XGBoost fit
      on the resampled data.

For severe imbalance (fraud rate ~1%), raw accuracy is nearly meaningless
(predicting "not fraud" for everything gives ~99% accuracy) - so AUC-PR
(average_precision_score) is treated as the PRIMARY comparison metric
alongside AUC-ROC, with Precision/Recall/F1/Accuracy/confusion matrix
reported for completeness.

Robustness check: the winning approach is also re-trained on a stratified
random 75/25 split and both results are reported. The random split is NOT the
headline: transactions are a time series and fraud comes in bursts, so a random
split puts transactions from the same burst on both sides.

All features are past-only (they use only the account's earlier transactions).

Both approaches are evaluated on the SAME untouched, real-distribution test
set, and the better one (by AUC-PR) is selected as the final model, with the
comparison and justification written into model_report.txt.

SHAP explainability: global feature importance + per-transaction top-3
reasons for the highest-risk flagged transactions.

OUTPUTS:
  outputs/model_metrics.json        - both approaches' full metrics + winner
  outputs/model_report.txt          - human-readable comparison + discussion
  outputs/shap_summary.png          - SHAP global feature importance
  outputs/flagged_transactions.csv  - transaction_id, fraud_probability, top SHAP reasons

Optional MLflow tracking (log_to_mlflow, after all outputs are written; logging only):
params, every metric in model_metrics.json, the outputs above as artifacts, and the winning
model registered as 'fraud-xgboost'. Skipped with a warning if MLFLOW_TRACKING_URI is unset
or the server is unreachable.
"""
import json
import os

import numpy as np
import pandas as pd
import sqlalchemy
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "/opt/airflow/outputs")

NUMERIC_FEATURE_COLS = [
    "amount",
    "transaction_hour",
    "transaction_dow",
    "is_foreign_transaction",
    "minutes_since_last_transaction",
    "txn_count_trailing_1hr",
    "txn_count_trailing_24hr",
    "distinct_categories_trailing_24hr",
    "distinct_countries_trailing_24hr",
    "historical_avg_amount",
    "amount_deviation_ratio",
    "amount_to_avg_ratio",
    "is_night_hour",
    "prior_txn_count",
    "is_new_category_for_account",
    "is_new_country_for_account",
    "prior_distinct_categories",
    "prior_distinct_countries",
]
TEST_START = "2025-07-01"  # train Jan-Jun, test Jul-Aug
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.05, 1.0, 0.05)]
CATEGORICAL_FEATURE_COLS = ["merchant_category", "channel"]


def get_engine():
    user = os.environ.get("POSTGRES_USER", "fraud_user")
    password = os.environ.get("POSTGRES_PASSWORD", "fraud_pass")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5435")
    db = os.environ.get("POSTGRES_DB", "fraud_detection")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return sqlalchemy.create_engine(url)


def load_data(engine):
    # ORDER BY makes the seeded train/test split reproducible: without it, row order
    # (and so the split) changed whenever dbt rebuilt the table.
    df = pd.read_sql("select * from warehouse.mart_transaction_features order by transaction_id", engine)
    print(f"Loaded {len(df)} rows from warehouse.mart_transaction_features")
    return df


def build_features(df):
    df = df.copy()
    df["is_foreign_transaction"] = df["is_foreign_transaction"].astype(int)
    cat_dummies = pd.get_dummies(df[CATEGORICAL_FEATURE_COLS], prefix=CATEGORICAL_FEATURE_COLS)
    X = pd.concat([df[NUMERIC_FEATURE_COLS], cat_dummies], axis=1)
    X = X.fillna(0)
    y = df["is_fraud"].astype(int)
    return X, y


def evaluate(y_test, y_pred, y_pred_proba):
    return {
        "auc_roc": round(float(roc_auc_score(y_test, y_pred_proba)), 4),
        "auc_pr": round(float(average_precision_score(y_test, y_pred_proba)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def make_model(scale_pos_weight=None):
    params = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="aucpr", random_state=42, n_jobs=4)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return xgb.XGBClassifier(**params)


def fit_predict(approach, X_train, y_train, X_test):
    """Class weights / SMOTE are derived from the training rows only."""
    if approach == "class_weighted":
        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = make_model(spw)
        model.fit(X_train, y_train)
        return model, model.predict_proba(X_test)[:, 1], {"scale_pos_weight": round(float(spw), 4)}
    X_res, y_res = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train, y_train)
    model = make_model()
    model.fit(X_res, y_res)
    return model, model.predict_proba(X_test)[:, 1], {"smote_resampled_train_size": int(len(X_res))}


def check_for_leakage(X, importances_df, top_n=5):
    """Sanity check: flag any feature with suspiciously dominant importance
    that could indicate a label-proxy (leakage) rather than genuine signal."""
    warnings = []
    top_feats = importances_df.head(top_n)
    total_importance = importances_df["importance"].sum()
    for _, row in top_feats.iterrows():
        share = row["importance"] / total_importance if total_importance > 0 else 0
        if share > 0.85:
            warnings.append(
                f"Feature '{row['feature']}' accounts for {share:.1%} of total "
                f"importance alone - investigate for leakage."
            )
    return warnings


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    engine = get_engine()
    df = load_data(engine)

    fraud_rate = df["is_fraud"].mean()
    print(f"Overall fraud rate in mart_transaction_features: {fraud_rate:.4%}")

    X, y = build_features(df)
    feature_names = X.columns.tolist()

    is_test = pd.to_datetime(df["transaction_datetime"]) >= pd.Timestamp(TEST_START)
    idx_train, idx_test = df.index[~is_test], df.index[is_test]
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    print(f"Time split: train < {TEST_START}: {len(X_train)} ({y_train.mean():.4%} fraud) | "
          f"test >= {TEST_START}: {len(X_test)} ({y_test.mean():.4%} fraud)")

    # ---------- Approach (a): class-weighted XGBoost ----------
    model_weighted, proba_weighted, extra_weighted = fit_predict("class_weighted", X_train, y_train, X_test)
    scale_pos_weight = extra_weighted["scale_pos_weight"]
    print(f"\n[Approach A: class-weighted] scale_pos_weight={scale_pos_weight:.2f} (from train rows only)")
    pred_weighted = (proba_weighted >= 0.5).astype(int)
    metrics_weighted = evaluate(y_test, pred_weighted, proba_weighted)
    print(f"[Approach A results] {metrics_weighted}")

    # ---------- Approach (b): SMOTE-resampled XGBoost ----------
    print("\n[Approach B: SMOTE] Applying SMOTE to TRAINING SET ONLY...")
    model_smote, proba_smote, extra_smote = fit_predict("smote", X_train, y_train, X_test)
    X_train_smote_len = extra_smote["smote_resampled_train_size"]
    pred_smote = (proba_smote >= 0.5).astype(int)
    metrics_smote = evaluate(y_test, pred_smote, proba_smote)
    print(f"[Approach B results] {metrics_smote}")

    # ---------- Pick winner by AUC-PR (primary metric for severe imbalance) ----------
    if metrics_weighted["auc_pr"] >= metrics_smote["auc_pr"]:
        winner = "class_weighted"
        final_model = model_weighted
        final_metrics = metrics_weighted
        final_proba = proba_weighted
        final_pred = pred_weighted
    else:
        winner = "smote"
        final_model = model_smote
        final_metrics = metrics_smote
        final_proba = proba_smote
        final_pred = pred_smote

    print(f"\n=== WINNER (by AUC-PR): {winner} ===")

    # ---------- Robustness: same approach on a stratified random split ----------
    r_train, r_test = train_test_split(df.index, test_size=0.25, random_state=42, stratify=y)
    _, proba_r, _ = fit_predict(winner, X.loc[r_train], y.loc[r_train], X.loc[r_test])
    metrics_random = evaluate(y.loc[r_test], (proba_r >= 0.5).astype(int), proba_r)
    print(f"[Robustness: random 75/25 split, {winner}] {metrics_random}")

    # ---------- Precision / recall at alternative thresholds (winner, time split) ----------
    threshold_rows = []
    for t in THRESHOLDS:
        pred_t = (final_proba >= t).astype(int)
        threshold_rows.append({
            "threshold": t,
            "precision": round(float(precision_score(y_test, pred_t, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred_t, zero_division=0)), 4),
            "flagged": int(pred_t.sum()),
        })
    pd.DataFrame(threshold_rows).to_csv(os.path.join(OUTPUTS_DIR, "threshold_metrics.csv"), index=False)

    # ---------- Leakage sanity check ----------
    importances_df = pd.DataFrame({
        "feature": feature_names,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    leakage_warnings = check_for_leakage(X, importances_df)
    if leakage_warnings:
        print("\nLEAKAGE WARNINGS:")
        for w in leakage_warnings:
            print(f"  - {w}")
    else:
        print("\nNo single-feature-dominance leakage red flags detected "
              f"(top feature '{importances_df.iloc[0]['feature']}' accounts for "
              f"{importances_df.iloc[0]['importance']/importances_df['importance'].sum():.1%} "
              "of total importance).")
    print("\nTop 10 feature importances (winning model):")
    print(importances_df.head(10).to_string(index=False))

    if final_metrics["auc_roc"] > 0.995:  # red flag for a label proxy
        print(f"\nNOTE: AUC-ROC of {final_metrics['auc_roc']} is very high - "
              "investigated feature importances above for a dominant single "
              "feature that could be a label proxy. See leakage check output.")

    # ---------- Save metrics ----------
    metrics_out = {
        "overall_fraud_rate": round(float(fraud_rate), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_fraud_rate": round(float(y_train.mean()), 4),
        "test_fraud_rate": round(float(y_test.mean()), 4),
        "approach_a_class_weighted": {
            **metrics_weighted,
            "scale_pos_weight": round(float(scale_pos_weight), 4),
        },
        "approach_b_smote": {
            **metrics_smote,
            "smote_resampled_train_size": X_train_smote_len,
        },
        "winner": winner,
        "winner_metrics": final_metrics,
        "split": f"time-based: train < {TEST_START}, test >= {TEST_START}",
        "robustness_random_split_winner": metrics_random,
        "leakage_check_warnings": leakage_warnings,
        "top_10_features": importances_df.head(10).to_dict(orient="records"),
    }
    with open(os.path.join(OUTPUTS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ---------- Report ----------
    with open(os.path.join(OUTPUTS_DIR, "model_report.txt"), "w") as f:
        f.write("Bank Transaction Fraud Detection Model - Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total transactions: {len(df)}\n")
        f.write(f"Overall fraud rate: {fraud_rate:.4%}\n")
        f.write(f"Train: {len(X_train)} ({y_train.mean():.4%} fraud) | "
                f"Test: {len(X_test)} ({y_test.mean():.4%} fraud) "
                f"[time-based split: train Jan-Jun 2025, test Jul-Aug 2025]\n\n")

        f.write("Approach A: Class-Weighted XGBoost\n")
        f.write("-" * 40 + "\n")
        f.write(f"  scale_pos_weight (from actual train ratio): {scale_pos_weight:.2f}\n")
        for k, v in metrics_weighted.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Approach B: SMOTE-Resampled XGBoost\n")
        f.write("-" * 40 + "\n")
        f.write(f"  SMOTE applied to training set only ({len(X_train)} -> {X_train_smote_len} rows)\n")
        for k, v in metrics_smote.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write(f"WINNER (selected by AUC-PR, the primary metric for severe "
                f"class imbalance): {winner}\n")
        f.write(
            "Justification: with fraud at ~1% of transactions, accuracy is "
            "nearly meaningless (predicting 'no fraud' for every transaction "
            "already scores ~99% accuracy), so AUC-PR (average precision) - "
            "which focuses on the precision/recall tradeoff for the minority "
            "(fraud) class - is the more decision-relevant metric. "
        )
        if winner == "class_weighted":
            f.write(
                "Class-weighting won here: it lets the model learn from the "
                "true data distribution and calibrate directly on real "
                "minority-class density, whereas SMOTE's synthetic "
                "interpolated fraud examples can blur decision boundaries "
                "- here the interpolated examples did not improve the ranking "
                "of real fraud in the unseen test months.\n\n"
            )
        else:
            f.write(
                "SMOTE won here: oversampling the minority class in training "
                "gave the model more balanced exposure to fraud-pattern "
                "examples, improving its ability to separate the fraud "
                "decision boundary without hurting precision materially.\n\n"
            )

        f.write(f"Robustness check ({winner}, stratified random 75/25 split instead of time split):\n")
        for k, v in metrics_random.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write("Leakage investigation:\n")
        if leakage_warnings:
            for w in leakage_warnings:
                f.write(f"  WARNING: {w}\n")
        else:
            f.write(
                "  No single feature dominates model decisions "
                f"(top feature '{importances_df.iloc[0]['feature']}' = "
                f"{importances_df.iloc[0]['importance']/importances_df['importance'].sum():.1%} "
                "of total importance). All features are past-only.\n"
            )
        f.write("\nTop 10 feature importances:\n")
        f.write(importances_df.head(10).to_string(index=False))
        f.write("\n\nConfusion matrix [rows=actual, cols=predicted] (0=legit, 1=fraud), winning model:\n")
        f.write(f"{np.array(final_metrics['confusion_matrix'])}\n\n")
        f.write("Honest discussion:\n")
        f.write(
            "Synthetic dataset. An audit found that the first version of the "
            "generator leaked the label through the hour of day (hour 5 was "
            "100% fraud; legitimate activity never happened at most night "
            "hours) and that the model used lifetime account aggregates "
            "computed over all 8 months including the fraud rows. Both were "
            "fixed: night-time activity is now a realistic skew, features are "
            "past-only, and evaluation is on unseen later months. The "
            "generator's signal level was tuned on a validation split inside "
            "the training months only. Real-world fraud is adversarial and "
            "noisier, so real AUC-PR would typically be lower still. Results "
            "are reported exactly as computed.\n"
        )

    # ---------- SHAP ----------
    print("\nComputing SHAP values...")
    import shap
    explainer = shap.TreeExplainer(final_model)
    # Use a sample of the test set for SHAP if large, to keep runtime bounded
    shap_sample = X_test if len(X_test) <= 20000 else X_test.sample(20000, random_state=42)
    shap_values = explainer.shap_values(shap_sample)

    plt.figure()
    shap.summary_plot(shap_values, shap_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_summary.png"), dpi=150)
    plt.close()

    def top_reasons(row, n=3):
        ordered = row.abs().sort_values(ascending=False).index[:n]
        parts = []
        for feat in ordered:
            direction = "+" if row[feat] > 0 else "-"
            parts.append(f"{feat}({direction}{abs(row[feat]):.3f})")
        return "; ".join(parts)

    # Flagged high-risk transactions: every test-set transaction with
    # fraud_probability >= 0.5 (or the top 100 if none), each with its own SHAP
    # reasons. SHAP is computed on the flagged rows directly: restricting them to
    # the 20k plot sample above silently dropped ~44% of flagged transactions.
    result_df = df.loc[idx_test, ["transaction_id"]].copy()
    result_df["fraud_probability"] = final_proba
    result_df["predicted_fraud"] = final_pred
    result_df["actual_fraud"] = y_test.values

    flagged = result_df[result_df["fraud_probability"] >= 0.5].sort_values(
        "fraud_probability", ascending=False)
    if len(flagged) == 0:
        flagged = result_df.sort_values("fraud_probability", ascending=False).head(100)
    flagged_shap = pd.DataFrame(explainer.shap_values(X_test.loc[flagged.index]),
                                columns=feature_names, index=flagged.index)
    flagged = flagged.assign(top_shap_reasons=flagged_shap.apply(top_reasons, axis=1))

    flagged[["transaction_id", "fraud_probability", "top_shap_reasons"]].to_csv(
        os.path.join(OUTPUTS_DIR, "flagged_transactions.csv"), index=False)

    print(f"\nFlagged {len(flagged)} high-risk transactions "
          f"(saved to outputs/flagged_transactions.csv)")

    # Sample flagged transaction for README documentation
    if len(flagged) > 0:
        sample_row = flagged.iloc[0]
        print("\nSample flagged high-risk transaction:")
        print(f"  transaction_id: {sample_row['transaction_id']}")
        print(f"  fraud_probability: {sample_row['fraud_probability']:.4f}")
        print(f"  top_shap_reasons: {sample_row['top_shap_reasons']}")

    print("\n=== FINAL METRICS (winner) ===")
    print(json.dumps(final_metrics, indent=2))
    print(f"\nWrote outputs to {OUTPUTS_DIR}")

    # For the optional MLflow logging below (read-only; nothing above depends on it)
    return {"model": final_model, "winner": winner, "rows": int(len(df)),
            "first_txn": str(df["transaction_datetime"].min()),
            "last_txn": str(df["transaction_datetime"].max())}


# ---------- Optional MLflow experiment tracking (logging only) ----------
EXPERIMENT_NAME = "fraud_detection"
REGISTERED_MODEL_NAME = "fraud-xgboost"
RAW_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "data", "raw", "bank_transactions.csv")


def log_to_mlflow(run_info):
    """Runs after main() has written every output: reads them back and logs params, all metrics
    in model_metrics.json, the report/plot/CSV artifacts and the winning XGBoost model (registered
    as REGISTERED_MODEL_NAME). Guarded in mlflow_tracking: no server -> a warning, nothing else."""
    from mlflow_tracking import file_sha256, flatten_metrics, log_run

    model = run_info["model"]
    with open(os.path.join(OUTPUTS_DIR, "model_metrics.json")) as f:
        metrics_out = json.load(f)
    params = {f"xgb_{k}": v for k, v in model.get_params().items() if v is not None}
    params.update({
        "winner": run_info["winner"],
        "split": metrics_out["split"],
        "test_start": TEST_START,
        "smote_k_neighbors": 5,
        "smote_random_state": 42,
        "robustness_split": "stratified random 75/25, random_state=42",
        "decision_threshold": 0.5,
        "n_features": len(model.get_booster().feature_names or []),
        "numeric_features": ",".join(NUMERIC_FEATURE_COLS),
        "categorical_features": ",".join(CATEGORICAL_FEATURE_COLS),
    })
    raw_sha = file_sha256(RAW_CSV)
    data_version = (f"warehouse.mart_transaction_features: {run_info['rows']} rows, "
                    f"{run_info['first_txn']} .. {run_info['last_txn']}; "
                    f"bank_transactions.csv sha256 {raw_sha[:12] if raw_sha else 'n/a'}")

    def log_model(mlflow):
        import mlflow.xgboost
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME,
                                 pip_requirements=[f"xgboost=={xgb.__version__}"])

    log_run(
        EXPERIMENT_NAME, run_name=f"xgboost_{run_info['winner']}", params=params,
        metrics=flatten_metrics(metrics_out),
        artifacts=[os.path.join(OUTPUTS_DIR, name) for name in (
            "model_metrics.json", "model_report.txt", "shap_summary.png",
            "threshold_metrics.csv", "flagged_transactions.csv")],
        tags={"data_version": data_version, "raw_csv_sha256": raw_sha, "winner": run_info["winner"],
              "project": "bank_fraud_detection"},
        log_model=log_model,
    )


if __name__ == "__main__":
    _run_info = main()
    try:
        log_to_mlflow(_run_info)
    except Exception as e:  # tracking must never fail the training task
        print(f"WARNING [mlflow]: tracking skipped ({type(e).__name__}: {e}); training outputs are unaffected.",
              flush=True)
