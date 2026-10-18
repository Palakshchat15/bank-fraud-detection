# Upgrade notes: MLflow experiment tracking (2026-10-17, code; run 2026-10-18)

**Run on 2026-10-18: done.** DAG run `mlflow_upgrade_20261018_013815`: all 9 tasks succeeded; run
`xgboost_class_weighted` logged in `fraud_detection` (winner `auc_pr` 0.7979); `fraud-xgboost` v1 registered;
all outputs byte-identical to the pre-upgrade backup. UI http://localhost:5005. Container memory not measured.
Results: `README_PIPELINE.md`, "Experiment tracking (MLflow)".

## What was added
- An `mlflow` service (tracking server + model registry) in the Docker Compose stack:
  official image `ghcr.io/mlflow/mlflow:v2.16.2`, SQLite metadata store and artifact store on
  the named volume `mlflow_data`, one gunicorn worker, `mem_limit: 512m`, a `/health`
  healthcheck, host port **5005** (UI http://localhost:5005).
- `MLFLOW_TRACKING_URI=http://mlflow:5000` in the shared Airflow environment.
- Logging in `train_fraud_model.py`, after every output has been written. Experiment
  `fraud_detection`, run name `xgboost_<winner>`:
  - params: all non-null XGBoost parameters of the winning model (`xgb_*`), split, test start,
    SMOTE settings, decision threshold, feature lists;
  - metrics: every number in `model_metrics.json`, flattened (e.g. `winner_metrics.auc_pr`,
    `approach_b_smote.recall`, `winner_metrics.confusion_matrix.1.1`);
  - artifacts: `model_metrics.json`, `model_report.txt`, `shap_summary.png`,
    `threshold_metrics.csv`, `flagged_transactions.csv`;
  - model: the winning XGBoost model via `mlflow.xgboost.log_model`, registered as
    **`fraud-xgboost`** (a new version per run);
  - tags: `airflow_dag_id`, `airflow_run_id`, `airflow_task_id`, `airflow_try_number` (from the
    `AIRFLOW_CTX_*` variables Airflow sets for every task; `manual` outside Airflow),
    `data_version` (mart rows + transaction date range + sha256 of `bank_transactions.csv`),
    `raw_csv_sha256`, `winner`.
- `mlflow==2.16.2` in `requirements-airflow.txt` (same version as the server image).

## Files changed
- `docker-compose.yml`: `mlflow` service, `MLFLOW_TRACKING_URI`, `mlflow_data` volume.
- `requirements-airflow.txt`: `mlflow==2.16.2`.
- `src/ml_pipeline/mlflow_tracking.py` (new): the guarded logging helper (identical file in the
  Flight and Sales projects).
- `src/ml_pipeline/train_fraud_model.py`: `main()` now returns the winning model and a data
  summary; new `log_to_mlflow()` called after `main()`. No change above the `return`.
- `README_PIPELINE.md`: new section "Experiment tracking (MLflow)" with `not run yet` placeholders (filled in after the 2026-10-18 run).

## Why the results cannot change
The logging runs after `main()` has written every output, only reads those files and the
in-memory winning model, and touches no random state. Model code, seeds, the SQL, the split and
all outputs are unchanged. If `MLFLOW_TRACKING_URI` is unset, `mlflow` cannot be imported, the
server does not answer `/health` within 5 s, or any MLflow call fails, the script prints
`WARNING [mlflow]: ...` and the task still succeeds (HTTP retries are capped at 2 so a dead
server cannot stall the task).

## How to run (in order)
1. Back up the current outputs to compare against (PowerShell, from this folder):
   `New-Item -ItemType Directory -Force ..\..\_work_in_progress\upgrades\baselines; Copy-Item outputs ..\..\_work_in_progress\upgrades\baselines\fraud_outputs_pre_mlflow -Recurse`
2. Rebuild the Airflow image (new package): `docker compose build`
3. Start the stack as in README_PIPELINE.md "How to run", plus the server:
   `docker compose up -d mlflow` then `curl http://localhost:5005/health` (expect `OK`).
4. Check the image: `docker compose exec airflow-scheduler pip check` and
   `docker compose exec airflow-scheduler python -c "import mlflow; print(mlflow.__version__)"` (2.16.2).
5. Trigger: `docker compose exec airflow-scheduler airflow dags trigger fraud_pipeline_dag`, then
   `airflow tasks states-for-dag-run fraud_pipeline_dag <run_id>` until all 9 tasks succeed.
6. `docker compose stop` when done.

## What to verify
- The `train_fraud_model` log ends with `[mlflow] logged run ...` and has no `WARNING [mlflow]`.
- MLflow UI (http://localhost:5005): experiment `fraud_detection` has a run whose
  `airflow_run_id` tag equals the DAG run id; `winner_metrics.auc_pr` equals the value in
  `outputs/model_metrics.json`; the artifacts and the `model/` folder are present.
- Models tab: `fraud-xgboost` has a new version.
- Outputs are byte-identical to the backup: `Get-FileHash` on `model_metrics.json`,
  `model_report.txt`, `threshold_metrics.csv`, `flagged_transactions.csv` in both folders.
- Guard: `docker compose stop mlflow`, clear and re-run only `train_fraud_model`; it must succeed
  with one `WARNING [mlflow]: tracking server ... is unreachable` line.
- Memory: `docker stats --no-stream` during the run; record the `mlflow` container and the total.
- Fill in the table in README_PIPELINE.md "Experiment tracking (MLflow)".

## Expected run time (estimates, not measured)
Image rebuild a few minutes (one extra package set); first pull of the MLflow image a few
minutes; the logging itself adds seconds to `train_fraud_model`.

## Risks
- **Dependencies:** the Dockerfile installs without Airflow's constraints file, so pip resolves
  `mlflow` against what is in the image. Checked against PyPI metadata (not installed):
  mlflow 2.16.2 needs `sqlalchemy>=1.4,<3`, `protobuf>=3.12,<6`, `packaging<25`, `Flask<4`,
  `alembic<2`, `gunicorn<24`, `pyarrow>=4,<18`, `docker>=4,<8`, `importlib-metadata<9`,
  `opentelemetry-api/sdk<3`, `sqlparse>=0.4,<1`, `numpy<3`, `pandas<3`. Airflow 2.9.3's Python
  3.11 constraints have SQLAlchemy 1.4.52, protobuf 4.25.3, packaging 24.1, Flask 2.2.5, alembic
  1.13.2, gunicorn 22.0.0, pyarrow 16.1.0, docker 7.1.0, importlib-metadata 6.11.0,
  opentelemetry 1.25.0, sqlparse 0.5.0; dbt-core 1.7.13 needs `protobuf<5`, `sqlparse<0.6`.
  All of these fall inside mlflow's bounds, so no downgrade is expected. New packages:
  `databricks-sdk`, `graphene`, `querystring_parser` (pure Python). `pip check` in step 4
  confirms it. Fallback if the build conflicts: pin `mlflow-skinny==2.16.2` instead (client
  only, no Flask/alembic/gunicorn/pyarrow/docker requirements).
- **Image:** needs network for the first pull from ghcr.io (tag `v2.16.2` exists). The server and
  client versions must stay equal.
- **Memory:** the server is capped at 512 MB; the real use is not measured yet. If it is
  OOM-killed (healthcheck fails), raise `mem_limit`. Fraud peaked at 2.3 GB in a 4 GB VM, so
  there is room.
- **Data:** runs and models live in the `mlflow_data` volume; `docker compose down -v` deletes
  them. Each DAG run adds a run and a registered-model version.
- MLflow prints a harmless warning that the model has no signature/input example (none is logged
  on purpose: it would add an extra prediction pass).
