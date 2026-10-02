"""
Bank Transaction Fraud Detection pipeline DAG.

Orchestrates: chunked CSV ingest into raw.transactions -> Great Expectations
validation -> dbt run -> dbt test -> mart export -> fraud model training
(class-weighted vs SMOTE comparison + SHAP) -> Excel dashboard -> Tableau
workbook export.

No GenAI/LLM step in this project's scope - not added.
"""

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor

logger = logging.getLogger(__name__)

PROJECT_ROOT = "/opt/airflow"

default_args = {
    "owner": "fraud-detection-pipeline",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def on_failure_callback(context):
    ti = context.get("task_instance")
    logger.error(
        "Task failed: dag=%s task=%s execution_date=%s try=%s",
        context.get("dag").dag_id if context.get("dag") else "?",
        ti.task_id if ti else "?",
        context.get("execution_date"),
        ti.try_number if ti else "?",
    )


def check_raw_csv_exists(**kwargs):
    path = f"{PROJECT_ROOT}/data/raw/bank_transactions.csv"
    return os.path.exists(path)


def run_ingestion(**kwargs):
    import subprocess

    cmd = ["python", f"{PROJECT_ROOT}/src/ingestion/load_raw_transactions.py",
           "--csv", f"{PROJECT_ROOT}/data/raw/bank_transactions.csv"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Ingestion failed: {result.stderr}")


def run_ge_validation(**kwargs):
    import subprocess

    cmd = ["python", f"{PROJECT_ROOT}/great_expectations/validate_fraud_transactions.py"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError("Great Expectations validation failed")


with DAG(
    dag_id="fraud_pipeline_dag",
    default_args=default_args,
    description="Ingest -> GE -> dbt -> fraud model (class-weighted vs SMOTE) -> Excel/Tableau export",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["fraud-detection", "portfolio"],
    on_failure_callback=on_failure_callback,
) as dag:

    wait_for_csv = PythonSensor(
        task_id="wait_for_raw_csv",
        python_callable=check_raw_csv_exists,
        poke_interval=10,
        timeout=120,
        mode="poke",
    )

    ingest = PythonOperator(
        task_id="ingest_raw_transactions",
        python_callable=run_ingestion,
    )

    ge_validate = PythonOperator(
        task_id="great_expectations_validate",
        python_callable=run_ge_validation,
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=(
            f"cd {PROJECT_ROOT}/dbt/fraud_dbt && "
            f"DBT_PROFILES_DIR={PROJECT_ROOT}/dbt dbt run"
        ),
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=(
            f"cd {PROJECT_ROOT}/dbt/fraud_dbt && "
            f"DBT_PROFILES_DIR={PROJECT_ROOT}/dbt dbt test"
        ),
    )

    export_marts = BashOperator(
        task_id="export_marts_to_csv",
        bash_command=f"cd {PROJECT_ROOT}/src/dashboards && python export_marts.py",
    )

    train_fraud_model = BashOperator(
        task_id="train_fraud_model",
        bash_command=f"cd {PROJECT_ROOT}/src/ml_pipeline && python train_fraud_model.py",
    )

    build_excel_dashboard = BashOperator(
        task_id="build_excel_dashboard",
        bash_command=f"cd {PROJECT_ROOT}/src/dashboards && python build_excel_dashboard.py",
    )

    build_tableau_workbook = BashOperator(
        task_id="build_tableau_workbook",
        bash_command=f"cd {PROJECT_ROOT}/src/dashboards && python build_tableau_workbook.py",
    )

    (
        wait_for_csv
        >> ingest
        >> ge_validate
        >> dbt_run
        >> dbt_test
        >> export_marts
        >> train_fraud_model
        >> build_excel_dashboard
        >> build_tableau_workbook
    )
