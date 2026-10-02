"""
Export warehouse marts from Postgres to CSV files under data/processed/.

These CSVs feed the Excel dashboard builder and the Tableau .twb datasources.
"""
import os

import pandas as pd
import sqlalchemy

OUT_DIR = os.environ.get("PROCESSED_DIR", "/opt/airflow/data/processed")


def get_engine():
    user = os.environ.get("POSTGRES_USER", "fraud_user")
    password = os.environ.get("POSTGRES_PASSWORD", "fraud_pass")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5435")
    db = os.environ.get("POSTGRES_DB", "fraud_detection")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return sqlalchemy.create_engine(url)


TABLES = {
    "mart_transaction_features": "transaction_features.csv",
    "fact_transactions": "fact_transactions.csv",
    "dim_account": "dim_account.csv",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = get_engine()
    for table, filename in TABLES.items():
        df = pd.read_sql(f"SELECT * FROM warehouse.{table}", engine)
        out_path = os.path.join(OUT_DIR, filename)
        df.to_csv(out_path, index=False)
        print(f"Exported warehouse.{table} -> {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
