"""
Chunked ingestion of the synthetic bank transaction CSV into
raw.transactions in Postgres.
"""
import argparse
import logging
import os
import time

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CSV = "/opt/airflow/data/raw/bank_transactions.csv"

COLUMNS = [
    "transaction_id", "account_id", "customer_id", "transaction_datetime",
    "amount", "merchant_category", "merchant_name", "transaction_city",
    "transaction_country", "customer_home_country", "channel", "is_fraud",
]

INSERT_SQL = f"""
    INSERT INTO raw.transactions
        ({", ".join(COLUMNS)})
    VALUES %s
    ON CONFLICT (transaction_id) DO NOTHING
"""


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ.get("POSTGRES_USER", "fraud_user"),
        password=os.environ.get("POSTGRES_PASSWORD", "fraud_pass"),
        dbname=os.environ.get("POSTGRES_DB", "fraud_detection"),
    )


def load(csv_path: str, chunksize: int = 50_000, truncate: bool = True):
    start = time.time()
    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    if truncate:
        logger.info("Truncating raw.transactions before load")
        cur.execute("TRUNCATE TABLE raw.transactions")
        conn.commit()

    total_rows = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        chunk["transaction_datetime"] = pd.to_datetime(chunk["transaction_datetime"])
        chunk = chunk[COLUMNS]

        records = [tuple(x) for x in chunk.to_numpy()]
        cleaned = []
        for row in records:
            cleaned.append(tuple(None if (isinstance(v, float) and pd.isna(v)) else v for v in row))

        execute_values(cur, INSERT_SQL, cleaned, page_size=chunksize)
        conn.commit()
        total_rows += len(chunk)
        logger.info("Loaded chunk %s: %s rows (cumulative %s)", chunk_idx, len(chunk), total_rows)

    cur.close()
    conn.close()
    elapsed = time.time() - start
    logger.info("Done. Total rows loaded: %s in %.1fs", total_rows, elapsed)
    return total_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--no-truncate", action="store_true")
    args = parser.parse_args()

    n = load(args.csv, chunksize=args.chunksize, truncate=not args.no_truncate)
    print(f"Loaded {n} rows into raw.transactions")
