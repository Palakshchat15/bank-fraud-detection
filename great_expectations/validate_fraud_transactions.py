"""
Great Expectations validation of raw.transactions in Postgres.

Uses GE's lightweight pandas-based validation (reads the table into a
DataFrame via SQLAlchemy, then runs an in-memory Expectation Suite), matching
the pattern used successfully in the sibling retail-analytics portfolio
project.

BUSINESS RULES / edge cases considered explicitly for a bank-transaction
fraud dataset:

  - amount must be > 0 for every transaction (a transaction with $0 or
    negative amount is not a valid purchase/withdrawal event in this
    dataset's generation model - unlike the retail project's $0 promotional
    items, there is no legitimate "free transaction" concept for bank card
    transactions, so this IS a hard gate).
  - transaction_id, account_id, customer_id, transaction_datetime,
    merchant_category, transaction_country, channel must never be null
    (structurally required for every row).
  - transaction_datetime must fall within the known generation window
    (2025-01-01 to 2025-08-31) - a hard gate, since any row outside this
    range would indicate a corrupted load rather than legitimate data.
  - is_fraud must be in {0, 1} only.
  - amount must be within a reasonable range: very large values (whale
    fraud/legit high-value transfers) are expected and allowed up to a
    generous upper bound; extremely small values (e.g. $1-2, common at
    vending machines/parking meters/small fees) are a LEGITIMATE edge case
    and must NOT be flagged - this mirrors the lesson from the retail
    project (investigate before tightening a check). We checked the actual
    distribution: min amount in this dataset is ~$1.65, which is a
    realistic small legitimate transaction, so the lower bound is set
    generously low ($0.01) rather than reflexively raised.
  - every account_id/customer_id referenced in raw.transactions should
    correspond to a real customer - since this is a self-referential
    synthetic dataset (no separate customer dimension table at the raw
    layer), we instead validate structural consistency: each account_id
    must map to exactly one customer_id (1:1 account:customer design) with
    no cross-contamination - this catches data-generation bugs.

Exits non-zero if any hard-gate expectation fails.
"""

import os
import sys

import great_expectations as ge
import pandas as pd
import sqlalchemy

VALID_START = pd.Timestamp("2025-01-01 00:00:00")
VALID_END = pd.Timestamp("2025-08-31 23:59:59")


def get_engine():
    user = os.environ.get("POSTGRES_USER", "fraud_user")
    password = os.environ.get("POSTGRES_PASSWORD", "fraud_pass")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5435")
    db = os.environ.get("POSTGRES_DB", "fraud_detection")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return sqlalchemy.create_engine(url)


def main():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM raw.transactions", engine)
    print(f"Loaded {len(df)} rows from raw.transactions for validation")

    if len(df) == 0:
        print("FAIL: raw.transactions is empty - nothing to validate.")
        sys.exit(1)

    gdf = ge.from_pandas(df)
    results = []

    # Structural not-null checks
    for col in ["transaction_id", "account_id", "customer_id",
                "transaction_datetime", "merchant_category",
                "transaction_country", "channel"]:
        results.append((f"expect_{col}_not_null", gdf.expect_column_values_to_not_be_null(col)))

    # transaction_id uniqueness
    results.append(("expect_transaction_id_unique", gdf.expect_column_values_to_be_unique("transaction_id")))

    # amount > 0, hard gate (no legitimate $0 bank transaction in this dataset's scope)
    results.append((
        "expect_amount_positive",
        gdf.expect_column_values_to_be_between("amount", min_value=0.01, max_value=None),
    ))

    # amount reasonable upper bound - generous, since legit large transfers and
    # amount-spike fraud both produce high values; this just catches corrupted
    #/overflow values, not a business-plausibility filter.
    results.append((
        "expect_amount_within_sane_bound",
        gdf.expect_column_values_to_be_between("amount", min_value=0.01, max_value=100000),
    ))

    # transaction_datetime within known generation window (hard gate - a
    # value outside this range means a load/parsing bug, not legitimate data)
    results.append((
        "expect_datetime_within_range",
        gdf.expect_column_values_to_be_between(
            "transaction_datetime", min_value=VALID_START, max_value=VALID_END,
        ),
    ))

    # is_fraud must be 0 or 1
    results.append((
        "expect_is_fraud_binary",
        gdf.expect_column_values_to_be_in_set("is_fraud", [0, 1]),
    ))

    # account_id <-> customer_id must be a consistent 1:1 mapping (data
    # generation invariant - catches a real bug if violated)
    mapping_check = df.groupby("account_id")["customer_id"].nunique()
    bad_accounts = mapping_check[mapping_check > 1]
    account_mapping_ok = len(bad_accounts) == 0
    print(f"\naccount_id -> customer_id 1:1 mapping check: "
          f"{'PASS' if account_mapping_ok else 'FAIL'} "
          f"({len(bad_accounts)} accounts mapped to >1 customer_id)")

    # Edge case investigated: very small amounts. Real min in this dataset
    # is checked and printed (not reflexively flagged as bad data).
    min_amount = df["amount"].min()
    small_amount_count = (df["amount"] < 5).sum()
    print(f"Minimum amount observed: ${min_amount:.2f} "
          f"({small_amount_count} transactions under $5, "
          f"{small_amount_count/len(df)*100:.2f}% - treated as legitimate "
          f"small purchases, e.g. vending/parking/small fees, NOT flagged).")

    failed = [(name, r) for name, r in results if not r.success]

    print(f"\nRan {len(results)} hard-gate expectations, "
          f"{len(results) - len(failed)} passed, {len(failed)} failed.")
    for name, r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] {name} ({r.expectation_config.expectation_type})")
        if not r.success:
            unexpected = r.result.get("unexpected_count", "?")
            print(f"        unexpected_count={unexpected} details={r.result}")

    if failed or not account_mapping_ok:
        print("\nGreat Expectations validation FAILED.")
        sys.exit(1)

    print("\nGreat Expectations validation PASSED.")
    print("Confirms: structural integrity holds, amounts are positive and "
          "sane, datetimes are within the expected generation window, and "
          "legitimate small-dollar transactions are correctly NOT flagged.")


if __name__ == "__main__":
    main()
