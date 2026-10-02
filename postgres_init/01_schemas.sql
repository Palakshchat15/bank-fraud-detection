-- Schemas for the fraud detection pipeline
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS intermediate;
CREATE SCHEMA IF NOT EXISTS warehouse;

-- Raw transactions table (loaded by src/ingestion/load_raw_transactions.py)
CREATE TABLE IF NOT EXISTS raw.transactions (
    transaction_id          VARCHAR(20) PRIMARY KEY,
    account_id              VARCHAR(20) NOT NULL,
    customer_id             VARCHAR(20) NOT NULL,
    transaction_datetime    TIMESTAMP NOT NULL,
    amount                  NUMERIC(12,2) NOT NULL,
    merchant_category       VARCHAR(50) NOT NULL,
    merchant_name           VARCHAR(100) NOT NULL,
    transaction_city        VARCHAR(100) NOT NULL,
    transaction_country     VARCHAR(50) NOT NULL,
    customer_home_country   VARCHAR(50) NOT NULL,
    channel                 VARCHAR(20) NOT NULL,
    is_fraud                INTEGER NOT NULL,
    loaded_at               TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_transactions_account ON raw.transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_raw_transactions_datetime ON raw.transactions(transaction_datetime);
