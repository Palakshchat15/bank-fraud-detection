-- Per-account lifetime summary stats, used to build dim_account and to
-- provide "customer baseline" context features in the ML mart.
with base as (
    select * from {{ ref('stg_transactions') }}
),

agg as (
    select
        account_id,
        customer_id,
        min(customer_home_country) as home_country,
        count(*) as lifetime_txn_count,
        sum(is_fraud) as lifetime_fraud_count,
        avg(amount) as lifetime_avg_amount,
        stddev(amount) as lifetime_stddev_amount,
        min(transaction_datetime) as first_transaction_datetime,
        max(transaction_datetime) as last_transaction_datetime,
        count(distinct merchant_category) as lifetime_distinct_categories,
        count(distinct transaction_country) as lifetime_distinct_countries
    from base
    group by account_id, customer_id
)

select * from agg
