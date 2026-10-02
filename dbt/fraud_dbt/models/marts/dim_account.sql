select
    account_id,
    customer_id,
    home_country,
    lifetime_txn_count,
    lifetime_fraud_count,
    round(lifetime_fraud_count::numeric / nullif(lifetime_txn_count, 0), 4) as lifetime_fraud_rate,
    round(lifetime_avg_amount::numeric, 2) as lifetime_avg_amount,
    round(coalesce(lifetime_stddev_amount, 0)::numeric, 2) as lifetime_stddev_amount,
    first_transaction_datetime,
    last_transaction_datetime,
    lifetime_distinct_categories,
    lifetime_distinct_countries
from {{ ref('int_account_summary') }}
