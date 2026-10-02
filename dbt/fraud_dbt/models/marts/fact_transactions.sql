select
    transaction_id,
    account_id,
    customer_id,
    transaction_datetime,
    transaction_date,
    transaction_hour,
    transaction_dow,
    amount,
    merchant_category,
    transaction_country,
    is_foreign_transaction,
    channel,
    is_fraud
from {{ ref('int_account_velocity') }}
