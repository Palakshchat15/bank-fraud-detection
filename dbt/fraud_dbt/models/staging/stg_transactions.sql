-- Staging: clean/standardize raw transactions.
with source as (
    select * from {{ source('raw', 'transactions') }}
),

cleaned as (
    select
        transaction_id,
        account_id,
        customer_id,
        transaction_datetime,
        date_trunc('day', transaction_datetime)::date as transaction_date,
        extract(hour from transaction_datetime)::int as transaction_hour,
        extract(dow from transaction_datetime)::int as transaction_dow,
        amount::numeric(12,2) as amount,
        lower(trim(merchant_category)) as merchant_category,
        merchant_name,
        transaction_city,
        transaction_country,
        customer_home_country,
        (transaction_country <> customer_home_country) as is_foreign_transaction,
        lower(trim(channel)) as channel,
        is_fraud
    from source
    where amount > 0
      and transaction_id is not null
      and account_id is not null
)

select * from cleaned
