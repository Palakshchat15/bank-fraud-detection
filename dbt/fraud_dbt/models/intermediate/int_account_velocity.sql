-- Velocity + behavioral features per transaction, computed via trailing
-- window functions over each account's own transaction history (ordered by
-- time). Uses Postgres RANGE window frames over the transaction_datetime
-- column to compute true trailing-time-window counts (not just row counts).
with base as (
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
    from {{ ref('stg_transactions') }}
),

with_lag as (
    select
        *,
        lag(transaction_datetime) over (
            partition by account_id order by transaction_datetime, transaction_id
        ) as prev_transaction_datetime
    from base
),

with_time_since_last as (
    select
        *,
        case
            when prev_transaction_datetime is null then null
            else extract(epoch from (transaction_datetime - prev_transaction_datetime)) / 60.0
        end as minutes_since_last_transaction
    from with_lag
),

-- Trailing 1hr / 24hr counts and rolling stats using RANGE window frames
-- (Postgres supports RANGE BETWEEN INTERVAL ... PRECEDING for timestamp
-- order-by columns).
windowed as (
    select
        *,
        count(*) over (
            partition by account_id
            order by transaction_datetime
            range between interval '1 hour' preceding and current row
        ) as txn_count_trailing_1hr,
        count(*) over (
            partition by account_id
            order by transaction_datetime
            range between interval '24 hours' preceding and current row
        ) as txn_count_trailing_24hr,
        -- Postgres does not support DISTINCT inside any window function
        -- (neither COUNT(DISTINCT ...) OVER nor array_agg(DISTINCT ...) OVER
        -- are implemented), so distinct-count-in-trailing-window is instead
        -- computed via array_agg(...) OVER (non-distinct, which IS
        -- supported) followed by de-duplicating the array with a plain
        -- scalar expression (array(select distinct unnest(...))) in the
        -- next CTE below.
        array_agg(merchant_category) over (
            partition by account_id
            order by transaction_datetime
            range between interval '24 hours' preceding and current row
        ) as categories_trailing_24hr_arr,
        array_agg(transaction_country) over (
            partition by account_id
            order by transaction_datetime
            range between interval '24 hours' preceding and current row
        ) as countries_trailing_24hr_arr,
        avg(amount) over (
            partition by account_id
            order by transaction_datetime, transaction_id
            rows between unbounded preceding and 1 preceding
        ) as cumulative_avg_amount_before_this_txn,
        count(*) over (
            partition by account_id
            order by transaction_datetime, transaction_id
            rows between unbounded preceding and 1 preceding
        ) as prior_txn_count,
        -- first time this account uses this merchant category / country
        -- (depends only on earlier rows, so it is past-only)
        row_number() over (
            partition by account_id, merchant_category
            order by transaction_datetime, transaction_id
        ) as category_use_seq,
        row_number() over (
            partition by account_id, transaction_country
            order by transaction_datetime, transaction_id
        ) as country_use_seq
    from with_time_since_last
),

with_novelty as (
    select
        *,
        case when category_use_seq = 1 then 1 else 0 end as first_category_use,
        case when country_use_seq = 1 then 1 else 0 end as first_country_use
    from windowed
),

with_prior_distinct as (
    select
        *,
        -- distinct categories/countries seen BEFORE this transaction (past-only
        -- replacement for the old lifetime_distinct_* columns, which were
        -- computed over all 8 months including the label period)
        coalesce(sum(first_category_use) over (
            partition by account_id order by transaction_datetime, transaction_id
            rows between unbounded preceding and 1 preceding
        ), 0) as prior_distinct_categories,
        coalesce(sum(first_country_use) over (
            partition by account_id order by transaction_datetime, transaction_id
            rows between unbounded preceding and 1 preceding
        ), 0) as prior_distinct_countries
    from with_novelty
)

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
    is_fraud,
    coalesce(minutes_since_last_transaction, 999999) as minutes_since_last_transaction,
    txn_count_trailing_1hr,
    txn_count_trailing_24hr,
    cardinality(array(select distinct unnest(categories_trailing_24hr_arr))) as distinct_categories_trailing_24hr,
    cardinality(array(select distinct unnest(countries_trailing_24hr_arr))) as distinct_countries_trailing_24hr,
    prior_txn_count,
    -- new-for-this-account flags (0 for an account's very first transaction)
    case when prior_txn_count > 0 then first_category_use else 0 end as is_new_category_for_account,
    case when prior_txn_count > 0 then first_country_use else 0 end as is_new_country_for_account,
    prior_distinct_categories,
    prior_distinct_countries,
    -- historical average amount for this account BEFORE this transaction
    -- (avoids leakage: never uses the current or future transactions)
    case
        when prior_txn_count > 0 then cumulative_avg_amount_before_this_txn
        else amount
    end as historical_avg_amount,
    -- deviation of this transaction's amount from the account's own history
    case
        when prior_txn_count > 0 and cumulative_avg_amount_before_this_txn > 0
            then (amount - cumulative_avg_amount_before_this_txn) / cumulative_avg_amount_before_this_txn
        else 0
    end as amount_deviation_ratio
from with_prior_distinct
