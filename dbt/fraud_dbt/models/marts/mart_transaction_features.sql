-- ML-ready mart: one row per transaction with all engineered features + label.
select
    v.transaction_id,
    v.account_id,
    v.customer_id,
    v.transaction_datetime,
    v.transaction_date,
    v.transaction_hour,
    v.transaction_dow,
    v.amount,
    v.merchant_category,
    v.transaction_country,
    v.is_foreign_transaction,
    v.channel,

    -- velocity features
    v.minutes_since_last_transaction,
    v.txn_count_trailing_1hr,
    v.txn_count_trailing_24hr,
    v.distinct_categories_trailing_24hr,
    v.distinct_countries_trailing_24hr,

    -- amount / behavioral deviation features
    v.historical_avg_amount,
    v.amount_deviation_ratio,
    case when v.historical_avg_amount > 0 then v.amount / v.historical_avg_amount else 1 end as amount_to_avg_ratio,

    -- generic night-time flag (midnight-6am). Earlier versions used 2-5am,
    -- which copied the old generator's hard-coded fraud rule.
    case when v.transaction_hour between 0 and 5 then 1 else 0 end as is_night_hour,

    -- account novelty features, past-only: they depend only on the account's
    -- earlier transactions. (The old lifetime_* columns from dim_account were
    -- computed over all 8 months, including fraud rows and future data, and
    -- have been removed from the ML mart; dim_account keeps them for BI only.)
    v.prior_txn_count,
    v.is_new_category_for_account,
    v.is_new_country_for_account,
    v.prior_distinct_categories,
    v.prior_distinct_countries,
    a.home_country,

    v.is_fraud
from {{ ref('int_account_velocity') }} v
left join {{ ref('dim_account') }} a on v.account_id = a.account_id
