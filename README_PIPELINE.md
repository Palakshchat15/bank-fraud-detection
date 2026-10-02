# Bank Transaction Fraud Detection Pipeline

End-to-end data engineering + ML portfolio project on a **synthetic bank
transaction dataset** (149,995 transactions, Jan-Aug 2025, 3,000
accounts/customers) with deliberately engineered, learnable fraud signal.
Every number in this document was produced by actually running the pipeline
in this repo on 2026-09-25 — see "Verified run evidence" at the bottom for
exact commands and outputs.

**Headline result (after the audit fixes below):** class-weighted XGBoost,
trained on Jan-Jun 2025 and tested on the unseen months Jul-Aug 2025:
**AUC-PR 0.798**, AUC-ROC 0.978, precision 51.6% / recall 80.8% at the 0.5
threshold. An earlier version reported AUC-PR 0.976; an audit showed that
number came from label leakage in the synthetic data and features, not from a
better model. See "Audit fixes".

## Architecture

```
data/generator/generate_fraud_data.py --> data/raw/bank_transactions.csv (149,995 rows)
        |
        v  src/ingestion/load_raw_transactions.py  (chunked, 50k rows/chunk)
Postgres: raw.transactions
        |
        v  great_expectations/validate_fraud_transactions.py
   (12 hard-gate expectations: not-null, amount>0, datetime range, is_fraud binary, etc.)
        |
        v  dbt run  (dbt/fraud_dbt)
staging.stg_transactions
        |
        v
intermediate.int_account_velocity (trailing 1hr/24hr velocity, amount deviation, past-only novelty features)
intermediate.int_account_summary  (per-account lifetime aggregates - BI only, not model features)
        |
        v
warehouse.dim_account, warehouse.fact_transactions,
warehouse.mart_transaction_features (ML-ready, one row per transaction)
        |
        v  dbt test  (27 schema tests: not_null/unique/relationships/accepted_values)
        |
        +--> src/dashboards/export_marts.py --> data/processed/*.csv
        |
        +--> src/ml_pipeline/train_fraud_model.py --> outputs/ (metrics, SHAP, flagged transactions)
        |
        +--> src/dashboards/build_excel_dashboard.py --> outputs/Fraud_Detection_Dashboard.xlsx
        |
        +--> src/dashboards/build_tableau_workbook.py --> tableau/Fraud_Detection.twbx
```

Orchestrated by a single Airflow DAG: `airflow/dags/fraud_pipeline_dag.py`
(`fraud_pipeline_dag`), task order:

```
wait_for_raw_csv -> ingest_raw_transactions -> great_expectations_validate
  -> dbt_run -> dbt_test -> export_marts_to_csv -> train_fraud_model
  -> build_excel_dashboard -> build_tableau_workbook
```

There is **no GenAI/LLM step** in this project's scope — not added.

## Folder structure

```
data/generator/          Synthetic fraud data generator (documented assumptions below)
data/raw/                Generated CSV (not committed; see .gitignore)
data/processed/          CSV exports of the warehouse marts + Tableau-derived aggregates
dbt/fraud_dbt/           dbt project: staging -> intermediate -> marts (warehouse)
dbt/profiles.yml         dbt Postgres connection profile (env-var driven)
great_expectations/      Standalone GE validation script (pandas-based, not a full DataContext)
postgres_init/           SQL run once on first Postgres container init (schemas + raw table DDL)
src/ingestion/           Chunked CSV -> Postgres loader
src/ml_pipeline/         Fraud model training (class-weighted vs SMOTE XGBoost + SHAP)
src/dashboards/          Marts export, Excel dashboard builder, Tableau .twbx builder
outputs/                 Model metrics/report/SHAP plot/flagged transactions + Excel workbook
tableau/                 Tableau .twbx workbook (generated, extract-based)
docker-compose.yml       postgres, postgres-init-airflow-db, airflow-init, airflow-webserver, airflow-scheduler
```

## Libraries / stack

- **Orchestration**: Apache Airflow 2.9.3 (LocalExecutor), Python 3.11
- **Warehouse**: PostgreSQL 15 (schemas: `raw`, `staging`, `intermediate`, `warehouse`)
- **Transformation**: dbt-core 1.7.13 + dbt-postgres 1.7.13
- **Data quality**: Great Expectations 0.18.19 (lightweight pandas-based usage)
- **ML**: scikit-learn 1.4.2, XGBoost 2.0.3, imbalanced-learn 0.12.3 (SMOTE), SHAP 0.45.0
- **Dashboards**: openpyxl 3.1.2 (Excel), Tableau .twbx generated with the Tableau Hyper API
- **Pinning rationale**: `sqlalchemy>=1.4.36,<2.0` because both Airflow 2.9.3
  and dbt-postgres 1.7.13 require SQLAlchemy 1.x; numpy/pandas/xgboost/shap
  versions pinned together to avoid known ABI incompatibilities; the DAG
  uses `PythonSensor` (checking `os.path.exists`), not `FileSensor`, to
  avoid requiring an unconfigured `fs_default` Airflow connection.

## Ports (this project, chosen to avoid collision with sibling portfolio projects)

- Postgres: host `localhost:5435` -> container `5432`
- Airflow webserver: host `localhost:8084` -> container `8080` (login `admin`/`admin`)

## How to run

```bash
# 1. Generate the synthetic dataset (one-time, or to regenerate; fixed seed 42,
#    so the output is identical every time). The DAG does NOT run the generator:
#    it ingests whatever data/raw/bank_transactions.csv is present.
python data/generator/generate_fraud_data.py

# 2. Bring up Postgres and let it initialize schemas
docker compose up -d postgres postgres-init-airflow-db

# 3. Build and initialize Airflow's metadata DB + admin user
docker compose build airflow-init
docker compose up airflow-init

# 4. Start the Airflow webserver (http://localhost:8084, admin/admin) and scheduler
docker compose up -d airflow-webserver airflow-scheduler

# 5. Confirm the DAG has no import errors
docker exec <airflow-webserver-container> airflow dags list-import-errors

# 6. Trigger the full pipeline
docker exec <airflow-webserver-container> airflow dags trigger fraud_pipeline_dag

# 7. Watch it to completion
docker exec <airflow-webserver-container> airflow tasks states-for-dag-run fraud_pipeline_dag <run_id>
```

Postgres is reachable from the host at `localhost:5435` — this is the port
to use for the Excel/Power Query live connection below, and for any local
`psql`/DBeaver/etc. client.

## Fraud-signal-generation logic (exact assumptions, as implemented)

Implemented in `data/generator/generate_fraud_data.py` (seed 42). Population:
3,000 customers, each with exactly one account. Date range: 2025-01-01 to
2025-08-31. 149,995 transactions, 1,499 fraud (0.999%).

**Legitimate behaviour** is drawn from per-customer baselines (average amount,
3-6 preferred merchant categories, activity level, personal active window
starting 6-9am and ending 9-11pm):

- *Time of day*: most activity falls inside the personal window, but every
  hour has some activity. Outside the window the hourly weight is 0.15 (0.06 at
  midnight-6am) against 1.0 inside it, so 2.5% of legitimate transactions
  happen between midnight and 6am. No hour is fraud-only.
- *Shopping sessions*: 12% of transactions follow the customer's previous one
  within 5-90 minutes, so short gaps and several transactions an hour also
  happen legitimately.
- *Travel*: 35% of customers take 1-2 trips abroad of 3-10 days, and 90% of
  purchases on trip days are made in the trip country. Another 1.5% of
  transactions are cross-border online purchases. 2.7% of legitimate
  transactions are foreign in total.
- *Unusual but legitimate*: 2% are one-off large purchases (3-8x the customer's
  average) and 15% are outside the preferred categories.

**Fraud**: 22% of customers (660) are victims. Every victim gets at least one
compromise episode: the allocation only allows a burst when enough budget
is left to give every remaining victim a transaction, and leftover budget
becomes repeat single hits. Each episode has one attacker profile:

1. **Velocity burst** (27% of episodes, 67% of fraud transactions): 3-8
   transactions, each 0.3-3.5 minutes after the previous one, and the whole
   burst always within 15 minutes (176 bursts, median span 8.4 minutes, max
   14.5). In 40% of bursts the first transaction is a small $1-10 "card test".
2. **Geographic anomaly** (45% of episodes): transactions from another
   country, half of them from high-risk countries (Nigeria, Russia, China,
   UAE). 40.8% of fraud transactions are foreign.
3. **Merchant anomaly**: 55% of fraud transactions use cash-out categories
   (crypto_exchange, money_transfer, electronics, jewelry, gambling,
   online_retail).
4. **Amount anomaly** (40% of episodes): 3-12x the customer's average,
   sometimes rounded to a clean hundred.
5. **Time-of-day skew**: fraud times follow the population's daily curve,
   with midnight-6am weighted 6x. 15.1% of fraud happens at night, compared
   with 2.5% of legitimate transactions. The fraud rate is 5.8% at night and
   0.87% during the day (maximum 8.4% in any single hour), and most night
   transactions are still legitimate.

**How the signal level was chosen.** The target was a credible fraud model
for synthetic data: AUC-PR roughly 0.5-0.85, clearly better than the 1%
base rate but short of the near-perfect 0.97+ that indicated leakage. The
settings were judged only on a validation split *inside the training months*
(train Jan-Apr, validate May-Jun), using the same features and XGBoost setup.
The Jul-Aug test months were not used to choose anything. The first
candidate scored AUC-PR 0.764 on validation (AUC-ROC 0.964, precision 0.60,
recall 0.75 at 0.5) with no dominant feature (top feature 12% of importance),
so it was kept as is. The test result reported below (0.798) was computed
afterwards and is close to the validation number.

## Data sanity-check (before building anything downstream)

Real output from `python data/generator/generate_fraud_data.py`:

- Total transactions: 149,995. Fraud: 1,499 (0.999%). Unique accounts: 3,000.
  Victim accounts: 660 (22.0%).
- Amounts: legit mean $55.28 (median $37.46), fraud mean $158.03 (median
  $70.68). The range is $1.03 to $2,201.45.
- Fraud rate by hour: 2.6-8.4% for hours 0-5 and 0.5-1.2% for hours 6-23.
  60% of fraud used to fall in hours 0-5; now 15% does.
- Top fraud categories: online_retail 209, electronics 172, jewelry 154,
  gambling 153, crypto_exchange 153. Legit is led by grocery (23,481),
  restaurant (20,786) and online_retail (18,812).

## Great Expectations: 12 hard-gate checks, all passed cleanly

Unlike the sibling Retail E-commerce project (which found a real edge case
requiring a WARN-severity downgrade), this run's GE validation **passed all
12 checks on the first try** with no failures requiring investigation. This
was still investigated deliberately rather than assumed: the minimum
transaction amount in the real data is $1.03 (in the regenerated data;
$1.65 in the first version). These are legitimate small purchases and are explicitly
documented as NOT flaggable, but since they didn't trip any actual
expectation threshold, no severity downgrade was needed. All checks
(not-null on 7 key columns, transaction_id uniqueness, amount > 0, amount
within a sane upper bound, datetime within the known generation window,
is_fraud in {0,1}, and a structural account_id<->customer_id 1:1 mapping
check) passed against the real 149,995-row dataset, again after the data
was regenerated (12/12 passed).

## dbt marts (real row counts, this run)

| Table | Rows |
|---|---|
| `staging.stg_transactions` | 149,995 |
| `warehouse.dim_account` | 3,000 |
| `warehouse.fact_transactions` | 149,995 |
| `warehouse.mart_transaction_features` | 149,995 |

`dbt test` passed: 27/27 schema tests (not_null / unique / relationships /
accepted_values across staging/intermediate/mart models, including the new
0/1 novelty and night-hour flags).

**Real Postgres bug found and fixed**: Postgres does not support `DISTINCT`
inside *any* window function (`COUNT(DISTINCT ...) OVER (...)` and even
`array_agg(DISTINCT ...) OVER (...)` both error with "DISTINCT is not
implemented for window functions"). Fixed by computing a non-distinct
`array_agg(...)  OVER (...)` for the trailing-24hr window, then
de-duplicating with `cardinality(array(select distinct unnest(...)))` in the
next SQL step. This is documented in `int_account_velocity.sql`.

### Feature engineering validated against the injected fraud signal (real query)

All model features are **past-only**: each value depends only on the
account's own earlier transactions (trailing windows, prior averages, and
"first time this account used this category/country" flags).

```sql
select is_fraud, count(*),
  avg(txn_count_trailing_1hr), avg(txn_count_trailing_24hr),
  avg(distinct_categories_trailing_24hr), avg(amount_deviation_ratio),
  avg(is_night_hour), avg(is_foreign_transaction::int),
  avg(is_new_country_for_account), avg(is_new_category_for_account)
from warehouse.mart_transaction_features group by is_fraud;
```

| is_fraud | n | txn/1hr | txn/24hr | distinct categories/24hr | amount deviation | % night (0-5h) | % foreign | % new country | % new category |
|---|---|---|---|---|---|---|---|---|---|
| 0 (legit) | 148,496 | 1.09 | 1.41 | 1.31 | 0.03 | 2.5% | 2.7% | 1.9% | 13.1% |
| 1 (fraud) | 1,499 | 2.79 | 3.01 | 2.61 | 1.43 | 15.1% | 40.8% | 17.8% | 54.8% |

Each signal separates the classes on average, but none does so cleanly on
its own. For example, 85% of fraud happens during the day, and 59% of fraud
is in the customer's home country.

## Real model metrics (holdout test months, 2026-09-25)

```
Total transactions: 149,995 | Overall fraud rate: 0.999%
Train: 111,471 (1.01% fraud) = Jan-Jun 2025 | Test: 38,524 (0.96% fraud, 370 frauds) = Jul-Aug 2025
[time-based split; class weights and SMOTE computed from the training rows only]
```

### Approach A: Class-Weighted XGBoost (scale_pos_weight=97.73, from the train ratio) - winner

```
AUC-ROC:   0.9777
AUC-PR:    0.7979
Precision: 0.5164
Recall:    0.8081
F1 Score:  0.6301
Accuracy:  0.9909
Confusion matrix [rows=actual, cols=predicted] (0=legit, 1=fraud):
[[37874   280]
 [   71   299]]
```

### Approach B: SMOTE-Resampled XGBoost (training set resampled to 50/50, 111,471 -> 220,684 rows)

```
AUC-ROC:   0.9750
AUC-PR:    0.7825
Precision: 0.8098
Recall:    0.7135
F1 Score:  0.7586
Accuracy:  0.9956
Confusion matrix:
[[38092    62]
 [  106   264]]
```

### Winner: Class-Weighted (selected by AUC-PR)

At ~1% fraud, accuracy is nearly meaningless, so the approaches are compared
on AUC-PR, which measures how well each one ranks fraud above legitimate
transactions. Class-weighting ranks slightly better (0.798 vs 0.783). The
precision/recall difference at 0.5 mostly reflects calibration rather than
ranking. SMOTE's probabilities sit lower, so its 0.5 cut-off is stricter.
The class-weighted model reaches similar operating points by moving the
threshold (`outputs/threshold_metrics.csv`, test months):

| threshold | precision | recall | flagged |
|---|---|---|---|
| 0.3 | 34.5% | 85.7% | 918 |
| 0.5 | 51.6% | 80.8% | 579 |
| 0.7 | 66.4% | 74.3% | 414 |
| 0.8 | 78.0% | 71.1% | 337 |
| 0.9 | 87.5% | 66.0% | 279 |

The right threshold depends on review capacity and on what a missed fraud
costs compared with a false alarm. The dashboards use 0.5.

### Robustness: random split vs time split

The headline uses a time-based split because transactions are a time series
and fraud comes in bursts. In the first version, 183 of 375 test frauds under
the random split had a fraud on the same account within 2 hours that was in
the training set. Re-training the winning approach on a stratified random
75/25 split (seed 42) gives **AUC-PR 0.787**, AUC-ROC 0.962, precision 37.6% /
recall 82.7% at 0.5, close to the time-split result. The random split did
not inflate the metrics here. A grouped-by-account split was not added; if it
is, use a seed other than 42, because victims were drawn with
`random_state=42`.

**Reproducibility.** Training was run three times (twice in the DAG, once
manually) and `model_metrics.json` and `flagged_transactions.csv` were
byte-identical each time. The training query orders by `transaction_id`, and
the row-based dbt windows break timestamp ties on `transaction_id`.

### What the model relies on

The top features by importance are `is_foreign_transaction` (11.2%),
`is_new_country_for_account` (10.4%), `minutes_since_last_transaction` (8.8%),
`txn_count_trailing_1hr` (6.7%) and `is_new_category_for_account` (5.6%).
`is_night_hour` is 9th (3.4%). No feature is close to the 85% single-feature
red-flag threshold in `train_fraud_model.py`. Removing all hour features
lowers test AUC-PR only from 0.800 to 0.791 (scratch re-run, same data). In
the first version, the same ablation dropped it from 0.976 to 0.900.

This is still synthetic data with injected patterns. Real fraud adapts to
detection and is noisier, so a real model would typically score lower.

### Sample flagged high-risk transaction with real SHAP reasons

```
transaction_id: TXN00120743
fraud_probability: 0.9999
top_shap_reasons: minutes_since_last_transaction(+4.510); merchant_category_crypto_exchange(+1.754); is_foreign_transaction(+1.268)
```

All 579 test-period transactions with predicted fraud probability >= 0.5 are
in `outputs/flagged_transactions.csv` (299 real frauds + 280 false alarms,
matching the confusion matrix), each with its own top-3 SHAP reasons. SHAP
global summary plot: `outputs/shap_summary.png`.

An earlier version listed only 249 of 438 flagged transactions: SHAP ran on a
random 20,000-row sample, and the flagged list was restricted to that sample.
SHAP is now computed on every flagged row, and the sample is used only for
the global plot.

## Audit fixes

An audit of the first version found that its 0.976 AUC-PR came mostly from
leakage. Each issue, the fix, and its effect:

| # | Issue (severity) | Evidence | Fix |
|---|---|---|---|
| 1 | Hour of day nearly gave the label away (HIGH) | Legitimate transactions only happened inside each customer's 6am-11pm window, except 3% "noise" limited to hours 0-4. Non-odd-hour fraud used any hour, so hour 5 was 200/200 fraud, hours 2-4 were ~31% fraud, and 60% of fraud fell in hours 0-5. `is_odd_hour` copied the generator's 2-5am rule. | The generator now gives legitimate activity a realistic low rate at every hour, including hour 5 (2.5% of legit at 0-5am). Fraud time of day is a 6x night skew, not a rule: the maximum hourly fraud rate is 8.4%. `is_odd_hour` (2-5) is replaced by a generic `is_night_hour` (0-5). |
| 2 | Velocity bursts were broken (HIGH) | Burst offsets reached ~98 minutes, then each burst transaction's hour/minute was overwritten. Median burst span was 15.9 hours and 0% fell within 15 minutes, although the README promised 3-8 transactions in 0-15 minutes. | Bursts are built from 0.3-3.5 minute gaps capped at a 15-minute total, and their timestamps are no longer overwritten. There are 176 bursts, median span 8.4 minutes, max 14.5. |
| 3 | Lifetime features leaked labels and future data (MEDIUM) | `lifetime_avg_amount`, `lifetime_distinct_categories` and `lifetime_distinct_countries` were computed over all 8 months including fraud rows, then joined into the ML mart. | Removed from the mart and the model. They are replaced by past-only `prior_txn_count`, `prior_distinct_categories/countries` and `is_new_category/country_for_account`. `dim_account` keeps the lifetime stats for BI only. |
| 4 | Random split on time-series data (MEDIUM) | 183 of 375 test frauds had a same-account fraud within 2 hours that was in the training set. | Headline is now a time split: train Jan-Jun, test Jul-Aug, with SMOTE and class weights fitted on the training rows only. The random split is kept as a robustness check. |
| minor | Victim share | The README said ~22% of customers are victims, but only 401 (13.4%) got fraud because the budget ran out. | Every one of the 660 victims (22.0%) now gets at least one episode. |
| minor | Excel KPI label | The README said the KPI card shows AUC-ROC, but it shows AUC-PR / Precision / Recall. | README corrected (see Excel section). |

Also changed: fractional-minute burst timestamps are floored to whole seconds
(the loader rejected mixed timestamp formats), and the row-based dbt windows
now break ties on `transaction_id`.

**Before -> after (winner = class-weighted in both):**

| | Before: random split | Before: time split* | After: time split (headline) | After: random split |
|---|---|---|---|---|
| AUC-PR | 0.9759 | 0.9745 | **0.7979** | 0.7865 |
| AUC-ROC | 0.9995 | 0.9994 | 0.9777 | 0.9618 |
| Precision @0.5 | 0.8265 | 0.8219 | 0.5164 | 0.3762 |
| Recall @0.5 | 0.9653 | 0.9505 | 0.8081 | 0.8267 |
| Top feature | distinct_countries_trailing_24hr (57%) | same | is_foreign_transaction (11%) | - |

*Before/time split = the old data and old features, re-trained with the new
Jan-Jun / Jul-Aug split in the Airflow container for comparison. The old
narrative ("a well-engineered dataset, not a data leak") was wrong. The
single-feature-dominance check could not catch leakage spread across
`transaction_hour` and `is_odd_hour`, or a leak in the data generator itself.

## Excel dashboard (verified programmatically)

`outputs/Fraud_Detection_Dashboard.xlsx`, verified with openpyxl after the
real pipeline run:

- Sheets: `Summary`, `Dashboard`, `RawData`, `TransactionFeatures`, `FlaggedTransactions`
- `Dashboard` sheet: 6 KPI cards (real values: Total Transactions 149,995,
  Total Fraud Count 1,499, Fraud Rate 0.999%, Total Fraud $ Amount $236,890,
  Avg Fraud Transaction $158.03, and Model AUC-PR / Precision / Recall
  0.798 / 0.516 / 0.808) + 4 charts (BarChart "Fraud Rate by
  Merchant Category (%)", LineChart "Fraud Rate by Hour of Day (%)",
  PieChart "Fraud Amount Distribution", BarChart "Top Flagged High-Risk
  Transactions (Fraud Probability)")
- `RawData`: 30,000-row real sample of `fact_transactions` (149,995 total
  rows in Postgres; sampled for Excel file-size/performance, not because
  the full data doesn't exist)
- `TransactionFeatures`: 30,000-row real sample of `mart_transaction_features`
  with all engineered features
- `FlaggedTransactions`: all 579 real flagged high-risk transactions with
  fraud probability and SHAP reasons (= TP + FP)

### Setting up a LIVE Power Query connection to Postgres

The psqlODBC "PostgreSQL Unicode(x64)" driver is installed on this machine
(verified working in the sibling Retail E-commerce project's session, same
machine/driver). The steps below follow that proven, verified pattern,
adapted to this project's table/column names and port.

**Exact click-by-click steps** (Excel with Power Query, Windows):

1. Make sure the stack is running: `docker compose up -d postgres` (or the
   full stack) so Postgres is listening on `localhost:5435`.
2. Open `outputs/Fraud_Detection_Dashboard.xlsx` in Excel.
3. Go to the **Data** ribbon tab -> **Get Data** -> **From Database** ->
   **From PostgreSQL Database**.
   - If you don't see this option, instead use **Get Data** -> **From
     Other Sources** -> **ODBC**, then pick the DSN-less option and enter
     the connection string manually (see step 5 fallback).
4. In the "PostgreSQL database" dialog:
   - **Server**: `localhost:5435`
   - **Database**: `fraud_detection`
   - Leave "Data Connectivity mode" as **Import** (or **DirectQuery** for
     live pass-through queries instead of a cached snapshot).
   - Click **OK**.
5. If prompted for credentials, choose **Database** authentication:
   - **User name**: `fraud_user`
   - **Password**: `fraud_pass`
   - (These match `POSTGRES_USER`/`POSTGRES_PASSWORD` in `.env`.)
   - Click **Connect**.
   - *Fallback if the PostgreSQL-native dialog isn't available*: use **Get
     Data -> From Other Sources -> ODBC -> DSN-less**, and enter this
     connection string in the "Connection string" box:
     `Driver={PostgreSQL Unicode(x64)};Server=localhost;Port=5435;Database=fraud_detection;Uid=fraud_user;Pwd=fraud_pass;`
6. In the **Navigator** window that appears, expand the `warehouse` schema
   and check the tables you want (`mart_transaction_features`,
   `fact_transactions`, `dim_account`). Click **Transform Data** if you
   want to preview/rename columns first, or **Load** to load directly.
7. In the **Load To** dialog, choose **Table** and **New worksheet** (to
   keep the existing `RawData`/`TransactionFeatures` static sheets intact
   for comparison), or **Existing worksheet** to replace a static sheet
   with the live one.
8. Click **Load**. Power Query will run the live query and populate the
   sheet.
9. To refresh with the latest data after re-running the pipeline: **Data**
   ribbon -> **Refresh All** (or right-click the query in **Queries &
   Connections** -> **Refresh**).
10. Optional — auto-refresh on open: right-click the query in **Queries &
    Connections** -> **Properties** -> check **Refresh data when opening
    the file**.

This exact ODBC connection pattern (same driver, same machine) was already
verified end-to-end from the command line in the sibling Retail E-commerce
project's session, so these steps are known-working for this driver/host
combination — the specific database name/port/credentials above are unique
to this project and were not re-tested live in this session, but follow the
identical, already-proven mechanism.

## Tableau workbook

`tableau/Fraud_Detection.twbx` opens directly in Tableau Public (double-click).
It has one dashboard, "Fraud Detection Overview", in the same style as the
Retail and Sales portfolio dashboards: dark green background (`#0e1f16`),
sized automatically to the screen, borderless tiles, and no gridlines, zero
lines, axis lines or dividers. It uses the same green theme values as the
Sales dashboard.

**Eight KPI tiles** across the top (centred big numbers): Transactions
149,995 · Fraud Cases 1,499 · Fraud Rate 1.00% · Fraud Losses $237K · Avg Fraud
$158 · Model AUC-PR 0.798 · Precision 51.6% · Recall 80.8%. The model KPIs
come from the winning model on the Jul-Aug test months.

**Six charts** in a 3 x 2 grid, with data labels and a light-to-dark green
gradient where colour encodes a measure:

| Chart | Type | What it shows |
|---|---|---|
| Fraud Rate: Hour of Day x Weekday | Heatmap (Square, rate label per cell) | Night hours (0-5) run 2-11% fraud and daytime about 0.3-1.5%: a skew, not a rule |
| Top 10 Merchant Categories by Fraud Rate | Bar, sorted, gradient | crypto_exchange 23.1%, then jewelry 8.5%, gambling 7.5%, money_transfer 7.2% |
| Weekly Fraud Losses ($) | Area | Fraud $ per week, Jan-Aug 2025 |
| Fraud Losses by Transaction Country | Treemap (size + colour = fraud $) | USA 43.7% of losses, then China 11.2%, Canada 8.4%, Russia 8.3% |
| Flagged in Test (Jul-Aug): Fraud (dark) vs False Alarm (light) | Stacked bar by probability band | 244 of the 279 flagged at 0.9-1.0 are real fraud; below 0.8 most flags are false alarms |
| Precision vs Recall by Threshold (Test) | Scatter, one dot per threshold (lod detail) | The trade-off from t=0.1 (91% recall, 16% precision) to t=0.9 (66% / 87%) |

Legend: the stacked bar colours a 0/1 "confirmed fraud" flag with the green
gradient, with the outcome on detail, and states the key in its title. Fixed
colours for a categorical field, written as a colour map in the XML, were
ignored by Tableau in every placement tried (datasource, worksheet and pane
styles), so a categorical legend was not used.

Built by `src/dashboards/build_tableau_workbook.py`, the DAG's last task.
`prepare_dashboard_data()` aggregates `fact_transactions.csv` and the model
outputs (`model_metrics.json`, `flagged_transactions.csv`,
`threshold_metrics.csv`) into small `data/processed/dash_*.csv` tables. Each
table is written to a `.hyper` extract with the Tableau Hyper API and packaged
with the workbook XML into a `.twbx`. Tableau Public only opens extract-based
workbooks, which is why this is a `.twbx`. The builder also asserts that the
flagged rows equal TP + FP from the confusion matrix. The old builder read
three aggregate CSVs (`fraud_daily_trend.csv` etc.) that no pipeline step
regenerated. They were stale and have been deleted.

Styling techniques that work, found while building this:

- **Green gradient**: in the worksheet `<style>`, a `<style-rule element='mark'>`
  containing `<encoding attr='color' field='[federated.x].[sum:col:qk]'
  type='custom-interpolated'>` with a `<color-palette custom='true'
  type='ordered-sequential'>` of two colours (`#c7ebd2` -> `#14532d`).
- **Readable header text** when mark labels keep Tableau's automatic contrast
  colour: colour the `axis`, `header` *and* `label` elements instead of the
  whole `worksheet`. Without `label`, the heatmap's row and column headers
  stayed dark grey.

**How it was verified.** The DAG-produced workbook was opened in Tableau
Public 2026.2 by a script that waits for Tableau's log to show
`end-workspace.load-workbook` (loaded) or an error dialog, then screenshots
the window. Result: loaded with no error dialog. The screenshot showed all
8 KPI tiles and 6 charts rendering in green with real data (see "Verified run
evidence").

## What was built in this session

This project was built from a completely empty folder in this session: the
synthetic data generator, all dbt models (staging/intermediate/marts with
velocity/behavioral feature engineering), the fraud model (class-weighted
vs SMOTE comparison + SHAP), the Excel dashboard builder, the Tableau .twb
builder, the Airflow DAG, and all Docker/dbt/GE infrastructure were written
from scratch and then actually run end-to-end (not just written and
assumed to work).

**Two real bugs were found and fixed by actually running the pipeline**:
1. dbt's default schema-naming behavior concatenated `staging_staging`,
   `staging_intermediate`, `staging_warehouse` instead of the intended
   `staging`/`intermediate`/`warehouse` schema names (because a custom
   `+schema` config is appended to the profile's target schema by default).
   Fixed with a `generate_schema_name.sql` macro override (same pattern
   used successfully in the sibling Retail project).
2. Postgres does not support `DISTINCT` inside window functions at all
   (neither `COUNT(DISTINCT...)  OVER` nor `array_agg(DISTINCT...) OVER`) —
   discovered when `dbt run` failed on `int_account_velocity.sql`. Fixed by
   using a plain (non-distinct) `array_agg(...) OVER (...)` followed by
   `cardinality(array(select distinct unnest(...)))` to de-duplicate.

No Great Expectations check needed loosening in the original run — the 12 hard-gate
checks passed cleanly on the real generated data on the first try, unlike
the sibling Retail project's real-world dataset which had a genuine edge
case. This was still checked, not assumed.

## Verified run evidence (this session, 2026-09-25)

- `docker ps -a` before: no fraud-project containers existed (clean start
  in a fresh empty folder); ports 5435 (Postgres) and 8084 (Airflow) chosen
  to avoid collision with the Retail (5434/8083) and Sales (5433/8082)
  sibling projects, confirmed via `docker ps -a` beforehand.
- `python data/generator/generate_fraud_data.py` generated 149,995 real
  rows to `data/raw/bank_transactions.csv`, confirmed via `wc -l` (149,996
  lines incl. header) and pandas dtype/null checks (zero nulls, amount
  range $1.65-$3,800).
- Postgres brought up healthy in ~5s; schemas (`raw`, `staging`,
  `intermediate`, `warehouse`) present from `postgres_init/01_schemas.sql`.
- `airflow-init` built successfully (including `shap`, `xgboost`, `dbt-core`
  compiling/installing cleanly with VC++ Build Tools) and ran `airflow db
  migrate` + created the admin user.
- `airflow dags list-import-errors` returned no errors both before and
  after the dbt bug fixes.
- Ingestion, GE validation, dbt run/test, mart feature spot-check, and
  model training were all run manually first (inside the airflow-webserver
  container) to find and fix the two real bugs above before triggering the
  full DAG.
- `airflow dags trigger fraud_pipeline_dag` -> run
  `manual__2026-09-25T08:19:14+00:00`.
- All 9 tasks reached `success` (`wait_for_raw_csv`,
  `ingest_raw_transactions`, `great_expectations_validate`, `dbt_run`,
  `dbt_test`, `export_marts_to_csv`, `train_fraud_model`,
  `build_excel_dashboard`, `build_tableau_workbook`), confirmed via
  `airflow tasks states-for-dag-run` — full DAG completed in under 90
  seconds end-to-end.
- `SELECT count(*) FROM raw.transactions` = 149,995 (matches CSV row count
  exactly).
- Mart row counts (`dim_account`=3,000, `fact_transactions`=149,995,
  `mart_transaction_features`=149,995) and the feature-vs-label sanity
  query above pulled directly via `psql` against the real warehouse.
- `outputs/model_metrics.json`, `model_report.txt`, `shap_summary.png`,
  `flagged_transactions.csv` all present with real computed values,
  including the class-weighted vs SMOTE comparison and the leakage
  investigation documented above.
- Excel workbook verified programmatically with openpyxl: 5 sheets, 4
  charts with correct real titles, RawData/TransactionFeatures at 30,000
  sampled real rows each (149,995 total rows exist in Postgres),
  FlaggedTransactions with all 438 real flagged rows, sample row values
  spot-checked.
- After the Tableau builder was replaced, the full DAG was re-triggered: all 9
  tasks `success`, and the DAG-produced `tableau/Fraud_Detection.twbx` loaded in
  Tableau Public with no error dialog (log check + screenshot, see "Tableau
  workbook").
- **Audit-fix run (2026-09-25, after regenerating the data):**
  `airflow dags trigger fraud_pipeline_dag` -> runs
  `manual__2026-09-25T17:13:39+00:00` (Part A fixes) and
  `manual__2026-09-25T17:30:03+00:00` (Part A + new dashboard). All 9 tasks
  `success` in both runs. GE 12/12 passed, dbt test PASS=27 WARN=0 ERROR=0.
  `model_metrics.json` and `flagged_transactions.csv` were byte-identical
  between the first run, a manual re-run of `train_fraud_model.py` and the
  second DAG run.
- The DAG-produced `tableau/Fraud_Detection.twbx` (second run) loaded in Tableau
  Public with no error dialog. The DAG-produced Excel workbook opened in Excel
  with 0 formula errors and 4 charts, KPIs as listed in the Excel section, and
  579 FlaggedTransactions rows.
- Dashboard CSV exports on the host filesystem after the audit-fix run:
  `fact_transactions.csv` (149,995 rows), `transaction_features.csv` (149,995
  rows; lifetime_* columns replaced by past-only features, `is_odd_hour` by
  `is_night_hour`), `dim_account.csv` (3,000 rows) and the dashboard tables
  `dash_kpis.csv`, `dash_fraud_heatmap.csv`, `dash_fraud_by_category.csv`,
  `dash_fraud_weekly.csv`, `dash_fraud_by_country.csv`,
  `dash_flagged_by_band.csv`, `dash_threshold_curve.csv`.
- *(Original run, first version:)* Dashboard CSV exports confirmed present on the host filesystem (not just
  inside the container) at the end of this run, with real headers and row
  counts:
  - `data/processed/fraud_daily_trend.csv`
    (243 data rows + header) — columns: `txn_date,fraud_rate_pct,fraud,total`
  - `data/processed/fraud_by_category.csv`
    (16 data rows + header) — columns: `merchant_category,total,fraud,fraud_rate_pct`
  - `data/processed/fraud_amount_distribution.csv`
    (6 data rows + header) — columns: `amount_range,fraud_count`
  - `data/processed/fact_transactions.csv`
    (149,995 data rows + header) — columns:
    `transaction_id,account_id,customer_id,transaction_datetime,transaction_date,transaction_hour,transaction_dow,amount,merchant_category,transaction_country,is_foreign_transaction,channel,is_fraud`
  - `data/processed/transaction_features.csv`
    (149,995 data rows + header) — columns:
    `transaction_id,account_id,customer_id,transaction_datetime,transaction_date,transaction_hour,transaction_dow,amount,merchant_category,transaction_country,is_foreign_transaction,channel,minutes_since_last_transaction,txn_count_trailing_1hr,txn_count_trailing_24hr,distinct_categories_trailing_24hr,distinct_countries_trailing_24hr,historical_avg_amount,amount_deviation_ratio,amount_to_avg_ratio,is_odd_hour,lifetime_avg_amount,lifetime_distinct_categories,lifetime_distinct_countries,home_country,is_fraud`
  - `data/processed/dim_account.csv`
    (3,000 data rows + header) — columns:
    `account_id,customer_id,home_country,lifetime_txn_count,lifetime_fraud_count,lifetime_fraud_rate,lifetime_avg_amount,lifetime_stddev_amount,first_transaction_datetime,last_transaction_datetime,lifetime_distinct_categories,lifetime_distinct_countries`
