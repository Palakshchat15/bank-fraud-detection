"""
Synthetic bank transaction fraud data generator.

Design goals (explicit, documented assumptions - also repeated in
README_PIPELINE.md):

- 3,000 customers/accounts (1:1 account:customer: each customer has exactly
  one debit account in this dataset).
- 8 months of transaction history (2025-01-01 -> 2025-08-31).
- ~150,000 total transactions, ~1.0% fraud (realistic card-fraud range).

LEGITIMATE behaviour (per customer, so "normal" varies by customer):
- Personal average amount, 3-6 preferred merchant categories, activity level.
- Time of day: a personal active window (start 6-9am, end 9-11pm). Most
  activity falls inside it, but there is a realistic low rate of activity at
  every other hour, including 0-5am (about 2-3% of legitimate transactions
  happen between midnight and 6am). No hour is fraud-only.
- Shopping sessions: ~12% of transactions follow the previous one within
  5-90 minutes (errands, a meal after shopping), so short gaps and several
  transactions per hour also occur legitimately.
- Travel: ~35% of customers take 1-2 trips abroad (3-10 days); purchases on
  those days are mostly in the trip country. Another ~1.5% of transactions are
  cross-border online purchases.
- Occasional large one-off purchases (~2% of transactions, 3-8x the
  customer's average) and ~15% of purchases outside the preferred categories.

FRAUD (a randomly selected 22% of customers are victims; every victim gets at
least one "compromise episode" and the fraud budget is spread so the share is
actually reached). Each episode has one attacker profile:
  1. VELOCITY BURST (~27% of episodes, ~60% of fraud transactions): 3-8
     transactions within 15 minutes of each other, sometimes starting with a
     small "card test" amount.
  2. GEO anomaly (~45% of episodes): the attacker transacts from another
     country, weighted towards a few high-risk countries.
  3. MERCHANT anomaly: ~55% of fraud transactions use cash-out categories
     (crypto, money transfer, electronics, jewelry, gambling, online retail).
  4. AMOUNT anomaly (~40% of episodes): 3-12x the customer's average,
     sometimes a round number.
  5. TIME-OF-DAY skew: fraud times follow the same daily curve as the
     population, but hours 0-5 are 6x over-weighted. Fraud is therefore
     several times more common at night, yet most fraud still happens during
     the day and most night transactions are legitimate.

The signal strength was tuned on a validation split inside the training
period only (train Jan-Apr, validate May-Jun); the Jul-Aug test months were
not used to choose these settings. See README_PIPELINE.md "Audit fixes".

Output: data/raw/bank_transactions.csv
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_CUSTOMERS = 3000
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 8, 31, 23, 59, 59)
TOTAL_DAYS = (END_DATE - START_DATE).days
TARGET_TOTAL_TXNS = 150_000
TARGET_FRAUD_RATE = 0.01  # ~1.0%

# --- behaviour knobs (documented in README_PIPELINE.md) ---
NIGHT_HOURS = range(0, 6)          # midnight to 6am
LEGIT_NIGHT_WEIGHT = 0.06          # relative hourly weight at night vs 1.0 inside the personal window
LEGIT_SHOULDER_WEIGHT = 0.15       # other hours outside the personal window
SESSION_FOLLOWUP_PROB = 0.12       # legit txn follows the previous one within 5-90 minutes
TRIP_PROBS = (0.65, 0.25, 0.10)    # customers with 0 / 1 / 2 trips abroad
TRIP_COUNTRY_PROB = 0.9            # share of purchases made in the trip country during a trip
CROSS_BORDER_ONLINE_PROB = 0.015
LARGE_PURCHASE_PROB = 0.02
VICTIM_SHARE = 0.22
BURST_EPISODE_PROB = 0.27
BURST_MAX_SPAN_MINUTES = 15
FRAUD_NIGHT_MULTIPLIER = 6.0
FRAUD_GEO_PROB = 0.45
FRAUD_CATEGORY_PROB = 0.55
FRAUD_SPIKE_PROB = 0.40
CARD_TEST_PROB = 0.4

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail", "electronics",
    "travel", "entertainment", "utilities", "pharmacy", "clothing",
    "home_improvement", "jewelry", "atm_withdrawal", "money_transfer",
    "crypto_exchange", "gambling",
]

# Categories that are much rarer overall and disproportionately used in
# fraud (crypto_exchange, gambling, jewelry, money_transfer, electronics) -
# real-world cash-out categories for fraudsters.
_cat_raw_weights = {
    "grocery": 0.20, "restaurant": 0.16, "gas_station": 0.12,
    "online_retail": 0.14, "electronics": 0.05, "travel": 0.04,
    "entertainment": 0.06, "utilities": 0.05, "pharmacy": 0.05,
    "clothing": 0.06, "home_improvement": 0.03, "jewelry": 0.008,
    "atm_withdrawal": 0.03, "money_transfer": 0.01,
    "crypto_exchange": 0.003, "gambling": 0.009,
}
_cat_weight_sum = sum(_cat_raw_weights.values())
CATEGORY_BASE_WEIGHTS = {k: v / _cat_weight_sum for k, v in _cat_raw_weights.items()}
CATS = list(CATEGORY_BASE_WEIGHTS.keys())
CAT_P = np.array(list(CATEGORY_BASE_WEIGHTS.values()))
FRAUD_PREFERRED_CATEGORIES = [
    "crypto_exchange", "money_transfer", "electronics", "jewelry",
    "gambling", "online_retail",
]
LARGE_PURCHASE_CATEGORIES = ["electronics", "travel", "home_improvement", "jewelry", "clothing"]

MERCHANT_NAMES = {
    "grocery": ["FreshMart", "Kroger", "Whole Foods", "Safeway", "Trader Joe's"],
    "restaurant": ["Olive Garden", "Chipotle", "Local Diner", "Sushi House", "Pizza Palace"],
    "gas_station": ["Shell", "Chevron", "BP", "Exxon", "Circle K"],
    "online_retail": ["Amazon", "eBay", "Walmart.com", "Target.com", "Best Buy Online"],
    "electronics": ["Best Buy", "Apple Store", "GameStop", "Micro Center", "B&H Photo"],
    "travel": ["Delta Airlines", "Marriott", "Expedia", "Airbnb", "Hertz"],
    "entertainment": ["AMC Theatres", "Netflix", "Spotify", "Live Nation", "Steam"],
    "utilities": ["City Power & Light", "Comcast", "AT&T", "Water Utility Co", "Waste Mgmt"],
    "pharmacy": ["CVS", "Walgreens", "Rite Aid", "Duane Reade", "Costco Pharmacy"],
    "clothing": ["Nike", "H&M", "Zara", "Nordstrom", "Gap"],
    "home_improvement": ["Home Depot", "Lowe's", "Ace Hardware", "Menards", "IKEA"],
    "jewelry": ["Tiffany & Co", "Kay Jewelers", "Zales", "Local Pawn Shop", "Blue Nile"],
    "atm_withdrawal": ["Chase ATM", "BofA ATM", "Wells Fargo ATM", "Citibank ATM", "Generic ATM"],
    "money_transfer": ["Western Union", "MoneyGram", "Zelle", "Wise", "Remitly"],
    "crypto_exchange": ["Coinbase", "Binance", "Kraken", "Crypto.com", "Gemini"],
    "gambling": ["DraftKings", "FanDuel", "BetMGM", "Caesars Sportsbook", "Local Casino"],
}

COUNTRIES = ["USA", "Canada", "UK", "Mexico", "Germany", "France", "Nigeria",
             "Russia", "China", "Brazil", "India", "UAE"]
HOME_COUNTRY_WEIGHTS = np.array([0.775, 0.08, 0.04, 0.03, 0.02, 0.015, 0.005, 0.005,
                                 0.01, 0.01, 0.01, 0.005])
HOME_COUNTRY_WEIGHTS = HOME_COUNTRY_WEIGHTS / HOME_COUNTRY_WEIGHTS.sum()
# Holiday destinations for legitimate trips (popular, not uniform)
TRIP_COUNTRY_WEIGHTS = dict(zip(COUNTRIES, [0.20, 0.15, 0.12, 0.18, 0.08, 0.10, 0.01,
                                            0.01, 0.03, 0.04, 0.04, 0.04]))
# High-risk countries disproportionately used for cross-border fraud
HIGH_RISK_FRAUD_COUNTRIES = ["Nigeria", "Russia", "China", "UAE"]

US_CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
             "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
             "Seattle", "Denver", "Boston", "Atlanta", "Miami"]
FOREIGN_CITIES = {
    "Canada": ["Toronto", "Vancouver", "Montreal"],
    "UK": ["London", "Manchester", "Birmingham"],
    "Mexico": ["Mexico City", "Cancun", "Guadalajara"],
    "Germany": ["Berlin", "Munich", "Frankfurt"],
    "France": ["Paris", "Lyon", "Marseille"],
    "Nigeria": ["Lagos", "Abuja", "Port Harcourt"],
    "Russia": ["Moscow", "St. Petersburg", "Novosibirsk"],
    "China": ["Shanghai", "Beijing", "Shenzhen"],
    "Brazil": ["Sao Paulo", "Rio de Janeiro", "Brasilia"],
    "India": ["Mumbai", "Delhi", "Bangalore"],
    "UAE": ["Dubai", "Abu Dhabi", "Sharjah"],
}

CHANNELS = ["online", "in-store", "atm", "mobile"]
CHANNEL_WEIGHTS = [0.34, 0.38, 0.08, 0.20]
FRAUD_CHANNEL_WEIGHTS = [0.50, 0.10, 0.15, 0.25]  # fraud skews online/mobile/atm


def city_for(country):
    return rng.choice(US_CITIES) if country == "USA" else rng.choice(FOREIGN_CITIES[country])


def other_country(home, weights=None):
    options = [c for c in COUNTRIES if c != home]
    if weights is None:
        return rng.choice(options)
    p = np.array([weights[c] for c in options])
    return rng.choice(options, p=p / p.sum())


print("Generating customer population...")

customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(1, N_CUSTOMERS + 1)]
account_ids = [f"ACCT{str(i).zfill(6)}" for i in range(1, N_CUSTOMERS + 1)]

home_countries = rng.choice(COUNTRIES, size=N_CUSTOMERS, p=HOME_COUNTRY_WEIGHTS)
home_cities = [city_for(hc) for hc in home_countries]

avg_txn_amounts = np.round(rng.lognormal(mean=3.6, sigma=0.65, size=N_CUSTOMERS), 2)  # ~ $30-150 typical
avg_txn_amounts = np.clip(avg_txn_amounts, 8, 600)

# Personal "normal activity window" (start hour, end hour)
activity_start_hours = rng.integers(6, 10, size=N_CUSTOMERS)   # wake up 6-9am
activity_end_hours = rng.integers(21, 24, size=N_CUSTOMERS)    # wind down 9-11pm

customer_pref_categories = []
for _ in range(N_CUSTOMERS):
    k = rng.integers(3, 7)
    customer_pref_categories.append(list(rng.choice(CATS, size=k, replace=False, p=CAT_P)))

base_txn_counts = rng.lognormal(mean=3.6, sigma=0.55, size=N_CUSTOMERS)
base_txn_counts = np.clip(base_txn_counts, 5, 300)
base_txn_counts = base_txn_counts / base_txn_counts.sum() * (TARGET_TOTAL_TXNS * 0.99)
base_txn_counts = np.clip(np.round(base_txn_counts).astype(int), 3, None)

customers_df = pd.DataFrame({
    "customer_id": customer_ids,
    "account_id": account_ids,
    "home_country": home_countries,
    "home_city": home_cities,
    "avg_txn_amount": avg_txn_amounts,
    "activity_start_hour": activity_start_hours,
    "activity_end_hour": activity_end_hours,
    "pref_categories": customer_pref_categories,
    "n_txns": base_txn_counts,
})
print(f"Total planned legitimate transactions: {customers_df['n_txns'].sum()}")


def legit_hour_weights(start_h, end_h):
    w = np.full(24, LEGIT_SHOULDER_WEIGHT)
    w[list(NIGHT_HOURS)] = LEGIT_NIGHT_WEIGHT
    w[start_h:end_h + 1] = 1.0
    return w / w.sum()


# Population daily curve (average of personal curves); fraud uses it with the
# night hours over-weighted.
POPULATION_HOUR_P = np.mean(
    [legit_hour_weights(s, e) for s, e in zip(activity_start_hours, activity_end_hours)], axis=0)
_fraud_w = POPULATION_HOUR_P.copy()
_fraud_w[list(NIGHT_HOURS)] *= FRAUD_NIGHT_MULTIPLIER
FRAUD_HOUR_P = _fraud_w / _fraud_w.sum()


def random_time(day, hour_p):
    hour = int(rng.choice(24, p=hour_p))
    return day.replace(hour=hour, minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60)))


def random_day():
    return START_DATE + timedelta(days=int(rng.integers(0, TOTAL_DAYS + 1)))


def make_amount(cust_row, spike=False):
    if spike:
        amt = cust_row.avg_txn_amount * rng.uniform(3, 12)
        if rng.random() < 0.25:
            amt = round(amt / 100) * 100  # suspiciously round
        return round(float(np.clip(amt, 50, 15000)), 2)
    amt = rng.lognormal(mean=np.log(max(cust_row.avg_txn_amount, 5)), sigma=0.5)
    return round(float(np.clip(amt, 1, cust_row.avg_txn_amount * 4)), 2)


def txn_record(cust_row, dt, amount, category, country, channel, is_fraud):
    city = cust_row.home_city if country == cust_row.home_country else city_for(country)
    return {
        "account_id": cust_row.account_id,
        "customer_id": cust_row.customer_id,
        "transaction_datetime": dt,
        "amount": amount,
        "merchant_category": category,
        "merchant_name": rng.choice(MERCHANT_NAMES[category]),
        "transaction_city": city,
        "transaction_country": country,
        "customer_home_country": cust_row.home_country,
        "channel": channel,
        "is_fraud": is_fraud,
    }


def customer_trips(cust_row):
    """List of (first_day, last_day, country) trips abroad for one customer."""
    n_trips = int(rng.choice(3, p=TRIP_PROBS))
    trips = []
    for _ in range(n_trips):
        first = random_day()
        trips.append((first, first + timedelta(days=int(rng.integers(3, 11))),
                      other_country(cust_row.home_country, TRIP_COUNTRY_WEIGHTS)))
    return trips


def legit_transactions(cust_row):
    hour_p = legit_hour_weights(cust_row.activity_start_hour, cust_row.activity_end_hour)
    trips = customer_trips(cust_row)
    out, prev_dt = [], None
    for _ in range(int(cust_row.n_txns)):
        if prev_dt is not None and rng.random() < SESSION_FOLLOWUP_PROB:
            dt = prev_dt + timedelta(minutes=float(rng.uniform(5, 90)))
            if dt > END_DATE:
                dt = random_time(random_day(), hour_p)
        else:
            dt = random_time(random_day(), hour_p)
        prev_dt = dt

        if rng.random() < 0.85:
            category = rng.choice(cust_row.pref_categories)
        else:
            category = rng.choice(CATS, p=CAT_P)
        channel = rng.choice(CHANNELS, p=CHANNEL_WEIGHTS)

        country = cust_row.home_country
        day = dt.replace(hour=0, minute=0, second=0)
        for first, last, trip_country in trips:
            if first <= day <= last and rng.random() < TRIP_COUNTRY_PROB:
                country = trip_country
                if channel == "online":
                    channel = "in-store"
        if country == cust_row.home_country and rng.random() < CROSS_BORDER_ONLINE_PROB:
            country, channel = other_country(cust_row.home_country), "online"

        if rng.random() < LARGE_PURCHASE_PROB:
            amount = round(cust_row.avg_txn_amount * rng.uniform(3, 8), 2)
            if rng.random() < 0.5:
                category = rng.choice(LARGE_PURCHASE_CATEGORIES)
        else:
            amount = make_amount(cust_row)
        out.append(txn_record(cust_row, dt, amount, category, country, channel, 0))
    return out


def fraud_episode(cust_row, n_txns):
    """One compromise episode: n_txns fraud transactions sharing an attacker profile.
    n_txns > 1 is a velocity burst: every transaction within BURST_MAX_SPAN_MINUTES."""
    start = random_time(random_day(), FRAUD_HOUR_P)
    foreign = rng.random() < FRAUD_GEO_PROB
    if foreign:
        country = (rng.choice(HIGH_RISK_FRAUD_COUNTRIES) if rng.random() < 0.5
                   else other_country(cust_row.home_country))
        if country == cust_row.home_country:
            country = other_country(cust_row.home_country)
    else:
        country = cust_row.home_country
    spike = rng.random() < FRAUD_SPIKE_PROB

    times = [start]
    if n_txns > 1:
        gaps = rng.uniform(0.3, 3.5, size=n_txns - 1)  # minutes between transactions
        if gaps.sum() > BURST_MAX_SPAN_MINUTES - 0.5:
            gaps *= (BURST_MAX_SPAN_MINUTES - 0.5) / gaps.sum()
        times += [start + timedelta(minutes=float(m)) for m in np.cumsum(gaps)]
    card_test = n_txns > 1 and rng.random() < CARD_TEST_PROB

    out = []
    for i, dt in enumerate(times):
        if dt > END_DATE:
            dt = END_DATE - timedelta(minutes=(len(times) - i))
        if rng.random() < FRAUD_CATEGORY_PROB:
            category = rng.choice(FRAUD_PREFERRED_CATEGORIES)
        else:
            category = rng.choice(CATS, p=CAT_P)
        if card_test and i == 0:
            amount = round(float(rng.uniform(1, 10)), 2)
        else:
            amount = make_amount(cust_row, spike=spike)
        channel = rng.choice(CHANNELS, p=FRAUD_CHANNEL_WEIGHTS)
        out.append(txn_record(cust_row, dt, amount, category, country, channel, 1))
    return out


print("Generating legitimate transactions...")
records = []
for cust_row in customers_df.itertuples(index=False):
    records.extend(legit_transactions(cust_row))
print(f"Legit transactions generated: {len(records)}")

# --- Inject fraud ---
current_legit = len(records)
n_fraud_needed = int(current_legit / (1 - TARGET_FRAUD_RATE)) - current_legit
print(f"Fraud transactions to inject: {n_fraud_needed}")

n_victims = int(N_CUSTOMERS * VICTIM_SHARE)
victim_customers = customers_df.sample(n=n_victims, random_state=RNG_SEED).reset_index(drop=True)

# Every victim gets one episode; a burst only happens if enough budget is left to
# still give each remaining victim at least one transaction. Leftover budget
# becomes repeat single-transaction hits on random victims.
fraud_records = []
remaining = n_fraud_needed
for i, cust_row in enumerate(victim_customers.itertuples(index=False)):
    victims_left_after = n_victims - i - 1
    spare = remaining - victims_left_after - 1
    if spare >= 2 and rng.random() < BURST_EPISODE_PROB:
        size = min(int(rng.integers(3, 9)), spare + 1)
    else:
        size = 1
    fraud_records.extend(fraud_episode(cust_row, size))
    remaining -= size
while remaining > 0:
    cust_row = victim_customers.iloc[int(rng.integers(0, n_victims))]
    fraud_records.extend(fraud_episode(cust_row, 1))
    remaining -= 1
print(f"Fraud transactions generated: {len(fraud_records)}")

df = pd.DataFrame(records + fraud_records)
# whole seconds only (burst offsets are fractional minutes)
df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"]).dt.floor("s")
df = df.sort_values(["transaction_datetime", "account_id"], kind="mergesort").reset_index(drop=True)
df.insert(0, "transaction_id", [f"TXN{i + 1:08d}" for i in range(len(df))])

# --- Summary / sanity checks ---
hours = df["transaction_datetime"].dt.hour
fraud = df[df.is_fraud == 1]
print("\n--- Summary ---")
print(f"Total transactions: {len(df)}")
print(f"Fraud transactions: {len(fraud)} ({df['is_fraud'].mean() * 100:.3f}%)")
print(f"Unique accounts: {df['account_id'].nunique()}")
print(f"Victim accounts: {fraud['account_id'].nunique()} "
      f"({fraud['account_id'].nunique() / N_CUSTOMERS:.1%} of customers)")
print(f"Date range: {df['transaction_datetime'].min()} to {df['transaction_datetime'].max()}")
print("\nAmount stats - fraud vs legit:")
print(df.groupby("is_fraud")["amount"].describe())
night = hours.isin(list(NIGHT_HOURS))
print(f"\nShare of legit txns at 0-5am: {night[df.is_fraud == 0].mean():.2%}; "
      f"share of fraud at 0-5am: {night[df.is_fraud == 1].mean():.2%}")
print(f"Fraud rate at 0-5am: {df.loc[night, 'is_fraud'].mean():.2%} vs "
      f"6am-11pm: {df.loc[~night, 'is_fraud'].mean():.2%}")
print("Fraud rate by hour (%):", (df.groupby(hours)["is_fraud"].mean() * 100).round(2).to_dict())
f_sorted = fraud.sort_values(["account_id", "transaction_datetime"])
gap = f_sorted.groupby("account_id")["transaction_datetime"].diff().dt.total_seconds() / 60
in_burst = (gap <= BURST_MAX_SPAN_MINUTES) | (gap.shift(-1) <= BURST_MAX_SPAN_MINUTES)
print(f"Fraud txns within {BURST_MAX_SPAN_MINUTES} min of another fraud txn on the same account: "
      f"{in_burst.mean():.1%}")
print(f"\nMerchant category distribution, fraud vs legit (top 5 each):")
print("Fraud:", fraud["merchant_category"].value_counts().head(5).to_dict())
print("Legit:", df[df.is_fraud == 0]["merchant_category"].value_counts().head(5).to_dict())
print(f"Foreign share: legit {(df.transaction_country != df.customer_home_country)[df.is_fraud == 0].mean():.2%}, "
      f"fraud {(fraud.transaction_country != fraud.customer_home_country).mean():.2%}")

out_dir = os.path.join(os.path.dirname(__file__), "..", "raw")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "bank_transactions.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved to {os.path.abspath(out_path)}")
