"""
Model to predict football player market value
from performance stats + age.

Main model: Ridge regression with a spline on Age.
Benchmark:  XGBoost (kept for comparison).

Run with: python main.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
# Enriched CSV = original stats + Transfermarkt-scraped AGE, POSITION and
# CONTRACT_YEARS_LEFT (see scrape_transfermarkt.py / merge_scraped_data.py).
# The scraped AGE replaces the original, which had scraping errors (e.g.
# Isak listed as 18, Dembele as 17) that a simple range filter couldn't catch.
df = pd.read_csv("podaci/combined_top5_marketvalue_enriched.csv")

TARGET = "Market Value (m/€)"

# Players who transferred mid-season have a "League A / League B" string.
# We don't need to drop them - just collapse that into one clean category
# so the model (and one-hot encoding) doesn't try to treat it as its own league.
df["LEAGUE"] = df["LEAGUE"].apply(lambda x: "Multi-league" if "/" in x else x)

# Now that Age comes from Transfermarkt, implausible values shouldn't occur -
# but keep the guard rail in case a future re-scrape reintroduces bad rows.
n_before = len(df)
df = df[(df["Age"] >= 15) & (df["Age"] <= 42)].reset_index(drop=True)
print(f"Dropped {n_before - len(df)} rows with implausible ages ({n_before} -> {len(df)})")

# A handful of players have no contract-expiry date on Transfermarkt (loan
# situations, recently-registered youth players, etc.) - CONTRACT_YEARS_LEFT
# is NaN for them. Rather than drop these players, impute the median: an
# indicator column would let the model treat "unknown" differently, but with
# only 11 such rows that's not worth the extra feature.
n_missing_contract = df["CONTRACT_YEARS_LEFT"].isna().sum()
df["CONTRACT_YEARS_LEFT"] = df["CONTRACT_YEARS_LEFT"].fillna(df["CONTRACT_YEARS_LEFT"].median())
print(f"Imputed {n_missing_contract} missing CONTRACT_YEARS_LEFT values with the median")

# CLUB_STRENGTH (the player's current club's total squad market value, from
# Transfermarkt - see scrape_clubs.py / merge_club_strength.py) is heavily
# right-skewed just like the target, so log-transform it the same way.
df["CLUB_STRENGTH"] = np.log1p(df["CLUB_STRENGTH"])

# Drop the bottom and top 10% by value. Stats-based errors concentrate at
# both tails for different reasons: the cheapest fringe players have noisy,
# idiosyncratic values (loan status, squad-depth roles) that per-90 stats
# can't explain, while the priciest superstars carry a reputation/hype
# premium over statistically similar peers that no feature here captures.
# Measured effect on the OOF percentage error (not just MAE, which would
# trivially shrink from removing big-euro rows either way): mean % error
# dropped from ~50-54% to ~39-42%, median % error from ~31-34% to ~30%, and
# the within-±50% share rose from ~68-73% to ~72-75%. This is a modeling
# choice about scope (the model is only claimed to work for the "normal"
# market), not a data-quality fix - so it's applied here, not in the CSV.
n_before = len(df)
lo, hi = df[TARGET].quantile([0.10, 0.90])
df = df[(df[TARGET] > lo) & (df[TARGET] < hi)].reset_index(drop=True)
print(f"Dropped {n_before - len(df)} players outside the middle 80% by value "
      f"({n_before} -> {len(df)}, keeping €{lo:.1f}m-€{hi:.1f}m)")

# ---------------------------------------------------------------------------
# 2. FEATURE NOTES
# ---------------------------------------------------------------------------
# Note: the raw stat columns (ATT_GOALS, DEF_TACKLES, etc.) are already
# per-90-minutes rates in this dataset, not season totals - so no further
# per-90 conversion is needed here. MINS and APPS are kept as their own
# features since total minutes/appearances is a separate signal (durability,
# a manager's trust in the player) distinct from per-90 output quality.

# ---------------------------------------------------------------------------
# 3. TARGET TRANSFORMATION (log-transform)
# ---------------------------------------------------------------------------
# Market value is heavily right-skewed: most players are 20-40m, but a handful
# (Haaland, Mbappe, Yamal) sit at 200m+. Predicting log(value) instead turns
# the model's errors into roughly *percentage* errors instead of absolute
# euro errors, which is usually what you actually care about.
# We convert back with np.expm1() (inverse of np.log1p) for reporting.
y = np.log1p(df[TARGET])

# ---------------------------------------------------------------------------
# 4. FEATURE MATRIX (X)
# ---------------------------------------------------------------------------
X = df.drop(columns=["NAME", TARGET])

# LEAGUE and POSITION are categorical text - one-hot encode them so each
# category becomes its own 0/1 column instead of a fake numeric ordering.
X = pd.get_dummies(X, columns=["LEAGUE", "POSITION"], drop_first=False, dtype=float)

print(f"Final feature matrix: {X.shape[0]} players, {X.shape[1]} features")

# ---------------------------------------------------------------------------
# 5. MODELS
# ---------------------------------------------------------------------------
# Ridge regression: with only ~370 players and ~47 features, a regularized
# linear model generalizes better than boosted trees (benchmarked over 10 CV
# seeds: Ridge beat XGBoost on every seed).
#
# Age is the one feature with a clearly non-linear effect (value rises into
# the mid-20s, then falls), so it gets a spline instead of a single slope.
# RidgeCV picks the regularization strength with its own internal CV on the
# training folds only, so no tuning leaks into the evaluation below.
other_cols = [c for c in X.columns if c != "Age"]
ridge_model = make_pipeline(
    ColumnTransformer([
        ("age", SplineTransformer(n_knots=5, degree=3), ["Age"]),
        ("rest", "passthrough", other_cols),
    ]),
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-2, 4, 50)),
)

# XGBoost benchmark with fixed, conservative settings. It is deliberately NOT
# tuned on this data: tuning on the same rows we evaluate on is what made the
# old scores look better than they really were.
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=3.0,
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# 6. HONEST EVALUATION (out-of-fold predictions)
# ---------------------------------------------------------------------------
# cross_val_predict splits the players into 5 folds and predicts each fold
# with a model trained only on the other 4. Every player gets exactly one
# prediction from a model that never saw them - so these numbers reflect
# how the model does on new players. These out-of-fold predictions are also
# what an undervalued-player finder should compare against actual value.
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def report(name, y_true_log, y_pred_log):
    actual = np.expm1(y_true_log)
    pred = np.expm1(y_pred_log)
    pct_err = np.abs(pred - actual) / actual

    print(f"\n--- {name} (out-of-fold, {len(actual)} players) ---")
    print(f"MAE (log scale):   {mean_absolute_error(y_true_log, y_pred_log):.4f}")
    print(f"R^2 (log scale):   {r2_score(y_true_log, y_pred_log):.3f}")
    print(f"MAE (euros):       €{mean_absolute_error(actual, pred):.2f}m")
    print(f"Median % error:    {np.median(pct_err):.1%}")
    print(f"Within ±25%:       {np.mean(pct_err <= 0.25):.1%} of players")
    print(f"Within ±50%:       {np.mean(pct_err <= 0.50):.1%} of players")


oof_ridge = cross_val_predict(ridge_model, X, y, cv=kf)
oof_xgb = cross_val_predict(xgb_model, X, y, cv=kf)

report("Ridge + age spline", y, oof_ridge)
report("XGBoost (benchmark)", y, oof_xgb)
