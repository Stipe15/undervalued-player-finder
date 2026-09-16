"""
Produces a normalized version of podaci/combined_top5_marketvalue.csv,
ready to feed into an XGBoost training script.

Run with: python normalize_data.py

Note: XGBoost itself doesn't require normalized inputs (it's tree-based,
so it only cares about value order, not scale). This script exists because
a normalized CSV is still useful to have on hand - e.g. for comparing
players on a common 0-1 scale, or for use with other model types later.
"""

import pandas as pd

SRC = "podaci/combined_top5_marketvalue.csv"
DST = "podaci/combined_top5_marketvalue_normalized.csv"
TARGET = "Market Value (m/€)"

df = pd.read_csv(SRC)

# Same cleaning as main.py, so the normalized file matches what the model trains on.
df["LEAGUE"] = df["LEAGUE"].apply(lambda x: "Multi-league" if "/" in x else x)
n_before = len(df)
df = df[(df["Age"] >= 15) & (df["Age"] <= 42)].reset_index(drop=True)
print(f"Dropped {n_before - len(df)} rows with implausible ages ({n_before} -> {len(df)})")

# One-hot encode LEAGUE so every column in the output is numeric.
df = pd.get_dummies(df, columns=["LEAGUE"], drop_first=False)

# Min-max normalize the numeric feature columns to [0, 1].
# NAME (identifier) and the target (kept in real euros, for interpretability)
# are excluded, along with the one-hot LEAGUE columns which are already 0/1.
league_cols = [c for c in df.columns if c.startswith("LEAGUE_")]
exclude = {"NAME", TARGET, *league_cols}
feature_cols = [c for c in df.columns if c not in exclude]

for col in feature_cols:
    lo, hi = df[col].min(), df[col].max()
    df[col] = (df[col] - lo) / (hi - lo) if hi > lo else 0.0

df.to_csv(DST, index=False)
print(f"Wrote {len(df)} rows x {len(df.columns)} columns -> {DST}")
