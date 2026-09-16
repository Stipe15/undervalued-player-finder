"""
Merges the Transfermarkt-scraped AGE / POSITION / CONTRACT_EXPIRY onto the
original stats CSV, producing podaci/combined_top5_marketvalue_enriched.csv.

- AGE from Transfermarkt replaces the original Age column (the original had
  scraping errors - see main.py's age filter comment history).
- POSITION is added as a new categorical column (one-hot encoded downstream).
- CONTRACT_EXPIRY becomes CONTRACT_YEARS_LEFT = CONTRACT_EXPIRY - SEASON_YEAR.
  A handful of players have no contract date on Transfermarkt; those rows
  get an empty CONTRACT_YEARS_LEFT rather than a guessed number, and
  main.py's modeling code needs to handle that (impute or drop rows).

Note: a RECENTLY_TRANSFERRED flag (current club in a different top-5 league
than LEAGUE says) was tried and dropped - it measured a real data quirk
(~40/378 players) but didn't move CV R^2, so it wasn't worth the extra
feature. See conversation history / git log if revisiting this.

Run with: python merge_scraped_data.py
"""
import pandas as pd

SEASON_YEAR = 2026  # the season this dataset's stats/values are drawn from

df = pd.read_csv("podaci/combined_top5_marketvalue.csv")
scraped = pd.read_csv("podaci/players_scraped.csv")

assert (scraped["STATUS"] == "OK").all(), "Some scraped rows are not OK - resolve before merging"

scraped = scraped[["NAME", "AGE", "POSITION", "CONTRACT_EXPIRY"]]
merged = df.merge(scraped, on="NAME", how="left", validate="one_to_one")

missing = merged["AGE"].isna().sum()
if missing:
    raise SystemExit(f"{missing} players failed to merge - check names match exactly")

merged["Age"] = merged["AGE"]
merged["CONTRACT_YEARS_LEFT"] = merged["CONTRACT_EXPIRY"] - SEASON_YEAR
merged = merged.drop(columns=["AGE", "CONTRACT_EXPIRY"])

n_no_contract = merged["CONTRACT_YEARS_LEFT"].isna().sum()
print(f"{n_no_contract} players have no contract-expiry date on Transfermarkt "
      f"(left as blank in CONTRACT_YEARS_LEFT)")

out = "podaci/combined_top5_marketvalue_enriched.csv"
merged.to_csv(out, index=False)
print(f"Wrote {len(merged)} rows x {len(merged.columns)} columns -> {out}")
print("Position counts:\n", merged["POSITION"].value_counts())
