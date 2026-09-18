"""
Extends the modeling dataset with the ~1,433 players below the original
EUR 20m floor (see scrape_remaining_players.py / scrape_needs_review.py),
producing the final podaci/combined_top5_marketvalue_enriched.csv used by
main.py - now covering close to the full top5_stats_combined.csv population
instead of just the original top-500-by-value list.

Excluded on purpose:
  - rows still STATUS == NEEDS_REVIEW (2 links that resolved to the wrong
    player - pending a corrected link)
  - rows with no market value at all on Transfermarkt (1 player: nothing to
    predict, not a data gap to impute)

CLUB_STRENGTH for "Without Club" (unattached free agents) is set to 0 - a
real fact about them (no squad backing them), not a missing value to impute.

Run with: python extend_dataset.py
"""
import numpy as np
import pandas as pd

SEASON_YEAR = 2026

stats = pd.read_csv("podaci/top5_stats_combined.csv")
extra = pd.read_csv("podaci/players_scraped_extra.csv")
clubs = pd.read_csv("podaci/clubs_scraped.csv")
original = pd.read_csv("podaci/combined_top5_marketvalue_enriched.csv")

usable = extra[(extra["STATUS"] == "OK") & extra["TM_MARKET_VALUE_M_EUR"].notna()].copy()
print(f"{len(extra) - len(usable)} of {len(extra)} scraped extra rows excluded "
      f"(still needs-review or no market value)")

new = stats.merge(usable, on="NAME", how="inner", validate="one_to_one", suffixes=("", "_scraped"))
assert len(new) == len(usable), "not every usable row matched a stats row"

new["Age"] = new["AGE"]
new["Market Value (m/€)"] = new["TM_MARKET_VALUE_M_EUR"]
new["CONTRACT_YEARS_LEFT"] = new["CONTRACT_EXPIRY"] - SEASON_YEAR

club_strength = clubs.set_index("CLUB")["SQUAD_MARKET_VALUE_M_EUR"]
new["CLUB_STRENGTH"] = new["MATCHED_CLUB"].map(club_strength)
new.loc[new["MATCHED_CLUB"] == "Without Club", "CLUB_STRENGTH"] = 0.0
missing_strength = new["CLUB_STRENGTH"].isna().sum()
if missing_strength:
    raise SystemExit(f"{missing_strength} new players' clubs have no CLUB_STRENGTH - re-run scrape_clubs.py")

new = new.rename(columns={"POSITION": "POSITION"})  # already named right, kept for clarity
new = new[original.columns]

n_missing_contract_new = new["CONTRACT_YEARS_LEFT"].isna().sum()
print(f"{n_missing_contract_new} new players have no contract-expiry date (left blank, "
      f"main.py imputes the median same as before)")

combined = pd.concat([original, new], ignore_index=True)
dup = combined["NAME"].duplicated().sum()
if dup:
    raise SystemExit(f"{dup} duplicate NAME rows after concat - investigate before overwriting")

out = "podaci/combined_top5_marketvalue_enriched.csv"
combined.to_csv(out, index=False)
print(f"Wrote {len(combined)} rows x {len(combined.columns)} columns -> {out} "
      f"({len(original)} original + {len(new)} new)")
print(combined["Market Value (m/€)"].describe())
