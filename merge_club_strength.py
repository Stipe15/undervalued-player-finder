"""
Adds a CLUB_STRENGTH feature (the player's current club's total squad market
value, from Transfermarkt - see scrape_clubs.py) to the enriched dataset.

Run merge_scraped_data.py first to produce combined_top5_marketvalue_enriched.csv;
this script reads that file, joins on MATCHED_CLUB from players_scraped.csv, and
overwrites it with the CLUB_STRENGTH column added.

Run with: python merge_club_strength.py
"""
import pandas as pd

df = pd.read_csv("podaci/combined_top5_marketvalue_enriched.csv")
df = df.drop(columns=["CLUB_STRENGTH"], errors="ignore")  # idempotent if re-run
scraped = pd.read_csv("podaci/players_scraped.csv")
clubs = pd.read_csv("podaci/clubs_scraped.csv")

player_club = scraped[["NAME", "MATCHED_CLUB"]]
merged = df.merge(player_club, on="NAME", how="left", validate="one_to_one")

missing = merged["MATCHED_CLUB"].isna().sum()
if missing:
    raise SystemExit(f"{missing} players failed to merge with players_scraped.csv - check names match exactly")

merged = merged.merge(
    clubs[["CLUB", "SQUAD_MARKET_VALUE_M_EUR"]],
    left_on="MATCHED_CLUB", right_on="CLUB", how="left", validate="many_to_one",
)
merged = merged.rename(columns={"SQUAD_MARKET_VALUE_M_EUR": "CLUB_STRENGTH"})
merged = merged.drop(columns=["MATCHED_CLUB", "CLUB"])

missing_strength = merged["CLUB_STRENGTH"].isna().sum()
if missing_strength:
    raise SystemExit(f"{missing_strength} players' clubs failed to merge with clubs_scraped.csv")

out = "podaci/combined_top5_marketvalue_enriched.csv"
merged.to_csv(out, index=False)
print(f"Wrote {len(merged)} rows x {len(merged.columns)} columns -> {out}")
print(merged["CLUB_STRENGTH"].describe())
