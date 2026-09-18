# Undervalued Player Finder

Predicts a football player's market value from performance stats, age, position, contract situation, and club strength, to surface players the market may be under- or over-pricing. Covers the top-5 European leagues.

See [documentation.docx](documentation.docx) for the full write-up (data issues found and fixed, feature engineering, scraping methodology, and complete results). This README is a short summary.

## Dataset

- **1,813 players**, €75k–€220m in market value (originally 378 players, all ≥€20m — the floor was removed by scraping ~1,450 additional players).
- Features: per-90 performance stats, age, position, contract years left, league, and club strength (squad market value).
- Final `podaci/combined_top5_marketvalue_enriched.csv` is built from `podaci/top5_stats_combined.csv` plus data scraped from Transfermarkt.

## Model

- **Ridge regression** with a spline on age (main model) and **XGBoost** (benchmark), both trained on the log-transformed market value.
- Evaluated honestly via 5-fold `cross_val_predict` (every prediction comes from a model that never saw that player).
- Scoped to the middle 80% by value (**€1.8m–€40m**) for evaluation, since errors concentrate at both tails: cheap fringe players have noisy values stats can't explain, and superstars carry a reputation premium no feature here captures.

### Results (middle 80%, n=1,393)

| Metric | Ridge | XGBoost |
|---|---|---|
| MAE | €3.98m | €3.86m |
| Median % error | 30.5% | 29.6% |
| Within ±25% | 41.3% | 41.9% |
| Within ±50% | 71.9% | 75.2% |

## Known limitations

- No assists/xA/key-pass data anywhere in the source stats, and no goalkeepers at all.
- Systematically under-predicts superstars (reputation/hype isn't modeled) — hence the €40m evaluation ceiling.
- A handful of players have imputed contract dates, no club, or were excluded for having no listed market value.

## Reproducing the dataset

Run in order (each script's docstring has details):

```
scrape_transfermarkt.py        -> podaci/players_scraped.csv
merge_scraped_data.py          -> combined_top5_marketvalue_enriched.csv (378 rows)
scrape_clubs.py                -> podaci/clubs_scraped.csv
merge_club_strength.py         -> adds CLUB_STRENGTH (still 378 rows)
scrape_remaining_players.py    -> podaci/players_scraped_extra.csv (--batch 1..5)
scrape_needs_review.py         -> resolves manually-reviewed rows
extend_dataset.py              -> final combined_top5_marketvalue_enriched.csv (1,813 rows)
main.py                        -> trains and evaluates the model
```

```bash
python main.py
```
