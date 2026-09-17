"""
Scrapes Transfermarkt market value, age, position and contract expiry for the
~1,450 players in podaci/top5_stats_combined.csv that are NOT in the original
top-500 list (i.e. the ones below the old EUR 20m floor).

Unlike scrape_transfermarkt.py, there is no known market value to match
against, so a candidate is accepted when:
  - "league":      its club played in the player's league (2025-26 or 2026-27
                   season club lists, cached in podaci/league_clubs.csv), the
                   name matches, and no other same-league candidate matches; or
  - "unique_name": no league match (e.g. transferred abroad), but it is the only
                   near-exact name match that has a market value.
Everything else is NEEDS_REVIEW. MATCH_METHOD records which rule was used.

Runs in 5 batches (python scrape_remaining_players.py --batch N, N = 1..5).
Progress is written row-by-row to podaci/players_scraped_extra.csv, so a
batch can be interrupted and re-run.
"""

import argparse
import csv
import difflib
import math
import os
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

import scrape_transfermarkt as st

STATS = "podaci/top5_stats_combined.csv"
DONE_378 = "podaci/combined_top5_marketvalue.csv"
OUT = "podaci/players_scraped_extra.csv"
LEAGUE_CLUBS = "podaci/league_clubs.csv"
N_BATCHES = 5
REQUEST_DELAY = st.REQUEST_DELAY

FIELDNAMES = [
    "NAME", "LEAGUE", "STATUS", "MATCH_METHOD", "MATCHED_NAME", "MATCHED_CLUB",
    "TM_MARKET_VALUE_M_EUR", "AGE", "POSITION", "POSITION_RAW", "CONTRACT_EXPIRY", "PROFILE_URL",
]

LEAGUE_PAGES = {
    "Premier League": "premier-league/startseite/wettbewerb/GB1",
    "La Liga": "laliga/startseite/wettbewerb/ES1",
    "Bundesliga": "bundesliga/startseite/wettbewerb/L1",
    "Serie A": "serie-a/startseite/wettbewerb/IT1",
    "Ligue 1": "ligue-1/startseite/wettbewerb/FR1",
}
SEASONS = [2025, 2026]  # stats season, plus current season (promotions/transfers)


def fetch_league_clubs(session):
    rows = []
    for league, path in LEAGUE_PAGES.items():
        for season in SEASONS:
            url = f"https://www.transfermarkt.com/{path}/saison_id/{season}"
            resp = session.get(url, headers=st.HEADERS, timeout=20)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.select('table.items tbody td.hauptlink a[href*="/startseite/verein/"]'):
                for club in {a.get("title"), a.get_text(strip=True)} - {None, ""}:
                    rows.append({"LEAGUE": league, "SEASON": season, "CLUB": club})
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(LEAGUE_CLUBS, index=False)
    return df


def load_league_clubs(session):
    df = pd.read_csv(LEAGUE_CLUBS) if os.path.exists(LEAGUE_CLUBS) else fetch_league_clubs(session)
    return {lg: set(g["CLUB"]) for lg, g in df.groupby("LEAGUE")}


def pick_match(name, leagues, candidates, league_clubs):
    target = st.strip_accents(name).lower()
    allowed_clubs = set().union(*(league_clubs.get(lg, set()) for lg in leagues))

    scored = []
    for c in candidates:
        ratio = difflib.SequenceMatcher(None, target, st.strip_accents(c["name"]).lower()).ratio()
        scored.append((ratio, c))

    in_league = [(r, c) for r, c in scored if r >= 0.75 and c["club"] in allowed_clubs]
    if len(in_league) == 1:
        return in_league[0][1], "league"
    if len(in_league) > 1:
        best = max(in_league, key=lambda t: (t[0], t[1]["market_value"] or 0))[1]
        return best, None

    exact_valued = [(r, c) for r, c in scored if r >= 0.9 and c["market_value"]]
    if len(exact_valued) == 1:
        return exact_valued[0][1], "unique_name"

    if not scored:
        return None, None
    best = max(scored, key=lambda t: (t[0], t[1]["market_value"] or 0))[1]
    return best, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True, choices=range(1, N_BATCHES + 1))
    args = parser.parse_args()

    stats = pd.read_csv(STATS)
    already = set(pd.read_csv(DONE_378)["NAME"])
    remaining = stats[~stats["NAME"].isin(already)][["NAME", "LEAGUE"]].reset_index(drop=True)

    size = math.ceil(len(remaining) / N_BATCHES)
    batch = remaining.iloc[(args.batch - 1) * size: args.batch * size]

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(prev.loc[~prev["STATUS"].astype(str).str.startswith("ERROR"), "NAME"])

    session = requests.Session()
    league_clubs = load_league_clubs(session)

    out_exists = os.path.exists(OUT)
    out_f = open(OUT, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not out_exists:
        writer.writeheader()

    counts = {"OK": 0, "NEEDS_REVIEW": 0, "NOT_FOUND": 0, "ERROR": 0}
    todo = batch[~batch["NAME"].isin(done)]
    print(f"Batch {args.batch}/{N_BATCHES}: {len(batch)} players, {len(todo)} still to scrape")
    t0 = time.time()

    for i, p in enumerate(todo.itertuples(index=False), 1):
        name, league = p.NAME, p.LEAGUE
        row = {"NAME": name, "LEAGUE": league}
        try:
            candidates = st.search_player(session, name)
            time.sleep(REQUEST_DELAY)
            if not candidates and st.strip_accents(name) != name:
                candidates = st.search_player(session, st.strip_accents(name))
                time.sleep(REQUEST_DELAY)
            best, method = pick_match(name, [lg.strip() for lg in league.split("/")], candidates, league_clubs)

            if best is None:
                row["STATUS"] = "NOT_FOUND"
            else:
                row.update({
                    "MATCHED_NAME": best["name"],
                    "MATCHED_CLUB": best["club"],
                    "TM_MARKET_VALUE_M_EUR": best["market_value"],
                })
                if method is None:
                    row["STATUS"] = "NEEDS_REVIEW"
                else:
                    details = st.fetch_contract_and_confirm(session, best)
                    time.sleep(REQUEST_DELAY)
                    row.update({
                        "STATUS": "OK",
                        "MATCH_METHOD": method,
                        "AGE": details["age"] or best["age"],
                        "POSITION": st.POSITION_BUCKET.get(details["position_raw"], ""),
                        "POSITION_RAW": details["position_raw"] or best["position"],
                        "CONTRACT_EXPIRY": details["contract_expiry"],
                        "PROFILE_URL": details["url"],
                    })
        except requests.exceptions.RequestException as e:
            row["STATUS"] = f"ERROR: {e}"
        except RuntimeError as e:
            print(f"STOPPING: {e}")
            out_f.close()
            sys.exit(1)

        counts[row["STATUS"].split(":")[0]] += 1
        writer.writerow(row)
        out_f.flush()
        print(f"[{i}/{len(todo)}] {name}: {row['STATUS']} "
              f"{row.get('MATCH_METHOD') or ''} -> {row.get('MATCHED_NAME')} "
              f"({row.get('MATCHED_CLUB')}, {row.get('TM_MARKET_VALUE_M_EUR')}m)", flush=True)

    out_f.close()
    print(f"\nBatch {args.batch} done in {(time.time() - t0) / 60:.1f} min. {counts}")


if __name__ == "__main__":
    main()
