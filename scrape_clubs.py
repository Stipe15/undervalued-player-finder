"""
Scrapes each club's total squad market value from Transfermarkt, as a
club-strength feature for main.py.

podaci/players_scraped.csv already has a MATCHED_CLUB for all 378 players,
and there are only 71 unique clubs among them - so unlike the player scrape,
this needs one search request per club (no profile-page fetch): Transfermarkt's
own club search results table already shows "Total Market Value" per club.

Progress is written to podaci/clubs_scraped.csv after every club, so the
script can be safely re-run/resumed (it skips clubs already present with a
status).

Run with: python scrape_clubs.py
"""

import csv
import difflib
import os
import re
import time

import requests
from bs4 import BeautifulSoup

import scrape_transfermarkt as st

SRC = "podaci/players_scraped.csv"
OUT = "podaci/clubs_scraped.csv"
FIELDNAMES = ["CLUB", "STATUS", "MATCHED_NAME", "LEAGUE", "SQUAD_MARKET_VALUE_M_EUR", "CLUB_URL"]

# The exact league labels Transfermarkt's own search results use for the
# top-5 leagues (note "LaLiga", no space - different from our CSV's "La Liga").
# Every club we're looking for currently plays in one of these, so a candidate
# whose league matches one of these is almost certainly the right club.
TOP5_LEAGUES = {"Premier League", "LaLiga", "Bundesliga", "Serie A", "Ligue 1"}


def search_club(session, name):
    """Returns a list of candidate dicts parsed from the club search results table."""
    url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    resp = session.get(url, params={"query": name}, headers=st.HEADERS, timeout=20)
    resp.raise_for_status()
    if "captcha" in resp.text.lower()[:5000]:
        raise RuntimeError("CAPTCHA encountered - stopping.")

    soup = BeautifulSoup(resp.text, "html.parser")
    header = soup.find(id="club-grid_c0")
    if header is None:
        return []
    table = header.find_parent("table")
    body_rows = table.find("tbody").find_all("tr", recursive=False)

    candidates = []
    for row in body_rows:
        name_link = row.select_one('a[href*="/startseite/verein/"]')
        if not name_link:
            continue
        href = name_link["href"]
        m_id = re.search(r"/verein/(\d+)", href)
        slug = href.split("/startseite/verein/")[0].strip("/")
        cand_name = name_link.get_text(strip=True)

        league_link = row.select_one('a[href*="/wettbewerb/"]')
        league = league_link.get_text(strip=True) if league_link else None

        market_value = None
        for td in row.find_all("td"):
            txt = td.get_text(strip=True)
            if txt.startswith("€"):  # only the market-value cell starts with "€"
                market_value = st.parse_market_value(txt)
                break

        candidates.append({
            "name": cand_name,
            "slug": slug,
            "id": m_id.group(1) if m_id else None,
            "league": league,
            "market_value": market_value,
        })
    return candidates


def pick_best_club(target_name, candidates):
    """Prefer a candidate playing in one of the top-5 leagues; break ties by name match."""
    if not candidates:
        return None, False

    target_norm = st.strip_accents(target_name).lower()
    scored = []
    for c in candidates:
        ratio = difflib.SequenceMatcher(None, target_norm, st.strip_accents(c["name"]).lower()).ratio()
        is_top5 = c["league"] in TOP5_LEAGUES
        scored.append((is_top5, ratio, c))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    is_top5, best_ratio, best = scored[0]
    confident = is_top5 and best_ratio >= 0.6
    return best, confident


def load_already_done():
    done = {}
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done[row["CLUB"]] = row
    return done


def main():
    with open(SRC, newline="", encoding="utf-8-sig") as f:
        clubs = sorted({row["MATCHED_CLUB"] for row in csv.DictReader(f) if row["MATCHED_CLUB"]})

    done = load_already_done()
    session = requests.Session()

    out_exists = os.path.exists(OUT)
    out_f = open(OUT, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not out_exists:
        writer.writeheader()

    n_ok, n_review = 0, 0

    for i, club in enumerate(clubs, 1):
        if club in done and done[club]["STATUS"] in ("OK", "NEEDS_REVIEW"):
            continue

        row = {"CLUB": club}
        try:
            candidates = search_club(session, club)
            time.sleep(st.REQUEST_DELAY)

            # Transfermarkt's search chokes on "&" in a multi-word query
            # (e.g. "Brighton & Hove Albion" -> no result); retry with "and".
            if not candidates and "&" in club:
                candidates = search_club(session, club.replace("&", "and"))
                time.sleep(st.REQUEST_DELAY)

            if not candidates:
                row["STATUS"] = "NOT_FOUND"
                print(f"[{i}/{len(clubs)}] {club}: NOT FOUND")
            else:
                best, confident = pick_best_club(club, candidates)
                row.update({
                    "STATUS": "OK" if confident else "NEEDS_REVIEW",
                    "MATCHED_NAME": best["name"],
                    "LEAGUE": best["league"],
                    "SQUAD_MARKET_VALUE_M_EUR": best["market_value"],
                    "CLUB_URL": f"https://www.transfermarkt.com/{best['slug']}/startseite/verein/{best['id']}",
                })
                if confident:
                    n_ok += 1
                else:
                    n_review += 1
                print(f"[{i}/{len(clubs)}] {club}: {row['STATUS']} -> "
                      f"{best['name']} ({best['league']}, {best['market_value']}m)")
        except requests.exceptions.RequestException as e:
            row["STATUS"] = f"ERROR: {e}"
            print(f"[{i}/{len(clubs)}] {club}: ERROR {e}")

        writer.writerow(row)
        out_f.flush()

    out_f.close()
    print(f"\nDone. OK={n_ok} NEEDS_REVIEW={n_review} out of {len(clubs)}. Wrote {OUT}")


if __name__ == "__main__":
    main()
