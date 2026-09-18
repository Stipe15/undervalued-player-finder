"""
Resolves the NEEDS_REVIEW rows in podaci/players_scraped_extra.csv using the
Transfermarkt profile links the user manually confirmed (in PROFILE_URL).

For each such row:
  1. Fetch the given profile page directly - Transfermarkt profile URLs are
     keyed by the numeric id at the end, not the slug, so the id is what
     actually gets us the right player.
  2. Re-run the normal player search on the page's own displayed name and
     match it back to the same id, to reuse the canonical club spelling
     (matching podaci/clubs_scraped.csv) instead of the profile page's
     shorthand club name (e.g. "Getafe" vs "Getafe CF").
  3. Sanity-check that the page's name actually resembles the target name.
     A provided link that resolves to a clearly different player is not
     silently accepted - it's left as NEEDS_REVIEW with a NAME_MISMATCH note
     printed for manual double-checking, since a bad link would otherwise
     corrupt that row silently.

Run with: python scrape_needs_review.py
"""
import csv
import difflib
import re
import time

import requests
from bs4 import BeautifulSoup

import scrape_transfermarkt as st

SRC = OUT = "podaci/players_scraped_extra.csv"


def fetch_profile_page(session, url):
    resp = session.get(url, headers=st.HEADERS, timeout=20)
    resp.raise_for_status()
    if "captcha" in resp.text.lower()[:5000]:
        raise RuntimeError("CAPTCHA encountered - stopping.")
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.select_one("h1")
    name = re.sub(r"^#\d+\s*", "", h1.get_text(" ", strip=True)) if h1 else None

    mv_el = soup.select_one("a.data-header__market-value-wrapper")
    market_value = None
    if mv_el:
        market_value = st.parse_market_value(mv_el.get_text(" ", strip=True).split("Last update")[0])

    club_el = soup.select_one("span.data-header__club a")
    club = club_el.get_text(strip=True) if club_el else None

    age = None
    m = re.search(r"Date of birth/Age:.*?data-header__content[^>]*>(.*?)</span>", html, re.S)
    if m:
        am = re.search(r"\((\d+)\)", m.group(1))
        if am:
            age = int(am.group(1))

    position_raw = None
    m = re.search(r"Position:.*?data-header__content[^>]*>(.*?)</span>", html, re.S)
    if m:
        position_raw = re.sub(r"<[^>]+>", "", m.group(1)).strip()

    contract_expiry = None
    m = re.search(r"Contract expires:.*?data-header__content[^>]*>([^<]*)", html, re.S)
    if m:
        dm = re.search(r"(\d{2})/(\d{2})/(\d{4})", m.group(1))
        if dm:
            contract_expiry = int(dm.group(3))

    return {
        "name": name, "market_value": market_value, "club": club,
        "age": age, "position_raw": position_raw, "contract_expiry": contract_expiry,
    }


def canonical_club_and_value(session, name, player_id, fallback_club, fallback_value):
    """Re-run the normal search to get the club spelling used everywhere else."""
    candidates = st.search_player(session, name)
    time.sleep(st.REQUEST_DELAY)
    for c in candidates:
        if c["id"] == player_id:
            return c["club"] or fallback_club, c["market_value"] or fallback_value
    return fallback_club, fallback_value


def main():
    with open(SRC, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    session = requests.Session()
    n_ok, n_mismatch, n_error = 0, 0, 0

    for row in rows:
        if row["STATUS"] != "NEEDS_REVIEW" or not row.get("PROFILE_URL"):
            continue

        url = row["PROFILE_URL"]
        player_id = re.search(r"/spieler/(\d+)", url)
        player_id = player_id.group(1) if player_id else None

        try:
            profile = fetch_profile_page(session, url)
            time.sleep(st.REQUEST_DELAY)

            target_norm = st.strip_accents(row["NAME"]).lower()
            page_norm = st.strip_accents(profile["name"] or "").lower()
            ratio = difflib.SequenceMatcher(None, target_norm, page_norm).ratio()

            if ratio < 0.5:
                n_mismatch += 1
                print(f"NAME_MISMATCH: {row['NAME']!r} (league={row['LEAGUE']}) -> "
                      f"link resolves to {profile['name']!r} ({profile['club']}) - {url}")
                continue

            club, market_value = canonical_club_and_value(
                session, profile["name"], player_id, profile["club"], profile["market_value"]
            )

            row.update({
                "STATUS": "OK",
                "MATCH_METHOD": "manual_link",
                "MATCHED_NAME": profile["name"],
                "MATCHED_CLUB": club,
                "TM_MARKET_VALUE_M_EUR": profile["market_value"],
                "AGE": profile["age"],
                "POSITION": st.POSITION_BUCKET.get(profile["position_raw"], ""),
                "POSITION_RAW": profile["position_raw"],
                "CONTRACT_EXPIRY": profile["contract_expiry"],
            })
            n_ok += 1
            print(f"OK: {row['NAME']} -> {profile['name']} ({club}, {profile['market_value']}m, "
                  f"age={profile['age']}, pos={row['POSITION']}, contract={profile['contract_expiry']})")
        except requests.exceptions.RequestException as e:
            n_error += 1
            print(f"ERROR: {row['NAME']}: {e}")
        except RuntimeError as e:
            print(f"STOPPING: {e}")
            break

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. OK={n_ok} NAME_MISMATCH={n_mismatch} ERROR={n_error}. Wrote {OUT}")


if __name__ == "__main__":
    main()
