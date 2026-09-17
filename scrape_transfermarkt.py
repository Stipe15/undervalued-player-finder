"""
Scrapes age, position, and contract expiry for every player in
podaci/combined_top5_marketvalue.csv from Transfermarkt.

For each player:
  1. Query Transfermarkt's quick search for the player's name.
  2. Parse the results table (name, club, position, age, market value).
  3. Pick the candidate whose market value is closest to the value already
     in our CSV (a strong disambiguator for common names), requiring a
     reasonably close name match too. Low-confidence matches are left for
     manual review rather than guessed.
  4. Visit the matched player's profile page for contract expiry (not
     shown in the search table).

Progress is written to podaci/players_scraped.csv after every player, so
the script can be safely re-run/resumed (it skips names already present
with a status).

Run with: python scrape_transfermarkt.py
"""

import csv
import difflib
import os
import re
import sys
import time
import unicodedata

import requests
from bs4 import BeautifulSoup

SRC = "podaci/combined_top5_marketvalue.csv"
OUT = "podaci/players_scraped.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_DELAY = 1.5  # seconds between requests, polite rate limit
FIELDNAMES = [
    "NAME", "LEAGUE", "MARKET_VALUE_M_EUR", "AGE_IN_DATA",
    "STATUS", "MATCHED_NAME", "MATCHED_CLUB", "TM_MARKET_VALUE_M_EUR",
    "AGE", "POSITION", "POSITION_RAW", "CONTRACT_EXPIRY", "PROFILE_URL",
]

POSITION_BUCKET = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB",
    "Right-Back": "FB",
    "Defensive Midfield": "DM",
    "Central Midfield": "CM",
    "Attacking Midfield": "AM",
    "Left Winger": "W",
    "Right Winger": "W",
    "Left Midfield": "W",
    "Right Midfield": "W",
    "Centre-Forward": "ST",
    "Second Striker": "ST",
}


# Characters unicodedata.normalize("NFKD", ...) does NOT decompose, so a
# plain ascii-encode/ignore would silently DROP them (e.g. "Yıldız" -> "Yldz").
# That previously caused a mismatch: a namesake with plain ASCII spelling
# scored a "better" match than the correct player. Map these explicitly first.
_EXTRA_CHAR_MAP = str.maketrans({
    "ı": "i", "İ": "I", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
    "ł": "l", "Ł": "L", "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
})


def strip_accents(s):
    s = s.translate(_EXTRA_CHAR_MAP)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def parse_market_value(text):
    """'€100.00m' -> 100.0, '€500Th.' -> 0.5, '€1.33bn' -> 1330.0, '-' -> None"""
    text = text.strip()
    if not text or text == "-":
        return None
    m = re.match(r"[^\d]*([\d.]+)\s*(bn|m|Th\.)?", text)
    if not m:
        return None
    val = float(m.group(1))
    if m.group(2) == "Th.":
        val /= 1000.0
    elif m.group(2) == "bn":
        val *= 1000.0
    return val


def search_player(session, name):
    """Returns a list of candidate dicts parsed from the quick-search results table."""
    url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    resp = session.get(url, params={"query": name}, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    if "captcha" in resp.text.lower()[:5000]:
        raise RuntimeError("CAPTCHA encountered - stopping.")

    soup = BeautifulSoup(resp.text, "html.parser")
    header = soup.find(id="player-grid_c0")
    if header is None:
        return []
    table = header.find_parent("table")
    body_rows = table.find("tbody").find_all("tr", recursive=False)

    candidates = []
    for row in body_rows:
        cells = row.find_all("td", recursive=False)
        if len(cells) < 6:
            continue

        name_link = cells[0].select_one('a[href*="/profil/spieler/"]')
        if not name_link:
            continue
        href = name_link["href"]
        m_id = re.search(r"/profil/spieler/(\d+)", href)
        slug = href.split("/profil/spieler/")[0].strip("/")
        cand_name = name_link.get_text(strip=True)

        club_cell = cells[0].find("table").find_all("tr")
        club = None
        if len(club_cell) > 1:
            club_link = club_cell[1].find("a")
            club = club_link.get_text(strip=True) if club_link else club_cell[1].get_text(strip=True)

        position = cells[1].get_text(strip=True) or None
        age_text = cells[3].get_text(strip=True)
        age = int(age_text) if age_text.isdigit() else None
        market_value = parse_market_value(cells[5].get_text(strip=True))

        candidates.append({
            "name": cand_name,
            "slug": slug,
            "id": m_id.group(1) if m_id else None,
            "club": club,
            "position": position,
            "age": age,
            "market_value": market_value,
        })
    return candidates


def pick_best_match(target_name, target_mv, candidates):
    """Score candidates by name similarity + market value closeness.

    Our players are all current top-5-league players with a real market
    value, so a candidate with NO market value on Transfermarkt (retired,
    reserve team, a lower league) is treated as suspect even if the name
    matches exactly - that combination is exactly what previously matched
    a wrong, retired "Kenan Yildiz" instead of the real (accented) one.

    Returns (candidate, confident: bool).
    """
    if not candidates:
        return None, False

    target_name_norm = strip_accents(target_name).lower()
    scored = []
    for c in candidates:
        name_ratio = difflib.SequenceMatcher(
            None, target_name_norm, strip_accents(c["name"]).lower()
        ).ratio()
        mv_diff = None
        if target_mv and c["market_value"]:
            mv_diff = abs(c["market_value"] - target_mv) / target_mv
        scored.append((name_ratio, mv_diff, c))

    # Prefer candidates with a market value close to ours; push valueless
    # candidates (mv_diff unknown) to the back rather than letting a bare
    # name match win outright.
    scored.sort(key=lambda t: (t[1] if t[1] is not None else 1e9, -t[0]))
    best_ratio, best_mv_diff, best = scored[0]

    confident = (
        (best_mv_diff is not None and best_mv_diff <= 0.35 and best_ratio >= 0.85)
        or (best_mv_diff is not None and best_mv_diff <= 0.15 and best_ratio >= 0.6)
    )
    return best, confident


def fetch_contract_and_confirm(session, candidate):
    """Visits the profile page; returns dict with age, position_raw, contract_expiry."""
    url = f"https://www.transfermarkt.com/{candidate['slug']}/profil/spieler/{candidate['id']}"
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    if "captcha" in resp.text.lower()[:5000]:
        raise RuntimeError("CAPTCHA encountered - stopping.")
    html = resp.text

    result = {"age": None, "position_raw": None, "contract_expiry": None, "url": url}

    m = re.search(
        r"Date of birth/Age:.*?data-header__content[^>]*>(.*?)</span>", html, re.S
    )
    if m:
        age_m = re.search(r"\((\d+)\)", m.group(1))
        if age_m:
            result["age"] = int(age_m.group(1))

    m = re.search(
        r"Position:.*?data-header__content[^>]*>(.*?)</span>", html, re.S
    )
    if m:
        result["position_raw"] = re.sub(r"<[^>]+>", "", m.group(1)).strip()

    m = re.search(r"Contract expires:.*?data-header__content[^>]*>([^<]*)", html, re.S)
    if m:
        date_m = re.search(r"(\d{2})/(\d{2})/(\d{4})", m.group(1))
        if date_m:
            result["contract_expiry"] = int(date_m.group(3))

    return result


def load_already_done():
    done = {}
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done[row["NAME"]] = row
    return done


def main():
    with open(SRC, newline="", encoding="utf-8") as f:
        players = list(csv.DictReader(f))

    done = load_already_done()
    session = requests.Session()

    out_exists = os.path.exists(OUT)
    out_f = open(OUT, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not out_exists:
        writer.writeheader()

    n_ok, n_review, n_notfound = 0, 0, 0

    for i, p in enumerate(players, 1):
        name = p["NAME"]
        if name in done and done[name]["STATUS"] in ("OK", "NEEDS_REVIEW"):
            continue  # already processed in a previous run

        target_mv = float(p["Market Value (m/€)"]) if "Market Value (m/€)" in p else None
        row = {
            "NAME": name,
            "LEAGUE": p.get("LEAGUE", ""),
            "MARKET_VALUE_M_EUR": target_mv,
            "AGE_IN_DATA": p.get("Age", ""),
        }

        try:
            candidates = search_player(session, name)
            time.sleep(REQUEST_DELAY)

            if not candidates:
                row.update({"STATUS": "NOT_FOUND"})
                n_notfound += 1
                print(f"[{i}/{len(players)}] {name}: NOT FOUND")
            else:
                best, confident = pick_best_match(name, target_mv, candidates)
                if not confident:
                    top3 = ", ".join(
                        f"{c['name']} ({c['club']}, {c['market_value']}m)"
                        for c in candidates[:3]
                    )
                    row.update({
                        "STATUS": "NEEDS_REVIEW",
                        "MATCHED_NAME": best["name"],
                        "MATCHED_CLUB": best["club"],
                        "TM_MARKET_VALUE_M_EUR": best["market_value"],
                    })
                    n_review += 1
                    print(f"[{i}/{len(players)}] {name}: NEEDS REVIEW - candidates: {top3}")
                else:
                    details = fetch_contract_and_confirm(session, best)
                    time.sleep(REQUEST_DELAY)
                    position_bucket = POSITION_BUCKET.get(details["position_raw"], "")
                    row.update({
                        "STATUS": "OK",
                        "MATCHED_NAME": best["name"],
                        "MATCHED_CLUB": best["club"],
                        "TM_MARKET_VALUE_M_EUR": best["market_value"],
                        "AGE": details["age"] or best["age"],
                        "POSITION": position_bucket,
                        "POSITION_RAW": details["position_raw"] or best["position"],
                        "CONTRACT_EXPIRY": details["contract_expiry"],
                        "PROFILE_URL": details["url"],
                    })
                    n_ok += 1
                    print(f"[{i}/{len(players)}] {name}: OK -> "
                          f"age={row['AGE']} pos={row['POSITION']} "
                          f"contract={row['CONTRACT_EXPIRY']}")
        except requests.exceptions.RequestException as e:
            row.update({"STATUS": f"ERROR: {e}"})
            print(f"[{i}/{len(players)}] {name}: ERROR {e}")
        except RuntimeError as e:
            print(f"STOPPING: {e}")
            writer.writerow(row)
            out_f.flush()
            sys.exit(1)

        writer.writerow(row)
        out_f.flush()

    out_f.close()
    print(f"\nDone. OK={n_ok} NEEDS_REVIEW={n_review} NOT_FOUND={n_notfound} "
          f"out of {len(players)}. Wrote {OUT}")


if __name__ == "__main__":
    main()
