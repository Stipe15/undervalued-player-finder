"""
Re-checks every row in podaci/players_scraped.csv against the fixed
matching logic in scrape_transfermarkt.py (see its pick_best_match /
strip_accents for why the original matching had a bug). Only re-searches
(cheap) for every player; only re-fetches the profile page (expensive) for
rows whose match actually changes or that were already flagged.

Run with: python revalidate_scrape.py
"""
import csv
import time

import requests

import scrape_transfermarkt as st

SRC = "podaci/players_scraped.csv"
OUT = "podaci/players_scraped.csv"


def main():
    with open(SRC, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    session = requests.Session()
    n_changed = 0
    n_checked = 0

    for row in rows:
        name = row["NAME"]
        target_mv = float(row["MARKET_VALUE_M_EUR"]) if row["MARKET_VALUE_M_EUR"] else None
        old_url = row.get("PROFILE_URL", "")

        candidates = st.search_player(session, name)
        time.sleep(st.REQUEST_DELAY)
        n_checked += 1

        best, confident = st.pick_best_match(name, target_mv, candidates)
        if best is None:
            if row["STATUS"] == "OK":
                print(f"[{n_checked}/{len(rows)}] {name}: WAS OK, now NOT_FOUND on recheck - flagging")
                row["STATUS"] = "NEEDS_REVIEW"
            continue

        new_url = f"https://www.transfermarkt.com/{best['slug']}/profil/spieler/{best['id']}"
        changed = (new_url != old_url) or (row["STATUS"] != "OK") or (not confident)

        if not changed:
            continue

        n_changed += 1
        if not confident:
            top3 = ", ".join(f"{c['name']} ({c['club']}, {c['market_value']}m)" for c in candidates[:3])
            print(f"[{n_checked}/{len(rows)}] {name}: NEEDS_REVIEW - candidates: {top3}")
            row.update({
                "STATUS": "NEEDS_REVIEW",
                "MATCHED_NAME": best["name"],
                "MATCHED_CLUB": best["club"],
                "TM_MARKET_VALUE_M_EUR": best["market_value"],
                "AGE": "", "POSITION": "", "POSITION_RAW": "",
                "CONTRACT_EXPIRY": "", "PROFILE_URL": "",
            })
            continue

        print(f"[{n_checked}/{len(rows)}] {name}: match changed ({old_url} -> {new_url}), re-fetching profile")
        details = st.fetch_contract_and_confirm(session, best)
        time.sleep(st.REQUEST_DELAY)
        position_bucket = st.POSITION_BUCKET.get(details["position_raw"], "")
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
        print(f"    -> age={row['AGE']} pos={row['POSITION']} contract={row['CONTRACT_EXPIRY']}")

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=st.FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nChecked {n_checked} players, {n_changed} rows changed. Wrote {OUT}")


if __name__ == "__main__":
    main()
