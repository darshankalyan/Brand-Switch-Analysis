"""
Pull r/CreditCardsIndia from Arctic Shift for 2025 and 2026, no record cap.

Goes newest-first and walks backward until it reaches CUTOFF (1 Jan 2025)
or runs out of data. Every page is written to disk as it arrives, so:
  - Ctrl+C is safe, nothing already pulled is lost
  - rerunning the script resumes where it stopped
  - the final CSVs are rebuilt from everything saved so far

Timeouts are handled by shrinking the page size and retrying the same
window. Only after repeated failures does it skip one hour, and the
number of skipped windows is reported at the end so gaps are visible.

pip install requests pandas tqdm
"""

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://arctic-shift.photon-reddit.com"
SUBREDDIT = "CreditCardsIndia"
CUTOFF = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())

BRANDS = [
    "HDFC", "Regalia", "Infinia",
    "ICICI", "Amazon Pay",
    "SBI Card", "Cashback SBI", "SBI",
    "Axis", "Magnus",
    "Amex", "American Express", "MRCC",
    "Kotak",
    "IDFC First", "IDFC FIRST",
    "Yes Bank",
    "IndusInd",
    "HSBC",
    "RBL",
]

ENDPOINTS = {
    "posts": "id,title,selftext,author,created_utc,score,num_comments",
    "comments": "id,body,author,created_utc,score,link_id",
}


def fmt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def contains_brand(text):
    if pd.isna(text):
        return None
    t = str(text).lower()
    hits = [b for b in BRANDS if b.lower() in t]
    return hits or None


def load_checkpoint(path):
    seen, oldest = set(), None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                seen.add(rec["id"])
                ts = rec["created_utc"]
                oldest = ts if oldest is None else min(oldest, ts)
    return seen, oldest


def pull(endpoint, fields):
    path = f"{endpoint}_recent.jsonl"
    seen, oldest = load_checkpoint(path)
    before = oldest if oldest else None
    if oldest:
        print(f"Resuming {endpoint} from {fmt(oldest)} ({len(seen)} already saved)")

    limit = 100
    errs = dead_streak = empty = skipped = 0
    stop_reason = "unknown"
    bar = tqdm(initial=len(seen), desc=endpoint, unit="rec")

    try:
        while True:
            params = {
                "subreddit": SUBREDDIT,
                "limit": limit,
                "sort": "desc",
                "fields": fields,
            }
            if before:
                params["before"] = before

            try:
                r = requests.get(f"{BASE_URL}/api/{endpoint}/search", params=params, timeout=30)
                status = r.status_code
            except requests.exceptions.RequestException as e:
                r, status = None, type(e).__name__

            if status == 429:
                bar.write("  Rate limited, waiting 15s")
                time.sleep(15)
                continue

            if status != 200:
                errs += 1
                limit = max(10, limit // 2)
                detail = r.text[:100] if r is not None else ""
                bar.write(f"  Error {status} (retry {errs}, page size now {limit}) {detail}")
                if errs >= 6 and before:
                    before -= 3600
                    skipped += 1
                    dead_streak += 1
                    errs = 0
                    bar.write("  Skipping one hour after repeated failures")
                    if dead_streak >= 20:
                        stop_reason = "too many consecutive failures, rerun later to resume"
                        break
                time.sleep(min(5 * max(errs, 1), 30))
                continue

            errs = dead_streak = 0
            limit = min(100, limit + 10)
            batch = r.json().get("data", [])

            if not batch:
                empty += 1
                if empty >= 3:
                    stop_reason = "no more data"
                    break
                time.sleep(2)
                continue
            empty = 0

            fresh = [b for b in batch if b["created_utc"] >= CUTOFF]
            new = [b for b in fresh if b["id"] not in seen]
            if new:
                with open(path, "a", encoding="utf-8") as f:
                    for b in new:
                        f.write(json.dumps(b) + "\n")
                        seen.add(b["id"])
                bar.update(len(new))

            oldest_in_batch = batch[-1]["created_utc"]
            bar.set_postfix(date=fmt(oldest_in_batch), page=limit)

            if len(fresh) < len(batch):
                stop_reason = f"reached cutoff {fmt(CUTOFF)}"
                break

            # keep ties at the page boundary; dedupe by id handles repeats
            next_before = oldest_in_batch
            if next_before == before:
                next_before -= 1
            before = next_before
            time.sleep(0.7)

    except KeyboardInterrupt:
        stop_reason = "interrupted by you, rerun to resume"
        bar.write("Interrupted. Everything pulled so far is saved.")

    bar.close()
    return path, skipped, stop_reason


def build_outputs(endpoint, path, skipped, stop_reason):
    if not os.path.exists(path):
        print(f"\n{endpoint}: nothing saved.")
        return
    df = pd.read_json(path, lines=True, dtype=False).drop_duplicates(subset="id")
    df["date"] = pd.to_datetime(df["created_utc"], unit="s")
    if endpoint == "comments":
        text = df["body"]
    else:
        text = df["title"].fillna("") + " " + df["selftext"].fillna("")
    df["matched_brands"] = text.apply(contains_brand)
    rel = df[df["matched_brands"].notna()]

    df.to_csv(f"{endpoint}_recent_raw.csv", index=False)
    rel.to_csv(f"{endpoint}_recent_filtered.csv", index=False)

    print(f"\n{endpoint}: {len(df)} records, {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  mentioning a brand: {len(rel)}")
    print(f"  stopped because: {stop_reason}")
    print(f"  skipped one-hour windows (gaps): {skipped}")
    print("  records per month:")
    print(df.groupby(df["date"].dt.to_period("M")).size().to_string())


if __name__ == "__main__":
    results = {}
    for ep, flds in ENDPOINTS.items():
        results[ep] = pull(ep, flds)
        if results[ep][2].startswith("interrupted"):
            break  # Ctrl+C stops the whole run, not just this endpoint
    for ep, (p, sk, why) in results.items():
        build_outputs(ep, p, sk, why)
    print("\nDone.")
