import time
import requests
import feedparser
import pandas as pd
from typing import List
from datetime import datetime, timedelta


ARXIV_API_URL = "http://export.arxiv.org/api/query"
HEADERS = {
    "User-Agent": "TopicModelingBot/1.0 (contact: your_email@example.com)"
}

MAX_RESULTS = 100
REQUEST_SLEEP = 6
MAX_START = 9000


def build_query(categories: List[str], start_date: str, end_date: str) -> str:
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    date_query = f"submittedDate:[{start_date} TO {end_date}]"
    return f"({cat_query}) AND {date_query}"


def fetch_batch(query: str, start: int):
    params = {
        "search_query": query,
        "start": start,
        "max_results": MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "ascending"
    }

    backoff = 10
    max_backoff = 120

    while True:
        try:
            response = requests.get(
                ARXIV_API_URL,
                params=params,
                headers=HEADERS,
                timeout=30
            )

            if response.status_code == 429:
                print(f"⏳ 429 rate limit. Sleeping {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            response.raise_for_status()
            return feedparser.parse(response.text)

        except requests.RequestException as e:
            print(f"⚠️ Request error: {e}. Sleeping {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

def download_window(categories, start_date, end_date):
    """
    Download ONE date window.
    If pagination approaches 10k, signal overflow.
    """
    query = build_query(categories, start_date, end_date)
    records = []
    start = 0

    while True:
        if start >= MAX_START:
            # arXiv hard limit reached
            return None

        feed = fetch_batch(query, start)

        if not feed.entries:
            break

        for entry in feed.entries:
            records.append({
                "paper_id": entry.id,
                "title": entry.title.replace("\n", " ").strip(), # type: ignore
                "abstract": entry.summary.replace("\n", " ").strip(), # type: ignore
                "published": entry.published[:10],
                "categories": [t["term"] for t in entry.tags]
            })

        start += MAX_RESULTS
        time.sleep(REQUEST_SLEEP)

    return pd.DataFrame(records)

def recursive_download(categories, start_dt, end_dt, min_days=7):
    """
    Recursively split windows if they exceed arXiv limits.
    """
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")

    print(f"📦 Attempting {start_str} → {end_str}")

    df = download_window(categories, start_str, end_str)

    if df is not None:
        return [df]

    # Window too large → split
    delta = (end_dt - start_dt).days
    if delta <= min_days:
        print(f"⚠️ Window too dense, keeping partial: {start_str} → {end_str}")
        return []

    mid_dt = start_dt + timedelta(days=delta // 2)

    return (
        recursive_download(categories, start_dt, mid_dt) +
        recursive_download(categories, mid_dt, end_dt)
    )

