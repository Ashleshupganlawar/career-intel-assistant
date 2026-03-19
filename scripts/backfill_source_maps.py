"""Regenerates/repairs company source-map JSON files from available raw inputs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

CSV_PATH = Path("data/input/list_of_companies.csv")
SOURCE_MAPS_DIR = Path("data/source_maps")
SOURCE_MAPS_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    import re

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required = {"company", "industry"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV must contain columns: {required}")

    created = 0
    for _, row in df.iterrows():
        company = str(row["company"]).strip()
        industry = str(row["industry"]).strip()
        slug = slugify(company)
        path = SOURCE_MAPS_DIR / f"{slug}.json"
        if path.exists():
            continue

        payload = {
            "company": company,
            "industry": industry,
            "official_sources": {
                "careers_url": "",
                "jobs_url": "",
                "hiring_process_url": "",
                "interview_structure_url": "",
                "culture_url": "",
                "faq_url": "",
            },
            "community_sources": {
                "salary_source": [],
                "review_source": [],
                "reddit_threads": [],
            },
            "last_updated": time.strftime("%Y-%m-%d"),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        created += 1

    print(f"[OK] Backfilled missing source maps: {created}")


if __name__ == "__main__":
    main()
