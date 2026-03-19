"""Loads processed company profile JSON files for culture/interview/backstory lookups."""

from __future__ import annotations

import json
from pathlib import Path

from job_intel.utils.text import slugify


class CompanyInsightsStore:
    def __init__(self, profiles_dir: str = "data/processed_company_profiles"):
        self.profiles_dir = Path(profiles_dir)

    def load_by_company(self, company_name: str) -> dict:
        slug = slugify(company_name)
        return self.load_by_slug(slug)

    def load_by_slug(self, slug: str) -> dict:
        path = self.profiles_dir / f"{slug}_profile.json"
        if not path.exists():
            return {
                "company": slug,
                "status": "missing",
                "message": "No generated company profile found yet. Run the data pipeline first.",
            }
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
