"""Job-board connectors that normalize external API/provider responses into JobRecord rows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import os
from typing import Any
from urllib.parse import urlparse

import requests

from job_intel.types import JobRecord
from job_intel.utils.text import normalize_spaces, tokenize


class JobSource(ABC):
    source_name: str

    @abstractmethod
    def fetch_jobs(self, query: str, location: str | None, time_window_hours: int) -> list[JobRecord]:
        raise NotImplementedError


class JobSpyDirectSource(JobSource):
    """Fetch jobs directly via python-jobspy (no MCP HTTP server required)."""

    source_name = "jobspy_direct"

    def __init__(self):
        self.site_names = [s.strip() for s in os.getenv("JOBSPY_DIRECT_SITE_NAMES", "indeed,linkedin").split(",") if s.strip()]
        self.results_wanted = int(os.getenv("JOBSPY_DIRECT_RESULTS_WANTED", "40"))
        self.country_indeed = os.getenv("JOBSPY_DIRECT_COUNTRY_INDEED", "usa")
        self.linkedin_fetch_description = os.getenv("JOBSPY_DIRECT_LINKEDIN_FETCH_DESCRIPTION", "0").strip() in {
            "1",
            "true",
            "yes",
        }

    def health_check(self) -> tuple[bool, str]:
        try:
            import jobspy  # noqa: F401

            return True, "jobspy_import_ok"
        except Exception as exc:
            return False, str(exc)

    def fetch_jobs(self, query: str, location: str | None, time_window_hours: int) -> list[JobRecord]:
        from jobspy import scrape_jobs

        jobs_df = scrape_jobs(
            site_name=self.site_names,
            search_term=query or "software engineer",
            location=location or "United States",
            results_wanted=max(1, self.results_wanted),
            hours_old=max(1, int(time_window_hours)),
            country_indeed=self.country_indeed,
            linkedin_fetch_description=self.linkedin_fetch_description,
        )
        if jobs_df is None or getattr(jobs_df, "empty", True):
            return []

        rows: list[JobRecord] = []
        for _, row in jobs_df.iterrows():
            data = row.to_dict()
            title = normalize_spaces(str(data.get("title", "")))
            company = normalize_spaces(str(data.get("company", "")))
            loc = normalize_spaces(str(data.get("location", "")))
            desc = normalize_spaces(str(data.get("description", "")))
            url = str(data.get("job_url", data.get("url", "")) or "")
            site = str(data.get("site", self.source_name) or self.source_name)
            posted_raw = str(data.get("date_posted", data.get("posted_at", "")) or "")
            posted = _normalize_date(posted_raw)
            if not title and not company:
                continue
            rows.append(
                JobRecord(
                    id=f"{self.source_name}:{data.get('id', title + company + loc)}",
                    title=title or "Unknown role",
                    company=company or "Unknown company",
                    location=loc or "Unknown",
                    posted_at=posted,
                    description=desc,
                    skills=[],
                    source_url=url,
                    source_name=site,
                )
            )
        return rows


class JobSpyMCPSource(JobSource):
    """Fetch jobs from a locally hosted JobSpy MCP server over HTTP."""

    source_name = "jobspy_mcp"

    def __init__(self):
        # Preferred MCP web endpoint is /mcp/request.
        self.url = os.getenv("JOBSPY_MCP_URL", "http://localhost:9423/mcp/request")
        self.site_names = os.getenv("JOBSPY_SITE_NAMES", "linkedin,indeed")
        self.results_wanted = int(os.getenv("JOBSPY_RESULTS_WANTED", "40"))
        self.country_indeed = os.getenv("JOBSPY_COUNTRY_INDEED", "USA")
        self.mode = os.getenv("JOBSPY_MCP_MODE", "auto").strip().lower()  # auto|mcp|search|api

    def configured(self) -> bool:
        return bool(self.url)

    def fetch_jobs(self, query: str, location: str | None, time_window_hours: int) -> list[JobRecord]:
        if not self.configured():
            return []

        params = {
            "search_term": query or "software engineer",
            "location": location or "United States",
            "site_names": self.site_names,
            "results_wanted": self.results_wanted,
            "hours_old": max(1, int(time_window_hours)),
            "country_indeed": self.country_indeed,
            "format": "json",
        }
        rows_in = self._request_rows(params)
        if not isinstance(rows_in, list):
            return []

        out: list[JobRecord] = []
        for item in rows_in:
            title = normalize_spaces(item.get("title", item.get("job_title", "")))
            company = normalize_spaces(item.get("company", item.get("company_name", "")))
            loc = normalize_spaces(item.get("location", item.get("job_location", "")))
            url = item.get("job_url", item.get("url", "")) or ""
            posted = _normalize_date(
                item.get("date_posted")
                or item.get("posted_at")
                or item.get("date")
                or ""
            )
            desc = normalize_spaces(
                item.get("description")
                or item.get("job_description")
                or ""
            )
            source_name = str(item.get("site", item.get("source", self.source_name))).strip() or self.source_name
            if not title and not company:
                continue
            out.append(
                JobRecord(
                    id=f"{self.source_name}:{item.get('id', title + company + loc)}",
                    title=title or "Unknown role",
                    company=company or "Unknown company",
                    location=loc or "Unknown",
                    posted_at=posted,
                    description=desc,
                    skills=[],
                    source_url=url,
                    source_name=source_name,
                )
            )
        return out

    def health_check(self) -> tuple[bool, str]:
        """Best-effort MCP reachability check for diagnostics."""
        parsed = urlparse(self.url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        connect_url = f"{base}/mcp/connect"
        try:
            r = requests.get(connect_url, timeout=8)
            if r.status_code < 500:
                return True, f"ok:{connect_url}"
            return False, f"http_{r.status_code}:{connect_url}"
        except Exception as exc:
            return False, str(exc)

    def _request_rows(self, params: dict[str, Any]) -> Any:
        parsed = urlparse(self.url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        endpoints: list[tuple[str, str]] = []
        if self.mode in {"mcp", "auto"}:
            endpoints.append((f"{base}/mcp/request", "mcp"))
        if self.mode in {"search", "auto"}:
            endpoints.append((f"{base}/search", "search"))
        if self.mode in {"api", "auto"}:
            endpoints.append((f"{base}/api", "api"))

        # If caller provided a concrete URL path, try it first.
        if parsed.path and parsed.path not in {"/", ""}:
            concrete = f"{base}{parsed.path}"
            if concrete not in [u for u, _ in endpoints]:
                endpoints.insert(0, (concrete, "concrete"))

        last_exc: Exception | None = None
        for endpoint, mode in endpoints:
            try:
                if mode in {"mcp", "concrete"} and endpoint.endswith("/mcp/request"):
                    payload = {"tool": "search_jobs", "params": params}
                    resp = requests.post(endpoint, json=payload, timeout=45)
                elif mode in {"search", "concrete"} and endpoint.endswith("/search"):
                    resp = requests.get(endpoint, params=params, timeout=45)
                else:
                    payload = {"method": "search_jobs", "params": params}
                    resp = requests.post(endpoint, json=payload, timeout=45)

                resp.raise_for_status()
                body = resp.json()
                rows_in = body.get("result", body)
                if isinstance(rows_in, dict):
                    rows_in = rows_in.get("jobs", rows_in.get("data", []))
                return rows_in
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc:
            raise last_exc
        return []


class ArbeitnowSource(JobSource):
    source_name = "arbeitnow"
    url = "https://www.arbeitnow.com/api/job-board-api"

    def fetch_jobs(self, query: str, location: str | None, time_window_hours: int) -> list[JobRecord]:
        resp = requests.get(self.url, timeout=25)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        rows: list[JobRecord] = []
        q_tokens = tokenize(query)

        for item in data:
            title = normalize_spaces(item.get("title", ""))
            company = normalize_spaces(item.get("company_name", ""))
            description = normalize_spaces(item.get("description", ""))
            loc = normalize_spaces(item.get("location", ""))

            text = f"{title} {description}"
            if q_tokens and not (q_tokens & tokenize(text)):
                continue
            if location and location.lower() not in loc.lower():
                continue

            tags = item.get("tags", []) or []
            posted = _normalize_date(item.get("created_at") or "")
            rows.append(
                JobRecord(
                    id=f"arbeitnow:{item.get('slug', title)}",
                    title=title,
                    company=company,
                    location=loc or "Unknown",
                    posted_at=posted,
                    description=description,
                    skills=[str(t).strip().lower() for t in tags if str(t).strip()],
                    source_url=item.get("url", ""),
                    source_name=self.source_name,
                )
            )

        return rows


class RemotiveSource(JobSource):
    source_name = "remotive"
    url = "https://remotive.com/api/remote-jobs"

    def fetch_jobs(self, query: str, location: str | None, time_window_hours: int) -> list[JobRecord]:
        params = {"search": query} if query else None
        resp = requests.get(self.url, params=params, timeout=25)
        resp.raise_for_status()
        jobs = resp.json().get("jobs", [])

        rows: list[JobRecord] = []
        for item in jobs:
            title = normalize_spaces(item.get("title", ""))
            company = normalize_spaces(item.get("company_name", ""))
            loc = normalize_spaces(item.get("candidate_required_location", "Remote"))
            if location and location.lower() not in loc.lower() and "remote" not in loc.lower():
                continue

            skills = [str(x).strip().lower() for x in (item.get("tags") or []) if str(x).strip()]
            rows.append(
                JobRecord(
                    id=f"remotive:{item.get('id', title)}",
                    title=title,
                    company=company,
                    location=loc,
                    posted_at=_normalize_date(item.get("publication_date") or ""),
                    description=normalize_spaces(item.get("description", "")),
                    skills=skills,
                    source_url=item.get("url", ""),
                    source_name=self.source_name,
                )
            )

        return rows


class TheMuseSource(JobSource):
    source_name = "themuse"
    url = "https://www.themuse.com/api/public/jobs"

    def fetch_jobs(self, query: str, location: str | None, time_window_hours: int) -> list[JobRecord]:
        q_tokens = tokenize(query)
        rows: list[JobRecord] = []
        page = 0
        max_pages = 2

        while page < max_pages:
            resp = requests.get(self.url, params={"page": page}, timeout=25)
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("results", []) or []
            if not items:
                break

            for item in items:
                title = normalize_spaces(item.get("name", ""))
                company = normalize_spaces((item.get("company") or {}).get("name", ""))
                locations = item.get("locations", []) or []
                loc = normalize_spaces(", ".join(x.get("name", "") for x in locations if x.get("name")))
                levels = item.get("levels", []) or []
                categories = item.get("categories", []) or []
                tags = [normalize_spaces(x.get("name", "")).lower() for x in (levels + categories)]
                desc = normalize_spaces(item.get("contents", ""))

                if q_tokens and not (q_tokens & tokenize(f"{title} {desc}")):
                    continue
                if location and loc and location.lower() not in loc.lower():
                    continue

                rows.append(
                    JobRecord(
                        id=f"themuse:{item.get('id', title)}",
                        title=title,
                        company=company,
                        location=loc or "Unknown",
                        posted_at=_normalize_date(item.get("publication_date") or ""),
                        description=desc,
                        skills=[t for t in tags if t],
                        source_url=item.get("refs", {}).get("landing_page", "") or item.get("short_name", ""),
                        source_name=self.source_name,
                    )
                )
            page += 1
        return rows


class AdzunaSource(JobSource):
    source_name = "adzuna"

    def __init__(self, country: str = "us"):
        self.country = os.getenv("ADZUNA_COUNTRY", country)
        self.app_id = os.getenv("ADZUNA_APP_ID", "")
        self.app_key = os.getenv("ADZUNA_APP_KEY", "")
        self.url = f"https://api.adzuna.com/v1/api/jobs/{self.country}/search/1"

    def configured(self) -> bool:
        return bool(self.app_id and self.app_key)

    def fetch_jobs(self, query: str, location: str | None, time_window_hours: int) -> list[JobRecord]:
        if not self.configured():
            return []

        max_days = max(1, int(time_window_hours / 24) + 1)
        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "what": query or "software engineer",
            "results_per_page": 50,
            "max_days_old": max_days,
            "content-type": "application/json",
        }
        if location:
            params["where"] = location

        resp = requests.get(self.url, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json().get("results", []) or []
        rows: list[JobRecord] = []
        for item in data:
            title = normalize_spaces(item.get("title", ""))
            company = normalize_spaces((item.get("company") or {}).get("display_name", ""))
            loc = normalize_spaces((item.get("location") or {}).get("display_name", ""))
            desc = normalize_spaces(item.get("description", ""))
            rows.append(
                JobRecord(
                    id=f"adzuna:{item.get('id', title)}",
                    title=title,
                    company=company,
                    location=loc or "Unknown",
                    posted_at=_normalize_date(item.get("created") or ""),
                    description=desc,
                    skills=[],
                    source_url=item.get("redirect_url", ""),
                    source_name=self.source_name,
                )
            )
        return rows


class MockSource(JobSource):
    source_name = "mock"

    def fetch_jobs(self, query: str, location: str | None, time_window_hours: int) -> list[JobRecord]:
        now = datetime.now(timezone.utc).isoformat()
        seed = [
            JobRecord(
                id="mock:1",
                title="Machine Learning Engineer",
                company="Meta",
                location="Remote",
                posted_at=now,
                description="Build ML ranking systems with Python, SQL, and PyTorch.",
                skills=["python", "sql", "pytorch", "ml"],
                source_url="https://example.com/mock-1",
                source_name=self.source_name,
            ),
            JobRecord(
                id="mock:2",
                title="Data Scientist",
                company="Amazon",
                location="Seattle, WA",
                posted_at=now,
                description="Experimentation, statistics, causal inference, and dashboards.",
                skills=["python", "statistics", "sql", "experimentation"],
                source_url="https://example.com/mock-2",
                source_name=self.source_name,
            ),
        ]
        if not query:
            return seed
        q_tokens = tokenize(query)
        return [r for r in seed if q_tokens & tokenize(r.title + " " + r.description)]



def _normalize_date(value: Any) -> str:
    if value is None or value == "":
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except Exception:
            return datetime.now(timezone.utc).isoformat()
    value = str(value)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()
