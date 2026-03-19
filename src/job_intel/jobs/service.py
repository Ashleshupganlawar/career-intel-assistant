"""Aggregates multiple job connectors, filters by recency, and returns diagnostics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

from job_intel.types import JobRecord

from .connectors import (
    AdzunaSource,
    ArbeitnowSource,
    JobSource,
    JobSpyDirectSource,
    JobSpyMCPSource,
    MockSource,
    RemotiveSource,
    TheMuseSource,
)


class JobAggregator:
    def __init__(self, sources: list[JobSource] | None = None):
        self.sources = sources or self._default_sources()

    def fetch_jobs(self, query: str, location: str | None, time_window_hours: int = 48) -> list[JobRecord]:
        jobs, _diag = self.fetch_jobs_with_diagnostics(query, location, time_window_hours)
        return jobs

    def fetch_jobs_with_diagnostics(
        self,
        query: str,
        location: str | None,
        time_window_hours: int = 48,
    ) -> tuple[list[JobRecord], list[dict[str, str | int]]]:
        all_rows: list[JobRecord] = []
        diagnostics: list[dict[str, str | int]] = []
        for source in self.sources:
            try:
                if hasattr(source, "health_check"):
                    ok, detail = source.health_check()
                    if not ok:
                        diagnostics.append({"source": source.source_name, "status": "unreachable", "error": detail})
                        continue
                rows = source.fetch_jobs(query=query, location=location, time_window_hours=time_window_hours)
                all_rows.extend(rows)
                diagnostics.append({"source": source.source_name, "status": "ok", "rows": len(rows)})
            except Exception as exc:
                # keep app resilient if one source fails
                print(f"[WARN] source failed={source.source_name} err={exc}")
                diagnostics.append({"source": source.source_name, "status": "error", "error": str(exc)})

        if not all_rows:
            # Hard fallback so UI is never empty during API/source outages.
            all_rows = MockSource().fetch_jobs(query=query, location=location, time_window_hours=time_window_hours)
            if not all_rows:
                all_rows = MockSource().fetch_jobs(query="", location=location, time_window_hours=time_window_hours)
            diagnostics.append({"source": "mock", "status": "fallback", "rows": len(all_rows)})

        filtered = self._filter_by_time(all_rows, time_window_hours)
        deduped = self._dedupe(filtered)
        if not deduped:
            # Final fallback: return seed mock set rather than zero results.
            fallback = MockSource().fetch_jobs(query="", location=location, time_window_hours=time_window_hours)
            deduped = self._dedupe(self._filter_by_time(fallback, time_window_hours))
            diagnostics.append({"source": "mock", "status": "fallback_final", "rows": len(deduped)})
        return deduped, diagnostics

    @staticmethod
    def _default_sources() -> list[JobSource]:
        base: list[JobSource] = []
        if os.getenv("JOBSPY_DIRECT_ENABLED", "0").strip() in {"1", "true", "yes"}:
            base.append(JobSpyDirectSource())
        if os.getenv("JOBSPY_MCP_ENABLED", "0").strip() in {"1", "true", "yes"}:
            base.append(JobSpyMCPSource())
        base.extend([ArbeitnowSource(), RemotiveSource(), TheMuseSource()])
        if os.getenv("ADZUNA_APP_ID") and os.getenv("ADZUNA_APP_KEY"):
            base.append(AdzunaSource())
        return base

    @staticmethod
    def _filter_by_time(rows: list[JobRecord], hours: int) -> list[JobRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        out: list[JobRecord] = []
        for row in rows:
            try:
                posted_raw = str(row.posted_at or "")
                posted = datetime.fromisoformat(posted_raw.replace("Z", "+00:00"))
                # Some providers return naive timestamps; treat them as UTC.
                if posted.tzinfo is None:
                    posted = posted.replace(tzinfo=timezone.utc)
            except Exception:
                out.append(row)
                continue
            if posted >= cutoff:
                out.append(row)
        return out

    @staticmethod
    def _dedupe(rows: list[JobRecord]) -> list[JobRecord]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[JobRecord] = []
        for row in rows:
            key = (row.title.lower(), row.company.lower(), row.location.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped
