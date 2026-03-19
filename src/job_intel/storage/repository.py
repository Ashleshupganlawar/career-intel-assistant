"""Stores pipeline artifacts (candidate, jobs, matches) as JSON for reproducibility."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from job_intel.types import CandidateProfile, JobRecord, MatchResult


class JsonRepository:
    def __init__(self, base_dir: str = "data/cache"):
        self.base = Path(base_dir)
        self.jobs_dir = self.base / "jobs"
        self.candidates_dir = self.base / "candidates"
        self.matches_dir = self.base / "matches"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.matches_dir.mkdir(parents=True, exist_ok=True)

    def save_jobs(self, key: str, jobs: list[JobRecord]) -> Path:
        path = self.jobs_dir / f"{key}.json"
        self._write(path, [j.to_dict() for j in jobs])
        return path

    def save_candidate(self, key: str, candidate: CandidateProfile) -> Path:
        path = self.candidates_dir / f"{key}.json"
        self._write(path, candidate.to_dict())
        return path

    def save_matches(self, key: str, matches: list[MatchResult]) -> Path:
        path = self.matches_dir / f"{key}.json"
        self._write(path, [m.to_dict() for m in matches])
        return path

    @staticmethod
    def _write(path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
