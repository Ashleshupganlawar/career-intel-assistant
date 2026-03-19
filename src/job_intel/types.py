"""Shared dataclasses for normalized jobs, candidate profiles, and match outputs."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any


@dataclass
class JobRecord:
    id: str
    title: str
    company: str
    location: str
    posted_at: str
    description: str
    skills: list[str]
    source_url: str
    source_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateProfile:
    raw_text: str
    parsed_skills: list[str]
    parsed_roles: list[str]
    years_experience: float | None
    preferred_locations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    job_id: str
    overall_score: float
    lexical_score: float
    embedding_score: float
    explanation: str
    company_slug: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RankedJob:
    job: JobRecord
    match: MatchResult



def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
