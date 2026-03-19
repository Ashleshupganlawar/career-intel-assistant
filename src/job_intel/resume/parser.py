"""Parses uploaded resume files and extracts normalized candidate profile fields."""

from __future__ import annotations

import io
import re
from dataclasses import asdict

from job_intel.types import CandidateProfile
from job_intel.utils.text import COMMON_SKILLS, ROLE_HINTS, normalize_spaces, tokenize


class ResumeParser:
    def extract_text(self, filename: str, raw_bytes: bytes) -> str:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"
        if ext == "pdf":
            return self._from_pdf(raw_bytes)
        if ext == "docx":
            return self._from_docx(raw_bytes)
        return raw_bytes.decode("utf-8", errors="ignore")

    def parse_text(self, text: str) -> CandidateProfile:
        return self._extract_profile(normalize_spaces(text))

    def parse_upload(self, filename: str, raw_bytes: bytes) -> CandidateProfile:
        text = self.extract_text(filename, raw_bytes)
        return self.parse_text(text)

    def _from_pdf(self, raw_bytes: bytes) -> str:
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw_bytes))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return raw_bytes.decode("utf-8", errors="ignore")

    def _from_docx(self, raw_bytes: bytes) -> str:
        try:
            import docx

            doc = docx.Document(io.BytesIO(raw_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return raw_bytes.decode("utf-8", errors="ignore")

    def _extract_profile(self, text: str) -> CandidateProfile:
        tokens = tokenize(text)
        parsed_skills = sorted([skill for skill in COMMON_SKILLS if skill in tokens])

        lowered = text.lower()
        parsed_roles = sorted([role for role in ROLE_HINTS if role in lowered])

        years_experience = self._extract_years(lowered)

        preferred_locations = []
        for marker in ["location", "based in", "open to relocate", "remote"]:
            if marker in lowered:
                preferred_locations.append(marker)
        preferred_locations = sorted(set(preferred_locations))

        return CandidateProfile(
            raw_text=text,
            parsed_skills=parsed_skills,
            parsed_roles=parsed_roles,
            years_experience=years_experience,
            preferred_locations=preferred_locations,
        )

    @staticmethod
    def _extract_years(text: str) -> float | None:
        patterns = [
            r"(\d+(?:\.\d+)?)\+?\s+years?\s+of\s+experience",
            r"experience\s+of\s+(\d+(?:\.\d+)?)\+?\s+years?",
            r"(\d+(?:\.\d+)?)\+?\s+yrs",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
        return None


__all__ = ["ResumeParser"]
