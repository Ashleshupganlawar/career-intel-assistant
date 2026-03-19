"""Local fallback retriever over processed profiles and raw source text chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from job_intel.utils.text import slugify, tokenize


@dataclass
class RAGChunk:
    chunk_id: str
    company_slug: str
    source: str
    text: str


class LocalRAGRetriever:
    """Local retrieval over company intelligence docs.

    Priority order:
    1) processed_company_profiles/*.json
    2) raw_sources/*/*.txt
    """

    def __init__(
        self,
        profiles_dir: str = "data/processed_company_profiles",
        raw_sources_dir: str = "data/raw_sources",
    ):
        self.profiles_dir = Path(profiles_dir)
        self.raw_sources_dir = Path(raw_sources_dir)
        self.chunks: list[RAGChunk] = self._build_chunks()

    def query(self, question: str, company_filter: list[str] | None = None, top_k: int = 5) -> list[dict]:
        q_tokens = tokenize(question)
        if not q_tokens:
            return []

        filter_slugs = {slugify(c) for c in (company_filter or []) if c}

        scored: list[tuple[float, RAGChunk]] = []
        for chunk in self.chunks:
            if filter_slugs and chunk.company_slug not in filter_slugs:
                continue

            c_tokens = tokenize(chunk.text)
            if not c_tokens:
                continue

            overlap = len(q_tokens & c_tokens)
            if overlap == 0:
                continue

            base = overlap / max(1, len(q_tokens))
            company_boost = 0.15 if chunk.company_slug in filter_slugs else 0.0
            source_boost = 0.05 if chunk.source.startswith("profile:") else 0.0
            score = base + company_boost + source_boost
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict] = []
        for score, chunk in scored[:top_k]:
            out.append(
                {
                    "score": round(score, 4),
                    "company_slug": chunk.company_slug,
                    "source": chunk.source,
                    "text": chunk.text[:320],
                }
            )
        return out

    def _build_chunks(self) -> list[RAGChunk]:
        chunks = self._chunks_from_profiles()
        if chunks:
            return chunks
        return self._chunks_from_raw_sources()

    def _chunks_from_profiles(self) -> list[RAGChunk]:
        chunks: list[RAGChunk] = []
        if not self.profiles_dir.exists():
            return chunks

        for path in sorted(self.profiles_dir.glob("*_profile.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            slug = path.stem.replace("_profile", "")
            texts = []
            text_keys = [
                "company_overview",
                "official_hiring_process",
                "hiring_trends_summary",
                "salary_summary",
                "culture_summary_official",
                "culture_summary_community",
                "interview_experience_summary",
                "final_blended_insight",
                "notes",
            ]
            for key in text_keys:
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    texts.append((f"profile:{key}", val.strip()))

            for key in ["common_interview_topics", "reddit_common_themes", "reddit_red_flags", "reddit_positive_signals"]:
                val = payload.get(key)
                if isinstance(val, list) and val:
                    texts.append((f"profile:{key}", "; ".join(str(x) for x in val)))

            idx = 0
            for source, text in texts:
                for part in self._split_text(text):
                    idx += 1
                    chunks.append(
                        RAGChunk(
                            chunk_id=f"{slug}_{idx}",
                            company_slug=slug,
                            source=source,
                            text=part,
                        )
                    )
        return chunks

    def _chunks_from_raw_sources(self) -> list[RAGChunk]:
        chunks: list[RAGChunk] = []
        if not self.raw_sources_dir.exists():
            return chunks

        for company_dir in sorted(self.raw_sources_dir.iterdir()):
            if not company_dir.is_dir():
                continue
            slug = slugify(company_dir.name)
            idx = 0
            for path in sorted(company_dir.glob("*.txt")):
                try:
                    txt = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                cleaned = re.sub(r"\s+", " ", txt).strip()
                if not cleaned:
                    continue

                for part in self._split_text(cleaned):
                    idx += 1
                    chunks.append(
                        RAGChunk(
                            chunk_id=f"{slug}_{idx}",
                            company_slug=slug,
                            source=f"raw:{path.name}",
                            text=part,
                        )
                    )
        return chunks

    @staticmethod
    def _split_text(text: str, max_chars: int = 450) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        out: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            out.append(text[start:end])
            start = end
        return out
