"""Hybrid matching engine combining lexical and vector-style similarity scoring."""

from __future__ import annotations

import math
import re
from collections import Counter

from job_intel.types import CandidateProfile, JobRecord, MatchResult, RankedJob
from job_intel.utils.text import slugify, tokenize


class HybridMatcher:
    def __init__(self, lexical_weight: float = 0.45, embedding_weight: float = 0.55):
        total = lexical_weight + embedding_weight
        self.lexical_weight = lexical_weight / total
        self.embedding_weight = embedding_weight / total

    def rank_jobs(self, candidate: CandidateProfile, jobs: list[JobRecord], top_k: int = 10) -> list[RankedJob]:
        if not jobs:
            return []

        embedding_scores = self._embedding_scores(candidate, jobs)
        ranked: list[RankedJob] = []

        for idx, job in enumerate(jobs):
            lexical = self._lexical_score(candidate, job)
            embedding = embedding_scores[idx]
            overall = round((self.lexical_weight * lexical) + (self.embedding_weight * embedding), 4)
            explanation = self._build_explanation(candidate, job, lexical, embedding)
            match = MatchResult(
                job_id=job.id,
                overall_score=overall,
                lexical_score=round(lexical, 4),
                embedding_score=round(embedding, 4),
                explanation=explanation,
                company_slug=slugify(job.company),
            )
            ranked.append(RankedJob(job=job, match=match))

        ranked.sort(key=lambda x: x.match.overall_score, reverse=True)
        return ranked[:top_k]

    def _embedding_scores(self, candidate: CandidateProfile, jobs: list[JobRecord]) -> list[float]:
        profile_text = " ".join([candidate.raw_text, " ".join(candidate.parsed_skills), " ".join(candidate.parsed_roles)])
        job_texts = [f"{j.title} {j.description} {' '.join(j.skills)}" for j in jobs]
        corpus_tokens = [set(self._term_list(profile_text))] + [set(self._term_list(t)) for t in job_texts]
        vocab = sorted(set().union(*corpus_tokens))
        if not vocab:
            return [0.0 for _ in jobs]

        # Smooth IDF to avoid zero division and stabilize small corpora.
        n_docs = len(corpus_tokens)
        df = {term: sum(1 for doc in corpus_tokens if term in doc) for term in vocab}
        idf = {term: math.log((1 + n_docs) / (1 + df[term])) + 1 for term in vocab}

        def tfidf_vector(text: str) -> dict[str, float]:
            terms = self._term_list(text)
            counts = Counter(terms)
            length = max(1, sum(counts.values()))
            return {term: (counts.get(term, 0) / length) * idf[term] for term in vocab}

        profile_vec = tfidf_vector(profile_text)
        out: list[float] = []
        for text in job_texts:
            job_vec = tfidf_vector(text)
            out.append(self._cosine(profile_vec, job_vec))
        return out

    @staticmethod
    def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        dot = sum(vec_a[k] * vec_b[k] for k in vec_a.keys())
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(max(0.0, min(1.0, dot / (norm_a * norm_b))))

    @staticmethod
    def _term_list(text: str) -> list[str]:
        return re.findall(r"[a-z0-9+#.]+", (text or "").lower())

    def _lexical_score(self, candidate: CandidateProfile, job: JobRecord) -> float:
        candidate_tokens = tokenize(" ".join(candidate.parsed_skills + candidate.parsed_roles + [candidate.raw_text]))
        job_tokens = tokenize(" ".join([job.title, job.description, " ".join(job.skills), job.location]))

        overlap = candidate_tokens & job_tokens
        denom = max(1, min(len(candidate_tokens), len(job_tokens)))
        skill_overlap = len(set(candidate.parsed_skills) & set(job.skills))
        role_hit = any(role in (job.title + " " + job.description).lower() for role in candidate.parsed_roles)

        score = (len(overlap) / denom) * 0.7
        score += min(1.0, skill_overlap / 4) * 0.2
        score += 0.1 if role_hit else 0.0
        return float(max(0.0, min(1.0, score)))

    def _build_explanation(self, candidate: CandidateProfile, job: JobRecord, lexical: float, embedding: float) -> str:
        matched_skills = sorted(set(candidate.parsed_skills) & set(job.skills))[:5]
        skill_text = ", ".join(matched_skills) if matched_skills else "general profile fit"
        return (
            f"Matched on {skill_text}; lexical={lexical:.2f}, semantic={embedding:.2f}. "
            f"Role considered: {job.title} at {job.company}."
        )
