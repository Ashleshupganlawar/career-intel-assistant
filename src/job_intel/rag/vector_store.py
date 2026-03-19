"""Embedding helpers and FAISS-backed vector index build/search utilities."""

from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VectorDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any]


class HFTextEmbedder:
    """Hugging Face-based embedding helper.

    Uses `sentence-transformers` locally when available.
    Falls back to `huggingface_hub.InferenceClient.feature_extraction`.
    """

    def __init__(self, model_name: str | None = None, token: str | None = None):
        self.model_name = model_name or os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.token = token or os.getenv("HF_TOKEN")
        self._local_model = None
        self._remote_client = None

        try:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.model_name)
        except Exception:
            self._local_model = None

        if self._local_model is None:
            try:
                from huggingface_hub import InferenceClient

                self._remote_client = InferenceClient(model=self.model_name, token=self.token)
            except Exception:
                self._remote_client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._local_model is not None:
            vectors = self._local_model.encode(texts, normalize_embeddings=True)
            return [list(map(float, row)) for row in vectors]

        if self._remote_client is not None:
            out: list[list[float]] = []
            for text in texts:
                feats = self._remote_client.feature_extraction(text)
                vec = self._pool_features(feats)
                out.append(vec)
            return out

        raise RuntimeError(
            "No embedding backend available. Install sentence-transformers or set HF token for Inference API."
        )

    @staticmethod
    def _pool_features(features: Any) -> list[float]:
        # Inference API can return either a single vector or token vectors.
        if isinstance(features, list) and features and isinstance(features[0], (int, float)):
            return [float(x) for x in features]

        if isinstance(features, list) and features and isinstance(features[0], list):
            # mean pool token vectors
            rows = features
            dim = len(rows[0])
            accum = [0.0] * dim
            count = 0
            for row in rows:
                if len(row) != dim:
                    continue
                count += 1
                for i, val in enumerate(row):
                    accum[i] += float(val)
            if count == 0:
                return [0.0] * dim
            return [x / count for x in accum]

        return []


class HashTextEmbedder:
    """Deterministic, offline embedder for fallback/index portability."""

    def __init__(self, dim: int = 384):
        self.dim = max(64, int(dim))
        self.model_name = f"local-hash-{self.dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


class FaissVectorDB:
    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "index.faiss"
        self.meta_path = self.persist_dir / "metadata.json"
        self.docs_path = self.persist_dir / "documents.json"
        self.config_path = self.persist_dir / "config.json"

    def build(self, docs: list[VectorDocument], embedder: HFTextEmbedder):
        if not docs:
            raise ValueError("No documents to index.")

        vectors = embedder.embed([d.text for d in docs])
        if not vectors or not vectors[0]:
            raise RuntimeError("Embedding output is empty.")

        import faiss  # lazy import
        import numpy as np

        matrix = np.array(vectors, dtype="float32")
        dim = matrix.shape[1]

        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(matrix)
        index.add(matrix)

        faiss.write_index(index, str(self.index_path))

        self.meta_path.write_text(
            json.dumps(
                {
                    "count": len(docs),
                    "dim": int(dim),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.docs_path.write_text(
            json.dumps(
                [
                    {
                        "doc_id": d.doc_id,
                        "text": d.text,
                        "metadata": d.metadata,
                    }
                    for d in docs
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        self.config_path.write_text(
            json.dumps({"embed_model": embedder.model_name}, indent=2),
            encoding="utf-8",
        )

    def search(self, query: str, embedder: HFTextEmbedder, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.index_path.exists() or not self.docs_path.exists():
            return []

        import faiss  # lazy import
        import numpy as np

        index = faiss.read_index(str(self.index_path))
        docs = json.loads(self.docs_path.read_text(encoding="utf-8"))

        q_vec = embedder.embed([query])
        if not q_vec or not q_vec[0]:
            return []

        q = np.array(q_vec, dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = index.search(q, top_k)

        out: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(docs):
                continue
            row = docs[idx]
            out.append(
                {
                    "score": float(score),
                    "doc_id": row.get("doc_id", ""),
                    "text": row.get("text", ""),
                    "metadata": row.get("metadata", {}),
                }
            )
        return out


def build_company_documents(
    profiles_dir: str = "data/processed_company_profiles",
    source_maps_dir: str = "data/source_maps",
    raw_sources_dir: str = "data/raw_sources",
    chunk_chars: int = 500,
) -> list[VectorDocument]:
    profiles_path = Path(profiles_dir)
    source_maps_path = Path(source_maps_dir)
    raw_sources_path = Path(raw_sources_dir)

    docs: list[VectorDocument] = []

    for profile_file in sorted(profiles_path.glob("*_profile.json")):
        company_slug = profile_file.stem.replace("_profile", "")

        try:
            profile = json.loads(profile_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        source_map_file = source_maps_path / f"{company_slug}.json"
        source_map = {}
        if source_map_file.exists():
            try:
                source_map = json.loads(source_map_file.read_text(encoding="utf-8"))
            except Exception:
                source_map = {}

        sections = _profile_sections(profile)
        chunk_index = 0
        for section_name, section_text in sections:
            for chunk in split_text(section_text, chunk_chars):
                chunk_index += 1
                docs.append(
                    VectorDocument(
                        doc_id=f"{company_slug}:{section_name}:{chunk_index}",
                        text=chunk,
                        metadata={
                            "company": profile.get("company", company_slug),
                            "company_slug": company_slug,
                            "industry": profile.get("industry", ""),
                            "section": section_name,
                            "last_updated": profile.get("last_updated", ""),
                            "official_sources": source_map.get("official_sources", {}),
                            "community_sources": source_map.get("community_sources", {}),
                        },
                    )
                )

    if docs:
        return docs

    # Fallback path: build docs from source maps + raw source text.
    for source_map_file in sorted(source_maps_path.glob("*.json")):
        company_slug = source_map_file.stem
        try:
            source_map = json.loads(source_map_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        raw_company_dir = raw_sources_path / company_slug
        raw_docs = []
        if raw_company_dir.exists():
            for raw_file in sorted(raw_company_dir.glob("*.txt")):
                try:
                    raw_text = raw_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                cleaned = re.sub(r"\s+", " ", raw_text).strip()
                if cleaned:
                    raw_docs.append((raw_file.name, cleaned))

        sections = _source_map_sections(source_map, raw_docs)
        chunk_index = 0
        for section_name, section_text in sections:
            for chunk in split_text(section_text, chunk_chars):
                chunk_index += 1
                docs.append(
                    VectorDocument(
                        doc_id=f"{company_slug}:{section_name}:{chunk_index}",
                        text=chunk,
                        metadata={
                            "company": source_map.get("company", company_slug),
                            "company_slug": company_slug,
                            "industry": source_map.get("industry", ""),
                            "section": section_name,
                            "last_updated": source_map.get("last_updated", ""),
                            "official_sources": source_map.get("official_sources", {}),
                            "community_sources": source_map.get("community_sources", {}),
                        },
                    )
                )

    return docs


def _profile_sections(profile: dict[str, Any]) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []

    text_fields = [
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
    for key in text_fields:
        val = profile.get(key)
        if isinstance(val, str) and val.strip():
            sections.append((key, re.sub(r"\s+", " ", val).strip()))

    list_fields = [
        "common_interview_topics",
        "reddit_common_themes",
        "reddit_red_flags",
        "reddit_positive_signals",
    ]
    for key in list_fields:
        val = profile.get(key)
        if isinstance(val, list) and val:
            text = "; ".join(str(x).strip() for x in val if str(x).strip())
            if text:
                sections.append((key, text))

    return sections


def _source_map_sections(source_map: dict[str, Any], raw_docs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []

    # Structured source-map context
    company = source_map.get("company", "")
    industry = source_map.get("industry", "")
    if company or industry:
        sections.append(("source_map_overview", f"company={company}; industry={industry}"))

    official = source_map.get("official_sources", {})
    if isinstance(official, dict) and official:
        parts = [f"{k}={v}" for k, v in official.items() if str(v).strip()]
        if parts:
            sections.append(("official_sources", "; ".join(parts)))

    community = source_map.get("community_sources", {})
    if isinstance(community, dict) and community:
        community_parts = []
        for key, val in community.items():
            if isinstance(val, list):
                rendered = ", ".join(
                    item.get("url", "") if isinstance(item, dict) else str(item)
                    for item in val
                )
                if rendered.strip():
                    community_parts.append(f"{key}={rendered}")
            elif str(val).strip():
                community_parts.append(f"{key}={val}")
        if community_parts:
            sections.append(("community_sources", "; ".join(community_parts)))

    # Raw crawled text context
    for filename, text in raw_docs:
        sections.append((f"raw_{filename}", text))

    return sections


def split_text(text: str, max_chars: int = 500) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        chunks.append(cleaned[start:end])
        start = end
    return chunks


def make_embedder(backend: str = "auto", model_name: str | None = None):
    backend = (backend or "auto").strip().lower()
    if backend == "hash":
        return HashTextEmbedder()

    if backend == "hf":
        return HFTextEmbedder(model_name=model_name)

    # auto: prefer HF, then fallback to local hash
    try:
        emb = HFTextEmbedder(model_name=model_name)
        probe = emb.embed(["embedder probe text"])
        if probe and probe[0]:
            return emb
    except Exception:
        pass
    return HashTextEmbedder()
