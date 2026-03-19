import json

from job_intel.rag.vector_store import build_company_documents, split_text


def test_split_text_chunks():
    text = "a" * 1200
    chunks = split_text(text, max_chars=500)
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == 1200


def test_build_company_documents_from_profile(tmp_path):
    profiles = tmp_path / "profiles"
    source_maps = tmp_path / "source_maps"
    profiles.mkdir()
    source_maps.mkdir()

    payload = {
        "company": "Amazon",
        "industry": "Technology",
        "company_overview": "Amazon overview text",
        "official_hiring_process": "Structured interview process",
        "common_interview_topics": ["Leadership Principles", "System design"],
        "last_updated": "2026-03-14",
    }
    (profiles / "amazon_profile.json").write_text(json.dumps(payload), encoding="utf-8")
    (source_maps / "amazon.json").write_text(
        json.dumps({"official_sources": {"careers_url": "https://example.com"}}),
        encoding="utf-8",
    )

    docs = build_company_documents(str(profiles), str(source_maps), chunk_chars=200)
    assert docs
    assert docs[0].metadata["company_slug"] == "amazon"
    assert "section" in docs[0].metadata


def test_build_company_documents_fallback_from_source_maps_and_raw(tmp_path):
    profiles = tmp_path / "profiles"
    source_maps = tmp_path / "source_maps"
    raw = tmp_path / "raw_sources"
    profiles.mkdir()
    source_maps.mkdir()
    (raw / "amazon").mkdir(parents=True)

    (source_maps / "amazon.json").write_text(
        json.dumps(
            {
                "company": "Amazon",
                "industry": "Technology",
                "official_sources": {"careers_url": "https://www.amazon.jobs"},
                "community_sources": {"reddit_threads": ["https://reddit.com/test"]},
                "last_updated": "2026-03-14",
            }
        ),
        encoding="utf-8",
    )
    (raw / "amazon" / "official_hiring_process.txt").write_text(
        "Interview loop, leadership principles, and behavioral process.",
        encoding="utf-8",
    )

    docs = build_company_documents(str(profiles), str(source_maps), str(raw), chunk_chars=200)
    assert docs
    assert any(d.metadata["company_slug"] == "amazon" for d in docs)
    assert any(str(d.metadata.get("section", "")).startswith("raw_") for d in docs)
