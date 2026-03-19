import json

from job_intel.rag import LocalRAGRetriever


def test_rag_retriever_profiles_priority(tmp_path):
    profiles = tmp_path / "profiles"
    raw = tmp_path / "raw"
    profiles.mkdir()
    raw.mkdir()

    payload = {
        "company_overview": "Company builds cloud services.",
        "final_blended_insight": "Interview loop has coding + system design with behavioral round.",
    }
    (profiles / "amazon_profile.json").write_text(json.dumps(payload), encoding="utf-8")

    retriever = LocalRAGRetriever(str(profiles), str(raw))
    rows = retriever.query("system design interview", company_filter=["Amazon"], top_k=3)
    assert rows
    assert rows[0]["company_slug"] == "amazon"


def test_rag_retriever_raw_fallback(tmp_path):
    profiles = tmp_path / "profiles"
    raw = tmp_path / "raw"
    (raw / "meta").mkdir(parents=True)
    profiles.mkdir()

    (raw / "meta" / "official_hiring_process.txt").write_text(
        "Meta process includes recruiter screen, coding rounds, and behavioral interview.",
        encoding="utf-8",
    )

    retriever = LocalRAGRetriever(str(profiles), str(raw))
    rows = retriever.query("coding rounds interview", company_filter=["Meta"], top_k=3)
    assert rows
    assert rows[0]["company_slug"] == "meta"
