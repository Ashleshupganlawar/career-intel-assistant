import json

from job_intel.insights import CompanyInsightsStore


def test_company_insight_load_path(tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    payload = {"company": "Amazon", "company_overview": "Test overview"}
    (profiles / "amazon_profile.json").write_text(json.dumps(payload), encoding="utf-8")

    store = CompanyInsightsStore(str(profiles))
    out = store.load_by_company("Amazon")
    assert out["company"] == "Amazon"
    assert out["company_overview"] == "Test overview"
