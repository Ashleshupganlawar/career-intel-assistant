from job_intel.matching import HybridMatcher
from job_intel.types import CandidateProfile, JobRecord


def test_hybrid_matcher_scores_sorted():
    candidate = CandidateProfile(
        raw_text="Machine learning engineer with Python SQL PyTorch",
        parsed_skills=["python", "sql", "pytorch"],
        parsed_roles=["machine learning engineer"],
        years_experience=3,
        preferred_locations=["remote"],
    )

    jobs = [
        JobRecord(
            id="1",
            title="ML Engineer",
            company="Meta",
            location="Remote",
            posted_at="2026-03-14T00:00:00+00:00",
            description="Build ML systems in Python and PyTorch",
            skills=["python", "pytorch"],
            source_url="u1",
            source_name="mock",
        ),
        JobRecord(
            id="2",
            title="Frontend Engineer",
            company="X",
            location="Remote",
            posted_at="2026-03-14T00:00:00+00:00",
            description="React and CSS",
            skills=["react"],
            source_url="u2",
            source_name="mock",
        ),
    ]

    ranked = HybridMatcher().rank_jobs(candidate, jobs, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].match.overall_score >= ranked[1].match.overall_score
    assert ranked[0].job.id == "1"
