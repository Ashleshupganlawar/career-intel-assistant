from job_intel.matching import HybridMatcher
from job_intel.resume.parser import ResumeParser
from job_intel.types import JobRecord


def test_resume_to_match_deterministic():
    parser = ResumeParser()
    candidate = parser.parse_upload(
        "resume.txt",
        b"AI Engineer with 4 years of experience. Skills: Python, SQL, AWS, machine learning.",
    )

    jobs = [
        JobRecord(
            id="a",
            title="AI Engineer",
            company="Amazon",
            location="Remote",
            posted_at="2026-03-14T00:00:00+00:00",
            description="Build AI products with Python and AWS",
            skills=["python", "aws", "sql"],
            source_url="x",
            source_name="mock",
        ),
        JobRecord(
            id="b",
            title="Sales Manager",
            company="Y",
            location="Chicago",
            posted_at="2026-03-14T00:00:00+00:00",
            description="Quota carrying role",
            skills=["sales"],
            source_url="y",
            source_name="mock",
        ),
    ]

    ranked = HybridMatcher().rank_jobs(candidate, jobs, top_k=2)
    assert ranked[0].job.id == "a"
    assert ranked[0].match.overall_score > ranked[1].match.overall_score
