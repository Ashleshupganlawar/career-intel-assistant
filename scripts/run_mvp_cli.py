"""Runs the MVP pipeline from CLI for resume -> jobs -> ranking -> summary output."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_intel.insights import CompanyInsightsStore
from job_intel.jobs import JobAggregator
from job_intel.matching import HybridMatcher
from job_intel.resume import ResumeParser
from job_intel.storage import JsonRepository


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("resume_path")
    parser.add_argument("--query", default="Machine Learning Engineer")
    parser.add_argument("--location", default="Remote")
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    resume_bytes = Path(args.resume_path).read_bytes()

    candidate = ResumeParser().parse_upload(Path(args.resume_path).name, resume_bytes)
    jobs = JobAggregator().fetch_jobs(query=args.query, location=args.location, time_window_hours=args.hours)
    ranked = HybridMatcher().rank_jobs(candidate, jobs, top_k=args.top_k)

    run_id = Path(args.resume_path).stem
    repo = JsonRepository()
    repo.save_candidate(run_id, candidate)
    repo.save_jobs(run_id, [r.job for r in ranked])
    repo.save_matches(run_id, [r.match for r in ranked])

    insights_store = CompanyInsightsStore()
    out = []
    for r in ranked:
        insight = insights_store.load_by_company(r.job.company)
        out.append(
            {
                "job": r.job.to_dict(),
                "match": r.match.to_dict(),
                "insight_summary": {
                    "company_overview": insight.get("company_overview", ""),
                    "final_blended_insight": insight.get("final_blended_insight", ""),
                },
            }
        )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
