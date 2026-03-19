"""Quick diagnostic script to test job connectors and print source health."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_intel.jobs import JobAggregator


def main():
    agg = JobAggregator()
    print("Configured sources:", [s.source_name for s in agg.sources])
    jobs, diagnostics = agg.fetch_jobs_with_diagnostics(
        query="data analyst",
        location="Remote",
        time_window_hours=24,
    )
    print(f"Fetched jobs (24h): {len(jobs)}")
    print("Diagnostics:")
    for d in diagnostics:
        src = d.get("source", "unknown")
        status = d.get("status", "unknown")
        rows = d.get("rows")
        err = d.get("error")
        if err:
            print(f"- {src}: {status} | error={err}")
        else:
            print(f"- {src}: {status} | rows={rows}")


if __name__ == "__main__":
    main()
