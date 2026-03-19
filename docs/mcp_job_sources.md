# MCP Job Sources

This project now supports an optional MCP-backed job source via `JobSpyMCPSource`.
It also supports direct in-process JobSpy via `JobSpyDirectSource` (recommended if MCP HTTP is unavailable).

## 0) JobSpy Direct (no HTTP MCP required)

Enable in `.env`:

```env
JOBSPY_DIRECT_ENABLED=1
JOBSPY_DIRECT_SITE_NAMES=indeed,linkedin
JOBSPY_DIRECT_RESULTS_WANTED=40
JOBSPY_DIRECT_COUNTRY_INDEED=usa
JOBSPY_DIRECT_LINKEDIN_FETCH_DESCRIPTION=0
```

Install dependency:

```bash
.venv/bin/python -m pip install python-jobspy
```

## 1) JobSpy MCP (optional HTTP mode)

Repo: `https://github.com/borgius/jobspy-mcp-server`

Why first:
- Broad coverage (LinkedIn/Indeed and more, depending on server config).
- Best fit for your "fetch latest jobs" requirement.

Enable in `.env`:

```env
JOBSPY_MCP_ENABLED=1
JOBSPY_MCP_URL=http://localhost:9423/mcp/request
JOBSPY_MCP_MODE=auto
JOBSPY_SITE_NAMES=linkedin,indeed
JOBSPY_RESULTS_WANTED=40
JOBSPY_COUNTRY_INDEED=USA
```

Recommended local MCP startup:

```bash
export HOST=0.0.0.0
export PORT=9423
export ENABLE_SSE=1
npm start
```

Quick checks before opening Streamlit:

```bash
curl http://localhost:9423/mcp/connect
PYTHONPATH=src .venv/bin/python scripts/check_job_sources.py
```

When enabled, aggregator order is:
1. `jobspy_mcp`
2. `arbeitnow`
3. `remotive`
4. `themuse`
5. `adzuna` (if keys set)

## 2) Greenhouse MCP

Repo: `https://github.com/alexmeckes/greenhouse-mcp`

Use this for company ATS-specific roles (great for target-company mode).
Not yet wired as a connector in this codebase. Add next as `GreenhouseMCPSource`.

## 3) H1B Job Search MCP

Repo: `https://github.com/aryaminus/h1b-job-search-mcp`

Use this as a specialist source for visa-aware searches and filter-based queries.
Not yet wired as a connector in this codebase.

## Verify the source used

In Streamlit, open "Top 20 Latest Jobs" and check each item's `Source`.
If `Source` shows LinkedIn/Indeed-like values, JobSpy MCP is active.
