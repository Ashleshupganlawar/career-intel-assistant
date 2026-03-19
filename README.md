# Career Intel Assistant

Career Intel Assistant is a Streamlit app for resume-aware job discovery and company research. It combines resume parsing, hybrid job matching, threaded chat, and retrieval-augmented generation (RAG) over curated company profile data.

## What It Does

- Upload a resume and extract a structured candidate profile
- Fetch and rank jobs against resume skills and target roles
- Keep a threaded chat history with job and resume context
- Surface company insights from curated source maps and a local vector store
- Blend deterministic matching with LLM-assisted responses

## Project Structure

- `app/streamlit_app.py`: Streamlit UI
- `src/job_intel/chat/`: Chat graph pipeline and orchestration
- `src/job_intel/matching/`: Hybrid job ranking logic
- `src/job_intel/rag/`: Local retriever and vector store helpers
- `src/job_intel/resume/`: Resume parsing utilities
- `src/job_intel/jobs/`: Job source connectors and service layer
- `src/job_intel/storage/`: Conversation and artifact persistence
- `data/`: Cached jobs, source maps, processed company profiles, and vector DB assets
- `scripts/`: Data prep and workflow utilities
- `tests/`: Unit and integration coverage

## Quick Start

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file for any API-backed features you want to use. The app currently reads OpenAI-related configuration from environment variables.

Example:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_RESUME_MODEL=gpt-4o-mini
```

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

Open the local Streamlit URL shown in the terminal.

## How It Works

1. A resume upload is parsed into a candidate profile.
2. The system derives a search query from parsed roles and skills.
3. Job records are fetched and ranked with lexical plus embedding-style signals.
4. Company context is retrieved from local processed sources and vector data.
5. The chat layer answers follow-up questions using thread context, jobs, resume text, and RAG evidence.

## Development

Run tests:

```bash
pytest -q
```

Useful docs:

- `docs/codex_workflow.md`
- `docs/mcp_job_sources.md`
- `docs/vector_db_workflow.md`

Useful scripts:

- `scripts/build_vector_db.py`
- `scripts/fetch_and_store_sources.py`
- `scripts/summarize_company_profiles.py`
- `scripts/query_vector_db.py`

## Notes

- The repository includes curated company profile and vector store assets so the app has useful local context out of the box.
- Local caches and secrets are ignored through `.gitignore`.

## Status

This is an actively evolving project focused on AI-assisted job search, matching, and company intelligence workflows.
