# Vector DB Workflow (Free + Local)

## Goal
Automate:
1. collect company sources
2. generate structured company JSON
3. chunk + embed with Hugging Face model
4. store in local FAISS vector DB

## Commands
- Build from existing company JSONs:
  - `PYTHONPATH=src python scripts/build_vector_db.py`
- Full pipeline (fetch + summarize + vectorize):
  - `PYTHONPATH=src python scripts/build_vector_db.py --run-fetch --run-summarize`
- Query vector DB:
  - `PYTHONPATH=src python scripts/query_vector_db.py "amazon interview process red flags" --top-k 5`

## Output Files
Created under `data/vector_store/`:
- `index.faiss`
- `metadata.json`
- `documents.json`
- `config.json`

## Notes
- Uses `sentence-transformers/all-MiniLM-L6-v2` by default (`HF_EMBED_MODEL` to override).
- If local sentence-transformers model is unavailable, it falls back to Hugging Face Inference API.
