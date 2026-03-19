"""This script is the command-line entry point that prepares your local semantic search database by optionally fetching and summarizing company data, chunking it into documents, embedding it, and saving it into a FAISS index.-"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Resolve the project root, locate the src directory, and add it to sys.path if it's not already included
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from job_intel.rag.vector_store import FaissVectorDB, build_company_documents, make_embedder


def run_step(step_cmd: list[str]):
    print(f"[RUN] {' '.join(step_cmd)}")
    subprocess.run(step_cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build local FAISS vector DB for company intelligence")
    parser.add_argument("--run-fetch", action="store_true", help="Run source collection first")
    parser.add_argument("--run-summarize", action="store_true", help="Run JSON profile summarization first")
    parser.add_argument("--persist-dir", default="data/vector_store")
    parser.add_argument("--chunk-chars", type=int, default=500)
    parser.add_argument(
        "--embed-backend",
        choices=["auto", "hf", "hash"],
    
        default="auto",
        help="Embedding backend: auto (HF then hash fallback), hf, or hash (offline).",
    )
    parser.add_argument("--embed-model", default=None, help="Optional HF embedding model override.")
    args = parser.parse_args()

    if args.run_fetch:
        run_step([sys.executable, "scripts/fetch_and_store_sources.py"])

    if args.run_summarize:
        run_step([sys.executable, "scripts/summarize_company_profiles.py"])

    docs = build_company_documents(chunk_chars=args.chunk_chars)
    if not docs:
        raise SystemExit(
            "No documents found. Ensure at least one of these has data: "
            "data/processed_company_profiles/*_profile.json OR data/source_maps/*.json + data/raw_sources/*/*.txt"
        )

    raw_doc_count = sum(1 for d in docs if str(d.metadata.get("section", "")).startswith("raw_"))

    embedder = make_embedder(backend=args.embed_backend, model_name=args.embed_model)
    db = FaissVectorDB(persist_dir=args.persist_dir)
    db.build(docs, embedder)

    print(f"[OK] Built vector DB at: {args.persist_dir}")
    print(f"[OK] Indexed chunks: {len(docs)}")
    print(f"[OK] Embedding model: {embedder.model_name}")
    if raw_doc_count > 0:
        print(f"[INFO] Used fallback raw/source-map chunks: {raw_doc_count}")


if __name__ == "__main__":
    main()
