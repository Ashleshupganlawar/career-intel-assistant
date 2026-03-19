"""CLI utility to query the local vector database and inspect top matching chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_intel.rag.vector_store import FaissVectorDB, make_embedder


def infer_backend_from_config(persist_dir: str) -> tuple[str, str | None]:
    cfg_path = Path(persist_dir) / "config.json"
    if not cfg_path.exists():
        return "auto", None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return "auto", None
    model = str(cfg.get("embed_model", "")).strip()
    if model.startswith("local-hash-"):
        return "hash", None
    if model:
        return "hf", model
    return "auto", None


def main():
    parser = argparse.ArgumentParser(description="Query local FAISS vector DB")
    parser.add_argument("query", help="Natural language question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--persist-dir", default="data/vector_store")
    parser.add_argument(
        "--embed-backend",
        choices=["auto", "hf", "hash"],
        default=None,
        help="Optional override. If omitted, inferred from vector_store/config.json",
    )
    parser.add_argument("--embed-model", default=None, help="Optional HF embed model override.")
    args = parser.parse_args()

    db = FaissVectorDB(args.persist_dir)
    backend, model = infer_backend_from_config(args.persist_dir)
    if args.embed_backend:
        backend = args.embed_backend
    if args.embed_model:
        model = args.embed_model
    embedder = make_embedder(backend=backend, model_name=model)

    rows = db.search(args.query, embedder, top_k=args.top_k)
    if not rows:
        print("No results found. Build index first.")
        return

    for i, row in enumerate(rows, start=1):
        md = row.get("metadata", {})
        print(f"\n#{i} score={row['score']:.4f} company={md.get('company_slug')} section={md.get('section')}")
        print(row.get("text", ""))


if __name__ == "__main__":
    main()
