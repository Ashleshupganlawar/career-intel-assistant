"""Builds normalized company profile summaries using LLM provider or heuristic fallback."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")
HF_ROUTER_MODEL = os.getenv("HF_ROUTER_MODEL", "deepseek-ai/DeepSeek-R1:novita")

RAW_SOURCES_DIR = Path(os.getenv("RAW_SOURCES_DIR", "data/raw_sources"))
SOURCE_MAPS_DIR = Path(os.getenv("SOURCE_MAPS_DIR", "data/source_maps"))
PROFILES_DIR = Path(os.getenv("PROFILES_DIR", "data/processed_company_profiles"))
MAX_CHARS_PER_FILE = int(os.getenv("MAX_CHARS_PER_FILE", "12000"))

PROFILES_DIR.mkdir(parents=True, exist_ok=True)

client = None
HF_DISABLED = False
HF_DISABLE_REASON = ""
BACKEND = "hf_hub"
router_client = None
DEFAULT_MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "tiiuae/falcon-7b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]


def init_client():
    global client, router_client, BACKEND
    if HF_PROVIDER == "hf-router-openai":
        from openai import OpenAI

        BACKEND = "hf_router_openai"
        router_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
        return

    from huggingface_hub import InferenceClient

    BACKEND = "hf_hub"
    client = InferenceClient(model=HF_MODEL, provider=HF_PROVIDER, token=HF_TOKEN)


def read_raw_texts(company_slug: str) -> dict[str, str]:
    company_dir = RAW_SOURCES_DIR / company_slug
    if not company_dir.exists():
        return {}

    docs: dict[str, str] = {}
    for file_path in sorted(company_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        docs[file_path.name] = text[:MAX_CHARS_PER_FILE]
    return docs


def load_source_map(company_slug: str) -> dict:
    path = SOURCE_MAPS_DIR / f"{company_slug}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(source_map: dict, raw_docs: dict[str, str]) -> tuple[str, str]:
    docs_text = []
    for name, text in raw_docs.items():
        docs_text.append(f"\n===== {name} =====\n{text}\n")

    combined_docs = "\n".join(docs_text)

    system_prompt = """
You are a data extraction and synthesis assistant.

Given official company pages, community review pages, and Reddit threads,
return ONE valid JSON object with this schema:

{
  "company": "",
  "industry": "",
  "official_sources": {
    "careers_url": "",
    "jobs_url": "",
    "hiring_process_url": "",
    "interview_structure_url": "",
    "culture_url": "",
    "faq_url": ""
  },
  "community_sources": {
    "salary_source": [],
    "review_source": [],
    "reddit_threads": []
  },
  "company_overview": "",
  "official_hiring_process": "",
  "hiring_trends_summary": "",
  "salary_summary": "",
  "culture_summary_official": "",
  "culture_summary_community": "",
  "interview_experience_summary": "",
  "common_interview_topics": [],
  "reddit_common_themes": [],
  "reddit_red_flags": [],
  "reddit_positive_signals": [],
  "final_blended_insight": "",
  "notes": "",
  "last_updated": ""
}

Rules:
- Official pages are source of truth for process and overview.
- Community/Reddit only for repeated patterns.
- Do not invent URLs.
- Return only valid JSON.
"""

    user_prompt = f"""
SOURCE MAP:
{json.dumps(source_map, indent=2)}

RAW DOCUMENTS:
{combined_docs}

Generate the final company profile JSON now.
"""

    return system_prompt, user_prompt


def call_huggingface(system_prompt: str, user_prompt: str) -> str:
    if BACKEND == "hf_router_openai":
        completion = router_client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2200,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e).lower()
        # Some models/providers do not expose /chat/completions but still support text generation.
        if "404" in err or "not found" in err:
            prompt = (
                "SYSTEM INSTRUCTIONS:\n"
                + system_prompt
                + "\n\nUSER INPUT:\n"
                + user_prompt
                + "\n\nReturn ONLY valid JSON."
            )
            return client.text_generation(
                prompt=prompt,
                max_new_tokens=2200,
                temperature=0.2,
                return_full_text=False,
            )
        raise


def check_hf_access() -> bool:
    global HF_DISABLED, HF_DISABLE_REASON
    try:
        _ = call_huggingface("Return short text only.", "ok")
        print(f"[OK] HF preflight passed (provider={HF_PROVIDER}, model={HF_MODEL})")
        return True
    except Exception as e:
        err_text = str(e)
        HF_DISABLED = True
        HF_DISABLE_REASON = err_text
        print(f"[WARN] HF preflight failed: {err_text}")
        print("[INFO] Will use heuristic fallback summarizer for this run.")
        return False


def try_model_candidates(candidates: list[str]) -> bool:
    global HF_MODEL, HF_DISABLED, HF_DISABLE_REASON
    for model in candidates:
        if not model.strip():
            continue
        HF_MODEL = model.strip()
        init_client()
        print(f"[INFO] Trying HF model candidate: {HF_MODEL}")
        if check_hf_access():
            HF_DISABLED = False
            HF_DISABLE_REASON = ""
            print(f"[OK] Using HF model: {HF_MODEL}")
            return True
    return False


def parse_json_from_model_output(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def heuristic_profile(source_map: dict, raw_docs: dict[str, str]) -> dict:
    company = source_map.get("company", "")
    industry = source_map.get("industry", "")
    official_sources = source_map.get("official_sources", {})
    community_sources = source_map.get("community_sources", {})

    combined = " ".join(raw_docs.values()).lower()

    def contains_any(words: list[str]) -> bool:
        return any(w in combined for w in words)

    common_topics = []
    if contains_any(["leadership principle", "leadership principles"]):
        common_topics.append("Leadership Principles")
    if contains_any(["star format", "behavioral"]):
        common_topics.append("behavioral questions in STAR format")
    if contains_any(["system design", "architecture"]):
        common_topics.append("system design")
    if contains_any(["loop interview", "interview loop", "multi-round"]):
        common_topics.append("loop interview structure")
    if contains_any(["coding", "dsa", "leetcode"]):
        common_topics.append("coding rounds / DSA")

    reddit_themes = []
    if contains_any(["team-dependent", "team dependent"]):
        reddit_themes.append("team quality matters a lot")
    if contains_any(["intense", "demanding", "long process"]):
        reddit_themes.append("interview loop can be intense")
    if contains_any(["leadership principle", "leadership principles"]):
        reddit_themes.append("Leadership Principles are central")
    if contains_any(["rigid", "formulaic"]):
        reddit_themes.append("process can feel rigid or formulaic")

    red_flags = []
    if contains_any(["long process", "multiple rounds", "4-5 rounds"]):
        red_flags.append("long interview process")
    if contains_any(["mechanical", "checkbox"]):
        red_flags.append("mechanical interview experience")
    if contains_any(["work-life balance", "wlb", "burnout"]):
        red_flags.append("mixed work-life balance")
    if contains_any(["team-dependent", "team dependent"]):
        red_flags.append("team-dependent experience")

    positive = []
    if contains_any(["good work-life balance", "great team"]):
        positive.append("some teams report good work-life balance")
    if contains_any(["compensation", "salary", "pay"]):
        positive.append("strong compensation on some tracks")
    if contains_any(["hiring", "openings", "jobs"]):
        positive.append("large hiring footprint")

    return {
        "company": company,
        "industry": industry,
        "official_sources": {
            "careers_url": official_sources.get("careers_url", ""),
            "jobs_url": official_sources.get("jobs_url", ""),
            "hiring_process_url": official_sources.get("hiring_process_url", ""),
            "interview_structure_url": official_sources.get("interview_structure_url", ""),
            "culture_url": official_sources.get("culture_url", ""),
            "faq_url": official_sources.get("faq_url", ""),
        },
        "community_sources": {
            "salary_source": community_sources.get("salary_source", []),
            "review_source": community_sources.get("review_source", []),
            "reddit_threads": community_sources.get("reddit_threads", []),
        },
        "company_overview": f"{company} is a major employer with role-specific hiring tracks and broad job families.",
        "official_hiring_process": "Structured, role-specific hiring with recruiter screening and interview rounds where applicable.",
        "hiring_trends_summary": "Use official jobs pages and recurring job families to track hiring patterns by location and role type.",
        "salary_summary": "Community salary references are available; interpret as directional rather than definitive.",
        "culture_summary_official": "Official messaging emphasizes inclusive hiring, preparation guidance, and role fit.",
        "culture_summary_community": "Community sentiment appears mixed and team-dependent based on repeated discussion patterns.",
        "interview_experience_summary": "Community reports commonly mention multi-round interviews and behavioral + technical depth.",
        "common_interview_topics": common_topics or ["behavioral interview", "role-specific technical depth"],
        "reddit_common_themes": reddit_themes or ["team quality varies by org", "interview process requires preparation"],
        "reddit_red_flags": red_flags or ["long interview process", "team-dependent experience"],
        "reddit_positive_signals": positive or ["large hiring footprint"],
        "final_blended_insight": "Official process appears structured; community feedback suggests outcomes depend strongly on team and role.",
        "notes": "Generated via heuristic fallback because model inference was unavailable. Treat as draft profile.",
        "last_updated": source_map.get("last_updated", ""),
    }


def summarize_company(company_slug: str):
    global HF_DISABLED, HF_DISABLE_REASON

    source_map = load_source_map(company_slug)
    raw_docs = read_raw_texts(company_slug)

    profile_json = None
    system_prompt, user_prompt = build_prompt(source_map, raw_docs)

    if not raw_docs:
        print(f"[WARN] No raw docs found for {company_slug}; generating minimal heuristic profile.")
        profile_json = heuristic_profile(source_map, raw_docs)
        out_path = PROFILES_DIR / f"{company_slug}_profile.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(profile_json, f, indent=2)
        print(f"[OK] Saved {out_path}")
        return

    if not HF_DISABLED:
        try:
            model_output = call_huggingface(system_prompt, user_prompt)
            profile_json = parse_json_from_model_output(model_output)
        except Exception as e:
            err_text = str(e)
            print(f"[WARN] HF summarization failed for {company_slug}: {err_text}")
            if "403" in err_text or "sufficient permissions" in err_text.lower():
                HF_DISABLED = True
                HF_DISABLE_REASON = (
                    "Hugging Face token lacks Inference Providers permission for current model/provider."
                )
                print("[INFO] Disabling HF calls for the rest of this run to avoid repeated 403 errors.")
                print("[FIX] Update .env with a permitted token/provider/model, e.g.:")
                print("      HF_PROVIDER=hf-inference")
                print("      HF_MODEL=<a model your account can access>")
                print("      HF_TOKEN=<token with inference permissions>")
            print(f"[INFO] Using heuristic fallback summarizer for {company_slug}.")
            profile_json = heuristic_profile(source_map, raw_docs)
    else:
        print(f"[INFO] HF disabled ({HF_DISABLE_REASON}). Using heuristic fallback for {company_slug}.")
        profile_json = heuristic_profile(source_map, raw_docs)

    out_path = PROFILES_DIR / f"{company_slug}_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile_json, f, indent=2)

    print(f"[OK] Saved {out_path}")


def main():
    global HF_MODEL, HF_PROVIDER, HF_DISABLED, HF_DISABLE_REASON

    parser = argparse.ArgumentParser(description="Summarize company profiles from source maps + raw docs.")
    parser.add_argument("--model", default=HF_MODEL, help="HF model id override")
    parser.add_argument(
        "--provider",
        default=HF_PROVIDER,
        help="Provider: hf-inference or hf-router-openai",
    )
    parser.add_argument("--max-companies", type=int, default=0, help="Limit number of companies")
    parser.add_argument("--check-only", action="store_true", help="Only test HF access and exit")
    parser.add_argument("--heuristic-only", action="store_true", help="Skip HF and use heuristic summarization only")
    parser.add_argument("--auto-model", action="store_true", help="Try multiple model candidates automatically")
    parser.add_argument(
        "--model-candidates",
        default=",".join(DEFAULT_MODEL_CANDIDATES),
        help="Comma-separated HF model candidates for --auto-model",
    )
    args = parser.parse_args()

    HF_MODEL = args.model
    HF_PROVIDER = args.provider
    if HF_PROVIDER == "hf-router-openai" and args.model == os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"):
        HF_MODEL = HF_ROUTER_MODEL
    init_client()

    ok = False
    if args.heuristic_only:
        print("[INFO] Running in heuristic-only mode; HF calls are skipped.")
        HF_DISABLED = True
        HF_DISABLE_REASON = "heuristic-only mode"
    else:
        ok = check_hf_access()
        if (not ok) and args.auto_model:
            candidates = [x.strip() for x in args.model_candidates.split(",")]
            ok = try_model_candidates(candidates)
        if not ok:
            print("[INFO] HF not available for this run; heuristic fallback will be used.")
    if args.check_only:
        return

    if not SOURCE_MAPS_DIR.exists():
        raise SystemExit("SOURCE_MAPS_DIR does not exist. Run fetch_and_store_sources.py first.")

    maps = sorted(SOURCE_MAPS_DIR.glob("*.json"))
    if not maps:
        raise SystemExit("No source maps found. Run fetch_and_store_sources.py first.")

    if args.max_companies > 0:
        maps = maps[: args.max_companies]
        print(f"[INFO] Limiting summarization to first {args.max_companies} companies.")

    for path in maps:
        summarize_company(path.stem)


if __name__ == "__main__":
    main()
