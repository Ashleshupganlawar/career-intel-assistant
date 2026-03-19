"""Fetches official/community web sources per company and stores raw/source-map artifacts."""

import os
import re
import json
import time
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

CSV_PATH = os.getenv("CSV_PATH", "data/input/list_of_companies.csv")
RAW_SOURCES_DIR = Path(os.getenv("RAW_SOURCES_DIR", "data/raw_sources"))
SOURCE_MAPS_DIR = Path(os.getenv("SOURCE_MAPS_DIR", "data/source_maps"))

SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "5"))
MAX_REDDIT_THREADS = int(os.getenv("MAX_REDDIT_THREADS", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "25"))
SLEEP_BETWEEN_REQUESTS = float(os.getenv("SLEEP_BETWEEN_REQUESTS", "1.0"))
MAX_COMPANIES = int(os.getenv("MAX_COMPANIES", "0"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CompanyProfileBot/0.1; +https://example.com/bot)"
}

RAW_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_MAPS_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def web_search(query: str, max_results: int = 5):
    """
    Stable fallback search using Bing RSS + requests.
    Avoids ddgs TLS issues on some macOS LibreSSL environments.
    """
    search_url = "https://www.bing.com/search?format=rss&q=" + quote_plus(query)
    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        results = []
        for item in root.findall(".//item"):
            title = item.findtext("title", default="")
            link = item.findtext("link", default="")
            if link:
                results.append({"title": title, "url": link})
            if len(results) >= max_results:
                break
        return results
    except Exception as e:
        print(f"[WARN] Search failed for query={query}: {e}")
        return []
    

    
def company_tokens(company: str):
    tokens = re.findall(r"[a-z0-9]+", company.lower())
    return [t for t in tokens if len(t) > 1]


def score_url_for_company(url: str, company: str, kind: str):
    """
    Heuristic ranking for official pages and site-specific pages.
    """
    tokens = company_tokens(company)
    lowered = url.lower()
    score = 0

    for token in tokens:
        if token in lowered:
            score += 3

    keyword_map = {
        "careers": ["career", "careers"],
        "jobs": ["job", "jobs", "search"],
        "hiring_process": ["hire", "hiring", "interview"],
        "interview_structure": ["interview", "loop", "bar-raiser", "process"],
        "culture": ["culture", "workplace", "life", "about"],
        "faq": ["faq", "application", "candidate"],
        "salary_levels": ["levels.fyi", "salary"],
        "salary_indeed": ["indeed", "salary"],
        "review_glassdoor": ["glassdoor"],
        "review_comparably": ["comparably"],
    }

    for kw in keyword_map.get(kind, []):
        if kw in lowered:
            score += 2

    if "reddit.com" in lowered and kind.startswith("reddit"):
        score += 5

    return score


def choose_best_result(results, company, kind):
    if not results:
        return None

    ranked = sorted(
        results,
        key=lambda x: score_url_for_company(x["url"], company, kind),
        reverse=True,
    )
    return ranked[0]


def extract_text_from_url(url: str) -> str:
    """
    Generic page fetch + BeautifulSoup extraction.
    This avoids trafilatura network path issues on some SSL stacks.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text("\n", strip=True)
    except Exception as e:
        return f"[ERROR] Could not extract text from {url}: {e}"


def fetch_reddit_thread_text(thread_url: str) -> str:
    """
    Fetches Reddit thread JSON and extracts post + top comments.
    """
    json_url = thread_url.rstrip("/") + ".json?limit=5"
    try:
        resp = requests.get(json_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        post = data[0]["data"]["children"][0]["data"]
        title = post.get("title", "")
        body = post.get("selftext", "")

        comments_out = []
        comments = data[1]["data"]["children"]
        for c in comments[:5]:
            if c.get("kind") != "t1":
                continue
            comment_data = c["data"]
            comment_body = comment_data.get("body", "")
            if comment_body:
                comments_out.append(comment_body)

        return "\n\n".join(
            [
                f"TITLE: {title}",
                f"POST: {body}",
                "COMMENTS:",
                *comments_out,
            ]
        ).strip()
    except Exception as e:
        return f"[ERROR] Could not fetch reddit thread {thread_url}: {e}"


def build_queries(company: str):
    return {
        "official": {
            "careers": f'{company} official careers',
            "jobs": f'{company} official jobs',
            "hiring_process": f'{company} official hiring process',
            "interview_structure": f'{company} official interview process',
            "culture": f'{company} official culture workplace',
            "faq": f'{company} official application faq',
        },
        "community": {
            "salary_levels": f'{company} levels.fyi salary',
            "salary_indeed": f'{company} indeed salary',
            "review_glassdoor": f'{company} glassdoor interview',
            "review_comparably": f'{company} comparably culture',
            "reddit": [
                f'site:reddit.com "{company}" interview',
                f'site:reddit.com "{company}" culture',
                f'site:reddit.com "{company}" work life balance',
                f'site:reddit.com/r/cscareerquestions "{company}"',
                f'site:reddit.com/r/recruitinghell "{company}"',
            ],
        },
    }


def save_text(file_path: Path, text: str):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")


def fetch_company(company: str, industry: str):
    slug = slugify(company)
    company_dir = RAW_SOURCES_DIR / slug
    company_dir.mkdir(parents=True, exist_ok=True)

    queries = build_queries(company)

    official_sources = {}
    community_sources = {
        "salary_source": [],
        "review_source": [],
        "reddit_threads": [],
    }

    # Official pages
    for kind, query in queries["official"].items():
        results = web_search(query, max_results=SEARCH_RESULTS_PER_QUERY)
        chosen = choose_best_result(results, company, kind)

        if chosen:
            official_sources[f"{kind}_url"] = chosen["url"]
            text = extract_text_from_url(chosen["url"])
            save_text(company_dir / f"official_{kind}.txt", text)
        else:
            official_sources[f"{kind}_url"] = ""

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Community salary/review pages
    community_label_map = {
        "salary_levels": "Levels.fyi",
        "salary_indeed": "Indeed",
        "review_glassdoor": "Glassdoor",
        "review_comparably": "Comparably",
    }

    for kind, query in queries["community"].items():
        if kind == "reddit":
            continue

        results = web_search(query, max_results=SEARCH_RESULTS_PER_QUERY)
        chosen = choose_best_result(results, company, kind)

        if chosen:
            text = extract_text_from_url(chosen["url"])
            save_text(company_dir / f"community_{kind}.txt", text)

            if kind.startswith("salary"):
                community_sources["salary_source"].append(
                    {"label": community_label_map[kind], "url": chosen["url"]}
                )
            else:
                community_sources["review_source"].append(
                    {"label": community_label_map[kind], "url": chosen["url"]}
                )

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Reddit threads
    reddit_seen = set()
    reddit_threads = []

    for query in queries["community"]["reddit"]:
        results = web_search(query, max_results=SEARCH_RESULTS_PER_QUERY)

        for r in results:
            url = r["url"]
            if "reddit.com" not in url:
                continue
            if "/comments/" not in url:
                continue
            if url in reddit_seen:
                continue

            reddit_seen.add(url)
            reddit_threads.append(url)

            if len(reddit_threads) >= MAX_REDDIT_THREADS:
                break

        if len(reddit_threads) >= MAX_REDDIT_THREADS:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    for idx, thread_url in enumerate(reddit_threads, start=1):
        text = fetch_reddit_thread_text(thread_url)
        save_text(company_dir / f"reddit_thread_{idx}.txt", text)
        community_sources["reddit_threads"].append(thread_url)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    source_map = {
        "company": company,
        "industry": industry,
        "official_sources": official_sources,
        "community_sources": community_sources,
        "last_updated": time.strftime("%Y-%m-%d"),
    }

    with open(SOURCE_MAPS_DIR / f"{slug}.json", "w", encoding="utf-8") as f:
        json.dump(source_map, f, indent=2)

    return source_map


def main():
    df = pd.read_csv(CSV_PATH)

    required_cols = {"company", "industry"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required_cols}")

    if MAX_COMPANIES > 0:
        df = df.head(MAX_COMPANIES)
        print(f"[INFO] Limiting fetch to first {MAX_COMPANIES} companies.")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching companies"):
        company = str(row["company"]).strip()
        industry = str(row["industry"]).strip()
        fetch_company(company, industry)


if __name__ == "__main__":
    main()
