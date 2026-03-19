"""Common text normalization, tokenization, and slug utilities used across modules."""

import re


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#.]+", (text or "").lower()))


COMMON_SKILLS = {
    "python", "sql", "java", "javascript", "typescript", "react", "node", "aws",
    "azure", "gcp", "docker", "kubernetes", "pytorch", "tensorflow", "nlp",
    "machine", "learning", "llm", "streamlit", "fastapi", "flask", "pandas",
    "numpy", "spark", "hadoop", "postgres", "mongodb", "redis", "linux", "git",
}


ROLE_HINTS = {
    "data scientist", "machine learning engineer", "ai engineer", "software engineer",
    "data analyst", "backend engineer", "frontend engineer", "full stack engineer",
    "product manager", "research engineer",
}
