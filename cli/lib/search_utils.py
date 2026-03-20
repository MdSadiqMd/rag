import json
from pathlib import Path
from typing import Any

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
BM25_K1 = 1.5
BM25_B = 0.75
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"
CACHE_PATH = PROJECT_ROOT / "cache"
PROMPT_PATH = PROJECT_ROOT / "cli" / "prompts"


def load_movies() -> list[dict]:
    with open(DATA_DIR / "movies.json", "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_FILE, "r") as f:
        data = f.read().splitlines()
    return data


def load_dataset() -> list[str]:
    with open(STOPWORDS_FILE, "r") as f:
        data = json.load(f)["test_cases"]
    return data


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
