import json
from pathlib import Path

BM25_K1 = 1.5
BM25_B = 0.75
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"
CACHE_PATH = PROJECT_ROOT / "cache"


def load_movies() -> list[dict]:
    with open(DATA_DIR / "movies.json", "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_FILE, "r") as f:
        data = f.read().splitlines()
    return data
