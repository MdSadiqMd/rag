import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"

def load_movies() -> list[dict]:
    with open(DATA_DIR / "movies.json", "r") as f:
        data= json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_FILE, "r") as f:
        data= f.read().splitlines()
    return data
