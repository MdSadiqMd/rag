from lib.search_utils import CACHE_PATH, load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
import pickle

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}  # map document id: document
        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.docmap[doc_id] = text

    def get_document(self, term: str) -> set[int]:
        return sorted(list(self.index[term]))

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie["title"] + " " + movie["description"]}"
            self.__add_document(doc_id, text)
        self.__save()

    def __save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)


def build_command():
    idx = InvertedIndex()
    idx.build()
    docs = idx.get_document("merida")
    print(f"First document for token 'merida'={docs[0]}")


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]
    return [stemmer.stem(tok) for tok in tokens]


def has_matching_tokens(query_tokens: list[str], movie_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False


def search_movies(query: str, n_results: int = 10) -> list[dict]:
    movies = load_movies()
    stopwords = load_stopwords()
    res = []
    query_tokens = [tok for tok in tokenize_text(query) if tok not in stopwords]
    for movie in movies:
        if has_matching_tokens(query_tokens, tokenize_text(movie["title"])):
            res.append(movie)
        if len(res) >= n_results:
            break
    return res
