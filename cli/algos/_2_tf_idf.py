from lib.search_utils import CACHE_PATH, load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import pickle
import math

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}  # map document id: document
        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.term_frequencies = defaultdict(Counter)
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for t in set(tokens):
            self.index[t].add(doc_id)
        self.docmap[doc_id] = text
        self.term_frequencies[doc_id].update(tokens)

    def get_document(self, term: str) -> set[int]:
        return self.index.get(term, set())

    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Can only have 1 token")
        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only have 1 tokens")

        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
        self.__save()

    def __save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        return self


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]
    return [stemmer.stem(tok) for tok in tokens]


def search_movies(query: str, n_results: int = 10) -> list[dict]:
    movies = load_movies()
    stopwords = load_stopwords()
    idx = InvertedIndex()
    idx.load()
    res = []
    query_tokens = [tok for tok in tokenize_text(query) if tok not in stopwords]
    if not query_tokens:
        return res

    candidate_docs = set()
    for q in query_tokens:
        candidate_docs.update(idx.get_document(q))

    for movie in movies:
        if movie["id"] in candidate_docs:
            res.append(movie)
        if len(res) >= n_results:
            break

    return res


def search_command(query, n_results):
    idx = InvertedIndex().load()
    seen, res = set(), []
    query_tokens = tokenize_text(query)
    for qt in query_tokens:
        matching_docs_ids = idx.get_documents(qt)
        for matching_doc_id in matching_docs_ids:
            if matching_doc_id in seen:
                continue
            seen.add(matching_doc_id)
            matching_doc = idx.docmap[matching_doc_id]
            res.append(matching_doc)

            if len(res) >= n_results:
                return res
    return res


def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))


def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tf_idf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tf_idf = idx.get_tf_idf(doc_id, term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def build_command():
    idx = InvertedIndex()
    idx.build()
    docs = idx.get_document("merida")
    print(f"First document for token 'merida'={sorted(docs)[0]}")
