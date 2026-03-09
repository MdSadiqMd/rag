"""
TF-IDF and BM25 Ranking Algorithms
====================================

This module implements an inverted index with support for TF-IDF and BM25 scoring,
enabling ranked full-text search over a movie dataset.

-------------------------------------------------------------------------------
1. TF (Term Frequency)
-------------------------------------------------------------------------------
Measures how often a term appears in a document. A higher count means the term
is more relevant to that specific document.

    TF(t, d) = count of term t in document d

-------------------------------------------------------------------------------
2. IDF (Inverse Document Frequency)
-------------------------------------------------------------------------------
Measures how rare a term is across all documents. Terms that appear in many
documents (e.g. "the", "is") get a low IDF, while rare terms get a high IDF.

    IDF(t) = log( (N + 1) / (df(t) + 1) )

    where:
        N      = total number of documents in the corpus
        df(t)  = number of documents containing term t
        +1     is Laplace smoothing to avoid log(0) or division by zero

-------------------------------------------------------------------------------
3. TF-IDF Score
-------------------------------------------------------------------------------
The product of TF and IDF. Used to rank how relevant a term is to a document
relative to the entire corpus.

    TF-IDF(t, d) = TF(t, d) * IDF(t)

-------------------------------------------------------------------------------
4. BM25 IDF (Okapi BM25 variant)
-------------------------------------------------------------------------------
A smoothed IDF variant used in BM25 scoring. Different from standard IDF in
that it penalizes terms appearing in more than half the documents.

    BM25_IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

    where:
        N      = total number of documents
        df(t)  = number of documents containing term t
        +1     ensures the result is always positive

-------------------------------------------------------------------------------
5. BM25 TF (Saturated Term Frequency)
-------------------------------------------------------------------------------
A length-normalised, saturating version of TF used in BM25. Unlike raw TF,
doubling the count of a term does not double the score — it converges to (k1+1).

    BM25_TF(t, d) = TF(t, d) * (k1 + 1)
                    ──────────────────────────────────────────────────
                    TF(t, d) + k1 * (1 - b + b * (|d| / avgdl))

    where:
        k1    = term saturation parameter (default 1.5)
                controls how quickly TF saturates; higher = slower saturation
        b     = length normalisation parameter (default 0.75)
                controls how much document length affects scoring; 0 = no,  1 = full normalisation
        |d|   = number of tokens in document d
        avgdl = average document length across all documents in the corpus

-------------------------------------------------------------------------------
6. BM25 Score
-------------------------------------------------------------------------------
The full BM25 score for a query-document pair is the sum of BM25_IDF * BM25_TF
for each query term.

    BM25(q, d) = Σ BM25_IDF(t) * BM25_TF(t, d)
                t ∈ q

-------------------------------------------------------------------------------
Implementation Notes
-------------------------------------------------------------------------------
- Text is lowercased, punctuation-stripped, and Porter-stemmed before indexing.
- The inverted index maps each stemmed token to the set of document IDs that
  contain it.
- Term frequencies per document are stored as collections.Counter objects.
- Document lengths (token counts) are cached for efficient BM25 normalisation.
- All index structures are serialised to disk via pickle for fast reuse.
- Query tokens are tokenized and stemmed the same way as documents for consistency.
- Candidate documents are filtered to only those containing at least one query term.
- Results are sorted by BM25 score in descending order (highest relevance first).
"""

from lib.search_utils import (
    CACHE_PATH,
    load_movies,
    load_stopwords,
    BM25_K1,
    BM25_B,
    format_search_result,
    DEFAULT_SEARCH_LIMIT,
)
import string
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import pickle
import math

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)  # term -> set of doc_ids
        self.docmap = {}  # doc_id -> document text
        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.term_frequencies = defaultdict(Counter)  # doc_id -> Counter of terms
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"
        self.doc_lengths = {}  # doc_id -> number of tokens
        self.doc_lengths_path = CACHE_PATH / "doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for t in set(tokens):
            self.index[t].add(doc_id)
        self.docmap[doc_id] = text
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        lengths = list(self.doc_lengths.values())
        if len(lengths) == 0:
            return 0
        return sum(lengths) / len(lengths)

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

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Can only have 1 tokens")

        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return tf * (k1 + 1) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0
        tf_component = self.get_bm25_tf(doc_id, term)
        idf_component = self.get_bm25_idf(term)
        return tf_component * idf_component

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = tokenize_text(query)
        candidate_docs = set()
        for token in query_tokens:
            candidate_docs.update(self.index.get(token, set()))

        scores = {}
        for doc_id in candidate_docs:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        movies = load_movies()
        movie_map = {m["id"]: m for m in movies}

        results = []
        for doc_id, score in sorted_docs[:limit]:
            movie = movie_map[doc_id]
            formatted_result = format_search_result(
                doc_id=doc_id,
                title=movie["title"],
                document=movie["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
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


def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    bm25idf = idx.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    idx.load()
    bm25tf = idx.get_bm25_tf(doc_id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")


def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)


def build_command():
    idx = InvertedIndex()
    idx.build()
    docs = idx.get_document("merida")
    print(f"First document for token 'merida'={sorted(docs)[0]}")
