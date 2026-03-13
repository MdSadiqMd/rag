"""
Semantic Search and Chunking Algorithms
========================================

This module implements semantic search using sentence transformers and vector embeddings,
enabling similarity-based search over a movie dataset. It supports both full-document
and chunk-based semantic search with configurable overlap.

-------------------------------------------------------------------------------
1. Vector Embeddings
-------------------------------------------------------------------------------
Text is converted into dense vector representations (embeddings) using a pre-trained
sentence transformer model. Similar texts have similar embeddings in vector space.

    embedding(text) = SentenceTransformer.encode(text)
    
    where:
        text      = input text (title + description)
        embedding = dense vector (typically 384 dimensions for all-MiniLM-L6-v2)

-------------------------------------------------------------------------------
2. Cosine Similarity
-------------------------------------------------------------------------------
Measures the similarity between two vectors by computing the cosine of the angle
between them. Values range from -1 (opposite) to 1 (identical).

    cosine_similarity(v1, v2) = (v1 · v2) / (||v1|| * ||v2||)
    
    where:
        v1, v2 = vector embeddings
        ·      = dot product
        ||v||  = vector magnitude (L2 norm)

-------------------------------------------------------------------------------
3. Semantic Search
-------------------------------------------------------------------------------
Finds documents most similar to a query by comparing embeddings:

    1. Generate query embedding: q = embedding(query)
    2. For each document d: compute similarity(q, embedding(d))
    3. Sort documents by similarity (descending)
    4. Return top-k results

Unlike keyword search, semantic search understands meaning and context, finding
relevant documents even when they don't share exact keywords.

-------------------------------------------------------------------------------
4. Fixed-Size Chunking
-------------------------------------------------------------------------------
Splits text into overlapping chunks of fixed word count. Useful for processing
long documents that exceed model context limits.

    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk = words[i : i + chunk_size]
        chunks.append(chunk)
    
    where:
        chunk_size = maximum words per chunk
        overlap    = number of words shared between consecutive chunks
        step_size  = how many words to advance for next chunk

Example with chunk_size=3, overlap=1:
    Text: "A B C D E F"
    Chunks: ["A B C"], ["C D E"], ["E F"]

-------------------------------------------------------------------------------
5. Semantic Chunking
-------------------------------------------------------------------------------
Splits text into overlapping chunks by sentences rather than words. Preserves
sentence boundaries for better semantic coherence.

    sentences = split_by_sentence(text)
    chunks = []
    step_size = max_chunk_size - overlap
    for i in range(0, len(sentences), step_size):
        chunk = sentences[i : i + max_chunk_size]
        chunks.append(chunk)
    
    where:
        max_chunk_size = maximum sentences per chunk
        overlap        = number of sentences shared between chunks
        step_size      = how many sentences to advance for next chunk

Example with max_chunk_size=2, overlap=1:
    Text: "S1. S2. S3. S4."
    Chunks: ["S1. S2."], ["S2. S3."], ["S3. S4."]

-------------------------------------------------------------------------------
6. Chunked Semantic Search
-------------------------------------------------------------------------------
Improves search accuracy by:
    1. Splitting documents into semantic chunks
    2. Embedding each chunk separately
    3. Searching at chunk granularity
    4. Aggregating chunk scores per document (using max score)

This allows finding relevant passages within long documents, even if the overall
document isn't highly relevant.

    For each document d:
        chunks = semantic_chunking(d)
        for each chunk c:
            score(c) = similarity(query_embedding, chunk_embedding(c))
        document_score(d) = max(score(c) for c in chunks)

-------------------------------------------------------------------------------
Implementation Notes
-------------------------------------------------------------------------------
- Uses sentence-transformers library with all-MiniLM-L6-v2 model (384 dimensions)
- Embeddings are cached to disk (NumPy .npy format) for fast reuse
- Chunk metadata stored as JSON mapping chunks to source documents
- Cosine similarity computed using NumPy dot product and norms
- Documents without descriptions are skipped during chunk embedding
- Default chunking: 4 sentences per chunk with 1 sentence overlap
- Search returns top-k results sorted by similarity score (descending)
"""
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from lib.search_utils import load_movies
from lib.cosine_similarity import cosine_similarity
import re
import json


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embedding_path = Path("cache/movie_embeddings.npy")

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_string = f"{doc['title']} {doc['description']}"
            movie_strings.append(movie_string)
            print(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embedding_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if self.embedding_path.exists():
            self.embeddings = np.load(self.embedding_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embeddings(self.documents)

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("must have text to create embedding")
        return self.model.encode([text])[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarities = []
        for document_embedding, doc in zip(self.embeddings, self.documents):
            _similarity = cosine_similarity(query_embedding, document_embedding)
            similarities.append((_similarity, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        res = []
        for sc, doc in similarities[:limit]:
            res.append(
                {"score": sc, "title": doc["title"], "description": doc["description"]}
            )
        return res


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search(query, limit=5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    search_results = ss.search(query, limit)
    for idx, res in enumerate(search_results):
        print(f"{idx + 1}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['description'][:100]}...")


def fixed_size_chunking(text, overlap, chunk_size=200):
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i : i + chunk_size]
        if len(chunks) > 0 and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


def chunk_text(text, overlap, chunk_size=200):
    chunks = fixed_size_chunking(text, overlap, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i}. {chunk}")


def semantic_chunking(text, max_chunk_size=200, overlap=0):
    text = text.strip()
    if not text:
        return []
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be less than max_chunk_size")

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return []

    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    chunks = []
    step_size = max_chunk_size - overlap
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i : i + max_chunk_size]
        chunks.append(" ".join(chunk_sentences))
        i += step_size
        if i >= len(sentences):
            break
    return chunks


def chunk_text_semantic(text, max_chunk_size=200, overlap=0):
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Divided {len(text)} characters into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"{i}. {chunk}")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_embeddings_path = Path("cache/chunk_embeddings.npy")
        self.chunk_metadata = None
        self.chunk_metadata_path = Path("cache/chunk_metadata.json")
        self.movie_index_map = {}  # Maps movie_idx to movie object

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        self.movie_index_map = {}
        for midx, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
            self.movie_index_map[midx] = doc

        all_chunks = []
        chunk_metadata = []
        for midx, doc in enumerate(documents):
            if doc["description"].strip() == "":
                continue
            _chunks = semantic_chunking(doc["description"], overlap=1, max_chunk_size=4)
            for cidx, chunk in enumerate(_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": midx, "chunk_idx": cidx, "total_chunks": len(_chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = {
            "chunks": chunk_metadata,
            "total_chunks": len(all_chunks),
        }
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(self.chunk_metadata, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        self.movie_index_map = {}
        for midx, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
            self.movie_index_map[midx] = doc

        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        movie_scores = defaultdict(lambda: 0)
        for idx in range(len(self.chunk_embeddings)):
            chunk_embedding = self.chunk_embeddings[idx]
            metadata = self.chunk_metadata["chunks"][idx]
            midx, cidx = metadata["movie_idx"], metadata["chunk_idx"]
            similarities = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {"movie_idx": midx, "chunk_idx": cidx, "score": similarities}
            )
            movie_scores[midx] = max(movie_scores[midx], similarities)
        movie_scores_sorted = sorted(
            movie_scores.items(), key=lambda x: x[1], reverse=True
        )

        res = []
        for midx, score in movie_scores_sorted[:limit]:
            doc = self.movie_index_map[midx]
            res.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score, 4),
                    "metadata": {},
                }
            )
        return res


def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query, limit=5):
    css = ChunkedSemanticSearch()
    movies = load_movies()
    css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, limit)
    for i, res in enumerate(results):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")
