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
sentence transformer model (BERT-based). Similar texts have similar embeddings in
vector space, enabling semantic similarity computation.

    embedding(text) = SentenceTransformer.encode(text)

    where:
        text      = input text (title + description)
        embedding = dense vector ∈ ℝᵈ (d = 384 for all-MiniLM-L6-v2)

The model uses:
    - Tokenization: text → token IDs
    - BERT encoding: token IDs → contextualized representations
    - Mean pooling: aggregate token embeddings → sentence embedding
    - L2 normalization: ||embedding|| = 1

-------------------------------------------------------------------------------
2. Cosine Similarity
-------------------------------------------------------------------------------
Measures the similarity between two vectors by computing the cosine of the angle
between them. For normalized vectors, this simplifies to the dot product.

    cosine_similarity(v₁, v₂) = (v₁ · v₂) / (||v₁|| × ||v₂||)

    For L2-normalized vectors (||v|| = 1):
        cosine_similarity(v₁, v₂) = v₁ · v₂ = Σᵢ v₁ᵢ × v₂ᵢ

    where:
        v₁, v₂ ∈ ℝᵈ = vector embeddings
        ·           = dot product
        ||v||       = L2 norm = √(Σᵢ vᵢ²)

Properties:
    - Range: [-1, 1]
    - cos(θ) = 1  → vectors are identical (θ = 0°)
    - cos(θ) = 0  → vectors are orthogonal (θ = 90°)
    - cos(θ) = -1 → vectors are opposite (θ = 180°)

-------------------------------------------------------------------------------
3. Semantic Search
-------------------------------------------------------------------------------
Finds documents most similar to a query by comparing embeddings in vector space.
Uses cosine similarity as the distance metric.

Algorithm:
    1. Encode query: q = embedding(query) ∈ ℝᵈ
    2. For each document dᵢ:
         score(dᵢ) = cosine_similarity(q, embedding(dᵢ))
    3. Sort documents by score (descending)
    4. Return top-k results

Mathematical formulation:

    ranking = argsort({score(d₁), score(d₂), ..., score(dₙ)}, descending=True)
    results = {d_ranking[0], d_ranking[1], ..., d_ranking[k-1]}

    where:
        n = total number of documents
        k = desired number of results (limit)

Unlike keyword search (exact token matching), semantic search captures:
    - Synonyms: "car" ≈ "automobile"
    - Paraphrases: "quick brown fox" ≈ "fast auburn canine"
    - Conceptual similarity: "king" - "man" + "woman" ≈ "queen"

-------------------------------------------------------------------------------
4. Fixed-Size Chunking
-------------------------------------------------------------------------------
Splits text into overlapping chunks of fixed word count. Useful for processing
long documents that exceed model context limits (typically 512 tokens).

Algorithm:

    words = text.split()
    chunks = []
    step_size = chunk_size - overlap

    for i in range(0, len(words), step_size):
        chunk = words[i : i + chunk_size]
        chunks.append(chunk)

    where:
        chunk_size = maximum words per chunk (default: 200)
        overlap    = number of words shared between consecutive chunks
        step_size  = stride = chunk_size - overlap

Mathematical properties:

    Number of chunks ≈ ⌈(n - overlap) / step_size⌉

    where n = total number of words

    Coverage: Each word appears in min(⌈overlap / step_size⌉ + 1, num_chunks) chunks

Example with chunk_size=3, overlap=1:
    Text: "A B C D E F" (6 words)
    step_size = 3 - 1 = 2

    Chunk 0: words[0:3] = "A B C"
    Chunk 1: words[2:5] = "C D E"  (C overlaps)
    Chunk 2: words[4:7] = "E F"    (E overlaps)

    Number of chunks = ⌈(6 - 1) / 2⌉ = 3

-------------------------------------------------------------------------------
5. Semantic Chunking
-------------------------------------------------------------------------------
Splits text into overlapping chunks by sentences rather than words. Preserves
sentence boundaries for better semantic coherence and grammatical completeness.

Algorithm:

    sentences = split_by_sentence(text)  # regex: (?<=[.!?])\\s+
    chunks = []
    step_size = max_chunk_size - overlap

    for i in range(0, len(sentences), step_size):
        chunk = sentences[i : i + max_chunk_size]
        chunks.append(join(chunk))

    where:
        max_chunk_size = maximum sentences per chunk (default: 200)
        overlap        = number of sentences shared between chunks
        step_size      = stride = max_chunk_size - overlap

Mathematical properties:

    Number of chunks = ⌈(s - overlap) / step_size⌉

    where s = total number of sentences

    Sentence coverage: Each sentence appears in min(⌈overlap / step_size⌉ + 1, num_chunks) chunks

Example with max_chunk_size=2, overlap=1:
    Text: "S1. S2. S3. S4." (4 sentences)
    step_size = 2 - 1 = 1

    Chunk 0: sentences[0:2] = "S1. S2."
    Chunk 1: sentences[1:3] = "S2. S3."  (S2 overlaps)
    Chunk 2: sentences[2:4] = "S3. S4."  (S3 overlaps)

    Number of chunks = ⌈(4 - 1) / 1⌉ = 3

Advantages over fixed-size chunking:
    - Preserves semantic units (complete sentences)
    - Maintains grammatical structure
    - Better context for embedding models
    - More interpretable results

-------------------------------------------------------------------------------
6. Chunked Semantic Search
-------------------------------------------------------------------------------
Improves search accuracy by operating at chunk granularity rather than document
level. Particularly effective for long documents where relevant content may be
buried within irrelevant context.

Algorithm:

    For each document dᵢ:
        chunks(dᵢ) = semantic_chunking(dᵢ)

        For each chunk cᵢⱼ ∈ chunks(dᵢ):
            score(cᵢⱼ) = cosine_similarity(q, embedding(cᵢⱼ))

        document_score(dᵢ) = max({score(cᵢⱼ) | cᵢⱼ ∈ chunks(dᵢ)})

    ranking = argsort({document_score(d₁), ..., document_score(dₙ)}, descending=True)
    results = {d_ranking[0], ..., d_ranking[k-1]}

Score aggregation strategies:

    1. Max pooling (implemented):
       score(d) = max({score(c) | c ∈ chunks(d)})

       Rationale: A document is relevant if ANY chunk is highly relevant

    2. Mean pooling (alternative):
       score(d) = (1/|chunks(d)|) × Σ score(c)

       Rationale: Overall document relevance

    3. Top-k pooling (alternative):
       score(d) = (1/k) × Σ top_k({score(c) | c ∈ chunks(d)})

       Rationale: Balance between max and mean

Mathematical advantages:

    Let d = [c₁, c₂, ..., cₘ] be a document with m chunks
    Let r be a relevant passage within d

    Whole-document embedding:
        e(d) = mean_pool([e(c₁), e(c₂), ..., e(cₘ)])

        If r is small relative to d, its signal is diluted:
        contribution(r) ≈ 1/m

    Chunk-based search:
        If r ⊆ cⱼ for some chunk cⱼ, then:
        score(cⱼ) ≈ 1 (high similarity)
        score(d) = max(..., score(cⱼ), ...) ≈ 1

        The relevant passage dominates the document score.

-------------------------------------------------------------------------------
Implementation Notes
-------------------------------------------------------------------------------
- Model: sentence-transformers/all-MiniLM-L6-v2
  - Architecture: 6-layer MiniLM (distilled from BERT)
  - Embedding dimension: 384
  - Max sequence length: 256 tokens
  - Training: Contrastive learning on 1B+ sentence pairs

- Embeddings cached to disk (NumPy .npy format) for O(1) loading
  - File size: ~7.3 MB for 5000 documents × 384 dimensions × 4 bytes/float32

- Chunk metadata stored as JSON:
  - Maps chunk index → (movie_idx, chunk_idx, total_chunks)
  - Enables reconstruction of document from chunks

- Cosine similarity computation:
  - Uses NumPy vectorized operations: np.dot(v1, v2) / (norm(v1) * norm(v2))
  - Time complexity: O(d) where d = embedding dimension
  - For normalized vectors: O(d) dot product only

- Default chunking parameters:
  - Sentences per chunk: 4
  - Sentence overlap: 1
  - Typical chunk size: 50-150 words
  - Chunks per document: ~15 (for average movie description)

- Search complexity:
  - Whole-document: O(n × d) where n = number of documents
  - Chunked: O(c × d) where c = total number of chunks (c ≈ 15n)
  - Trade-off: 15× more comparisons but better accuracy

- Results sorted by similarity score (descending):
  - Higher score = more similar to query
  - Typical score range: [0.2, 0.8] for relevant results
  - Scores < 0.3 usually indicate low relevance
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
