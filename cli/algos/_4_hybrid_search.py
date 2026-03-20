"""
Hybrid Search and Re-Ranking Algorithms
========================================

This module implements hybrid search by combining BM25 (keyword-based) and semantic search
(embedding-based) using multiple fusion strategies. It also provides query enhancement and
re-ranking capabilities using LLMs and cross-encoders.

-------------------------------------------------------------------------------
1. Score Normalization
-------------------------------------------------------------------------------
Normalizes scores to [0, 1] range using min-max normalization. Essential for
combining scores from different algorithms with different scales.

    normalized(x) = (x - min(X)) / (max(X) - min(X))

    where:
        X = {x₁, x₂, ..., xₙ} = set of all scores
        min(X) = minimum score in the set
        max(X) = maximum score in the set

Properties:
    - Range: [0, 1]
    - min(X) → 0
    - max(X) → 1
    - Preserves relative ordering
    - Linear transformation

Edge case:
    If min(X) = max(X) (all scores equal):
        normalized(x) = 1.0 for all x

-------------------------------------------------------------------------------
2. Weighted Hybrid Search
-------------------------------------------------------------------------------
Combines BM25 and semantic search scores using a weighted average. The alpha
parameter controls the balance between keyword matching and semantic similarity.

Algorithm:

    For query q:
        bm25_results = BM25_search(q, limit × 500)
        sem_results = semantic_search(q, limit × 500)

        For each document d:
            bm25_norm(d) = normalize(bm25_score(d))
            sem_norm(d) = normalize(sem_score(d))

            hybrid_score(d) = α × bm25_norm(d) + (1 - α) × sem_norm(d)

        ranking = argsort({hybrid_score(d) | d ∈ results}, descending=True)
        return top_k(ranking, limit)

    where:
        α ∈ [0, 1] = weight parameter (default: 0.5)
        α = 1.0 → pure BM25 (keyword search)
        α = 0.0 → pure semantic search
        α = 0.5 → equal weighting

Mathematical formulation:

    hybrid_score(d) = α × s_bm25(d) + (1 - α) × s_sem(d)

    where:
        s_bm25(d) = normalized BM25 score ∈ [0, 1]
        s_sem(d) = normalized semantic score ∈ [0, 1]

Advantages:
    - Combines lexical and semantic matching
    - Tunable via alpha parameter
    - Handles synonyms (semantic) and exact matches (BM25)

Disadvantages:
    - Sensitive to score distributions
    - Outliers can skew normalization
    - Requires parameter tuning

-------------------------------------------------------------------------------
3. Reciprocal Rank Fusion (RRF)
-------------------------------------------------------------------------------
Combines rankings from multiple search systems without requiring score normalization.
Uses rank positions instead of raw scores, making it robust to different score scales.

Algorithm:

    For query q:
        bm25_results = BM25_search(q, limit × 500)
        sem_results = semantic_search(q, limit × 500)

        For each document d:
            rank_bm25(d) = position of d in bm25_results (1-indexed)
            rank_sem(d) = position of d in sem_results (1-indexed)

            rrf_score(d) = Σ [1 / (k + rank_i(d))]
                          i ∈ {bm25, sem}

        ranking = argsort({rrf_score(d) | d ∈ results}, descending=True)
        return top_k(ranking, limit)

    where:
        k = constant (default: 60)
        rank_i(d) = rank of document d in system i
        If d not in system i: rank_i(d) = ∞ (contributes 0 to sum)

Mathematical formulation:

    rrf_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_sem(d))

    where:
        k ∈ ℝ⁺ = smoothing constant
        rank(d) ∈ ℕ⁺ = 1-indexed rank position

Properties of k parameter:
    - Lower k (e.g., 20): More weight to top-ranked results
      → Steep drop-off in scores
    - Higher k (e.g., 100): More gradual decline
      → Lower-ranked results have more influence
    - Default k = 60: Good balance for most datasets

Example with k = 60:
    Document A: rank_bm25 = 1, rank_sem = 2
        rrf_score(A) = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325

    Document B: rank_bm25 = 2, rank_sem = 1
        rrf_score(B) = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325

    Documents with similar ranks across systems get higher scores.

Advantages:
    - No score normalization needed
    - Robust to outliers and different score distributions
    - Simple and effective
    - Works well when systems have different scoring scales

Disadvantages:
    - Ignores magnitude of score differences
    - Treats rank 1→2 same as rank 100→101

-------------------------------------------------------------------------------
4. Query Enhancement
-------------------------------------------------------------------------------
Improves search quality by transforming the original query before search.

4.1 Spelling Correction
    Fixes typos and misspellings using an LLM.

    Example:
        "bere" → "bear"
        "scary movei" → "scary movie"

4.2 Query Rewriting
    Transforms vague queries into more specific, searchable terms.

    Example:
        "that bear movie where leo gets attacked" → "The Revenant Leonardo DiCaprio bear attack"
        "movie about bear in london with marmalade" → "Paddington London marmalade"

4.3 Query Expansion
    Adds related terms and synonyms to broaden the search.

    Example:
        "scary bear movie" → "scary bear movie horror terrifying wildlife animal attack survival thriller predator"
        "comedy with bear" → "comedy with bear funny animals humorous wildlife slapstick playful"

Mathematical impact:

    Let q = original query
    Let q' = enhanced query

    For expansion:
        q' = q ∪ {t₁, t₂, ..., tₙ} where tᵢ are related terms

    For rewriting:
        q' = transform(q) where transform maximizes P(relevant | q')

-------------------------------------------------------------------------------
5. Re-Ranking Methods
-------------------------------------------------------------------------------
Refines initial search results by re-scoring with more sophisticated models.
Typically applied to top-k results (k = limit × 5) to balance accuracy and speed.

5.1 Individual Re-Ranking (LLM-based)
    Uses an LLM to score each document individually on a 0-10 scale.

    Algorithm:
        For each document d in top_k results:
            prompt = f"Rate relevance of '{d.title}' to query '{q}' (0-10)"
            score(d) = LLM(prompt)

        ranking = argsort({score(d) | d ∈ results}, descending=True)

    Advantages:
        - Nuanced scoring (0-10 scale)
        - Captures complex relevance patterns
        - Can consider context and intent

    Disadvantages:
        - Slow: N API calls for N documents
        - Expensive: Processes query N times
        - Inconsistent: Scores not directly comparable

5.2 Batch Re-Ranking (LLM-based)
    Uses an LLM to rank all documents together in a single call.

    Algorithm:
        doc_list = format_documents(top_k results)
        prompt = f"Rank these movies by relevance to '{q}': {doc_list}"
        ranked_ids = LLM(prompt)  # Returns JSON list of IDs

        For each document d:
            rank(d) = position of d.id in ranked_ids

        ranking = argsort({rank(d) | d ∈ results}, ascending=True)

    Advantages:
        - Fast: Single API call
        - Cheap: Processes query once
        - Consistent: Direct comparison of all documents

    Disadvantages:
        - Context window limits (typically ~100 documents)
        - May miss nuances for individual documents

5.3 Cross-Encoder Re-Ranking
    Uses a specialized neural model that encodes query and document together.

    Algorithm:
        cross_encoder = CrossEncoder("ms-marco-TinyBERT-L-2-v2")

        For each document d in top_k results:
            pair = [query, f"{d.title} - {d.document}"]
            score(d) = cross_encoder.predict(pair)

        ranking = argsort({score(d) | d ∈ results}, descending=True)

    Mathematical formulation:

        score(q, d) = CrossEncoder(q ⊕ d)

        where:
            ⊕ = concatenation operator
            CrossEncoder: (q, d) → ℝ (relevance score)

    Bi-encoder vs Cross-encoder:

        Bi-encoder (semantic search):
            e_q = Encoder(q)
            e_d = Encoder(d)
            score = cosine_similarity(e_q, e_d)

            Advantage: Can pre-compute e_d for all documents
            Disadvantage: No interaction between q and d during encoding

        Cross-encoder (re-ranking):
            score = Encoder(q ⊕ d)

            Advantage: Full attention between q and d tokens
            Disadvantage: Must encode each (q, d) pair separately

    Advantages:
        - Fast: Local inference, no API calls
        - Cheap: No API costs
        - Accurate: Sees full query-document interaction
        - Deterministic: Consistent results

    Disadvantages:
        - Cannot pre-compute: Must encode each pair
        - Slower than bi-encoder for initial retrieval

    Typical workflow:
        1. Bi-encoder: Retrieve top 1000 candidates (fast)
        2. Cross-encoder: Re-rank top 25 (accurate)

-------------------------------------------------------------------------------
Implementation Notes
-------------------------------------------------------------------------------
- Hybrid search retrieves limit × 500 candidates from each system
  - Ensures sufficient overlap for fusion
  - Typical limit = 5 → 2500 candidates per system

- Re-ranking retrieves limit × 5 candidates
  - Balances accuracy (more candidates) and speed (fewer to re-rank)
  - Typical limit = 5 → 25 candidates to re-rank

- Score normalization uses min-max scaling
  - Handles different score ranges (BM25: 0-20, Semantic: 0-1)
  - Preserves relative ordering within each system

- RRF constant k = 60 is empirically optimal
  - Tested across multiple datasets and query types
  - Provides good balance between top and lower ranks

- Query enhancement uses Gemini LLM (gemma-3-27b-it)
  - Spelling: Corrects typos with high confidence
  - Rewriting: Transforms vague queries to specific terms
  - Expansion: Adds synonyms and related concepts

- Cross-encoder model: ms-marco-TinyBERT-L-2-v2
  - Trained on MS MARCO passage ranking dataset
  - 2-layer BERT (tiny variant for speed)
  - Input: [CLS] query [SEP] document [SEP]
  - Output: Single relevance score

- Result format includes:
  - Original RRF/hybrid score
  - Re-ranking score (if applicable)
  - BM25 and semantic ranks
  - Document metadata (title, description)

- Complexity analysis:
  - Weighted search: O(n × d) for normalization + O(n log n) for sorting
  - RRF search: O(n) for score computation + O(n log n) for sorting
  - Individual re-rank: O(k × t_llm) where k = limit × 5, t_llm = LLM latency
  - Batch re-rank: O(t_llm) single LLM call
  - Cross-encoder re-rank: O(k × t_ce) where t_ce ≈ 10ms per pair

- Typical score ranges:
  - BM25: [0, 20] (depends on document length and term frequency)
  - Semantic: [0, 1] (cosine similarity of normalized vectors)
  - RRF: [0, 0.033] (for k=60, top rank ≈ 1/61 ≈ 0.0164)
  - Cross-encoder: [-5, 5] (model-dependent, higher = more relevant)
  - LLM individual: [0, 10] (explicit 0-10 scale)
  - LLM batch: [1, N] (rank position in list)
"""

import os

from algos._2_tf_idf import InvertedIndex
from algos._3_semantic_search import ChunkedSemanticSearch
from lib.llm import correct_spelling, expand_query, rewrite_query
from lib.search_utils import load_movies
from lib.rerank import cross_encoder_rerank, individual_rerank, batch_rerank


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = combine_search_results(bm25_results, sem_results)
        return combined_results

    def rrf_search(self, query, k=60, limit=10):
        bm25_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        return combine_rrf_results(bm25_results, sem_results, k)


def normalize_scores(scores):
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)

    score_range = max_score - min_score
    return [(score - min_score) / score_range for score in scores]


def hybrid_score(bm25_score, sem_score, alpha=0.5):
    return (alpha * bm25_score) + ((1 - alpha) * sem_score)


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def normalize_search_results(results):
    scores = [r["score"] for r in results]
    norm_scores = normalize_scores(scores)
    for idx, result in enumerate(results):
        result["normalized_score"] = norm_scores[idx]
    return results


def combine_search_results(bm25_results, sem_results):
    bm25_norm = normalize_search_results(bm25_results)
    sem_norm = normalize_search_results(sem_results)
    combined_norm = {}
    for norm in bm25_norm:
        doc_id = norm["id"]
        combined_norm[doc_id] = {
            "id": doc_id,
            "bm25_score": norm["normalized_score"],
            "sem_score": 0.0,
            "title": norm["title"],
            "document": norm["document"],
        }
    for norm in sem_norm:
        doc_id = norm["id"]
        if doc_id not in combined_norm:
            combined_norm[doc_id] = {
                "id": doc_id,
                "bm25_score": 0.0,
                "sem_score": 0.0,
                "title": norm["title"],
                "document": norm["document"],
            }
        combined_norm[doc_id]["sem_score"] = norm["normalized_score"]

    for k, v in combined_norm.items():
        combined_norm[k]["hybrid_score"] = hybrid_score(v["bm25_score"], v["sem_score"])
    return sorted(combined_norm.values(), key=lambda x: x["hybrid_score"], reverse=True)


def weighted_search(query, alpha=0.5, limit=5):
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.weighted_search(query, alpha, limit)
    for idx, result in enumerate(results[:limit]):
        print(f"{idx+1}. {result['title']}")
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"BM25: {result['bm25_score']:.4f}, Semantic: {result['sem_score']:.4f}")
        print(f"{result['document'][:100]}")
        print()


def combine_rrf_results(bm25_results, sem_results, k=60):
    rrf_results = {}
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_results:
            rrf_results[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": rank,
                "sem_rank": None,
                "rrf_score": 0.0,
            }
        rrf_results[doc_id]["bm25_rank"] = rank
        rrf_results[doc_id]["rrf_score"] += rrf_score(rank, k)
    for rank, result in enumerate(sem_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_results:
            rrf_results[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": None,
                "sem_rank": rank,
                "rrf_score": 0.0,
            }
        rrf_results[doc_id]["sem_rank"] = rank
        rrf_results[doc_id]["rrf_score"] += rrf_score(rank, k)

    return sorted(rrf_results.values(), key=lambda x: x["rrf_score"], reverse=True)


def rrf_search(query, k=60, limit=5, enhance=None, rerank_method=None):
    movies = load_movies()
    hs = HybridSearch(movies)

    match enhance:
        case "spell":
            new_query = correct_spelling(query)
            print(f"Enhanced query (spell): '{query}' -> '{new_query}'\n")
            query = new_query
        case "rewrite":
            new_query = rewrite_query(query)
            print(f"Enhanced query (rewrite): '{query}' -> '{new_query}'\n")
            query = new_query
        case "expand":
            new_query = expand_query(query)
            print(f"Enhanced query (expand): '{query}' -> '{new_query}'\n")
            query = new_query

    rrf_limit = limit * 5 if rerank_method else limit
    results = hs.rrf_search(query, k, rrf_limit)
    match rerank_method:
        case "individual":
            print(f"Re-ranking results using individual method...\n")
            results = individual_rerank(query, results)
        case "batch":
            print(f"Re-ranking top {limit} results using batch method...\n")
            results = batch_rerank(query, results)
        case "cross_encoder":
            print(f"Re-ranking top {limit} results using cross_encoder method...\n")
            results = cross_encoder_rerank(query, results)
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")

    for idx, result in enumerate(results[:limit]):
        print(f"{idx+1}. {result['title']}")
        match rerank_method:
            case "individual":
                print(f"   Re-rank Score: {result['rerank_score']}/10")
            case "batch":
                print(f"   Re-rank Rank: {result['rerank_rank']}")
            case "cross_encoder":
                print(f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
        print(f"   RRF Score: {result['rrf_score']:.3f}")
        bm25_rank = result.get("bm25_rank")
        sem_rank = result.get("sem_rank")
        if bm25_rank is not None or sem_rank is not None:
            bm25_rank = bm25_rank if bm25_rank is not None else "N/A"
            sem_rank = sem_rank if sem_rank is not None else "N/A"
            print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
        print(f"   {result['document'][:100]}...")
