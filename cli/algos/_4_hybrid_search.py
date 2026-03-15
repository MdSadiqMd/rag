import os

from algos._2_tf_idf import InvertedIndex
from algos._3_semantic_search import ChunkedSemanticSearch
from lib.llm import correct_spelling, expand_query, rewrite_query
from lib.search_utils import load_movies
from lib.rerank import individual_rerank, batch_rerank


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
    if rerank_method == "individual":
        print(f"Re-ranking results using individual method...\n")
        results = individual_rerank(query, results)
    elif rerank_method == "batch":
        print(f"Re-ranking top {limit} results using batch method...\n")
        results = batch_rerank(query, results)
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")

    for idx, result in enumerate(results[:limit]):
        print(f"{idx+1}. {result['title']}")
        if rerank_method == "individual":
            print(f"   Re-rank Score: {result['rerank_score']}/10")
        elif rerank_method == "batch":
            print(f"   Re-rank Rank: {result['rerank_rank']}")
        print(f"   RRF Score: {result['rrf_score']:.3f}")
        bm25_rank = result.get("bm25_rank")
        sem_rank = result.get("sem_rank")
        if bm25_rank is not None or sem_rank is not None:
            bm25_rank = bm25_rank if bm25_rank is not None else "N/A"
            sem_rank = sem_rank if sem_rank is not None else "N/A"
            print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
        print(f"   {result['document'][:100]}...")
