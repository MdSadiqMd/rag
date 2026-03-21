from lib.llm import question_answer
from algos._4_hybrid_search import HybridSearch
from lib.search_utils import load_movies


def rag(query):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query=query, k=60, limit=5)
    print("Search Results:")
    for res in rrf_results:
        print(f"- {res['title']}")

    documents = "\n".join(
        [
            f"Title: {res['title']}\nDescription: {res['document']}"
            for res in rrf_results
        ]
    )

    rag_results = question_answer(query=query, documents=documents)
    print("\nRAG Response:")
    print(rag_results)
