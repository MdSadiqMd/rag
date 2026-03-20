from lib import search_utils
from algos import _4_hybrid_search


def evaluate(limit):
    print(f"k={limit}")
    test_cases = search_utils.load_dataset()
    movies = search_utils.load_movies()

    hs = _4_hybrid_search.HybridSearch(movies)
    for test_case in test_cases:
        query = test_case["query"]
        expected_results = test_case["relevant_docs"]
        rrf_results = hs.rrf_search(query, k=60, limit=limit)

        relevant_count = 0
        for rrf_result in rrf_results:
            relevant_count += rrf_result["title"] in expected_results

        precision = relevant_count / len(expected_results)
        retrived = ", ".join([r["title"] for r in rrf_results])

        print(query)
        print(f"- Precision@{limit}: {precision}")
        print(f"- Retrived: {retrived}")
        print(f"- Relevant: {expected_results}")
