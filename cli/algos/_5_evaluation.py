"""
Search Evaluation Metrics
==========================

This module implements evaluation metrics for assessing the quality of search results
by comparing retrieved documents against a golden dataset of expected relevant documents.

-------------------------------------------------------------------------------
1. Precision
-------------------------------------------------------------------------------
Measures the fraction of retrieved documents that are relevant.

Mathematical formulation:

    Precision@k = |Retrieved ∩ Relevant| / |Retrieved|

    where:
        Retrieved = set of k documents returned by search
        Relevant = set of documents marked as relevant in golden dataset
        |S| = cardinality (size) of set S
        ∩ = set intersection

Properties:
    - Range: [0, 1]
    - 1.0 = all retrieved documents are relevant (perfect precision)
    - 0.0 = no retrieved documents are relevant
    - Higher is better

Example:
    Query: "family movie about bears"
    Retrieved: [A, B, C, D, E] (5 documents)
    Relevant: [A, C, F, G] (4 documents)
    Retrieved ∩ Relevant: [A, C] (2 documents)

    Precision@5 = 2/5 = 0.4

Interpretation:
    - Precision@5 = 0.4 means 40% of retrieved results are relevant
    - Answers: "How many of the results shown to the user are useful?"
    - High precision = low false positive rate

Limitations:
    - Doesn't account for relevant documents not retrieved
    - Can be high even if many relevant documents are missed
    - Sensitive to k (number of results retrieved)

-------------------------------------------------------------------------------
2. Recall
-------------------------------------------------------------------------------
Measures the fraction of relevant documents that were successfully retrieved.

Mathematical formulation:

    Recall@k = |Retrieved ∩ Relevant| / |Relevant|

    where:
        Retrieved = set of k documents returned by search
        Relevant = set of documents marked as relevant in golden dataset
        |S| = cardinality (size) of set S
        ∩ = set intersection

Properties:
    - Range: [0, 1]
    - 1.0 = all relevant documents were retrieved (perfect recall)
    - 0.0 = no relevant documents were retrieved
    - Higher is better

Example:
    Query: "family movie about bears"
    Retrieved: [A, B, C, D, E] (5 documents)
    Relevant: [A, C, F, G] (4 documents)
    Retrieved ∩ Relevant: [A, C] (2 documents)

    Recall@5 = 2/4 = 0.5

Interpretation:
    - Recall@5 = 0.5 means 50% of relevant documents were found
    - Answers: "How many of the relevant documents did we find?"
    - High recall = low false negative rate

Limitations:
    - Doesn't account for irrelevant documents retrieved
    - Can be high even if many irrelevant documents are returned
    - Increases with k (more results = higher chance of finding relevant docs)

-------------------------------------------------------------------------------
3. F1 Score
-------------------------------------------------------------------------------
Harmonic mean of precision and recall, providing a single balanced metric.

Mathematical formulation:

    F1 = 2 × (Precision × Recall) / (Precision + Recall)

Alternative formulation:

    F1 = 2 × |Retrieved ∩ Relevant| / (|Retrieved| + |Relevant|)

Properties:
    - Range: [0, 1]
    - 1.0 = perfect precision and recall
    - 0.0 = either precision or recall is 0
    - Higher is better
    - Harmonic mean penalizes extreme values

Why harmonic mean?
    Arithmetic mean: (P + R) / 2
        Problem: (1.0 + 0.0) / 2 = 0.5 (misleading)

    Harmonic mean: 2PR / (P + R)
        Better: 2(1.0)(0.0) / (1.0 + 0.0) = 0.0 (accurate)

Example:
    Precision = 0.4, Recall = 0.5
    F1 = 2 × (0.4 × 0.5) / (0.4 + 0.5)
       = 2 × 0.2 / 0.9
       = 0.444

Interpretation:
    - Balances precision and recall
    - Useful when you care equally about both metrics
    - Lower than both P and R when they differ significantly

Comparison of metrics:

    Scenario 1: High Precision, Low Recall
        Retrieved: [A, B] (2 docs)
        Relevant: [A, C, D, E, F] (5 docs)
        P = 1/2 = 0.5, R = 1/5 = 0.2, F1 = 0.286
        → System is conservative, only returns confident matches

    Scenario 2: Low Precision, High Recall
        Retrieved: [A, B, C, D, E, F, G, H, I, J] (10 docs)
        Relevant: [A, C, E] (3 docs)
        P = 3/10 = 0.3, R = 3/3 = 1.0, F1 = 0.462
        → System is aggressive, returns many results

    Scenario 3: Balanced
        Retrieved: [A, B, C, D, E] (5 docs)
        Relevant: [A, B, C, F, G] (5 docs)
        P = 3/5 = 0.6, R = 3/5 = 0.6, F1 = 0.6
        → System has balanced performance

-------------------------------------------------------------------------------
Implementation Notes
-------------------------------------------------------------------------------
- Evaluation uses a golden dataset with query-document pairs
  - Each test case has a query and list of relevant document titles
  - Format: {"query": "...", "relevant_docs": ["Title 1", "Title 2", ...]}

- Metrics are computed per query:
  - Precision@k: fraction of top-k results that are relevant
  - Recall@k: fraction of relevant docs found in top-k results
  - F1@k: harmonic mean of precision and recall

- The evaluation function:
  1. Loads test cases from golden dataset
  2. For each test case:
     a. Runs RRF search with specified limit k
     b. Counts relevant documents in results
     c. Computes precision, recall, and F1
     d. Prints metrics and retrieved titles

- Precision calculation:
  - Numerator: count of relevant docs in retrieved results
  - Denominator: number of expected relevant docs (NOT k)
  - Note: This differs from standard Precision@k definition
  - Standard: relevant_count / k
  - This implementation: relevant_count / len(expected_results)

- Recall calculation:
  - Numerator: count of relevant docs in retrieved results
  - Denominator: total number of relevant docs in golden dataset
  - Same as standard Recall@k definition

- F1 calculation:
  - Uses standard harmonic mean formula
  - Handles edge case where P + R = 0 (would cause division by zero)

- Typical workflow:
  1. Create golden dataset with representative queries
  2. Run evaluation with different k values (e.g., k=3, k=5, k=10)
  3. Compare metrics across different search configurations
  4. Tune parameters (alpha, k, rerank methods) to optimize F1

- Use cases:
  - A/B testing different search algorithms
  - Measuring impact of query enhancement
  - Comparing rerank methods
  - Tracking search quality over time

- Limitations:
  - Requires manually curated golden dataset
  - Golden dataset may not cover all query types
  - Binary relevance (relevant/not relevant) ignores degrees of relevance
  - Doesn't account for result ordering beyond top-k cutoff

- Complexity:
  - Time: O(q × (s + k × r)) where:
    - q = number of test queries
    - s = search time per query
    - k = number of results to retrieve
    - r = number of relevant docs per query
  - Space: O(k) for storing retrieved results per query
"""

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
        recall = relevant_count / len(expected_results)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(query)
        print(f"- Precision@{limit}: {precision}")
        print(f"- Recall@{limit}: {recall}")
        print(f"- F1: {f1}")
        print(f"- Retrived: {retrived}")
        print(f"- Relevant: {expected_results}")
