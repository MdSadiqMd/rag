from lib.llm import document_citations, question_answer, summarize_documents
from algos._4_hybrid_search import HybridSearch
from lib.search_utils import load_movies


def question_answering(query):
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


def document_summarization(query, limit=5):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query=query, k=60, limit=limit)
    print("Search Results:")
    for res in rrf_results:
        print(f"- {res['title']}")

    documents = "\n".join(
        [
            f"Title: {res['title']}\nDescription: {res['document']}"
            for res in rrf_results
        ]
    )

    rag_results = summarize_documents(query=query, documents=documents)
    print("\nLLM Summary:")
    print(rag_results)


def answer_with_citations(query, limit=5):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query=query, k=60, limit=limit)
    print("Search Results:")
    for res in rrf_results:
        print(f"- {res['title']}")

    documents = "\n".join(
        [
            f"Title: {res['title']}\nDescription: {res['document']}"
            for res in rrf_results
        ]
    )

    rag_results = document_citations(query=query, documents=documents)
    print("\nLLM Answer:")
    print(rag_results)
