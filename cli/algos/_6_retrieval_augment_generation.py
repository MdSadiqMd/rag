"""
Retrieval-Augmented Generation (RAG)
=====================================

This module implements Retrieval-Augmented Generation, a technique that combines
information retrieval with large language model generation to produce accurate,
grounded responses based on retrieved documents.

-------------------------------------------------------------------------------
1. RAG Architecture
-------------------------------------------------------------------------------
RAG enhances LLM responses by providing relevant context from a knowledge base.

Traditional LLM approach:

    User Query → LLM → Response

    Problems:
        - Hallucinations (making up facts)
        - Outdated knowledge (training data cutoff)
        - No domain-specific knowledge
        - Cannot cite sources

RAG approach:

    User Query → Retrieval System → Top-k Documents → LLM + Context → Response

    Benefits:
        - Grounded in actual documents
        - Up-to-date information (from live database)
        - Domain-specific knowledge
        - Can provide citations

Mathematical formulation:

    P(response | query) = P(response | query, context)

    where:
        context = {d₁, d₂, ..., dₖ} = top-k retrieved documents
        dᵢ = document retrieved by search system

Two-stage process:

    Stage 1 (Retrieval):
        context = retrieve(query, k)

        Uses hybrid search (BM25 + semantic + RRF) to find relevant documents

    Stage 2 (Generation):
        response = LLM(query, context)

        LLM generates response conditioned on both query and retrieved context

-------------------------------------------------------------------------------
2. RAG Variants Implemented
-------------------------------------------------------------------------------

2.1 Basic Question Answering

    Purpose: Provide comprehensive answers to user queries

    Algorithm:
        1. Retrieve top-5 relevant documents using RRF search
        2. Format documents as context (title + description)
        3. Send query + context to LLM
        4. LLM generates detailed answer

    Prompt structure:
        "Answer the question based on provided documents.
         Query: {query}
         Documents: {docs}
         Provide a comprehensive answer..."

    Use case: General information requests about movies
    Example: "What is Jurassic Park about?"

2.2 Detailed Question Answering

    Purpose: Provide casual, conversational answers

    Algorithm:
        1. Retrieve top-k relevant documents (configurable limit)
        2. Format documents as context
        3. Send query + context to LLM with conversational instructions
        4. LLM generates casual, direct answer

    Prompt structure:
        "Answer the user's question based on provided movies.
         Question: {question}
         Documents: {docs}
         Instructions:
         - Answer directly and concisely
         - Be casual and conversational
         - Talk like a normal person..."

    Use case: Chatbot-style interactions
    Example: "who are the main characters in jurassic park?"

2.3 Document Summarization

    Purpose: Synthesize information from multiple documents

    Algorithm:
        1. Retrieve top-k relevant documents
        2. Format documents as context
        3. Send query + context to LLM with summarization instructions
        4. LLM generates synthesized summary

    Prompt structure:
        "Summarize the movies based on the query.
         Query: {query}
         Documents: {docs}
         Provide a summary..."

    Use case: Overview of multiple related movies
    Example: "movies about action and bear"

2.4 Answer with Citations

    Purpose: Provide answers with explicit source attribution

    Algorithm:
        1. Retrieve top-k relevant documents
        2. Format documents as context
        3. Send query + context to LLM with citation instructions
        4. LLM generates answer with inline citations

    Prompt structure:
        "Answer the question and cite sources.
         Query: {query}
         Documents: {docs}
         Include citations like [Title]..."

    Use case: Fact-checking, academic-style responses
    Example: "What year was Jurassic Park released?"

-------------------------------------------------------------------------------
3. RAG Quality Factors
-------------------------------------------------------------------------------

3.1 Retrieval Quality

    The quality of RAG responses depends heavily on retrieval:

    Precision impact:
        High precision → More relevant context → Better answers
        Low precision → Noisy context → Confused or incorrect answers

    Recall impact:
        High recall → Complete information → Comprehensive answers
        Low recall → Missing information → Incomplete answers

    Optimal k (number of documents):
        Too small (k=1-2): May miss important information
        Too large (k=20+): Context window limits, noise increases
        Sweet spot (k=5-10): Balance between coverage and focus

3.2 Context Window Constraints

    LLMs have limited context windows:

    Context budget allocation:
        System prompt: ~200 tokens
        User query: ~20-100 tokens
        Retrieved documents: k × avg_doc_length tokens
        Response generation: ~500-2000 tokens

    For k=5 documents, avg 200 tokens each:
        Total context: 200 + 50 + (5 × 200) + 1000 = 2250 tokens

    Trade-off:
        More documents → Better coverage, but less room for response
        Fewer documents → More response space, but may miss info

3.3 Prompt Engineering

    Prompt quality significantly affects RAG performance:

    Key elements:
        1. Clear task description
        2. Explicit instructions (tone, style, format)
        3. Context formatting (structured, easy to parse)
        4. Output constraints (length, format, citations)

    Example comparison:

        Bad prompt:
            "Answer: {query}\nDocs: {docs}"
            → Unclear expectations, poor results

        Good prompt:
            "Answer the question based on provided documents.
             Be concise and cite sources.
             Query: {query}
             Documents: {docs}
             Answer:"
            → Clear task, better results

-------------------------------------------------------------------------------
4. RAG vs Fine-Tuning
-------------------------------------------------------------------------------

Comparison of approaches:

    RAG:
        Pros:
            - No training required
            - Easy to update knowledge (just update documents)
            - Can cite sources
            - Works with any LLM
            - Cost-effective for dynamic knowledge

        Cons:
            - Depends on retrieval quality
            - Slower (two-stage process)
            - Context window limits
            - May struggle with complex reasoning

    Fine-Tuning:
        Pros:
            - Knowledge embedded in model weights
            - Single-stage (faster inference)
            - Better at complex reasoning
            - No retrieval dependency

        Cons:
            - Expensive to train
            - Hard to update knowledge (requires retraining)
            - Cannot cite sources
            - May hallucinate
            - Requires training data

When to use RAG:
    - Knowledge changes frequently
    - Need source attribution
    - Limited training resources
    - Domain-specific knowledge base

When to use fine-tuning:
    - Static knowledge domain
    - Need fast inference
    - Complex reasoning required
    - Have training resources

-------------------------------------------------------------------------------
5. RAG Evaluation
-------------------------------------------------------------------------------

Measuring RAG quality:

5.1 Retrieval Metrics (covered in _5_evaluation.py)
    - Precision@k: Are retrieved docs relevant?
    - Recall@k: Did we find all relevant docs?
    - F1@k: Balanced measure

5.2 Generation Metrics

    Factual accuracy:
        - Does response match document content?
        - Are there hallucinations?
        - Metric: Human evaluation or LLM-as-judge

    Relevance:
        - Does response address the query?
        - Is information useful?
        - Metric: Relevance score (0-3 scale)

    Citation accuracy:
        - Are citations correct?
        - Are claims properly attributed?
        - Metric: Citation precision/recall

5.3 End-to-End Metrics

    User satisfaction:
        - Did user find answer helpful?
        - Metric: Thumbs up/down, ratings

    Task completion:
        - Did user accomplish their goal?
        - Metric: Click-through rate, time on page

-------------------------------------------------------------------------------
Implementation Notes
-------------------------------------------------------------------------------
- All RAG functions use RRF search with k=60 for retrieval
  - Combines BM25 and semantic search
  - Robust to different query types
  - Default limit=5 documents (configurable)

- Document formatting:
  - Each document formatted as "Title: {title}\nDescription: {document}"
  - Documents joined with newlines
  - Provides structured context for LLM

- LLM configuration:
  - Model: gemma-3-27b-it (Gemini)
  - Temperature: Default (balanced creativity/accuracy)
  - Max tokens: Varies by task

- Function variants:
  - question_answering: Fixed limit=5, comprehensive answers
  - question_answering_detailed: Configurable limit, casual tone
  - document_summarization: Configurable limit, synthesis focus
  - answer_with_citations: Configurable limit, source attribution

- Output format:
  - Prints search results (titles only)
  - Prints LLM response with appropriate label
  - Clean, user-friendly presentation

- Error handling:
  - Retrieval failures: Returns empty context
  - LLM failures: May return partial response
  - Malformed prompts: LLM may refuse or give poor response

- Performance considerations:
  - Retrieval: ~100-500ms (depends on corpus size)
  - LLM generation: ~1-5s (depends on response length)
  - Total latency: ~1-6s per query
  - Can be optimized with caching, async processing

- Typical workflow:
  1. User submits query
  2. System retrieves relevant documents
  3. Documents formatted as context
  4. LLM generates response
  5. Response displayed to user

- Use case mapping:
  - Quick facts → question_answering_detailed
  - Detailed info → question_answering
  - Multiple movies → document_summarization
  - Fact-checking → answer_with_citations

- Complexity:
  - Time: O(retrieval) + O(generation)
    - Retrieval: O(n log n) for RRF search
    - Generation: O(context_length × response_length)
  - Space: O(k × doc_size) for context storage
"""

from lib.llm import (
    document_citations,
    question_answer,
    question_answer_detailed,
    summarize_documents,
)
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


def question_answering_detailed(query, limit):
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

    rag_results = question_answer_detailed(query=query, documents=documents)
    print("\nAnswer:")
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
