import os
from dotenv import load_dotenv
from google import genai
from lib.search_utils import PROMPT_PATH
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"


def individual_rerank(query, documents):
    with open(PROMPT_PATH / "individual_rerank.md", "r") as f:
        prompt = f.read()
    results = []
    for doc in documents:
        _prompt = prompt.format(
            query=query, title=doc["title"], description=doc["document"]
        )
        response = client.models.generate_content(model=model, contents=_prompt)
        try:
            score = int(response.text.strip())
        except (ValueError, AttributeError):
            score = 0
        results.append({**doc, "rerank_score": score})

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


def batch_rerank(query, documents):
    import json

    with open(PROMPT_PATH / "batch_rerank.md", "r") as f:
        prompt = f.read()
    doc_list = "\n".join(
        [
            f"{i+1}. ID: {doc['id']}, Title: {doc['title']}, Description: {doc['document'][:200]}..."
            for i, doc in enumerate(documents)
        ]
    )

    _prompt = prompt.format(query=query, doc_list=doc_list)
    response = client.models.generate_content(model=model, contents=_prompt)
    try:
        ranked_ids = json.loads(response.text.strip())
    except (json.JSONDecodeError, ValueError):
        return documents

    id_to_rank = {doc_id: rank + 1 for rank, doc_id in enumerate(ranked_ids)}
    results = []
    for doc in documents:
        doc_id = doc["id"]
        rerank_rank = id_to_rank.get(doc_id, len(documents) + 1)
        results.append({**doc, "rerank_rank": rerank_rank})

    return sorted(results, key=lambda x: x["rerank_rank"])


def cross_encoder_rerank(query, documents):
    pairs = []
    for doc in documents:
        pair = [query, f"{doc.get('title', '')} - {doc.get('document', '')}"]
        pairs.append(pair)

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    scores = cross_encoder.predict(pairs)
    results = []
    for doc, score in zip(documents, scores):
        results.append({**doc, "cross_encoder_score": float(score)})

    return sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)
