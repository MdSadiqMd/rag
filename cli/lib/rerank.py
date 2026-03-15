import os
from dotenv import load_dotenv
from google import genai
from lib.search_utils import PROMPT_PATH

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
