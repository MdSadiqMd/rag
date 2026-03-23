import json
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


def generate_content(prompt, **kwargs):
    prompt = prompt.format(**kwargs)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def correct_spelling(query):
    with open(PROMPT_PATH / "spelling.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query=query)


def rewrite_query(query):
    with open(PROMPT_PATH / "rewrite.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query=query)


def expand_query(query):
    with open(PROMPT_PATH / "expand.md", "r") as f:
        prompt = f.read()
    expanded_terms = generate_content(prompt, query=query)
    return f"{query} {expanded_terms}"


def llm_judge(query, formatted_results):
    with open(PROMPT_PATH / "llm_judge.md", "r") as f:
        prompt = f.read()
    results = generate_content(prompt, query=query, formatted_results=formatted_results)
    results = json.loads(results)
    return results


def question_answer(query, documents):
    with open(PROMPT_PATH / "question_answer.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query=query, docs=documents)


def summarize_documents(query, documents):
    with open(PROMPT_PATH / "summarization.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query=query, docs=documents)


def document_citations(query, documents):
    with open(PROMPT_PATH / "answer_with_citations.md", "r") as f:
        prompt = f.read()
    return generate_content(prompt, query=query, docs=documents)
