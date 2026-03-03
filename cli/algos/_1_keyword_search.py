"""
Keyword Search Algorithm

This algorithm performs a keyword search on a list of movies
It tokenizes the query and movie titles, and then checks if any of the query tokens are present in any of the movie titles
If any of the query tokens are present in any of the movie titles, the movie is added to the results
If any of the query tokens are stopwords, the movie is not added to the results
Added stemming to the query and movie tokens to improve the search results
The results are returned in descending order of relevance
"""
from lib.search_utils import load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]
    return [stemmer.stem(tok) for tok in tokens]

def has_matching_tokens(query_tokens: list[str], movie_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False

def search_movies(query: str, n_results: int = 10) -> list[dict]:
    movies = load_movies()
    stopwords = load_stopwords()
    res = []
    query_tokens = [tok for tok in tokenize_text(query) if tok not in stopwords]
    for movie in movies:
        if has_matching_tokens(query_tokens, tokenize_text(movie["title"])):
            res.append(movie)
        if len(res) >= n_results:
            break
    return res
