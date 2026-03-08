#!/usr/bin/env python3
import argparse

# from algos._1_keyword_search import search_movies
from algos._2_tf_idf import (
    search_movies,
    build_command,
    tf_command,
    idf_command,
    tf_idf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the TF-IDF index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="ID of the document")
    tf_parser.add_argument("term", type=str, help="Term to lookup")

    idf_parser = subparsers.add_parser(
        "idf", help="Caluclate Inverse document frequency"
    )
    idf_parser.add_argument("query", type=str, help="Search query")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="ID of the document")
    tfidf_parser.add_argument("term", type=str, help="Term to lookup")

    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = search_movies(args.query, 5)
            for i, result in enumerate(movies):
                print(f"{i}. {result['title']}")
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.query)
        case "tfidf":
            tf_idf_command(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
