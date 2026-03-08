#!/usr/bin/env python3
import argparse

# from algos._1_keyword_search import search_movies
from algos._2_tf_idf import search_movies, build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the TF-IDF index")

    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = search_movies(args.query, 5)
            for i, result in enumerate(movies):
                print(f"{i}. {result['title']}")
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
