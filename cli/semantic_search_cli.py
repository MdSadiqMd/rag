#!/usr/bin/env python3
import argparse
from algos._3_semantic_search import (
    verify_embeddings,
    verify_model,
    embed_text,
    embed_query_text,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    verify_parser = subparsers.add_parser(
        "verify", help="Verifies the embedding model loads properly"
    )
    embed_parser = subparsers.add_parser("embed_text", help="Text to be encoded")
    embed_parser.add_argument(
        "text",
        type=str,
        help="Text to be embedded",
    )
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verifying embeddings"
    )
    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Embedding the Query"
    )
    embed_query_parser.add_argument(
        "query",
        type=str,
        help="Query to be embedded",
    )
    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
