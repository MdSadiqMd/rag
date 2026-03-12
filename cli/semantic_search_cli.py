#!/usr/bin/env python3
import argparse
from algos._3_semantic_search import verify_model, embed_text


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
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
