#!/usr/bin/env python3
import argparse
from lib.semantic_search import verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    verify_parser = subparsers.add_parser(
        "verify", help="Verifies the embedding model loads properly"
    )
    args = parser.parse_args()

    if args.command == "verify":
        verify_model()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
