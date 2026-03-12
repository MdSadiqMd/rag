import argparse
from algos._3_semantic_search import (
    verify_embeddings,
    verify_model,
    embed_text,
    embed_query_text,
    search,
    chunk_text,
    chunk_text_semantic,
    embed_chunks,
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
    search_parser = subparsers.add_parser("search", help="Searching")
    search_parser.add_argument(
        "query",
        type=str,
        help="query to be search",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="limit of the search (default: 5)",
    )
    chunk_parser = subparsers.add_parser("chunk", help="Chunking")
    chunk_parser.add_argument(
        "text",
        type=str,
        help="chunk text",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="chunking overlap (default: 0)",
    )
    chunk_parser.add_argument(
        "--chunk",
        type=int,
        default=200,
        help="chunking window",
    )
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunking")
    semantic_chunk_parser.add_argument(
        "text",
        type=str,
        help="chunk text",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="chunking overlap (default: 0)",
    )
    semantic_chunk_parser.add_argument(
        "--chunk",
        type=int,
        default=200,
        help="chunking window",
    )
    embed_chunk_parser = subparsers.add_parser("embed_chunk", help="Embed Chunk")
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
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.overlap, args.chunk)
        case "semantic_chunk":
            chunk_text_semantic(args.text, args.chunk, args.overlap)
        case "embed_chunk":
            embed_chunks()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
