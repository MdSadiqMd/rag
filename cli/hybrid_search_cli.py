import argparse
from algos._4_hybrid_search import normalize_scores, weighted_search, rrf_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser("normalize", help="Normalize scores")
    norm_parser.add_argument(
        "scores", type=float, nargs="+", help="list of scores to normalize"
    )

    ws_parser = subparsers.add_parser(
        name="weighted-search", help="a hybrid search with weighted average combination"
    )
    ws_parser.add_argument("query", type=str, help="user query to find related docs")
    ws_parser.add_argument(
        "--alpha", type=float, default=0.5, help="precentage of weight for bm25"
    )
    ws_parser.add_argument(
        "--limit", type=int, default=5, help="number of results to return"
    )

    rrf_parser = subparsers.add_parser(
        name="rrf-search", help="a hybrid search using Reciprocal Rank Fusion"
    )
    rrf_parser.add_argument("query", type=str, help="user query to find related docs")
    rrf_parser.add_argument(
        "-k", type=int, default=60, help="k parameter for RRF scoring"
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="number of results to return"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Rerank method: 'individual' uses LLM to score each result individually (0-10), 'batch' ranks all results together in one LLM call, 'cross_encoder' uses a cross-encoder model for fast re-ranking",
    )
    rrf_parser.add_argument(
        "--debug",
        type=str,
        help="Debug method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(
                args.query,
                args.k,
                args.limit,
                args.enhance,
                args.rerank_method,
                args.debug,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
