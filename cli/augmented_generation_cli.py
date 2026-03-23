import argparse
from algos._6_retrieval_augment_generation import (
    question_answering,
    document_summarization,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    question_answer_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    question_answer_parser.add_argument("query", type=str, help="Search query for RAG")

    summarization_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarize)"
    )
    summarization_parser.add_argument("query", type=str, help="Search query for RAG")
    summarization_parser.add_argument(
        "--limit", type=int, default=5, help="Limit for returning of query"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            question_answering(args.query)
        case "summarize":
            document_summarization(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
