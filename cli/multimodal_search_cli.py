import argparse
from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    multimodal_parser = subparsers.add_parser(
        name="verify_image_embedding", help="Normalize a list of scores"
    )
    multimodal_parser.add_argument(
        "image_fpath", type=str, help="path of the image file"
    )

    image_search_parser = subparsers.add_parser(
        name="image_search", help="Image Search"
    )
    image_search_parser.add_argument(
        "image_fpath", type=str, help="path of the image file"
    )
    image_search_parser.add_argument(
        "--limit", type=int, default=5, help="count of return"
    )

    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_fpath)
        case "image_search":
            image_search_command(args.image_fpath, args.limit)


if __name__ == "__main__":
    main()
