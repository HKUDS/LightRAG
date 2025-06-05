#!/usr/bin/env python
"""
Example script demonstrating the basic usage of MinerU parser

This example shows how to:
1. Parse different types of documents (PDF, images, office documents)
2. Use different parsing methods
3. Display document statistics
"""

import os
import argparse
from lightrag.mineru_parser import MineruParser


def parse_document(
    file_path: str, output_dir: str = None, method: str = "auto", stats: bool = False
):
    """
    Parse a document using MinerU parser

    Args:
        file_path: Path to the document
        output_dir: Output directory for parsed results
        method: Parsing method (auto, ocr, txt)
        stats: Whether to display content statistics
    """
    try:
        # Parse the document
        content_list, md_content = MineruParser.parse_document(
            file_path=file_path, parse_method=method, output_dir=output_dir
        )

        # Display statistics if requested
        if stats:
            print("\nDocument Statistics:")
            print(f"Total content blocks: {len(content_list)}")

            # Count different types of content
            content_types = {}
            for item in content_list:
                content_type = item.get("type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1

            print("\nContent Type Distribution:")
            for content_type, count in content_types.items():
                print(f"- {content_type}: {count}")

        return content_list, md_content

    except Exception as e:
        print(f"Error parsing document: {str(e)}")
        return None, None


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU Parser Example")
    parser.add_argument("file_path", help="Path to the document to parse")
    parser.add_argument("--output", "-o", help="Output directory path")
    parser.add_argument(
        "--method",
        "-m",
        choices=["auto", "ocr", "txt"],
        default="auto",
        help="Parsing method (auto, ocr, txt)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Display content statistics"
    )

    args = parser.parse_args()

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Parse document
    content_list, md_content = parse_document(
        args.file_path, args.output, args.method, args.stats
    )


if __name__ == "__main__":
    main()
