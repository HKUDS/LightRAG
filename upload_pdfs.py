#!/usr/bin/env python3
"""Upload PDFs to LightRAG server."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import requests


def upload_pdfs(pdf_dir: Path, api_url: str, timeout: float) -> int:
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f'ERROR: PDF directory does not exist or is not a directory: {pdf_dir}')
        return 1

    pdf_files = list(pdf_dir.glob('*.pdf'))
    print(f'Found {len(pdf_files)} PDFs to upload in {pdf_dir}')

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f'[{i}/{len(pdf_files)}] Uploading: {pdf_path.name}')
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (pdf_path.name, f, 'application/pdf')}
                response = requests.post(api_url, files=files, timeout=timeout)
                response.raise_for_status()
                result: Any = response.json()
                print(f'  -> {result.get("status", "unknown")}: {result.get("message", "No message")[:80]}')
        except Exception as e:
            print(f'  -> ERROR: {e}')

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Upload PDFs to LightRAG')
    parser.add_argument(
        '--pdf-dir',
        default=os.getenv('PDF_DIR', 'documents/questions/docs/pdf'),
        help='Directory containing PDF files (default: env PDF_DIR or documents/questions/docs/pdf)',
    )
    parser.add_argument(
        '--api-url',
        default=os.getenv('API_URL', 'http://localhost:9621/documents/upload'),
        help='LightRAG upload endpoint (default: env API_URL or http://localhost:9621/documents/upload)',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=float(os.getenv('UPLOAD_TIMEOUT', '120')),
        help='Request timeout in seconds (default: env UPLOAD_TIMEOUT or 120)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    exit_code = upload_pdfs(Path(args.pdf_dir), args.api_url, args.timeout)
    sys.exit(exit_code)
