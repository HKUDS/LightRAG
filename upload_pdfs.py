#!/usr/bin/env python3
"""Upload PDFs to LightRAG server."""
import os
import sys
from pathlib import Path
import requests

PDF_DIR = Path("documents/questions/docs/pdf")
API_URL = "http://localhost:9621/documents/upload"

def upload_pdfs():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to upload")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Uploading: {pdf_path.name}")
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (pdf_path.name, f, 'application/pdf')}
                response = requests.post(API_URL, files=files, timeout=120)
                result = response.json()
                print(f"  -> {result.get('status', 'unknown')}: {result.get('message', 'No message')[:80]}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    print("\nDone! Checking processing status...")

if __name__ == "__main__":
    upload_pdfs()
