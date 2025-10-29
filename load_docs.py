#!/usr/bin/env python3
"""
Simplified script to load documentation into LightRAG
Loads all markdown files from a directory structure
"""

import asyncio
import httpx
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional


async def load_document_to_lightrag(
    content: str, 
    title: str, 
    doc_url: str,
    endpoint: str = "http://localhost:9621",
    headers: Optional[Dict[str, str]] = None
) -> bool:
    """Load a single document to LightRAG with URL reference"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            request_headers = {"Content-Type": "application/json"}
            if headers:
                request_headers.update(headers)
            response = await client.post(
                f"{endpoint}/documents/text",
                headers=request_headers,
                json={
                    "text": content,
                    "file_source": doc_url
                }
            )
            
            if response.status_code == 200:
                print(f"✅ Loaded: {title}")
                return True
            else:
                print(f"❌ Failed to load {title}: {response.status_code}")
                if response.status_code == 500:
                    try:
                        error_detail = response.json()
                        print(f"   Error details: {error_detail}")
                    except:
                        print(f"   Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error loading {title}: {e}")
        return False


def convert_file_path_to_url(relative_path: str, base_url: str) -> str:
    """Convert file path to documentation URL"""
    # Ensure base URL ends with /
    if not base_url.endswith('/'):
        base_url += '/'
    
    # Handle special cases
    if relative_path in ["README.md", "SUMMARY.md"]:
        return base_url.rstrip('/')
    
    # Remove .md extension and convert path
    url_path = relative_path.replace(".md", "")
    
    # Handle README files in subdirectories - they map to the directory URL
    if url_path.endswith("/README"):
        url_path = url_path[:-7]  # Remove "/README"
    
    # Clean up any double slashes
    url_path = url_path.strip("/")
    
    return f"{base_url}{url_path}"


def load_markdown_files(docs_path: Path, mode: str = "files", base_url: str = None) -> List[tuple]:
    """Load all markdown files from directory structure
    
    Args:
        docs_path: Path to documentation directory
        mode: 'files' for file paths, 'urls' for URL references
        base_url: Base URL for documentation site (required for 'urls' mode)
    """
    if not docs_path.exists():
        raise FileNotFoundError(f"Documentation directory not found: {docs_path}")
    
    if mode == "urls" and not base_url:
        raise ValueError("base_url is required when mode is 'urls'")
    
    # Find all markdown files, excluding SUMMARY.md as it's just the table of contents
    md_files = [f for f in docs_path.rglob("*.md") if f.name != "SUMMARY.md"]
    print(f"📚 Found {len(md_files)} markdown files")
    print(f"🔧 Mode: {mode}")
    if mode == "urls":
        print(f"🌐 Base URL: {base_url}")
    
    documents = []
    
    for file_path in md_files:
        try:
            # Load content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                continue
                
            # Generate title from filename
            title = file_path.stem.replace("-", " ").replace("_", " ").title()
            if title.lower() == "readme":
                # Use parent directory name for README files
                title = f"{file_path.parent.name.replace('-', ' ').replace('_', ' ').title()} Overview"
            
            # Get relative path for metadata
            relative_path = str(file_path.relative_to(docs_path))
            
            if mode == "files":
                # Use file path as reference
                reference = relative_path
                source_info = f"File: {file_path.name}"
                
                # Prepare content with file metadata
                content_with_metadata = f"""
Title: {title}
Path: {relative_path}
Source: {source_info}

{content}
"""
            else:  # urls mode
                # Convert file path to documentation URL
                reference = convert_file_path_to_url(relative_path, base_url)
                source_info = f"Documentation Site"
                
                # Prepare content with URL metadata
                content_with_metadata = f"""
Title: {title}
URL: {reference}
Source: {source_info}

{content}
"""
            
            documents.append((content_with_metadata, title, reference))
            
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
            continue
    
    return documents


async def test_lightrag_health(
    endpoint: str = "http://localhost:9621",
    headers: Optional[Dict[str, str]] = None
) -> bool:
    """Test if LightRAG is accessible"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{endpoint}/health", headers=headers)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ LightRAG is healthy: {health_data.get('status')}")
                return True
            else:
                print(f"❌ LightRAG health check failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Cannot connect to LightRAG: {e}")
        return False


async def test_query(
    endpoint: str = "http://localhost:9621",
    headers: Optional[Dict[str, str]] = None
) -> None:
    """Test a sample query"""
    print(f"\n🧪 Testing query...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            request_headers = {"Content-Type": "application/json"}
            if headers:
                request_headers.update(headers)
            response = await client.post(
                f"{endpoint}/query",
                headers=request_headers,
                json={"query": "What is this documentation about?", "mode": "local"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Query successful!")
                print(f"Response: {result['response'][:200]}...")
            else:
                print(f"❌ Query failed: {response.status_code}")
                if response.status_code == 500:
                    try:
                        error_detail = response.json()
                        print(f"   Error details: {error_detail}")
                    except:
                        print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Query error: {e}")


async def main():
    """Main loading function"""
    parser = argparse.ArgumentParser(
        description="Load documentation into LightRAG with file paths or URL references",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load with file path references (default mode)
  python load_docs.py ../apolo-copilot/docs/official-apolo-documentation/docs
  
  # Load with URL references
  python load_docs.py docs/ --mode urls --base-url https://docs.apolo.us/index/
  
  # Load Apolo docs with URL references (common use case)
  python load_docs.py ../apolo-copilot/docs/official-apolo-documentation/docs \\
    --mode urls --base-url https://docs.apolo.us/index/
  
  # Use custom endpoint
  python load_docs.py docs/ --endpoint https://lightrag.example.com
  
  # Load with different documentation base URL
  python load_docs.py docs/ --mode urls --base-url https://my-docs.example.com/docs/
"""
    )
    
    parser.add_argument(
        "docs_path",
        nargs="?",
        default="../apolo-copilot/docs/official-apolo-documentation/docs",
        help="Path to documentation directory (default: ../apolo-copilot/docs/official-apolo-documentation/docs)"
    )
    parser.add_argument(
        "--mode",
        choices=["files", "urls"],
        default="files",
        help="Reference mode: 'files' for file paths, 'urls' for URL references (default: files)"
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        help="Base URL for documentation site (required when mode=urls). Example: https://docs.apolo.us/index/"
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:9621",
        help="LightRAG endpoint URL (default: http://localhost:9621)"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip test query after loading"
    )
    
    args = parser.parse_args()
    api_key = os.getenv("LIGHTRAG_API_KEY")
    if api_key:
        auth_headers = {"X-API-Key": api_key}
    else:
        auth_headers = None
        print("ℹ️ LIGHTRAG_API_KEY not set, continuing without authentication.")
    
    print("🚀 Loading Documentation into LightRAG")
    print("=" * 60)
    print(f"📁 Documentation path: {args.docs_path}")
    print(f"🔧 Reference mode: {args.mode}")
    if args.mode == "urls":
        if args.base_url:
            print(f"🌐 Base URL: {args.base_url}")
        else:
            print("❌ Error: --base-url is required when mode is 'urls'")
            sys.exit(1)
    print(f"🌐 LightRAG endpoint: {args.endpoint}")
    print()
    
    # Test LightRAG connectivity
    if not await test_lightrag_health(args.endpoint, headers=auth_headers):
        print("❌ Cannot connect to LightRAG. Please ensure it's running and accessible.")
        sys.exit(1)
    
    # Load documents
    docs_path = Path(args.docs_path).resolve()
    try:
        documents = load_markdown_files(docs_path, args.mode, args.base_url)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    if not documents:
        print("❌ No markdown files found to load")
        sys.exit(1)
    
    # Calculate statistics
    total_content = sum(len(content) for content, _, _ in documents)
    avg_content = total_content // len(documents) if documents else 0
    
    print(f"📊 Total content: {total_content:,} characters")
    print(f"📊 Average length: {avg_content:,} characters")
    
    # Load documents
    successful = 0
    failed = 0
    
    print(f"\n🔄 Starting to load documents...")
    
    for i, (content, title, doc_url) in enumerate(documents):
        success = await load_document_to_lightrag(
            content,
            title,
            doc_url,
            args.endpoint,
            headers=auth_headers
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"📈 Progress: {i + 1}/{len(documents)} ({successful} success, {failed} failed)")
        
        # Small delay to avoid overwhelming the service
        await asyncio.sleep(0.3)
    
    print(f"\n✅ Loading complete!")
    print(f"📊 Successful: {successful}")
    print(f"📊 Failed: {failed}")
    
    # Test query unless disabled
    if not args.no_test and successful > 0:
        await test_query(args.endpoint, headers=auth_headers)


if __name__ == "__main__":
    asyncio.run(main())
