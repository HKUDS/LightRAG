#!/usr/bin/env python3
"""
Test script: Demonstrates usage of aquery_data FastAPI endpoint
Query content: Who is the author of LightRAG

Updated to handle the new data format where:
- Response includes status, message, data, and metadata fields at top level
- Actual query results (entities, relationships, chunks, references) are nested under 'data' field
- Includes backward compatibility with legacy format
"""

import requests
import time
from typing import Dict, Any

# API configuration
API_KEY = "your-secure-api-key-here-123"
BASE_URL = "http://localhost:9621"

# Unified authentication headers
AUTH_HEADERS = {"Content-Type": "application/json", "X-API-Key": API_KEY}


def test_aquery_data_endpoint():
    """Test the /query/data endpoint"""

    # Use unified configuration
    endpoint = f"{BASE_URL}/query/data"

    # Query request
    query_request = {
        "query": "who authored LighRAG",
        "mode": "mix",  # Use mixed mode to get the most comprehensive results
        "top_k": 20,
        "chunk_top_k": 15,
        "max_entity_tokens": 4000,
        "max_relation_tokens": 4000,
        "max_total_tokens": 16000,
        "enable_rerank": True,
    }

    print("=" * 60)
    print("LightRAG aquery_data endpoint test")
    print(
        "   Returns structured data including entities, relationships and text chunks"
    )
    print("   Can be used for custom processing and analysis")
    print("=" * 60)
    print(f"Query content: {query_request['query']}")
    print(f"Query mode: {query_request['mode']}")
    print(f"API endpoint: {endpoint}")
    print("-" * 60)

    try:
        # Send request
        print("Sending request...")
        start_time = time.time()

        response = requests.post(
            endpoint, json=query_request, headers=AUTH_HEADERS, timeout=30
        )

        end_time = time.time()
        response_time = end_time - start_time

        print(f"Response time: {response_time:.2f} seconds")
        print(f"HTTP status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print_query_results(data)
        else:
            print(f"Request failed: {response.status_code}")
            print(f"Error message: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed: Please ensure LightRAG API service is running")
        print("   Start command: python -m lightrag.api.lightrag_server")
    except requests.exceptions.Timeout:
        print("❌ Request timeout: Query processing took too long")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")


def print_query_results(data: Dict[str, Any]):
    """Format and print query results"""

    # Check for new data format with status and message
    status = data.get("status", "unknown")
    message = data.get("message", "")

    print(f"\n📋 Query Status: {status}")
    if message:
        print(f"📋 Message: {message}")

    # Handle new nested data format
    query_data = data.get("data", {})

    # Fallback to old format if new format is not present
    if not query_data and any(
        key in data for key in ["entities", "relationships", "chunks"]
    ):
        print("   (Using legacy data format)")
        query_data = data

    entities = query_data.get("entities", [])
    relationships = query_data.get("relationships", [])
    chunks = query_data.get("chunks", [])
    references = query_data.get("references", [])

    print("\n📊 Query result statistics:")
    print(f"   Entity count: {len(entities)}")
    print(f"   Relationship count: {len(relationships)}")
    print(f"   Text chunk count: {len(chunks)}")
    print(f"   Reference count: {len(references)}")

    # Print metadata (now at top level in new format)
    metadata = data.get("metadata", {})
    if metadata:
        print("\n🔍 Query metadata:")
        print(f"   Query mode: {metadata.get('query_mode', 'unknown')}")

        keywords = metadata.get("keywords", {})
        if keywords:
            high_level = keywords.get("high_level", [])
            low_level = keywords.get("low_level", [])
            if high_level:
                print(f"   High-level keywords: {', '.join(high_level)}")
            if low_level:
                print(f"   Low-level keywords: {', '.join(low_level)}")

        processing_info = metadata.get("processing_info", {})
        if processing_info:
            print("   Processing info:")
            for key, value in processing_info.items():
                print(f"     {key}: {value}")

    # Print entity information
    if entities:
        print("\n👥 Retrieved entities (first 5):")
        for i, entity in enumerate(entities[:5]):
            entity_name = entity.get("entity_name", "Unknown")
            entity_type = entity.get("entity_type", "Unknown")
            description = entity.get("description", "No description")
            file_path = entity.get("file_path", "Unknown source")
            reference_id = entity.get("reference_id", "No reference")

            print(f"   {i+1}. {entity_name} ({entity_type})")
            print(
                f"      Description: {description[:100]}{'...' if len(description) > 100 else ''}"
            )
            print(f"      Source: {file_path}")
            print(f"      Reference ID: {reference_id}")
            print()

    # Print relationship information
    if relationships:
        print("🔗 Retrieved relationships (first 5):")
        for i, rel in enumerate(relationships[:5]):
            src = rel.get("src_id", "Unknown")
            tgt = rel.get("tgt_id", "Unknown")
            description = rel.get("description", "No description")
            keywords = rel.get("keywords", "No keywords")
            file_path = rel.get("file_path", "Unknown source")
            reference_id = rel.get("reference_id", "No reference")

            print(f"   {i+1}. {src} → {tgt}")
            print(f"      Keywords: {keywords}")
            print(
                f"      Description: {description[:100]}{'...' if len(description) > 100 else ''}"
            )
            print(f"      Source: {file_path}")
            print(f"      Reference ID: {reference_id}")
            print()

    # Print text chunk information
    if chunks:
        print("📄 Retrieved text chunks (first 3):")
        for i, chunk in enumerate(chunks[:3]):
            content = chunk.get("content", "No content")
            file_path = chunk.get("file_path", "Unknown source")
            chunk_id = chunk.get("chunk_id", "Unknown ID")
            reference_id = chunk.get("reference_id", "No reference")

            print(f"   {i+1}. Text chunk ID: {chunk_id}")
            print(f"      Source: {file_path}")
            print(f"      Reference ID: {reference_id}")
            print(
                f"      Content: {content[:200]}{'...' if len(content) > 200 else ''}"
            )
            print()

    # Print references information (new in updated format)
    if references:
        print("📚 References:")
        for i, ref in enumerate(references):
            reference_id = ref.get("reference_id", "Unknown ID")
            file_path = ref.get("file_path", "Unknown source")
            print(f"   {i+1}. Reference ID: {reference_id}")
            print(f"      File Path: {file_path}")
            print()

    print("=" * 60)


def compare_with_regular_query():
    """Compare results between regular query and data query"""

    query_text = "LightRAG的作者是谁"

    print("\n🔄 Comparison test: Regular query vs Data query")
    print("-" * 60)

    # Regular query
    try:
        print("1. Regular query (/query):")
        regular_response = requests.post(
            f"{BASE_URL}/query",
            json={"query": query_text, "mode": "mix"},
            headers=AUTH_HEADERS,
            timeout=30,
        )

        if regular_response.status_code == 200:
            regular_data = regular_response.json()
            response_text = regular_data.get("response", "No response")
            print(
                f"   Generated answer: {response_text[:300]}{'...' if len(response_text) > 300 else ''}"
            )
        else:
            print(f"   Regular query failed: {regular_response.status_code}")
            if regular_response.status_code == 403:
                print("   Authentication failed - Please check API Key configuration")
            elif regular_response.status_code == 401:
                print("   Unauthorized - Please check authentication information")
            print(f"   Error details: {regular_response.text}")

    except Exception as e:
        print(f"   Regular query error: {str(e)}")


if __name__ == "__main__":
    # Run main test
    test_aquery_data_endpoint()

    # Run comparison test
    compare_with_regular_query()

    print("\n💡 Usage tips:")
    print("1. Ensure LightRAG API service is running")
    print("2. Adjust base_url and authentication information as needed")
    print("3. Modify query parameters to test different retrieval strategies")
    print("4. Data query results can be used for further analysis and processing")
