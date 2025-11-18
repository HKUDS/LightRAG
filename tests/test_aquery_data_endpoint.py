#!/usr/bin/env python3
"""
Test script: Demonstrates usage of aquery_data FastAPI endpoint
Query content: Who is the author of LightRAG

Updated to handle the new data format where:
- Response includes status, message, data, and metadata fields at top level
- Actual query results (entities, relationships, chunks, references) are nested under 'data' field
- Includes backward compatibility with legacy format
"""

import pytest
import requests
import time
import json
from typing import Dict, Any, List, Optional

# API configuration
API_KEY = "your-secure-api-key-here-123"
BASE_URL = "http://localhost:9621"

# Unified authentication headers
AUTH_HEADERS = {"Content-Type": "application/json", "X-API-Key": API_KEY}


def validate_references_format(references: List[Dict[str, Any]]) -> bool:
    """Validate the format of references list"""
    if not isinstance(references, list):
        print(f"❌ References should be a list, got {type(references)}")
        return False

    for i, ref in enumerate(references):
        if not isinstance(ref, dict):
            print(f"❌ Reference {i} should be a dict, got {type(ref)}")
            return False

        required_fields = ["reference_id", "file_path"]
        for field in required_fields:
            if field not in ref:
                print(f"❌ Reference {i} missing required field: {field}")
                return False

            if not isinstance(ref[field], str):
                print(
                    f"❌ Reference {i} field '{field}' should be string, got {type(ref[field])}"
                )
                return False

    return True


def parse_streaming_response(
    response_text: str,
) -> tuple[Optional[List[Dict]], List[str], List[str]]:
    """Parse streaming response and extract references, response chunks, and errors"""
    references = None
    response_chunks = []
    errors = []

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("data: "):
            if line.startswith("data: "):
                line = line[6:]  # Remove 'data: ' prefix

        if not line:
            continue

        try:
            data = json.loads(line)

            if "references" in data:
                references = data["references"]
            if "response" in data:
                response_chunks.append(data["response"])
            if "error" in data:
                errors.append(data["error"])

        except json.JSONDecodeError:
            # Skip non-JSON lines (like SSE comments)
            continue

    return references, response_chunks, errors


@pytest.mark.integration
@pytest.mark.requires_api
def test_query_endpoint_references():
    """Test /query endpoint references functionality"""

    print("\n" + "=" * 60)
    print("Testing /query endpoint references functionality")
    print("=" * 60)

    query_text = "who authored LightRAG"
    endpoint = f"{BASE_URL}/query"

    # Test 1: References enabled (default)
    print("\n🧪 Test 1: References enabled (default)")
    print("-" * 40)

    try:
        response = requests.post(
            endpoint,
            json={"query": query_text, "mode": "mix", "include_references": True},
            headers=AUTH_HEADERS,
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()

            # Check response structure
            if "response" not in data:
                print("❌ Missing 'response' field")
                return False

            if "references" not in data:
                print("❌ Missing 'references' field when include_references=True")
                return False

            references = data["references"]
            if references is None:
                print("❌ References should not be None when include_references=True")
                return False

            if not validate_references_format(references):
                return False

            print(f"✅ References enabled: Found {len(references)} references")
            print(f"   Response length: {len(data['response'])} characters")

            # Display reference list
            if references:
                print("   📚 Reference List:")
                for i, ref in enumerate(references, 1):
                    ref_id = ref.get("reference_id", "Unknown")
                    file_path = ref.get("file_path", "Unknown")
                    print(f"      {i}. ID: {ref_id} | File: {file_path}")

        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Test 1 failed: {str(e)}")
        return False

    # Test 2: References disabled
    print("\n🧪 Test 2: References disabled")
    print("-" * 40)

    try:
        response = requests.post(
            endpoint,
            json={"query": query_text, "mode": "mix", "include_references": False},
            headers=AUTH_HEADERS,
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()

            # Check response structure
            if "response" not in data:
                print("❌ Missing 'response' field")
                return False

            references = data.get("references")
            if references is not None:
                print("❌ References should be None when include_references=False")
                return False

            print("✅ References disabled: No references field present")
            print(f"   Response length: {len(data['response'])} characters")

        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Test 2 failed: {str(e)}")
        return False

    print("\n✅ /query endpoint references tests passed!")
    return True


@pytest.mark.integration
@pytest.mark.requires_api
def test_query_stream_endpoint_references():
    """Test /query/stream endpoint references functionality"""

    print("\n" + "=" * 60)
    print("Testing /query/stream endpoint references functionality")
    print("=" * 60)

    query_text = "who authored LightRAG"
    endpoint = f"{BASE_URL}/query/stream"

    # Test 1: Streaming with references enabled
    print("\n🧪 Test 1: Streaming with references enabled")
    print("-" * 40)

    try:
        response = requests.post(
            endpoint,
            json={"query": query_text, "mode": "mix", "include_references": True},
            headers=AUTH_HEADERS,
            timeout=30,
            stream=True,
        )

        if response.status_code == 200:
            # Collect streaming response
            full_response = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    # Ensure chunk is string type
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8")
                    full_response += chunk

            # Parse streaming response
            references, response_chunks, errors = parse_streaming_response(
                full_response
            )

            if errors:
                print(f"❌ Errors in streaming response: {errors}")
                return False

            if references is None:
                print("❌ No references found in streaming response")
                return False

            if not validate_references_format(references):
                return False

            if not response_chunks:
                print("❌ No response chunks found in streaming response")
                return False

            print(f"✅ Streaming with references: Found {len(references)} references")
            print(f"   Response chunks: {len(response_chunks)}")
            print(
                f"   Total response length: {sum(len(chunk) for chunk in response_chunks)} characters"
            )

            # Display reference list
            if references:
                print("   📚 Reference List:")
                for i, ref in enumerate(references, 1):
                    ref_id = ref.get("reference_id", "Unknown")
                    file_path = ref.get("file_path", "Unknown")
                    print(f"      {i}. ID: {ref_id} | File: {file_path}")

        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Test 1 failed: {str(e)}")
        return False

    # Test 2: Streaming with references disabled
    print("\n🧪 Test 2: Streaming with references disabled")
    print("-" * 40)

    try:
        response = requests.post(
            endpoint,
            json={"query": query_text, "mode": "mix", "include_references": False},
            headers=AUTH_HEADERS,
            timeout=30,
            stream=True,
        )

        if response.status_code == 200:
            # Collect streaming response
            full_response = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    # Ensure chunk is string type
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8")
                    full_response += chunk

            # Parse streaming response
            references, response_chunks, errors = parse_streaming_response(
                full_response
            )

            if errors:
                print(f"❌ Errors in streaming response: {errors}")
                return False

            if references is not None:
                print("❌ References should be None when include_references=False")
                return False

            if not response_chunks:
                print("❌ No response chunks found in streaming response")
                return False

            print("✅ Streaming without references: No references present")
            print(f"   Response chunks: {len(response_chunks)}")
            print(
                f"   Total response length: {sum(len(chunk) for chunk in response_chunks)} characters"
            )

        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Test 2 failed: {str(e)}")
        return False

    print("\n✅ /query/stream endpoint references tests passed!")
    return True


@pytest.mark.integration
@pytest.mark.requires_api
def test_references_consistency():
    """Test references consistency across all endpoints"""

    print("\n" + "=" * 60)
    print("Testing references consistency across endpoints")
    print("=" * 60)

    query_text = "who authored LightRAG"
    query_params = {
        "query": query_text,
        "mode": "mix",
        "top_k": 10,
        "chunk_top_k": 8,
        "include_references": True,
    }

    references_data = {}

    # Test /query endpoint
    print("\n🧪 Testing /query endpoint")
    print("-" * 40)

    try:
        response = requests.post(
            f"{BASE_URL}/query", json=query_params, headers=AUTH_HEADERS, timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            references_data["query"] = data.get("references", [])
            print(f"✅ /query: {len(references_data['query'])} references")
        else:
            print(f"❌ /query failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ /query test failed: {str(e)}")
        return False

    # Test /query/stream endpoint
    print("\n🧪 Testing /query/stream endpoint")
    print("-" * 40)

    try:
        response = requests.post(
            f"{BASE_URL}/query/stream",
            json=query_params,
            headers=AUTH_HEADERS,
            timeout=30,
            stream=True,
        )

        if response.status_code == 200:
            full_response = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    # Ensure chunk is string type
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8")
                    full_response += chunk

            references, _, errors = parse_streaming_response(full_response)

            if errors:
                print(f"❌ Errors: {errors}")
                return False

            references_data["stream"] = references or []
            print(f"✅ /query/stream: {len(references_data['stream'])} references")
        else:
            print(f"❌ /query/stream failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ /query/stream test failed: {str(e)}")
        return False

    # Test /query/data endpoint
    print("\n🧪 Testing /query/data endpoint")
    print("-" * 40)

    try:
        response = requests.post(
            f"{BASE_URL}/query/data",
            json=query_params,
            headers=AUTH_HEADERS,
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            query_data = data.get("data", {})
            references_data["data"] = query_data.get("references", [])
            print(f"✅ /query/data: {len(references_data['data'])} references")
        else:
            print(f"❌ /query/data failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ /query/data test failed: {str(e)}")
        return False

    # Compare references consistency
    print("\n🔍 Comparing references consistency")
    print("-" * 40)

    # Convert to sets of (reference_id, file_path) tuples for comparison
    def refs_to_set(refs):
        return set(
            (ref.get("reference_id", ""), ref.get("file_path", "")) for ref in refs
        )

    query_refs = refs_to_set(references_data["query"])
    stream_refs = refs_to_set(references_data["stream"])
    data_refs = refs_to_set(references_data["data"])

    # Check consistency
    consistency_passed = True

    if query_refs != stream_refs:
        print("❌ References mismatch between /query and /query/stream")
        print(f"   /query only: {query_refs - stream_refs}")
        print(f"   /query/stream only: {stream_refs - query_refs}")
        consistency_passed = False

    if query_refs != data_refs:
        print("❌ References mismatch between /query and /query/data")
        print(f"   /query only: {query_refs - data_refs}")
        print(f"   /query/data only: {data_refs - query_refs}")
        consistency_passed = False

    if stream_refs != data_refs:
        print("❌ References mismatch between /query/stream and /query/data")
        print(f"   /query/stream only: {stream_refs - data_refs}")
        print(f"   /query/data only: {data_refs - stream_refs}")
        consistency_passed = False

    if consistency_passed:
        print("✅ All endpoints return consistent references")
        print(f"   Common references count: {len(query_refs)}")

        # Display common reference list
        if query_refs:
            print("   📚 Common Reference List:")
            for i, (ref_id, file_path) in enumerate(sorted(query_refs), 1):
                print(f"      {i}. ID: {ref_id} | File: {file_path}")

    return consistency_passed


@pytest.mark.integration
@pytest.mark.requires_api
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


@pytest.mark.integration
@pytest.mark.requires_api
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


@pytest.mark.integration
@pytest.mark.requires_api
def run_all_reference_tests():
    """Run all reference-related tests"""

    print("\n" + "🚀" * 20)
    print("LightRAG References Test Suite")
    print("🚀" * 20)

    all_tests_passed = True

    # Test 1: /query endpoint references
    try:
        if not test_query_endpoint_references():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ /query endpoint test failed with exception: {str(e)}")
        all_tests_passed = False

    # Test 2: /query/stream endpoint references
    try:
        if not test_query_stream_endpoint_references():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ /query/stream endpoint test failed with exception: {str(e)}")
        all_tests_passed = False

    # Test 3: References consistency across endpoints
    try:
        if not test_references_consistency():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ References consistency test failed with exception: {str(e)}")
        all_tests_passed = False

    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)

    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ /query endpoint references functionality works correctly")
        print("✅ /query/stream endpoint references functionality works correctly")
        print("✅ References are consistent across all endpoints")
        print("\n🔧 System is ready for production use with reference support!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above and fix the issues.")
        print("\n🔧 System needs attention before production deployment.")

    return all_tests_passed


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--references-only":
        # Run only the new reference tests
        success = run_all_reference_tests()
        sys.exit(0 if success else 1)
    else:
        # Run original tests plus new reference tests
        print("Running original aquery_data endpoint test...")
        test_aquery_data_endpoint()

        print("\nRunning comparison test...")
        compare_with_regular_query()

        print("\nRunning new reference tests...")
        run_all_reference_tests()

        print("\n💡 Usage tips:")
        print("1. Ensure LightRAG API service is running")
        print("2. Adjust base_url and authentication information as needed")
        print("3. Modify query parameters to test different retrieval strategies")
        print("4. Data query results can be used for further analysis and processing")
        print("5. Run with --references-only flag to test only reference functionality")
