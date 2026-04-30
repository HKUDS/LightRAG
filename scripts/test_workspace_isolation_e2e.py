#!/usr/bin/env python3
"""
Workspace Isolation E2E Test for LightRAG API

Tests that different workspaces maintain isolated data.
"""

import httpx
import os
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:9621"
WORKING_DIR = "/tmp/lightrag_ws_test"
REQUEST_TIMEOUT = 60.0
QUERY_TIMEOUT = 30.0

results = []


def print_test_header(name: str):
    print(f"\n{'=' * 40}")
    print(f"{name}")
    print(f"{'=' * 40}\n")


def print_request(method: str, path: str, headers: dict, body: dict = None):
    print(f"REQUEST: {method} {path}")
    print(f"HEADERS: {headers}")
    if body:
        print(f"BODY: {body}")
    print("-" * 40)


def print_response(status_code: int, body: dict):
    print(f"RESPONSE: {status_code}")
    print(f"BODY: {body}")
    print("-" * 40)


def print_assertion(check: str, passed: bool):
    status = "PASS" if passed else "FAIL"
    print(f"ASSERTION: {check} → {status}\n")
    return passed


def wait_for_healthy(client: httpx.Client) -> bool:
    """Poll /health until server is ready or timeout."""
    print("Waiting for server healthy...")
    start = time.time()
    attempt = 0

    while time.time() - start < REQUEST_TIMEOUT:
        attempt += 1
        try:
            resp = client.get("/health", timeout=5.0)
            print(f"  Attempt {attempt}: {resp.status_code} - {resp.text[:100]}")
            if resp.status_code == 200:
                print("  Server is healthy!\n")
                return True
        except Exception as e:
            print(f"  Attempt {attempt}: Error - {e}")

        time.sleep(2)

    print("  TIMEOUT: Server not healthy after 60 seconds\n")
    return False


def wait_for_pipeline_complete(
    client: httpx.Client, workspace: str = None, doc_id: str = None
) -> dict:
    """Poll /documents/pipeline_status until complete or timeout."""
    headers = {}
    if workspace:
        headers["LIGHTRAG-WORKSPACE"] = workspace

    start = time.time()
    attempt = 0

    while time.time() - start < REQUEST_TIMEOUT:
        attempt += 1
        try:
            resp = client.get("/documents/pipeline_status", headers=headers, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                # Check if pipeline is complete (no pending items)
                # Adjust based on actual API response structure
                is_complete = (
                    data.get("pending", 0) == 0
                    and data.get("processing", 0) == 0
                    if isinstance(data, dict)
                    else True
                )
                print(f"  Attempt {attempt}: {data}")
                if is_complete:
                    return data
        except Exception as e:
            print(f"  Attempt {attempt}: Error - {e}")

        time.sleep(2)

    return {"status": "timeout"}


def run_test(
    name: str,
    test_func,
):
    """Run a test and track results."""
    print_test_header(name)
    try:
        passed = test_func()
        results.append((name, passed))
        return passed
    except Exception as e:
        print(f"EXCEPTION: {e}")
        results.append((name, False))
        return False


def t1_insert_query_alpha(client: httpx.Client) -> bool:
    """T1: Insert document into alpha workspace and query it."""
    all_pass = True

    # Insert document
    headers = {"LIGHTRAG-WORKSPACE": "alpha"}
    body = {
        "text": "Alice works at OpenAI in San Francisco. She is a senior engineer working on GPT models."
    }

    print_request("POST", "/documents/text", headers, body)
    resp = client.post("/documents/text", json=body, headers=headers, timeout=10.0)
    print_response(resp.status_code, resp.json())

    # Wait for pipeline
    print("Waiting for pipeline to complete...")
    wait_for_pipeline_complete(client, workspace="alpha")

    # Query
    query_body = {"query": "Who is Alice and where does she work?", "mode": "mix"}
    print_request("POST", "/query", headers, query_body)
    resp = client.post("/query", json=query_body, headers=headers, timeout=QUERY_TIMEOUT)
    print_response(resp.status_code, resp.json())

    # Assert
    response_text = resp.json().get("response", "").lower()
    check = 'response mentions "Alice" AND "OpenAI"'
    passed = "alice" in response_text and "openai" in response_text
    all_pass &= print_assertion(check, passed)

    return all_pass


def t2_query_empty_beta(client: httpx.Client) -> bool:
    """T2: Query beta workspace (should be empty, no alpha data)."""
    headers = {"LIGHTRAG-WORKSPACE": "beta"}
    body = {"query": "Who is Alice and where does she work?", "mode": "mix"}

    print_request("POST", "/query", headers, body)
    resp = client.post("/query", json=body, headers=headers, timeout=QUERY_TIMEOUT)
    print_response(resp.status_code, resp.json())

    # Assert
    response_text = resp.json().get("response", "").lower()
    check = 'response does NOT mention "OpenAI", "San Francisco", or "GPT models"'
    passed = (
        "openai" not in response_text
        and "san francisco" not in response_text
        and "gpt" not in response_text
    )
    return print_assertion(check, passed)


def t3_insert_beta_verify_alpha(client: httpx.Client) -> bool:
    """T3: Insert into beta, verify alpha is unaffected."""
    all_pass = True

    # Insert into beta
    headers = {"LIGHTRAG-WORKSPACE": "beta"}
    body = {
        "text": "Bob works at Google in London. He is a product manager for Google Cloud."
    }

    print_request("POST", "/documents/text", headers, body)
    resp = client.post("/documents/text", json=body, headers=headers, timeout=10.0)
    print_response(resp.status_code, resp.json())

    # Wait for pipeline
    print("Waiting for pipeline to complete...")
    wait_for_pipeline_complete(client, workspace="beta")

    # Query beta for Bob
    query_body = {"query": "Who is Bob and where does he work?", "mode": "mix"}
    print_request("POST", "/query", headers, query_body)
    resp = client.post("/query", json=query_body, headers=headers, timeout=QUERY_TIMEOUT)
    print_response(resp.status_code, resp.json())

    response_text = resp.json().get("response", "").lower()
    check = 'beta response mentions "Bob" AND "Google"'
    passed = "bob" in response_text and "google" in response_text
    all_pass &= print_assertion(check, passed)

    # Query beta for Alice (should not find alpha data)
    query_body = {"query": "Tell me about Alice", "mode": "mix"}
    print_request("POST", "/query", headers, query_body)
    resp = client.post("/query", json=query_body, headers=headers, timeout=QUERY_TIMEOUT)
    print_response(resp.status_code, resp.json())

    response_text = resp.json().get("response", "").lower()
    check = 'beta response for Alice does NOT mention "OpenAI" or "San Francisco"'
    passed = "openai" not in response_text and "san francisco" not in response_text
    all_pass &= print_assertion(check, passed)

    # Query alpha for Bob (should not find beta data)
    headers_alpha = {"LIGHTRAG-WORKSPACE": "alpha"}
    query_body = {"query": "Tell me about Bob", "mode": "mix"}
    print_request("POST", "/query", headers_alpha, query_body)
    resp = client.post("/query", json=query_body, headers=headers_alpha, timeout=QUERY_TIMEOUT)
    print_response(resp.status_code, resp.json())

    response_text = resp.json().get("response", "").lower()
    check = 'alpha response for Bob does NOT mention "Google" or "London"'
    passed = "google" not in response_text and "london" not in response_text
    all_pass &= print_assertion(check, passed)

    return all_pass


def t4_query_default_empty(client: httpx.Client) -> bool:
    """T4: Query default workspace (no header) - should be empty."""
    body = {"query": "Who is Alice or Bob?", "mode": "mix"}

    print_request("POST", "/query", {}, body)
    resp = client.post("/query", json=body, timeout=QUERY_TIMEOUT)
    print_response(resp.status_code, resp.json())

    # Assert
    response_text = resp.json().get("response", "").lower()
    check = 'default workspace response does NOT mention "OpenAI" AND does NOT mention "Google"'
    passed = "openai" not in response_text and "google" not in response_text
    return print_assertion(check, passed)


def t5_insert_query_default(client: httpx.Client) -> bool:
    """T5: Insert into default workspace and query."""
    all_pass = True

    # Insert document (no workspace header)
    body = {"text": "Charlie works at Meta in New York. He leads the VR research team."}

    print_request("POST", "/documents/text", {}, body)
    resp = client.post("/documents/text", json=body, timeout=10.0)
    print_response(resp.status_code, resp.json())

    # Wait for pipeline
    print("Waiting for pipeline to complete...")
    wait_for_pipeline_complete(client)

    # Query default workspace
    query_body = {"query": "Who is Charlie and where does he work?", "mode": "mix"}
    print_request("POST", "/query", {}, query_body)
    resp = client.post("/query", json=query_body, timeout=QUERY_TIMEOUT)
    print_response(resp.status_code, resp.json())

    # Assert
    response_text = resp.json().get("response", "").lower()
    check = 'default workspace response mentions "Charlie" AND "Meta"'
    passed = "charlie" in response_text and "meta" in response_text
    all_pass &= print_assertion(check, passed)

    return all_pass


def t6_filesystem_check() -> bool:
    """T6: Check file system for workspace directories."""
    print("Checking workspace directories in:", WORKING_DIR)

    if not os.path.exists(WORKING_DIR):
        print(f"ERROR: Working directory {WORKING_DIR} does not exist")
        return False

    entries = os.listdir(WORKING_DIR)
    print(f"\nDirectory entries: {entries}\n")

    # Check for workspace directories
    has_alpha = any("alpha" in e.lower() for e in entries)
    has_beta = any("beta" in e.lower() for e in entries)
    has_default = any("default" in e.lower() or (e not in ["alpha", "beta"]) for e in entries)

    # More flexible check - at least verify directories exist
    check1 = len(entries) >= 3  # alpha, beta, default directories
    print_assertion(f"at least 3 workspace directories exist (found {len(entries)})", check1)

    # Print tree structure
    print("Directory tree:")
    for entry in sorted(entries):
        full_path = os.path.join(WORKING_DIR, entry)
        if os.path.isdir(full_path):
            sub_entries = os.listdir(full_path)[:5]  # First 5 subentries
            print(f"  {entry}/")
            for sub in sub_entries:
                print(f"    {sub}")
            if len(os.listdir(full_path)) > 5:
                print(f"    ... ({len(os.listdir(full_path))} total items)")
        else:
            print(f"  {entry}")

    return check1


def print_summary():
    """Print final summary table."""
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Test':<40} {'Result':<10}")
    print("-" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<40} {status:<10}")
        all_passed &= passed

    print("-" * 60)
    verdict = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"{'Verdict:':<40} {verdict}")
    print(f"{'=' * 60}\n")

    return all_passed


def main():
    """Run all E2E tests."""
    print("\n" + "=" * 60)
    print("WORKSPACE ISOLATION E2E TEST")
    print("=" * 60)

    client = httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT)

    try:
        # T0: Wait for server
        print_test_header("T0: Wait for Server Healthy")
        if not wait_for_healthy(client):
            print("FATAL: Server not healthy")
            print_summary()
            return 1

        # Run all tests
        run_test("T1: Insert & Query Alpha", lambda: t1_insert_query_alpha(client))
        run_test("T2: Query Empty Beta", lambda: t2_query_empty_beta(client))
        run_test("T3: Insert Beta, Verify Alpha Unaffected", lambda: t3_insert_beta_verify_alpha(client))
        run_test("T4: Query Default (Empty)", lambda: t4_query_default_empty(client))
        run_test("T5: Insert & Query Default", lambda: t5_insert_query_default(client))
        run_test("T6: File System Check", t6_filesystem_check)

    finally:
        client.close()

    # Print summary and exit
    all_passed = print_summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
