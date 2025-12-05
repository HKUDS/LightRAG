import time
import os
import sys
from client import LightRAGClient, print_success, print_error, print_step

# Configuration
BASE_URL = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")
USERNAME = os.getenv("AUTH_USER", "admin")
PASSWORD = os.getenv("AUTH_PASS", "admin123")


def run_tests():
    client = LightRAGClient(BASE_URL, USERNAME, PASSWORD)
    client.login()

    suffix = int(time.time())

    # Scenario: Two tenants operating "simultaneously" (interleaved)
    # Tenant A: Ingests -> Queries
    # Tenant B: Ingests -> Queries -> Deletes -> Queries

    print_step("SETUP: Creating Tenants")
    tenant_a = client.create_tenant(f"Mixed_A_{suffix}", "Mixed Tenant A")
    tenant_b = client.create_tenant(f"Mixed_B_{suffix}", "Mixed Tenant B")

    kb_a = client.create_kb(tenant_a, f"KB_A_{suffix}", "KB A")
    kb_b = client.create_kb(tenant_b, f"KB_B_{suffix}", "KB B")

    # Interleaved Operations
    print_step("STEP 1: Interleaved Ingestion")

    secret_a = f"Project A is codenamed 'Apollo-{suffix}'."
    secret_b = f"Project B is codenamed 'Gemini-{suffix}'."

    print("   > Ingesting into Tenant A...")
    client.ingest_text(tenant_a, kb_a, secret_a)

    print("   > Ingesting into Tenant B...")
    client.ingest_text(tenant_b, kb_b, secret_b)

    print_step("STEP 2: Waiting for Indexing (Both)")
    client.wait_for_indexing(tenant_a, kb_a)
    client.wait_for_indexing(tenant_b, kb_b)

    # Get Doc ID for B
    docs_b = client.get_documents(tenant_b, kb_b)
    doc_b_id = docs_b[0]["id"] if docs_b else None
    if not doc_b_id:
        print_error("Failed to get document ID for Tenant B")
        sys.exit(1)
    print_success(f"Got Document ID for Tenant B: {doc_b_id}")

    print_step("STEP 3: Cross-Verification")

    # Verify A sees A
    resp_a = client.query(tenant_a, kb_a, "What is the codename for Project A?")
    if "Apollo" in resp_a and str(suffix) in resp_a:
        print_success("Tenant A sees Project A.")
    else:
        print_error(f"Tenant A failed to see Project A. Got: {resp_a}")
        sys.exit(1)

    # Verify B sees B
    resp_b = client.query(tenant_b, kb_b, "What is the codename for Project B?")
    if "Gemini" in resp_b and str(suffix) in resp_b:
        print_success("Tenant B sees Project B.")
    else:
        print_error(f"Tenant B failed to see Project B. Got: {resp_b}")
        sys.exit(1)

    # Verify A does NOT see B
    resp_a_b = client.query(tenant_a, kb_a, "What is the codename for Project B?")
    if "Gemini" in resp_a_b and str(suffix) in resp_a_b:
        print_error("Tenant A saw Project B! Isolation failure.")
        sys.exit(1)
    else:
        print_success("Tenant A did not see Project B.")

    print_step("STEP 4: Tenant B Deletion")
    client.delete_document(tenant_b, kb_b, doc_b_id)
    time.sleep(2)

    # Verify B cannot see B anymore
    resp_b_after = client.query(tenant_b, kb_b, "What is the codename for Project B?")
    if "Gemini" in resp_b_after and str(suffix) in resp_b_after:
        print_error("Tenant B still sees Project B after deletion.")
        sys.exit(1)
    else:
        print_success("Tenant B correctly forgot Project B.")

    # Verify A still sees A (Regression check)
    resp_a_after = client.query(tenant_a, kb_a, "What is the codename for Project A?")
    if "Apollo" in resp_a_after and str(suffix) in resp_a_after:
        print_success("Tenant A still sees Project A (Unaffected by B's deletion).")
    else:
        print_error("Tenant A lost data after Tenant B deleted data!")
        sys.exit(1)

    print_step("🎉 MIXED OPERATIONS TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
