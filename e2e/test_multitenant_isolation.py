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

    # 1. Create Tenants
    suffix = int(time.time())
    tenant_a_name = f"E2E_Tenant_A_{suffix}"
    tenant_b_name = f"E2E_Tenant_B_{suffix}"
    
    tenant_a_id = client.create_tenant(tenant_a_name, "Test Tenant A")
    tenant_b_id = client.create_tenant(tenant_b_name, "Test Tenant B")

    # 2. Create KBs
    kb_a_name = f"KB_A_{suffix}"
    kb_b_name = f"KB_B_{suffix}"
    
    kb_a_id = client.create_kb(tenant_a_id, kb_a_name, "KB for Tenant A")
    kb_b_id = client.create_kb(tenant_b_id, kb_b_name, "KB for Tenant B")

    # 3. Ingest Data
    secret_a = f"The secret code for Tenant A is ALPHA-{suffix}."
    secret_b = f"The secret code for Tenant B is BRAVO-{suffix}."
    
    client.ingest_text(tenant_a_id, kb_a_id, secret_a)
    client.ingest_text(tenant_b_id, kb_b_id, secret_b)

    # 4. Wait for Indexing
    client.wait_for_indexing(tenant_a_id, kb_a_id)
    client.wait_for_indexing(tenant_b_id, kb_b_id)

    # 5. Verify Isolation
    print_step("VERIFYING ISOLATION")
    
    # Test A: Tenant A should find Secret A
    response_a_a = client.query(tenant_a_id, kb_a_id, f"What is the secret code for Tenant A?")
    # Normalize response to handle different hyphens
    normalized_response = response_a_a.replace("‑", "-")
    if f"ALPHA-{suffix}" in normalized_response or (f"ALPHA" in response_a_a and str(suffix) in response_a_a):
        print_success("Tenant A found its own secret.")
    else:
        print_error(f"Tenant A FAILED to find its own secret. Got: {response_a_a}")
        sys.exit(1)

    # Test B: Tenant B should find Secret B
    response_b_b = client.query(tenant_b_id, kb_b_id, f"What is the secret code for Tenant B?")
    normalized_response_b = response_b_b.replace("‑", "-")
    if f"BRAVO-{suffix}" in normalized_response_b or (f"BRAVO" in response_b_b and str(suffix) in response_b_b):
        print_success("Tenant B found its own secret.")
    else:
        print_error(f"Tenant B FAILED to find its own secret. Got: {response_b_b}")
        sys.exit(1)

    # Test C: Tenant A should NOT find Secret B
    response_a_b = client.query(tenant_a_id, kb_a_id, f"What is the secret code for Tenant B?")
    if f"BRAVO-{suffix}" in response_a_b:
        print_error(f"DATA LEAKAGE DETECTED! Tenant A found Tenant B's secret: {response_a_b}")
        sys.exit(1)
    else:
        print_success("Tenant A did NOT find Tenant B's secret (Correct).")

    # Test D: Tenant B should NOT find Secret A
    response_b_a = client.query(tenant_b_id, kb_b_id, f"What is the secret code for Tenant A?")
    if f"ALPHA-{suffix}" in response_b_a:
        print_error(f"DATA LEAKAGE DETECTED! Tenant B found Tenant A's secret: {response_b_a}")
        sys.exit(1)
    else:
        print_success("Tenant B did NOT find Tenant A's secret (Correct).")

    print_step("🎉 ALL ISOLATION TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
