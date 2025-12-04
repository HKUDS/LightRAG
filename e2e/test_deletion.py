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
    tenant_name = f"Deletion_Test_Tenant_{suffix}"
    kb_name = f"Deletion_Test_KB_{suffix}"
    
    print_step("SETUP: Creating Tenant and KB")
    tenant_id = client.create_tenant(tenant_name, "Deletion Test Tenant")
    kb_id = client.create_kb(tenant_id, kb_name, "Deletion Test KB")

    # 1. Ingest Document 1
    print_step("STEP 1: Ingesting Document 1")
    doc1_content = f"Project Flamingo is codenamed 'Flamingo-{suffix}'."
    client.ingest_text(tenant_id, kb_id, doc1_content)
    
    # 2. Ingest Document 2 (Control)
    print_step("STEP 2: Ingesting Document 2 (Control)")
    doc2_content = f"Project Eagle is codenamed 'Eagle-{suffix}'."
    client.ingest_text(tenant_id, kb_id, doc2_content)

    # 3. Wait for Indexing
    client.wait_for_indexing(tenant_id, kb_id)
    
    # Get Document IDs
    docs = client.get_documents(tenant_id, kb_id)
    doc1_id = next((d['id'] for d in docs if "Flamingo" in d.get('content_summary', '')), None)
    doc2_id = next((d['id'] for d in docs if "Eagle" in d.get('content_summary', '')), None)
    
    if not doc1_id:
        # Fallback: try to find by content match if summary is truncated
        print("Warning: Could not find doc1 by summary, checking all docs...")
        pass

    if not doc1_id or not doc2_id:
        print_error("Failed to resolve document IDs")
        for d in docs:
             print(f"Doc: {d.get('content_summary')} ID: {d.get('id')}")
        sys.exit(1)

    # 4. Verify both exist
    print_step("STEP 3: Verifying both documents exist")
    
    resp1 = client.query(tenant_id, kb_id, "What is the codename for Project Flamingo?")
    if ("Flamingo" in resp1 and str(suffix) in resp1) or f"Flamingo-{suffix}" in resp1:
        print_success("Document 1 found.")
    else:
        print_error(f"Document 1 NOT found. Response: {resp1}")
        sys.exit(1)

    resp2 = client.query(tenant_id, kb_id, "What is the codename for Project Eagle?")
    if ("Eagle" in resp2 and str(suffix) in resp2) or f"Eagle-{suffix}" in resp2:
        print_success("Document 2 found.")
    else:
        print_error(f"Document 2 NOT found. Response: {resp2}")
        sys.exit(1)

    # 5. Delete Document 1
    print_step("STEP 4: Deleting Document 1")
    client.delete_document(tenant_id, kb_id, doc1_id)
    
    # Wait for deletion to complete
    client.wait_for_pipeline(tenant_id, kb_id)
    
    # Clear cache to ensure fresh results
    client.clear_cache(tenant_id, kb_id)

    # 6. Verify Document 1 is gone
    print_step("STEP 5: Verifying Document 1 is gone")
    resp1_after = client.query(tenant_id, kb_id, "What is the codename for Project Flamingo?")
    
    # We expect the answer to NOT contain the specific secret code (Flamingo-{suffix})
    # It might echo the word "Flamingo" in the question, but the actual secret should not be there
    secret_code = f"Flamingo-{suffix}"
    if secret_code not in resp1_after:
        print_success("Document 1 successfully deleted (Secret not found in answer).")
    else:
        print_error(f"Document 1 STILL FOUND after deletion! Response: {resp1_after}")
        sys.exit(1)

    # 7. Verify Document 2 still exists
    print_step("STEP 6: Verifying Document 2 still exists")
    resp2_after = client.query(tenant_id, kb_id, "What is the public announcement?")
    if "Eagle" in resp2_after and str(suffix) in resp2_after:
        print_success("Document 2 still exists.")
    else:
        print_error(f"Document 2 disappeared! Response: {resp2_after}")
        sys.exit(1)

    print_step("🎉 DELETION TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
