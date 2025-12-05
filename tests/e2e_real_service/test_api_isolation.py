import requests
import time
import os
import json
import sys

# Configuration
BASE_URL = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")
USERNAME = os.getenv("AUTH_USER", "admin")
PASSWORD = os.getenv("AUTH_PASS", "admin123")

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")


def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")


def print_step(msg):
    print(f"\n{BOLD}👉 {msg}{RESET}")


class LightRAGClient:
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.token = None
        self.session = requests.Session()

    def login(self):
        print_step("Authenticating...")
        try:
            # Try form data first (FastAPI OAuth2PasswordRequestForm)
            response = self.session.post(
                f"{self.base_url}/login",
                data={"username": self.username, "password": self.password},
            )
            if response.status_code != 200:
                # Try JSON if form data fails
                response = self.session.post(
                    f"{self.base_url}/login",
                    json={"username": self.username, "password": self.password},
                )

            if response.status_code != 200:
                raise Exception(f"Login failed: {response.text}")

            data = response.json()
            self.token = data.get("access_token")
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            print_success(f"Authenticated as {self.username}")
        except Exception as e:
            print_error(f"Authentication failed: {e}")
            sys.exit(1)

    def create_tenant(self, name, description):
        print_step(f"Creating Tenant: {name}")
        response = self.session.post(
            f"{self.base_url}/api/v1/tenants",
            json={"name": name, "description": description},
        )
        if response.status_code not in [200, 201]:
            # If tenant already exists, try to find it
            if response.status_code == 409 or "already exists" in response.text:
                print(f"Tenant {name} might already exist, fetching list...")
                # This is a simplification; in a real test we might want to delete/recreate
                # For now, we'll just try to proceed if we can find the ID
                pass
            else:
                raise Exception(f"Failed to create tenant: {response.text}")

        # We need the tenant_id. If created, it's in response.
        # If 409, we might need to list tenants to find it.
        # Let's assume for this test we create unique names or handle it.

        # Actually, let's list tenants to find the ID by name to be robust
        list_resp = self.session.get(f"{self.base_url}/api/v1/tenants")
        tenants = list_resp.json()
        # Handle pagination
        items = tenants.get("items", tenants) if isinstance(tenants, dict) else tenants

        for t in items:
            if t.get("name") == name:
                print_success(f"Tenant '{name}' ID: {t['tenant_id']}")
                return t["tenant_id"]

        # If not found in list but create failed, that's an issue
        if response.status_code in [200, 201]:
            data = response.json()
            print_success(f"Tenant '{name}' created with ID: {data['tenant_id']}")
            return data["tenant_id"]

        raise Exception(f"Could not resolve Tenant ID for {name}")

    def create_kb(self, tenant_id, name, description):
        print_step(f"Creating KB '{name}' for Tenant '{tenant_id}'")
        headers = {"X-Tenant-ID": tenant_id}
        response = self.session.post(
            f"{self.base_url}/api/v1/knowledge-bases",
            json={"name": name, "description": description},
            headers=headers,
        )

        if response.status_code not in [200, 201]:
            # Similar robustness check
            pass

        # List KBs to find ID
        list_resp = self.session.get(
            f"{self.base_url}/api/v1/knowledge-bases", headers=headers
        )
        kbs = list_resp.json()
        # Handle pagination if needed, but usually returns list or items
        items = kbs.get("items", kbs) if isinstance(kbs, dict) else kbs

        for kb in items:
            if kb.get("name") == name:
                print_success(f"KB '{name}' ID: {kb['kb_id']}")
                return kb["kb_id"]

        if response.status_code in [200, 201]:
            data = response.json()
            return data["kb_id"]

        raise Exception(f"Could not resolve KB ID for {name}")

    def ingest_text(self, tenant_id, kb_id, text):
        print_step(f"Ingesting text into Tenant: {tenant_id}, KB: {kb_id}")
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        response = self.session.post(
            f"{self.base_url}/documents/text", json={"text": text}, headers=headers
        )
        if response.status_code != 200:
            raise Exception(f"Ingestion failed: {response.text}")
        print_success("Text ingested successfully")

    def wait_for_indexing(self, tenant_id, kb_id, timeout=60):
        print_step(f"Waiting for indexing in Tenant: {tenant_id}, KB: {kb_id}...")
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.base_url}/documents", headers=headers)
            if response.status_code != 200:
                print(f"Error checking documents: {response.text}")
                time.sleep(2)
                continue

            data = response.json()
            print(f"DEBUG: /documents response: {json.dumps(data, indent=2)}")

            docs = []
            if "statuses" in data:
                # Flatten the statuses dictionary
                for status_key, status_list in data["statuses"].items():
                    # print(f"DEBUG: Processing status {status_key} with {len(status_list)} items")
                    docs.extend(status_list)
            elif "items" in data:
                docs = data["items"]
            elif isinstance(data, list):
                docs = data
            else:
                # Fallback or empty
                docs = []

            if not docs:
                print("No documents found yet...")
                time.sleep(2)
                continue

            # Debug print
            # statuses = [d.get("status") for d in docs]
            # print(f"Docs found: {len(docs)}, Statuses: {statuses}")

            all_processed = True
            for doc in docs:
                if isinstance(doc, str):
                    print(f"Unexpected doc format (str): {doc}")
                    # If we get a string, it might be an ID or something else, but definitely not processed
                    all_processed = False
                    continue

                if doc.get("status") != "processed":
                    all_processed = False
                    break

            if all_processed and len(docs) > 0:
                print_success("All documents processed")
                return

            time.sleep(2)

        raise Exception("Timeout waiting for indexing")

    def query(self, tenant_id, kb_id, query_text):
        print_step(f"Querying '{query_text}' in Tenant: {tenant_id}, KB: {kb_id}")
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        response = self.session.post(
            f"{self.base_url}/query",
            json={"query": query_text, "mode": "global"},  # Use global or hybrid
            headers=headers,
        )
        if response.status_code != 200:
            raise Exception(f"Query failed: {response.text}")

        result = response.json()
        print(f"Response: {result.get('response', '')[:100]}...")
        return result.get("response", "")


def run_tests():
    client = LightRAGClient(BASE_URL, USERNAME, PASSWORD)
    client.login()

    # 1. Create Tenants
    # Use unique names to avoid conflicts if running multiple times
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
    response_a_a = client.query(
        tenant_a_id, kb_a_id, "What is the secret code for Tenant A?"
    )
    if f"ALPHA-{suffix}" in response_a_a:
        print_success("Tenant A found its own secret.")
    else:
        print_error(f"Tenant A FAILED to find its own secret. Got: {response_a_a}")
        sys.exit(1)

    # Test B: Tenant B should find Secret B
    response_b_b = client.query(
        tenant_b_id, kb_b_id, "What is the secret code for Tenant B?"
    )
    if f"BRAVO-{suffix}" in response_b_b:
        print_success("Tenant B found its own secret.")
    else:
        print_error(f"Tenant B FAILED to find its own secret. Got: {response_b_b}")
        sys.exit(1)

    # Test C: Tenant A should NOT find Secret B
    response_a_b = client.query(
        tenant_a_id, kb_a_id, "What is the secret code for Tenant B?"
    )
    if f"BRAVO-{suffix}" in response_a_b:
        print_error(
            f"DATA LEAKAGE DETECTED! Tenant A found Tenant B's secret: {response_a_b}"
        )
        sys.exit(1)
    else:
        print_success("Tenant A did NOT find Tenant B's secret (Correct).")

    # Test D: Tenant B should NOT find Secret A
    response_b_a = client.query(
        tenant_b_id, kb_b_id, "What is the secret code for Tenant A?"
    )
    if f"ALPHA-{suffix}" in response_b_a:
        print_error(
            f"DATA LEAKAGE DETECTED! Tenant B found Tenant A's secret: {response_b_a}"
        )
        sys.exit(1)
    else:
        print_success("Tenant B did NOT find Tenant A's secret (Correct).")

    print_step("🎉 ALL ISOLATION TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
