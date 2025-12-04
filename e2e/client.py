import requests
import time
import os
import json
import sys

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ️  {msg}{RESET}")

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
                data={"username": self.username, "password": self.password}
            )
            if response.status_code != 200:
                # Try JSON if form data fails
                response = self.session.post(
                    f"{self.base_url}/login",
                    json={"username": self.username, "password": self.password}
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
            json={"name": name, "description": description}
        )
        if response.status_code not in [200, 201]:
             # If tenant already exists, try to find it
            if response.status_code == 409 or "already exists" in response.text:
                 print(f"Tenant {name} might already exist, fetching list...")
                 pass
            else:
                raise Exception(f"Failed to create tenant: {response.text}")
        
        list_resp = self.session.get(f"{self.base_url}/api/v1/tenants")
        tenants = list_resp.json()
        items = tenants.get("items", tenants) if isinstance(tenants, dict) else tenants

        for t in items:
            if t.get("name") == name:
                print_success(f"Tenant '{name}' ID: {t['tenant_id']}")
                return t['tenant_id']
        
        if response.status_code in [200, 201]:
             data = response.json()
             print_success(f"Tenant '{name}' created with ID: {data['tenant_id']}")
             return data['tenant_id']
             
        raise Exception(f"Could not resolve Tenant ID for {name}")

    def create_kb(self, tenant_id, name, description):
        print_step(f"Creating KB '{name}' for Tenant '{tenant_id}'")
        headers = {"X-Tenant-ID": tenant_id}
        response = self.session.post(
            f"{self.base_url}/api/v1/knowledge-bases",
            json={"name": name, "description": description},
            headers=headers
        )
        
        list_resp = self.session.get(f"{self.base_url}/api/v1/knowledge-bases", headers=headers)
        kbs = list_resp.json()
        items = kbs.get("items", kbs) if isinstance(kbs, dict) else kbs
        
        for kb in items:
            if kb.get("name") == name:
                print_success(f"KB '{name}' ID: {kb['kb_id']}")
                return kb['kb_id']
                
        if response.status_code in [200, 201]:
            data = response.json()
            return data['kb_id']
            
        raise Exception(f"Could not resolve KB ID for {name}")

    def ingest_text(self, tenant_id, kb_id, text):
        print_step(f"Ingesting text into Tenant: {tenant_id}, KB: {kb_id}")
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        response = self.session.post(
            f"{self.base_url}/documents/text",
            json={"text": text},
            headers=headers
        )
        if response.status_code != 200:
            raise Exception(f"Ingestion failed: {response.text}")
        print_success("Text ingested successfully")

    def wait_for_indexing(self, tenant_id, kb_id, timeout=300):
        print_step(f"Waiting for indexing in Tenant: {tenant_id}, KB: {kb_id}...")
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        start_time = time.time()
        last_status = ""
        poll_count = 0
        
        while time.time() - start_time < timeout:
            poll_count += 1
            elapsed = int(time.time() - start_time)
            
            response = self.session.get(f"{self.base_url}/documents", headers=headers)
            if response.status_code != 200:
                print(f"  [{elapsed}s] Error checking documents: {response.text}")
                time.sleep(2)
                continue
                
            data = response.json()
            
            docs = []
            if "statuses" in data:
                for status_key, status_list in data["statuses"].items():
                    docs.extend(status_list)
            elif "items" in data:
                docs = data["items"]
            elif isinstance(data, list):
                docs = data
            else:
                docs = []
            
            if not docs:
                if poll_count % 5 == 1:  # Print every 10 seconds
                    print(f"  [{elapsed}s] No documents found yet, waiting...")
                time.sleep(2)
                continue

            # Count statuses
            status_counts = {}
            for doc in docs:
                if isinstance(doc, str):
                    status = "pending"
                else:
                    status = doc.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            current_status = ", ".join([f"{k}: {v}" for k, v in sorted(status_counts.items())])
            
            # Only print if status changed or every 10 seconds
            if current_status != last_status or poll_count % 5 == 1:
                print(f"  [{elapsed}s] Documents: {current_status}")
                last_status = current_status

            all_processed = all(
                (isinstance(doc, dict) and doc.get("status") == "processed") 
                for doc in docs
            )
            
            if all_processed and len(docs) > 0:
                print_success(f"All {len(docs)} document(s) processed in {elapsed}s")
                return
            
            time.sleep(2)
            
        raise Exception(f"Timeout ({timeout}s) waiting for indexing. Last status: {last_status}")

    def clear_cache(self, tenant_id, kb_id):
        print_step(f"Clearing cache in Tenant: {tenant_id}, KB: {kb_id}")
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        response = self.session.post(f"{self.base_url}/documents/clear_cache", json={}, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Clear cache failed: {response.text}")
        print_success("Cache cleared")

    def wait_for_pipeline(self, tenant_id, kb_id, timeout=60):
        print_step(f"Waiting for pipeline to be idle in Tenant: {tenant_id}, KB: {kb_id}...")
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.base_url}/documents/pipeline_status", headers=headers)
            if response.status_code != 200:
                print(f"Error checking pipeline status: {response.text}")
                time.sleep(2)
                continue
            
            data = response.json()
            if not data.get("busy", False):
                print_success("Pipeline is idle")
                return
            
            time.sleep(1)
        
        raise Exception("Timeout waiting for pipeline to be idle")

    def query(self, tenant_id, kb_id, query_text, verbose=True):
        if verbose:
            print_step(f"Querying '{query_text}' in Tenant: {tenant_id}, KB: {kb_id}")
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/query",
            json={"query": query_text, "mode": "global"},
            headers=headers
        )
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            raise Exception(f"Query failed: {response.text}")
        
        result = response.json()
        response_text = result.get('response', '')
        if verbose:
            print(f"  Response ({elapsed:.1f}s): {response_text[:150]}{'...' if len(response_text) > 150 else ''}")
        return response_text

    def delete_document(self, tenant_id, kb_id, doc_id):
        print_step(f"Deleting document '{doc_id}' in Tenant: {tenant_id}, KB: {kb_id}")
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        response = self.session.request(
            "DELETE",
            f"{self.base_url}/documents/delete_document",
            json={"doc_ids": [doc_id]},
            headers=headers
        )
        if response.status_code != 200:
            raise Exception(f"Deletion failed: {response.text}")
        print_success(f"Document {doc_id} deleted successfully")

    def get_documents(self, tenant_id, kb_id):
        headers = {"X-Tenant-ID": tenant_id, "X-KB-ID": kb_id}
        response = self.session.get(f"{self.base_url}/documents", headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get documents: {response.text}")
        
        data = response.json()
        # print(f"DEBUG: RAW DATA: {data}") 
        docs = []
        if "statuses" in data:
            for status_key, status_list in data["statuses"].items():
                docs.extend(status_list)
        elif "items" in data:
            docs = data["items"]
        elif isinstance(data, list):
            docs = data
        
        print(f"DEBUG: get_documents returned {len(docs)} docs")
        for d in docs:
            print(f"DEBUG: Doc: {d}")
            
        return docs
