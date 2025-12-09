#!/usr/bin/env python3
"""Monitor LightRAG pipeline processing status."""
import time
import requests

API_URL = "http://localhost:9621"

def monitor():
    print("Monitoring LightRAG pipeline processing...")
    while True:
        try:
            resp = requests.get(f"{API_URL}/documents/pipeline_status", timeout=10)
            status = resp.json()
            busy = status.get("busy", False)
            pending = status.get("request_pending", False)
            msg = status.get("latest_message", "")[:80]
            batch = f"{status.get('cur_batch', 0)}/{status.get('batchs', 0)}"

            print(f"[{time.strftime('%H:%M:%S')}] batch={batch} busy={busy} pending={pending} | {msg}")

            if not busy and not pending:
                print("\n✅ Pipeline complete!")
                # Check document count
                docs_resp = requests.get(f"{API_URL}/documents", timeout=10)
                docs = docs_resp.json()
                print(f"Documents indexed: {len(docs.get('documents', []))}")
                break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(10)

if __name__ == "__main__":
    monitor()
