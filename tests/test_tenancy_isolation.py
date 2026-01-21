import requests
import time
import sys

BASE_URL = "http://localhost:9621"

def register_user(username, password, org_id):
    url = f"{BASE_URL}/register"
    payload = {"username": username, "password": password, "org_id": org_id}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()["access_token"]
        elif "Username already registered" in response.text:
            # Login instead
            return login_user(username, password, org_id)
        else:
            print(f"[-] Registration failed for {username}: {response.text}")
            return None
    except Exception as e:
        print(f"[-] Error registering {username}: {e}")
        return None

def login_user(username, password, org_id):
    url = f"{BASE_URL}/login"
    payload = {"username": username, "password": password, "org_id": org_id}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["access_token"]
    print(f"[-] Login failed for {username}: {response.text}")
    return None

def upload_text(token, text):
    url = f"{BASE_URL}/documents/text"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"text": text}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return True
    print(f"[-] Upload failed: {response.text}")
    return False

def query_rag(token, query):
    url = f"{BASE_URL}/query"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"query": query, "mode": "global"} # Use global or hybrid
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["response"]
    print(f"[-] Query failed: {response.text}")
    return None

def run_test():
    print("[*] Starting Multi-Tenancy Isolation Test")
    
    # 1. Setup Users
    org_a = "org_alpha"
    user_a = "alice"
    token_a = register_user(user_a, "password123", org_a)
    print(f"[+] User A ({org_a}) Token: {token_a[:10]}...")

    org_b = "org_beta"
    user_b = "bob"
    token_b = register_user(user_b, "password123", org_b)
    print(f"[+] User B ({org_b}) Token: {token_b[:10]}...")

    if not token_a or not token_b:
        print("[-] Failed to authenticate users. Exiting.")
        return

    # 2. User A inserts a secret
    secret = "The secret code for Operation Alpha is 'BLUE_HORIZON'."
    print(f"[*] User A uploading secret: '{secret}'")
    if upload_text(token_a, secret):
        print("[+] Upload successful. Waiting for indexing (20s)...")
        time.sleep(20) # Wait longer for indexing

    # 3. User A queries the secret
    print("[*] User A querying for secret...")
    ans_a = query_rag(token_a, "What is the secret code for Operation Alpha?")
    print(f"[+] User A Answer: {ans_a}")

    if ans_a and "BLUE_HORIZON" in str(ans_a):
        print("[+] SUCCESS: User A retrieved their data.")
    else:
        print("[-] FAILURE: User A could not retrieve their data. Trying naive mode...")
        # Try naive mode
        url = f"{BASE_URL}/query"
        headers = {"Authorization": f"Bearer {token_a}"}
        payload = {"query": "What is the secret code for Operation Alpha?", "mode": "naive"}
        try:
            resp = requests.post(url, json=payload, headers=headers)
            ans_naive = resp.json().get("response")
            print(f"[+] User A Naive Answer: {ans_naive}")
            if ans_naive and "BLUE_HORIZON" in str(ans_naive):
                print("[+] SUCCESS: User A retrieved their data (Naive Mode).")
            else:
                 print("[-] FAILURE: User A could not retrieve their data even in Naive mode.")
        except Exception as e:
            print(f"[-] Naive query error: {e}")

    # 4. User B queries the secret (Should fail)
    print("[*] User B querying for User A's secret...")
    ans_b = query_rag(token_b, "What is the secret code for Operation Alpha?")
    print(f"[+] User B Answer: {ans_b}")

    if "BLUE_HORIZON" not in str(ans_b):
        print("[+] SUCCESS: User B DID NOT see User A's data.")
    else:
        print("[-] FAILURE: User B SAW User A's data (Data Leak!)")

if __name__ == "__main__":
    run_test()
