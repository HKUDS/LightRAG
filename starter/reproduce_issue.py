import requests
import json
import sys

BASE_URL = "http://localhost:9621"

def test_default_tenant_access():
    print(f"Testing access to default tenant at {BASE_URL}...")
    
    # Headers with 'default' tenant and KB
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJndWVzdCIsImV4cCI6MTc2Mzc5NDEwMiwicm9sZSI6Imd1ZXN0IiwibWV0YWRhdGEiOnsiYXV0aF9tb2RlIjoiZGlzYWJsZWQifX0.R9zsH00LTYtvk_pEG2b3bSdO9SAAPgnUlcHwOPIefXY",
        "Content-Type": "application/json",
        "X-Tenant-ID": "default",
        "X-KB-ID": "default"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/documents/paginated",
            headers=headers,
            json={
                "page": 1,
                "page_size": 10
            }
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.text}")
            print("SUCCESS: Default tenant accessed successfully.")
            return True
        else:
            print(f"Error Response: {response.text}")
            print("FAILURE: Could not access default tenant.")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_default_tenant_access()
    sys.exit(0 if success else 1)
