import sys
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:9621"  # Default LightRAG API port

# Change to your document path
INPUT_DIR = "/home/js/LightRAG/jrs/work/mcfadden/mcfadden_data" 

def trigger_directory_scan(directory_path):
    """
    Calls the LightRAG API to scan a directory for new documents.
    """
    endpoint = f"{API_URL}/documents/scan"
    payload = {
        "input_dir": directory_path
    }
    
    print(f"Sending scan request for: {directory_path}...")
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status() # Raises error for 4xx/5xx responses
        
        data = response.json()
        print("Successfully triggered scan.")
        print(f"Server Response: {json.dumps(data, indent=2)}")
        
        # Check if the server returned status information
        if data.get("status") == "success":
            print("\nIndexing has started in the background.")
            print("You can monitor progress in your LightRAG terminal/logs.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to LightRAG server: {e}")


if __name__ == "__main__":
    # If you provide a path in the command line, use it. 
    # Otherwise, use the default hard-coded path.
    target_dir = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    trigger_directory_scan(target_dir)    