import sys
import os
import argparse

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.api import db

def create_admin(username, password):
    print(f"Initializing DB connection...")
    db.init_db()
    
    # Check if user exists
    existing = db.get_user_by_username(username)
    if existing:
        print(f"User '{username}' already exists.")
        return

    # Default Org
    org_id = "org_default"
    
    print(f"Creating admin user '{username}'...")
    user = db.create_user(username, password, org_id, role="admin")
    
    if user:
        print(f"Successfully created admin user: {username}")
        print(f"Org ID: {org_id}")
    else:
        print("Failed to create user.")

def main():
    parser = argparse.ArgumentParser(description="Create a LightRAG Admin User")
    parser.add_argument("username", nargs="?", help="Username")
    parser.add_argument("password", nargs="?", help="Password")
    
    args = parser.parse_args()
    
    username = args.username
    password = args.password
    
    if not username:
        username = input("Enter username: ")
    if not password:
        import getpass
        password = getpass.getpass("Enter password: ")
        
    create_admin(username, password)

if __name__ == "__main__":
    main()
