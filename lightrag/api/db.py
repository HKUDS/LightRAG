import sqlite3
import os
import secrets
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

DB_PATH = os.environ.get("LIGHTRAG_DB_PATH", "lightrag.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db_cursor():
    conn = get_db_connection()
    try:
        yield conn.cursor()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def hash_password(password: str) -> str:
    # simple sha256 for demo - in prod use bcrypt/argon2
    # but to minimize deps we use hashlib for now if bcrypt not available
    # Check if bcrypt is available (it is in pyproject.toml optional)
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        import bcrypt
        # bcrypt.checkpw requires bytes
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except ImportError:
        return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def init_db():
    with get_db_cursor() as cur:
        # Organizations
        cur.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Users
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                org_id TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (org_id) REFERENCES organizations (id)
            )
        """)

        # Chat Sessions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Chat Messages
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
            )
        """)

        # Create Default Admin & Org if not exists
        cur.execute("SELECT count(*) FROM organizations")
        if cur.fetchone()[0] == 0:
            default_org_id = "org_default"
            cur.execute("INSERT INTO organizations (id, name) VALUES (?, ?)", (default_org_id, "Default Organization"))
            
            # Default Admin
            admin_pass = os.environ.get("LIGHTRAG_ADMIN_PASSWORD", "admin")
            admin_hash = hash_password(admin_pass)
            cur.execute(
                "INSERT INTO users (id, username, password_hash, org_id, role) VALUES (?, ?, ?, ?, ?)",
                ("user_admin", "admin", admin_hash, default_org_id, "admin")
            )
            print(f"Initialized default DB. Admin user 'admin' created with password '{admin_pass}'")

# Initialize on import logic is moved to explicit call in server startup
# to avoid side effects during imports in tests

# --- User Operations ---
def get_organization(org_id: str) -> Optional[Dict[str, Any]]:
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM organizations WHERE id = ?", (org_id,))
        row = cur.fetchone()
        return dict(row) if row else None

def create_organization(org_id: str, name: str):
    with get_db_cursor() as cur:
        cur.execute("INSERT OR IGNORE INTO organizations (id, name) VALUES (?, ?)", (org_id, name))

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        return dict(row) if row else None

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None

def create_user(username: str, password: str, org_id: str, role: str = "user", email: str = None) -> Optional[Dict[str, Any]]:
    user_id = f"user_{secrets.token_hex(8)}"
    pw_hash = hash_password(password)
    try:
        with get_db_cursor() as cur:
            cur.execute(
                "INSERT INTO users (id, username, password_hash, org_id, role, email) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, username, pw_hash, org_id, role, email)
            )
        # Committed here
        return get_user_by_id(user_id)
    except sqlite3.IntegrityError:
        return None

# --- Chat Operations ---
def create_chat_session(user_id: str, name: str = "New Chat") -> Dict[str, Any]:
    session_id = f"chat_{secrets.token_hex(8)}"
    with get_db_cursor() as cur:
        cur.execute(
            "INSERT INTO chat_sessions (id, user_id, name) VALUES (?, ?, ?)",
            (session_id, user_id, name)
        )
        # return inserted
        cur.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
        return dict(cur.fetchone())

def get_user_chat_sessions(user_id: str) -> List[Dict[str, Any]]:
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
        return [dict(row) for row in cur.fetchall()]

def get_chat_messages(session_id: str) -> List[Dict[str, Any]]:
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
        return [dict(row) for row in cur.fetchall()]

def add_chat_message(session_id: str, role: str, content: str) -> Dict[str, Any]:
    msg_id = f"msg_{secrets.token_hex(8)}"
    with get_db_cursor() as cur:
        cur.execute(
            "INSERT INTO chat_messages (id, session_id, role, content) VALUES (?, ?, ?, ?)",
            (msg_id, session_id, role, content)
        )
        # Update session timestamp
        cur.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,)
        )
        cur.execute("SELECT * FROM chat_messages WHERE id = ?", (msg_id,))
        return dict(cur.fetchone())

def get_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        return dict(row) if row else None

def delete_chat_session(session_id: str):
    with get_db_cursor() as cur:
        # Delete messages first (FK constraint usually handles cascade if set, but let's be safe)
        cur.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
