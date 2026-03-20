import unittest
from unittest.mock import MagicMock
import sys
import bcrypt
import os

# Add the project root to sys.path to ensure lightrag can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestAuthHandler(unittest.TestCase):
    def setUp(self):
        # Mock global_args
        self.mock_global_args = MagicMock()
        self.mock_global_args.token_secret = "lightrag-jwt-default-secret-key!"
        self.mock_global_args.jwt_algorithm = "HS256"
        self.mock_global_args.token_expire_hours = 48
        self.mock_global_args.guest_token_expire_hours = 24

        # Set some test accounts: admin with plaintext, user with bcrypt hash
        user_pass = "user_pass"
        user_hash = bcrypt.hashpw(user_pass.encode("utf-8"), bcrypt.gensalt()).decode(
            "utf-8"
        )
        self.mock_global_args.auth_accounts = f"admin:admin_pass,user:{user_hash}"

        # Patch the global_args in the module
        with unittest.mock.patch(
            "lightrag.api.auth.global_args", self.mock_global_args
        ):
            from lightrag.api.auth import AuthHandler

            self.handler = AuthHandler()
            # Manually update accounts because AuthHandler.__init__ uses the patched global_args
            self.handler.accounts = {"admin": "admin_pass", "user": user_hash}

    def test_verify_plaintext_password(self):
        # Admin has plaintext password 'admin_pass'
        self.assertTrue(self.handler.verify_password("admin", "admin_pass"))
        self.assertFalse(self.handler.verify_password("admin", "wrong_pass"))

    def test_verify_bcrypt_password(self):
        # User has bcrypt hashed password
        self.assertTrue(self.handler.verify_password("user", "user_pass"))
        self.assertFalse(self.handler.verify_password("user", "wrong_pass"))

    def test_verify_bcrypt_y_prefix(self):
        # Test $2y$ prefix support
        password = "y_pass"
        # Manually create a $2y$ hash by replacing $2b$
        hash_b = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode(
            "utf-8"
        )
        hash_y = hash_b.replace("$2b$", "$2y$")
        self.handler.accounts["y_user"] = hash_y

        self.assertTrue(self.handler.verify_password("y_user", password))

    def test_nonexistent_user(self):
        self.assertFalse(self.handler.verify_password("nobody", "any_pass"))

    def test_hash_password(self):
        from lightrag.api.auth import AuthHandler

        password = "new_password"
        hashed = AuthHandler.hash_password(password)
        self.assertTrue(hashed.startswith("$2b$"))
        self.assertTrue(
            bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        )


if __name__ == "__main__":
    unittest.main()
