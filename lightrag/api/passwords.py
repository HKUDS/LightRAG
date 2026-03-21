import bcrypt

BCRYPT_PASSWORD_PREFIX = "{bcrypt}"


def hash_password(password: str) -> str:
    """Return an AUTH_ACCOUNTS-ready bcrypt password value."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    return f"{BCRYPT_PASSWORD_PREFIX}{hashed}"


def verify_password(plain_password: str, stored_password: str) -> bool:
    """Verify a plaintext password against a stored password spec."""
    if stored_password.startswith(BCRYPT_PASSWORD_PREFIX):
        hashed_password = stored_password[len(BCRYPT_PASSWORD_PREFIX) :]
        if not hashed_password:
            return False
        try:
            return bcrypt.checkpw(
                plain_password.encode("utf-8"), hashed_password.encode("utf-8")
            )
        except ValueError:
            return False

    return stored_password == plain_password
