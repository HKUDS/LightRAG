import hmac

import bcrypt

BCRYPT_PASSWORD_PREFIX = "{bcrypt}"

# bcrypt only reads the first 72 bytes of its input. bcrypt>=5 raises a bare
# ValueError for anything longer, and bcrypt 4 silently ignores the extra bytes,
# so a longer password would be either unhashable or only partly checked.
BCRYPT_MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    """Return an AUTH_ACCOUNTS-ready bcrypt password value.

    Raises:
        ValueError: If the password is longer than 72 bytes in UTF-8.
    """
    password_bytes = len(password.encode("utf-8"))
    if password_bytes > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(
            f"password is {password_bytes} bytes in UTF-8; bcrypt accepts at "
            f"most {BCRYPT_MAX_PASSWORD_BYTES}"
        )
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

    # Constant-time comparison for plaintext specs. A plain ``==`` short-circuits
    # on the first mismatched byte, leaking password length and content through
    # response timing (CWE-208, GHSA-c759-cx9p-mrwq). compare_digest runs in time
    # independent of where the first difference occurs.
    return hmac.compare_digest(
        stored_password.encode("utf-8"), plain_password.encode("utf-8")
    )
