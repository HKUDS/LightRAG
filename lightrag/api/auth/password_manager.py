"""
Enhanced password security manager for LightRAG.

Provides secure password hashing, validation, and policy enforcement
with bcrypt hashing and comprehensive security features.
"""

import bcrypt
import secrets
import re
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("lightrag.auth.password")


class PasswordStrength(Enum):
    """Password strength levels."""
    WEAK = "weak"
    FAIR = "fair"
    GOOD = "good"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class PasswordPolicy:
    """Password policy configuration."""
    min_length: int = 8
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    history_count: int = 5
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    complexity_score_threshold: int = 3
    
    # Special characters allowed
    special_chars: str = "!@#$%^&*(),.?\":{}|<>[]\\-_=+"
    
    @classmethod
    def from_env(cls) -> "PasswordPolicy":
        """Create password policy from environment variables."""
        import os
        
        def get_env_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, "").lower()
            if value in ("true", "1", "yes", "on"):
                return True
            elif value in ("false", "0", "no", "off"):
                return False
            return default
        
        return cls(
            min_length=int(os.getenv("PASSWORD_MIN_LENGTH", "8")),
            max_length=int(os.getenv("PASSWORD_MAX_LENGTH", "128")),
            require_uppercase=get_env_bool("PASSWORD_REQUIRE_UPPERCASE", True),
            require_lowercase=get_env_bool("PASSWORD_REQUIRE_LOWERCASE", True),
            require_numbers=get_env_bool("PASSWORD_REQUIRE_NUMBERS", True),
            require_special_chars=get_env_bool("PASSWORD_REQUIRE_SPECIAL_CHARS", True),
            history_count=int(os.getenv("PASSWORD_HISTORY_COUNT", "5")),
            max_failed_attempts=int(os.getenv("PASSWORD_LOCKOUT_ATTEMPTS", "5")),
            lockout_duration_minutes=int(os.getenv("PASSWORD_LOCKOUT_DURATION_MINUTES", "30")),
            complexity_score_threshold=int(os.getenv("PASSWORD_COMPLEXITY_THRESHOLD", "3"))
        )
    
    def validate(self, password: str) -> Tuple[bool, List[str], PasswordStrength]:
        """
        Validate password against policy.
        
        Returns:
            Tuple of (is_valid, error_messages, strength_level)
        """
        errors = []
        score = 0
        
        # Length check
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        elif len(password) >= self.min_length:
            score += 1
            
        if len(password) > self.max_length:
            errors.append(f"Password must not exceed {self.max_length} characters")
        
        # Character requirements
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        elif self.require_uppercase and re.search(r'[A-Z]', password):
            score += 1
            
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        elif self.require_lowercase and re.search(r'[a-z]', password):
            score += 1
            
        if self.require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        elif self.require_numbers and re.search(r'\d', password):
            score += 1
            
        if self.require_special_chars:
            special_pattern = f"[{re.escape(self.special_chars)}]"
            if not re.search(special_pattern, password):
                errors.append(f"Password must contain at least one special character: {self.special_chars}")
            else:
                score += 1
        
        # Additional complexity checks
        if len(password) >= 12:
            score += 1
        if len(set(password)) >= 8:  # Character diversity
            score += 1
        if not re.search(r'(.)\1{2,}', password):  # No repeated characters
            score += 1
        
        # Determine strength
        if score <= 2:
            strength = PasswordStrength.WEAK
        elif score <= 3:
            strength = PasswordStrength.FAIR
        elif score <= 4:
            strength = PasswordStrength.GOOD
        elif score <= 5:
            strength = PasswordStrength.STRONG
        else:
            strength = PasswordStrength.VERY_STRONG
        
        is_valid = len(errors) == 0 and score >= self.complexity_score_threshold
        
        return is_valid, errors, strength


class PasswordManager:
    """Enhanced password management with security features."""
    
    def __init__(self, policy: Optional[PasswordPolicy] = None):
        self.policy = policy or PasswordPolicy.from_env()
        self.rounds = 12  # bcrypt rounds for hashing
        
    def hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt with salt.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Bcrypt hashed password string
        """
        if not password:
            raise ValueError("Password cannot be empty")
        
        # Generate salt and hash
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        logger.info("Password hashed successfully")
        return hashed.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            plain_password: Plain text password to verify
            hashed_password: Stored bcrypt hash
            
        Returns:
            True if password matches, False otherwise
        """
        if not plain_password or not hashed_password:
            return False
        
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8'), 
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def validate_password(self, password: str, user_id: str = None) -> Tuple[bool, List[str], PasswordStrength]:
        """
        Validate password against policy.
        
        Args:
            password: Password to validate
            user_id: Optional user ID for history checking
            
        Returns:
            Tuple of (is_valid, error_messages, strength_level)
        """
        is_valid, errors, strength = self.policy.validate(password)
        
        # Check against common passwords (basic implementation)
        if self._is_common_password(password):
            errors.append("Password is too common, please choose a more unique password")
            is_valid = False
        
        return is_valid, errors, strength
    
    def generate_secure_password(self, length: int = 16) -> str:
        """
        Generate cryptographically secure password.
        
        Args:
            length: Desired password length (minimum 8)
            
        Returns:
            Generated secure password
        """
        if length < 8:
            length = 8
        
        # Character sets
        lowercase = "abcdefghijklmnopqrstuvwxyz"
        uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        numbers = "0123456789"
        special = self.policy.special_chars
        
        # Ensure at least one character from each required set
        password = []
        
        if self.policy.require_lowercase:
            password.append(secrets.choice(lowercase))
        if self.policy.require_uppercase:
            password.append(secrets.choice(uppercase))
        if self.policy.require_numbers:
            password.append(secrets.choice(numbers))
        if self.policy.require_special_chars:
            password.append(secrets.choice(special))
        
        # Fill remaining length with random characters from all sets
        all_chars = lowercase + uppercase + numbers + special
        for _ in range(length - len(password)):
            password.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate cryptographically secure token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            URL-safe base64 encoded token
        """
        return secrets.token_urlsafe(length)
    
    def generate_numeric_code(self, length: int = 6) -> str:
        """
        Generate numeric code for verification.
        
        Args:
            length: Code length
            
        Returns:
            Numeric code string
        """
        return ''.join(secrets.choice('0123456789') for _ in range(length))
    
    def hash_token(self, token: str) -> str:
        """
        Hash token for secure storage.
        
        Args:
            token: Token to hash
            
        Returns:
            SHA-256 hash of token
        """
        return hashlib.sha256(token.encode('utf-8')).hexdigest()
    
    def _is_common_password(self, password: str) -> bool:
        """
        Check if password is in common passwords list.
        
        Args:
            password: Password to check
            
        Returns:
            True if password is common
        """
        # Basic common passwords list - in production, use a comprehensive list
        common_passwords = {
            "password", "123456", "12345678", "qwerty", "abc123",
            "password123", "admin", "letmein", "welcome", "monkey",
            "dragon", "master", "trustno1", "shadow", "111111"
        }
        
        return password.lower() in common_passwords
    
    async def check_password_history(self, user_id: str, new_password: str, 
                                   db_connection) -> bool:
        """
        Check if password was used recently.
        
        Args:
            user_id: User identifier
            new_password: New password to check
            db_connection: Database connection
            
        Returns:
            True if password is allowed (not in recent history)
        """
        if self.policy.history_count <= 0:
            return True
        
        try:
            # Get recent password hashes
            recent_hashes = await db_connection.fetch(
                """
                SELECT password_hash FROM password_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
                """,
                user_id, self.policy.history_count
            )
            
            # Check if new password matches any recent hash
            for row in recent_hashes:
                if self.verify_password(new_password, row['password_hash']):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking password history: {e}")
            # Allow password change if we can't check history
            return True
    
    async def store_password_history(self, user_id: str, password_hash: str, 
                                   db_connection) -> None:
        """
        Store password hash in history.
        
        Args:
            user_id: User identifier
            password_hash: Hashed password to store
            db_connection: Database connection
        """
        try:
            # Store new password hash
            await db_connection.execute(
                """
                INSERT INTO password_history (user_id, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                user_id, password_hash, datetime.utcnow()
            )
            
            # Clean up old history beyond the limit
            await db_connection.execute(
                """
                DELETE FROM password_history 
                WHERE user_id = ? 
                AND created_at < (
                    SELECT created_at FROM (
                        SELECT created_at FROM password_history 
                        WHERE user_id = ? 
                        ORDER BY created_at DESC 
                        LIMIT 1 OFFSET ?
                    ) AS subquery
                )
                """,
                user_id, user_id, self.policy.history_count - 1
            )
            
        except Exception as e:
            logger.error(f"Error storing password history: {e}")
    
    async def check_account_lockout(self, user_id: str, db_connection) -> Tuple[bool, Optional[datetime]]:
        """
        Check if account is locked due to failed attempts.
        
        Args:
            user_id: User identifier
            db_connection: Database connection
            
        Returns:
            Tuple of (is_locked, unlock_time)
        """
        try:
            user_data = await db_connection.fetchrow(
                "SELECT password_attempts, account_locked_until FROM users WHERE id = ?",
                user_id
            )
            
            if not user_data:
                return False, None
            
            locked_until = user_data.get('account_locked_until')
            if locked_until and locked_until > datetime.utcnow():
                return True, locked_until
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking account lockout: {e}")
            return False, None
    
    async def record_failed_attempt(self, user_id: str, db_connection) -> bool:
        """
        Record failed login attempt and lock account if threshold reached.
        
        Args:
            user_id: User identifier
            db_connection: Database connection
            
        Returns:
            True if account was locked
        """
        try:
            # Increment failed attempts
            result = await db_connection.execute(
                """
                UPDATE users 
                SET password_attempts = COALESCE(password_attempts, 0) + 1 
                WHERE id = ?
                """,
                user_id
            )
            
            # Check if lockout threshold reached
            user_data = await db_connection.fetchrow(
                "SELECT password_attempts FROM users WHERE id = ?",
                user_id
            )
            
            if user_data and user_data['password_attempts'] >= self.policy.max_failed_attempts:
                # Lock account
                lockout_until = datetime.utcnow() + timedelta(
                    minutes=self.policy.lockout_duration_minutes
                )
                
                await db_connection.execute(
                    "UPDATE users SET account_locked_until = ? WHERE id = ?",
                    lockout_until, user_id
                )
                
                logger.warning(f"Account locked for user {user_id} due to {self.policy.max_failed_attempts} failed attempts")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error recording failed attempt: {e}")
            return False
    
    async def reset_failed_attempts(self, user_id: str, db_connection) -> None:
        """
        Reset failed login attempts counter.
        
        Args:
            user_id: User identifier
            db_connection: Database connection
        """
        try:
            await db_connection.execute(
                """
                UPDATE users 
                SET password_attempts = 0, account_locked_until = NULL 
                WHERE id = ?
                """,
                user_id
            )
            
        except Exception as e:
            logger.error(f"Error resetting failed attempts: {e}")


# Global password manager instance
password_manager = PasswordManager()