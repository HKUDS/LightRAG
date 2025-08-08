"""
Database migration script for LightRAG Authentication Phase 1.

This script implements the database changes required for:
- Enhanced password security
- Password history tracking
- Account lockout functionality
- Audit logging tables
"""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from lightrag.api.auth.password_manager import PasswordManager
except ImportError:
    print(
        "Warning: Could not import PasswordManager. Password migration will be skipped."
    )
    PasswordManager = None

logger = logging.getLogger("lightrag.migration.auth_phase1")


class AuthPhase1Migration:
    """Authentication Phase 1 database migration."""

    def __init__(self, db_connection):
        self.db = db_connection
        self.migration_name = "auth_phase1_enhancement"
        self.version = "1.0.0"
        self.password_manager = PasswordManager() if PasswordManager else None

    async def migrate(self) -> bool:
        """
        Execute Phase 1 migration.

        Returns:
            True if migration succeeded, False otherwise
        """
        try:
            logger.info(f"Starting {self.migration_name} migration v{self.version}")

            # Log migration start
            await self._log_migration_start()

            # Execute migration steps
            await self._create_migration_tracking()
            await self._enhance_users_table()
            await self._create_password_history_table()
            await self._create_audit_log_tables()
            await self._migrate_existing_passwords()
            await self._create_indexes()

            # Log migration success
            await self._log_migration_success()

            logger.info(f"Migration {self.migration_name} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            await self._log_migration_failure(str(e))
            return False

    async def rollback(self) -> bool:
        """
        Rollback Phase 1 migration.

        Returns:
            True if rollback succeeded, False otherwise
        """
        try:
            logger.info(f"Rolling back {self.migration_name} migration")

            # Drop created tables and columns in reverse order
            await self._rollback_indexes()
            await self._rollback_audit_tables()
            await self._rollback_password_history()
            await self._rollback_users_enhancements()

            # Update migration log
            await self._log_migration_rollback()

            logger.info(f"Rollback of {self.migration_name} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    async def _log_migration_start(self):
        """Log migration start."""
        # Create migration log table if it doesn't exist
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS migration_log (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(255) NOT NULL,
                version VARCHAR(50) NOT NULL,
                status ENUM('started', 'completed', 'failed', 'rolled_back') NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP NULL,
                error_message TEXT NULL,
                migration_data JSON NULL,
                INDEX idx_migration_name (migration_name),
                INDEX idx_status (status)
            )
        """)

        await self.db.execute(
            """
            INSERT INTO migration_log (migration_name, version, status, migration_data)
            VALUES (?, ?, 'started', ?)
            """,
            self.migration_name,
            self.version,
            '{"phase": 1, "description": "Enhanced password security and audit logging"}',
        )

    async def _create_migration_tracking(self):
        """Create migration tracking table."""
        logger.info("Creating migration tracking table")

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(255) UNIQUE NOT NULL,
                version VARCHAR(50) NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum VARCHAR(64),
                INDEX idx_migration_name (migration_name)
            )
        """)

    async def _enhance_users_table(self):
        """Enhance users table with security columns."""
        logger.info("Enhancing users table with security columns")

        # Check if users table exists
        table_exists = await self._table_exists("users")
        if not table_exists:
            # Create users table if it doesn't exist
            await self.db.execute("""
                CREATE TABLE users (
                    id VARCHAR(255) PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE,
                    password VARCHAR(255),
                    role VARCHAR(50) DEFAULT 'user',
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_username (username),
                    INDEX idx_email (email),
                    INDEX idx_role (role)
                )
            """)
            logger.info("Created users table")

        # Add new security columns
        columns_to_add = [
            ("password_hash_new", "VARCHAR(255)", "New bcrypt password hash"),
            ("password_changed_at", "TIMESTAMP", "Last password change timestamp"),
            ("password_attempts", "INTEGER DEFAULT 0", "Failed login attempts counter"),
            ("account_locked_until", "TIMESTAMP NULL", "Account lockout expiration"),
            ("last_login_at", "TIMESTAMP NULL", "Last successful login"),
            ("last_login_ip", "VARCHAR(45)", "Last login IP address"),
            (
                "security_questions_set",
                "BOOLEAN DEFAULT FALSE",
                "Security questions configured",
            ),
            (
                "two_factor_enabled",
                "BOOLEAN DEFAULT FALSE",
                "Two-factor authentication enabled",
            ),
            ("email_verified", "BOOLEAN DEFAULT FALSE", "Email verification status"),
            ("password_reset_token", "VARCHAR(255)", "Password reset token"),
            (
                "password_reset_expires",
                "TIMESTAMP NULL",
                "Password reset token expiration",
            ),
        ]

        for column_name, column_type, description in columns_to_add:
            if not await self._column_exists("users", column_name):
                await self.db.execute(f"""
                    ALTER TABLE users
                    ADD COLUMN {column_name} {column_type}
                    COMMENT '{description}'
                """)
                logger.info(f"Added column: users.{column_name}")

    async def _create_password_history_table(self):
        """Create password history table."""
        logger.info("Creating password history table")

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS password_history (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR(45),
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_id (user_id),
                INDEX idx_created_at (created_at)
            ) COMMENT 'Password change history for security policy enforcement'
        """)

    async def _create_audit_log_tables(self):
        """Create audit logging tables."""
        logger.info("Creating audit log tables")

        # Main audit events table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(36) UNIQUE NOT NULL,
                event_type VARCHAR(100) NOT NULL,
                severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- User and session info
                user_id VARCHAR(255),
                session_id VARCHAR(255),
                ip_address VARCHAR(45),
                user_agent TEXT,

                -- Request info
                endpoint VARCHAR(500),
                method VARCHAR(10),
                status_code INTEGER,
                response_time FLOAT,

                -- Event details
                message TEXT,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,

                -- Additional context
                resource_id VARCHAR(255),
                resource_type VARCHAR(100),
                action VARCHAR(100),
                correlation_id VARCHAR(36),

                -- Metadata
                details JSON,
                tags JSON,
                metadata JSON,

                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                INDEX idx_event_type (event_type),
                INDEX idx_severity (severity),
                INDEX idx_timestamp (timestamp),
                INDEX idx_user_id (user_id),
                INDEX idx_ip_address (ip_address),
                INDEX idx_correlation_id (correlation_id),
                INDEX idx_event_user_time (event_type, user_id, timestamp),
                INDEX idx_severity_time (severity, timestamp)
            ) COMMENT 'Comprehensive audit log for security events'
        """)

        # Security incidents table for high-priority events
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS security_incidents (
                id SERIAL PRIMARY KEY,
                incident_id VARCHAR(36) UNIQUE NOT NULL,
                incident_type VARCHAR(100) NOT NULL,
                severity ENUM('medium', 'high', 'critical') NOT NULL,
                status ENUM('open', 'investigating', 'resolved', 'false_positive') DEFAULT 'open',

                -- Detection info
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detected_by VARCHAR(100) DEFAULT 'system',

                -- Incident details
                title VARCHAR(500) NOT NULL,
                description TEXT,
                affected_user_id VARCHAR(255),
                affected_resources JSON,

                -- Response info
                assigned_to VARCHAR(255),
                resolved_at TIMESTAMP NULL,
                resolution_notes TEXT,

                -- Related events
                related_event_ids JSON,

                FOREIGN KEY (affected_user_id) REFERENCES users(id) ON DELETE SET NULL,
                INDEX idx_incident_type (incident_type),
                INDEX idx_severity (severity),
                INDEX idx_status (status),
                INDEX idx_detected_at (detected_at),
                INDEX idx_affected_user (affected_user_id)
            ) COMMENT 'Security incidents requiring investigation'
        """)

        # Login attempts table for detailed authentication tracking
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id SERIAL PRIMARY KEY,
                attempt_id VARCHAR(36) UNIQUE NOT NULL,
                username VARCHAR(255),
                user_id VARCHAR(255),
                ip_address VARCHAR(45) NOT NULL,
                user_agent TEXT,

                -- Attempt details
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN NOT NULL,
                failure_reason VARCHAR(255),
                auth_method VARCHAR(50) DEFAULT 'password',

                -- Additional context
                session_id VARCHAR(255),
                request_headers JSON,
                geolocation JSON,

                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                INDEX idx_username (username),
                INDEX idx_user_id (user_id),
                INDEX idx_ip_address (ip_address),
                INDEX idx_attempted_at (attempted_at),
                INDEX idx_success (success),
                INDEX idx_ip_time (ip_address, attempted_at),
                INDEX idx_user_time (user_id, attempted_at)
            ) COMMENT 'Detailed login attempt tracking for security analysis'
        """)

    async def _migrate_existing_passwords(self):
        """Migrate existing passwords to new bcrypt hashes."""
        if not self.password_manager:
            logger.warning(
                "Password manager not available, skipping password migration"
            )
            return

        logger.info("Migrating existing passwords to bcrypt")

        # Get users with passwords that need migration
        users = await self.db.fetch("""
            SELECT id, username, password
            FROM users
            WHERE password IS NOT NULL
            AND password != ''
            AND password_hash_new IS NULL
        """)

        migrated_count = 0
        failed_count = 0

        for user in users:
            try:
                # Note: This assumes existing passwords are plain text
                # Adjust based on your current password storage method
                if user["password"]:
                    new_hash = self.password_manager.hash_password(user["password"])

                    await self.db.execute(
                        """
                        UPDATE users
                        SET password_hash_new = ?, password_changed_at = ?
                        WHERE id = ?
                    """,
                        new_hash,
                        datetime.now(timezone.utc),
                        user["id"],
                    )

                    migrated_count += 1
                    logger.debug(f"Migrated password for user: {user['username']}")

            except Exception as e:
                logger.error(
                    f"Failed to migrate password for user {user['username']}: {e}"
                )
                failed_count += 1

        logger.info(
            f"Password migration completed: {migrated_count} migrated, {failed_count} failed"
        )

    async def _create_indexes(self):
        """Create additional indexes for performance."""
        logger.info("Creating performance indexes")

        indexes = [
            # Users table indexes
            "CREATE INDEX IF NOT EXISTS idx_users_password_attempts ON users(password_attempts)",
            "CREATE INDEX IF NOT EXISTS idx_users_account_locked ON users(account_locked_until)",
            "CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login_at)",
            # Audit events composite indexes
            "CREATE INDEX IF NOT EXISTS idx_audit_severity_user ON audit_events(severity, user_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_type_time ON audit_events(event_type, timestamp DESC)",
            # Login attempts indexes
            "CREATE INDEX IF NOT EXISTS idx_login_ip_time ON login_attempts(ip_address, attempted_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_login_failures ON login_attempts(success, ip_address, attempted_at)",
        ]

        for index_sql in indexes:
            try:
                await self.db.execute(index_sql)
            except Exception as e:
                logger.warning(f"Index creation skipped (may already exist): {e}")

    async def _table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        try:
            result = await self.db.fetch(
                """
                SELECT COUNT(*) as count
                FROM information_schema.tables
                WHERE table_name = ? AND table_schema = DATABASE()
            """,
                table_name,
            )
            return result[0]["count"] > 0
        except Exception:
            # Fallback method
            try:
                await self.db.fetch(f"SELECT 1 FROM {table_name} LIMIT 1")
                return True
            except Exception:
                return False

    async def _column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in table."""
        try:
            result = await self.db.fetch(
                """
                SELECT COUNT(*) as count
                FROM information_schema.columns
                WHERE table_name = ? AND column_name = ? AND table_schema = DATABASE()
            """,
                table_name,
                column_name,
            )
            return result[0]["count"] > 0
        except Exception:
            return False

    async def _log_migration_success(self):
        """Log successful migration."""
        await self.db.execute(
            """
            UPDATE migration_log
            SET status = 'completed', completed_at = CURRENT_TIMESTAMP
            WHERE migration_name = ? AND status = 'started'
        """,
            self.migration_name,
        )

        # Record in schema migrations
        await self.db.execute(
            """
            INSERT INTO schema_migrations (migration_name, version, applied_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE version = VALUES(version), applied_at = CURRENT_TIMESTAMP
        """,
            self.migration_name,
            self.version,
        )

    async def _log_migration_failure(self, error_message: str):
        """Log migration failure."""
        await self.db.execute(
            """
            UPDATE migration_log
            SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error_message = ?
            WHERE migration_name = ? AND status = 'started'
        """,
            error_message,
            self.migration_name,
        )

    async def _log_migration_rollback(self):
        """Log migration rollback."""
        await self.db.execute(
            """
            UPDATE migration_log
            SET status = 'rolled_back', completed_at = CURRENT_TIMESTAMP
            WHERE migration_name = ?
        """,
            self.migration_name,
        )

        # Remove from schema migrations
        await self.db.execute(
            """
            DELETE FROM schema_migrations WHERE migration_name = ?
        """,
            self.migration_name,
        )

    # Rollback methods
    async def _rollback_indexes(self):
        """Rollback created indexes."""
        logger.info("Rolling back indexes")

        indexes_to_drop = [
            "idx_users_password_attempts",
            "idx_users_account_locked",
            "idx_users_last_login",
            "idx_audit_severity_user",
            "idx_audit_type_time",
            "idx_login_ip_time",
            "idx_login_failures",
        ]

        for index_name in indexes_to_drop:
            try:
                await self.db.execute(f"DROP INDEX IF EXISTS {index_name}")
            except Exception as e:
                logger.warning(f"Could not drop index {index_name}: {e}")

    async def _rollback_audit_tables(self):
        """Rollback audit tables."""
        logger.info("Rolling back audit tables")

        tables_to_drop = ["login_attempts", "security_incidents", "audit_events"]

        for table in tables_to_drop:
            try:
                await self.db.execute(f"DROP TABLE IF EXISTS {table}")
                logger.info(f"Dropped table: {table}")
            except Exception as e:
                logger.error(f"Could not drop table {table}: {e}")

    async def _rollback_password_history(self):
        """Rollback password history table."""
        logger.info("Rolling back password history table")

        try:
            await self.db.execute("DROP TABLE IF EXISTS password_history")
        except Exception as e:
            logger.error(f"Could not drop password_history table: {e}")

    async def _rollback_users_enhancements(self):
        """Rollback users table enhancements."""
        logger.info("Rolling back users table enhancements")

        columns_to_remove = [
            "password_hash_new",
            "password_changed_at",
            "password_attempts",
            "account_locked_until",
            "last_login_at",
            "last_login_ip",
            "security_questions_set",
            "two_factor_enabled",
            "email_verified",
            "password_reset_token",
            "password_reset_expires",
        ]

        for column in columns_to_remove:
            try:
                await self.db.execute(
                    f"ALTER TABLE users DROP COLUMN IF EXISTS {column}"
                )
                logger.info(f"Dropped column: users.{column}")
            except Exception as e:
                logger.warning(f"Could not drop column users.{column}: {e}")


async def run_migration(db_connection, rollback: bool = False):
    """
    Run the authentication Phase 1 migration.

    Args:
        db_connection: Database connection
        rollback: If True, perform rollback instead of migration
    """
    migration = AuthPhase1Migration(db_connection)

    if rollback:
        success = await migration.rollback()
        action = "Rollback"
    else:
        success = await migration.migrate()
        action = "Migration"

    if success:
        print(f"✅ {action} completed successfully!")
        return 0
    else:
        print(f"❌ {action} failed!")
        return 1


if __name__ == "__main__":
    """Run migration script directly."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LightRAG Authentication Phase 1 Migration"
    )
    parser.add_argument(
        "--rollback", action="store_true", help="Rollback the migration"
    )
    parser.add_argument("--db-url", type=str, help="Database connection URL")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🚀 LightRAG Authentication Phase 1 Migration")
    print("=" * 50)

    if args.rollback:
        print("⚠️  ROLLBACK MODE - This will undo Phase 1 changes")
        confirm = input("Are you sure you want to proceed? (yes/no): ")
        if confirm.lower() != "yes":
            print("Rollback cancelled.")
            sys.exit(0)

    # For now, print instructions since we don't have database connection setup
    print("\n📋 Migration Ready")
    print("To run this migration:")
    print("1. Ensure your database is backed up")
    print("2. Import this module in your application")
    print("3. Call run_migration(db_connection) with your database connection")
    print("\nExample:")
    print("from lightrag.api.migrations.auth_phase1_migration import run_migration")
    print("success = await run_migration(your_db_connection)")

    sys.exit(0)
