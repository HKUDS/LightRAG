"""
LightRAG Database Migrations.

Database migration scripts for authentication enhancements and schema updates.
"""

from .auth_phase1_migration import AuthPhase1Migration, run_migration

__all__ = [
    "AuthPhase1Migration",
    "run_migration"
]