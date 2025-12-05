#!/usr/bin/env python
"""
Workspace-to-Tenant Migration Script

Migrates existing single-tenant workspace-based deployments to multi-tenant architecture.
This script:
1. Scans existing workspace directories
2. Creates a default tenant for each workspace
3. Creates a default knowledge base within each tenant
4. Preserves all existing data structure for backward compatibility

Usage:
    python migrate_workspace_to_tenant.py --working-dir /path/to/rag_storage
    python migrate_workspace_to_tenant.py --working-dir /path/to/rag_storage --dry-run
    python migrate_workspace_to_tenant.py --working-dir /path/to/rag_storage --skip-backup
"""

import asyncio
import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from lightrag.services.tenant_service import TenantService
from lightrag.utils import logger


class WorkspaceToTenantMigrator:
    """
    Handles migration from workspace-based to multi-tenant architecture.
    """

    def __init__(self, working_dir: str, dry_run: bool = False, backup: bool = True):
        """
        Initialize the migrator.

        Args:
            working_dir: Root directory containing workspace folders
            dry_run: If True, simulate migration without making changes
            backup: If True, create backup before migration
        """
        self.working_dir = Path(working_dir)
        self.dry_run = dry_run
        self.backup = backup
        self.tenant_service = TenantService()
        self.migration_log: List[str] = []
        self.error_log: List[str] = []

    def validate_working_dir(self) -> bool:
        """Validate that working directory exists."""
        if not self.working_dir.exists():
            self.error_log.append(
                f"Working directory does not exist: {self.working_dir}"
            )
            return False

        if not self.working_dir.is_dir():
            self.error_log.append(f"Path is not a directory: {self.working_dir}")
            return False

        return True

    def discover_workspaces(self) -> List[str]:
        """
        Discover existing workspace directories.

        Workspaces are identified by common RAG storage files like:
        - kv_store_*.json
        - doc_status_storage.json
        - rag_storage.db

        Returns:
            List of workspace directory names
        """
        workspaces = []

        # Check for common RAG storage files
        for item in self.working_dir.iterdir():
            if not item.is_dir():
                continue

            # Skip special directories
            if item.name.startswith((".", "__")):
                continue

            # Check if directory contains RAG storage files
            has_rag_files = (
                any(
                    [
                        (item / f"kv_store_{name}.json").exists()
                        for name in [
                            "full_docs",
                            "text_chunks",
                            "entities",
                            "relations",
                        ]
                    ]
                )
                or (item / "doc_status_storage.json").exists()
            )

            if has_rag_files or item.name.startswith("workspace_"):
                workspaces.append(item.name)

        return sorted(workspaces)

    def backup_working_dir(self) -> Optional[Path]:
        """
        Create a backup of the working directory.

        Returns:
            Path to backup directory, or None if backup failed
        """
        if not self.backup:
            return None

        backup_dir = (
            self.working_dir.parent
            / f"{self.working_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        try:
            msg = f"Creating backup at {backup_dir}"
            logger.info(msg)
            self.migration_log.append(msg)

            if not self.dry_run:
                shutil.copytree(self.working_dir, backup_dir)

            return backup_dir
        except Exception as e:
            msg = f"Failed to create backup: {e}"
            logger.error(msg)
            self.error_log.append(msg)
            return None

    async def migrate_workspace(self, workspace_name: str) -> bool:
        """
        Migrate a single workspace to multi-tenant structure.

        Args:
            workspace_name: Name of the workspace to migrate

        Returns:
            True if migration successful, False otherwise
        """
        try:
            msg = f"\nMigrating workspace: {workspace_name}"
            logger.info(msg)
            self.migration_log.append(msg)

            # Create tenant from workspace
            tenant_name = workspace_name if workspace_name != "" else "default"

            if not self.dry_run:
                tenant = await self.tenant_service.create_tenant(
                    tenant_name=tenant_name,
                    config=None,  # Use default config
                )

                msg = f"  ✓ Created tenant '{tenant_name}' with ID: {tenant.tenant_id}"
                logger.info(msg)
                self.migration_log.append(msg)

                # Create default knowledge base
                kb = await self.tenant_service.create_knowledge_base(
                    tenant_id=tenant.tenant_id,
                    kb_name="default",
                    description="Default knowledge base (migrated from workspace)",
                )

                msg = f"  ✓ Created default KB with ID: {kb.kb_id}"
                logger.info(msg)
                self.migration_log.append(msg)
            else:
                msg = f"  [DRY RUN] Would create tenant '{tenant_name}' with default KB"
                logger.info(msg)
                self.migration_log.append(msg)

            return True

        except Exception as e:
            msg = f"  ✗ Failed to migrate workspace '{workspace_name}': {e}"
            logger.error(msg)
            self.error_log.append(msg)
            return False

    async def migrate_all_workspaces(self, workspaces: List[str]) -> Dict[str, bool]:
        """
        Migrate all discovered workspaces.

        Args:
            workspaces: List of workspace names to migrate

        Returns:
            Dictionary mapping workspace name to migration status
        """
        results = {}

        for workspace in workspaces:
            success = await self.migrate_workspace(workspace)
            results[workspace] = success

        return results

    def generate_report(self, workspaces: List[str], results: Dict[str, bool]) -> str:
        """
        Generate a migration report.

        Args:
            workspaces: List of workspaces processed
            results: Migration results

        Returns:
            Formatted report string
        """
        successful = sum(1 for v in results.values() if v)
        failed = len(workspaces) - successful

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           WORKSPACE-TO-TENANT MIGRATION REPORT              ║
╚══════════════════════════════════════════════════════════════╝

Working Directory: {self.working_dir}
Dry Run Mode: {self.dry_run}
Workspaces Processed: {len(workspaces)}
Successfully Migrated: {successful}
Failed: {failed}

Migration Log:
"""
        for line in self.migration_log:
            report += f"\n{line}"

        if self.error_log:
            report += "\n\nErrors Encountered:"
            for error in self.error_log:
                report += f"\n{error}"

        report += "\n"
        return report

    async def run(self) -> bool:
        """
        Execute the migration process.

        Returns:
            True if migration completed successfully, False otherwise
        """
        # Validate setup
        if not self.validate_working_dir():
            logger.error("Validation failed")
            return False

        # Discover workspaces
        workspaces = self.discover_workspaces()

        if not workspaces:
            msg = "No workspaces found to migrate"
            logger.warning(msg)
            self.migration_log.append(msg)
            return True

        msg = f"Discovered {len(workspaces)} workspace(s): {', '.join(workspaces)}"
        logger.info(msg)
        self.migration_log.append(msg)

        # Create backup if not dry-run
        if not self.dry_run:
            backup_path = self.backup_working_dir()
            if not backup_path and self.backup:
                logger.warning("Backup failed but continuing with migration")

        # Migrate workspaces
        results = await self.migrate_all_workspaces(workspaces)

        # Generate and display report
        report = self.generate_report(workspaces, results)
        print(report)

        # Save report to file
        report_path = (
            self.working_dir
            / f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        try:
            if not self.dry_run:
                with open(report_path, "w") as f:
                    f.write(report)
                logger.info(f"Migration report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save migration report: {e}")

        # Return success if no failures
        return all(results.values())


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate workspace-based deployment to multi-tenant architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Perform actual migration
  python migrate_workspace_to_tenant.py --working-dir /path/to/rag_storage

  # Preview what would be migrated without making changes
  python migrate_workspace_to_tenant.py --working-dir /path/to/rag_storage --dry-run

  # Migrate without creating backup
  python migrate_workspace_to_tenant.py --working-dir /path/to/rag_storage --skip-backup
        """,
    )

    parser.add_argument(
        "--working-dir",
        required=True,
        help="Path to the working directory containing workspaces",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making actual changes",
    )

    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip creating a backup of the working directory",
    )

    args = parser.parse_args()

    # Create migrator
    migrator = WorkspaceToTenantMigrator(
        working_dir=args.working_dir, dry_run=args.dry_run, backup=not args.skip_backup
    )

    # Run migration
    try:
        success = asyncio.run(migrator.run())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
