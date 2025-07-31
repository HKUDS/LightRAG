#!/bin/bash
# LightRAG Production Backup Script

set -e

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/app/backups"

echo "Starting backup at $DATE"

# PostgreSQL backup
if [ "${POSTGRES_HOST}" ]; then
    echo "Backing up PostgreSQL database..."
    PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
        -h "${POSTGRES_HOST}" \
        -U "${POSTGRES_USER}" \
        -d "${POSTGRES_DB}" \
        -f "${BACKUP_DIR}/postgres_${DATE}.sql"

    # Compress the backup
    gzip "${BACKUP_DIR}/postgres_${DATE}.sql"
    echo "PostgreSQL backup completed: postgres_${DATE}.sql.gz"
fi

# Data directory backup
if [ -d "/app/data" ]; then
    echo "Backing up data directory..."
    tar -czf "${BACKUP_DIR}/data_${DATE}.tar.gz" -C /app data/
    echo "Data backup completed: data_${DATE}.tar.gz"
fi

# Cleanup old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "*.gz" -mtime +7 -delete

echo "Backup completed at $(date)"
