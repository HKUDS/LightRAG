#!/bin/bash
# ====================================================================
# PostgreSQL Database Backup Script for LightRAG
# ====================================================================

set -e

# Configuration
BACKUP_DIR="/app/backups/database"
LOG_FILE="/app/logs/backup-database.log"
RETENTION_DAYS=${DB_BACKUP_RETENTION_DAYS:-7}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="lightrag_db_${TIMESTAMP}"

# Database connection parameters
PGHOST=${POSTGRES_HOST:-postgres}
PGPORT=${POSTGRES_PORT:-5432}
PGUSER=${POSTGRES_USER:-lightrag_prod}
PGPASSWORD=${POSTGRES_PASSWORD}
PGDATABASE=${POSTGRES_DATABASE:-lightrag_production}

export PGHOST PGPORT PGUSER PGPASSWORD PGDATABASE

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

log "Starting database backup: $BACKUP_NAME"

# Check database connectivity
if ! pg_isready -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE"; then
    log "ERROR: Cannot connect to database"
    exit 1
fi

# Create database dump
BACKUP_FILE="$BACKUP_DIR/${BACKUP_NAME}.sql"
COMPRESSED_BACKUP="$BACKUP_DIR/${BACKUP_NAME}.sql.gz"

log "Creating database dump..."
if pg_dump \
    --verbose \
    --clean \
    --if-exists \
    --format=plain \
    --no-owner \
    --no-privileges \
    --file="$BACKUP_FILE" \
    "$PGDATABASE"; then
    log "Database dump created successfully"
else
    log "ERROR: Database dump failed"
    exit 1
fi

# Compress backup
log "Compressing backup..."
if gzip "$BACKUP_FILE"; then
    log "Backup compressed successfully: $COMPRESSED_BACKUP"
else
    log "ERROR: Backup compression failed"
    exit 1
fi

# Verify backup integrity
log "Verifying backup integrity..."
if gzip -t "$COMPRESSED_BACKUP"; then
    log "Backup integrity verified"
else
    log "ERROR: Backup integrity check failed"
    exit 1
fi

# Calculate backup size
BACKUP_SIZE=$(du -h "$COMPRESSED_BACKUP" | cut -f1)
log "Backup size: $BACKUP_SIZE"

# Upload to cloud storage (if configured)
if [ -n "$AWS_S3_BUCKET" ]; then
    log "Uploading to AWS S3..."
    if aws s3 cp "$COMPRESSED_BACKUP" "s3://$AWS_S3_BUCKET/database/"; then
        log "Backup uploaded to S3 successfully"
    else
        log "WARNING: S3 upload failed"
    fi
fi

if [ -n "$RCLONE_CONFIG" ]; then
    log "Uploading with rclone..."
    if rclone copy "$COMPRESSED_BACKUP" "$RCLONE_REMOTE:database/"; then
        log "Backup uploaded with rclone successfully"
    else
        log "WARNING: rclone upload failed"
    fi
fi

# Clean up old backups
log "Cleaning up old backups (retention: $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "lightrag_db_*.sql.gz" -mtime +$RETENTION_DAYS -delete

# Count remaining backups
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/lightrag_db_*.sql.gz 2>/dev/null | wc -l)
log "Local backup count: $BACKUP_COUNT"

log "Database backup completed successfully: $BACKUP_NAME"

# Create backup manifest
MANIFEST_FILE="$BACKUP_DIR/backup_manifest.json"
cat > "$MANIFEST_FILE" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_name": "$BACKUP_NAME",
    "database": "$PGDATABASE",
    "size": "$BACKUP_SIZE",
    "file": "$COMPRESSED_BACKUP",
    "retention_days": $RETENTION_DAYS,
    "local_backup_count": $BACKUP_COUNT
}
EOF

log "Backup manifest created: $MANIFEST_FILE"
