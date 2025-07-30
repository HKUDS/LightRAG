#!/bin/bash
# ====================================================================
# Data Directory Backup Script for LightRAG
# ====================================================================

set -e

# Configuration
BACKUP_DIR="/app/backups/data"
LOG_FILE="/app/logs/backup-data.log"
DATA_DIR="/app/data"
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="lightrag_data_${TIMESTAMP}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

log "Starting data backup: $BACKUP_NAME"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    log "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Calculate data directory size
DATA_SIZE=$(du -sh "$DATA_DIR" | cut -f1)
log "Data directory size: $DATA_SIZE"

# Create tarball backup
BACKUP_FILE="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"

log "Creating data backup archive..."
if tar -czf "$BACKUP_FILE" \
    --exclude="*.tmp" \
    --exclude="*.log" \
    --exclude="cache/*" \
    -C "$(dirname "$DATA_DIR")" \
    "$(basename "$DATA_DIR")"; then
    log "Data backup archive created successfully"
else
    log "ERROR: Data backup failed"
    exit 1
fi

# Verify backup integrity
log "Verifying backup integrity..."
if tar -tzf "$BACKUP_FILE" > /dev/null; then
    log "Backup integrity verified"
else
    log "ERROR: Backup integrity check failed"
    exit 1
fi

# Calculate backup size
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
log "Backup size: $BACKUP_SIZE"

# Encrypt backup if enabled
if [ "${BACKUP_ENCRYPTION:-false}" = "true" ] && [ -n "$BACKUP_ENCRYPTION_KEY" ]; then
    log "Encrypting backup..."
    ENCRYPTED_BACKUP="${BACKUP_FILE}.enc"
    if openssl enc -aes-256-cbc -salt -in "$BACKUP_FILE" -out "$ENCRYPTED_BACKUP" -k "$BACKUP_ENCRYPTION_KEY"; then
        log "Backup encrypted successfully"
        rm "$BACKUP_FILE"  # Remove unencrypted version
        BACKUP_FILE="$ENCRYPTED_BACKUP"
    else
        log "ERROR: Backup encryption failed"
        exit 1
    fi
fi

# Upload to cloud storage (if configured)
if [ -n "$AWS_S3_BUCKET" ]; then
    log "Uploading to AWS S3..."
    if aws s3 cp "$BACKUP_FILE" "s3://$AWS_S3_BUCKET/data/"; then
        log "Backup uploaded to S3 successfully"
    else
        log "WARNING: S3 upload failed"
    fi
fi

if [ -n "$RCLONE_CONFIG" ]; then
    log "Uploading with rclone..."
    if rclone copy "$BACKUP_FILE" "$RCLONE_REMOTE:data/"; then
        log "Backup uploaded with rclone successfully"
    else
        log "WARNING: rclone upload failed"
    fi
fi

# Clean up old backups
log "Cleaning up old backups (retention: $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "lightrag_data_*.tar.gz*" -mtime +$RETENTION_DAYS -delete

# Count remaining backups
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/lightrag_data_*.tar.gz* 2>/dev/null | wc -l)
log "Local backup count: $BACKUP_COUNT"

log "Data backup completed successfully: $BACKUP_NAME"

# Create backup manifest
MANIFEST_FILE="$BACKUP_DIR/data_backup_manifest.json"
cat > "$MANIFEST_FILE" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_name": "$BACKUP_NAME",
    "data_size": "$DATA_SIZE",
    "backup_size": "$BACKUP_SIZE",
    "file": "$BACKUP_FILE",
    "encrypted": $([ "${BACKUP_ENCRYPTION:-false}" = "true" ] && echo "true" || echo "false"),
    "retention_days": $RETENTION_DAYS,
    "local_backup_count": $BACKUP_COUNT
}
EOF

log "Backup manifest created: $MANIFEST_FILE"