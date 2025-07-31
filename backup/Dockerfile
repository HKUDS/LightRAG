# ====================================================================
# Backup Service Dockerfile for LightRAG
# ====================================================================

FROM postgres:15-alpine

# Install additional tools
RUN apk add --no-cache \
    curl \
    aws-cli \
    rclone \
    gzip \
    tar \
    cronie \
    bash

# Create backup user
RUN addgroup -S backup && adduser -S backup -G backup

# Create backup directories
RUN mkdir -p /app/backups /app/scripts /app/logs && \
    chown -R backup:backup /app

# Copy backup scripts
COPY scripts/ /app/scripts/
COPY crontab /etc/crontabs/backup

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Switch to backup user
USER backup

WORKDIR /app

# Start cron daemon
CMD ["crond", "-f", "-l", "2"]
