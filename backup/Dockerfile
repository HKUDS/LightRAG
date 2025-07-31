FROM alpine:latest

# Create non-root user
RUN addgroup -g 1000 backup && \
    adduser -D -u 1000 -G backup backup

# Install packages
RUN apk add --no-cache postgresql-client bash dcron

# Create directories and set permissions
RUN mkdir -p /app/backups && \
    chown -R backup:backup /app

COPY backup-script.sh /app/backup-script.sh
RUN chmod +x /app/backup-script.sh && \
    chown backup:backup /app/backup-script.sh

# Setup cron for backup user
RUN echo '${BACKUP_SCHEDULE:-0 2 * * *} /app/backup-script.sh' > /etc/crontabs/backup

# Switch to non-root user
USER backup
WORKDIR /app

CMD ["crond", "-f", "-l", "2"]
