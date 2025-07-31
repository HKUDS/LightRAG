#!/bin/bash
set -e

# Set default environment
export NODE_ENV=${NODE_ENV:-production}
export PYTHON_ENV=${PYTHON_ENV:-production}

# Health check function
health_check() {
    echo "Performing health checks..."

    # Check database connectivity
    if [ "${POSTGRES_HOST:-}" ]; then
        echo "Checking database connectivity..."
        timeout 10 bash -c "</dev/tcp/${POSTGRES_HOST}/${POSTGRES_PORT:-5432}" || {
            echo "ERROR: Cannot connect to database"
            exit 1
        }
    fi

    # Check Redis connectivity
    if [ "${REDIS_URI:-}" ]; then
        echo "Checking Redis connectivity..."
        redis_host=$(echo $REDIS_URI | cut -d/ -f3 | cut -d: -f1)
        redis_port=$(echo $REDIS_URI | cut -d/ -f3 | cut -d: -f2)
        timeout 5 bash -c "</dev/tcp/${redis_host}/${redis_port:-6379}" || {
            echo "ERROR: Cannot connect to Redis"
            exit 1
        }
    fi

    echo "Health checks passed!"
}

# Wait for dependencies
wait_for_dependencies() {
    echo "Waiting for dependencies..."

    if [ "${POSTGRES_HOST:-}" ]; then
        echo "Waiting for PostgreSQL..."
        while ! timeout 1 bash -c "</dev/tcp/${POSTGRES_HOST}/${POSTGRES_PORT:-5432}"; do
            echo "PostgreSQL is unavailable - sleeping"
            sleep 2
        done
        echo "PostgreSQL is up!"
    fi

    if [ "${REDIS_URI:-}" ]; then
        echo "Waiting for Redis..."
        redis_host=$(echo $REDIS_URI | cut -d/ -f3 | cut -d: -f1)
        redis_port=$(echo $REDIS_URI | cut -d/ -f3 | cut -d: -f2)
        while ! timeout 1 bash -c "</dev/tcp/${redis_host}/${redis_port:-6379}"; do
            echo "Redis is unavailable - sleeping"
            sleep 2
        done
        echo "Redis is up!"
    fi
}

# Initialize database if needed
initialize_database() {
    echo "Initializing database..."

    # Test database connectivity
    python <<EOF
import sys
import os
import psycopg2

sys.path.insert(0, '/app')

try:
    # --- Start Debugging ---
    print('--- DATABASE CONNECTION DEBUG INFO ---')
    print(f"DB Host: {os.environ.get('POSTGRES_HOST')}")
    print(f"DB Port: {os.environ.get('POSTGRES_PORT', 5432)}")
    print(f"DB Name: {os.environ.get('POSTGRES_DATABASE')}")
    print(f"DB User: {os.environ.get('POSTGRES_USER')}")
    password = os.environ.get('POSTGRES_PASSWORD', '')
    print(f"DB Password Length: {len(password)}")
    if not password:
        print('WARNING: POSTGRES_PASSWORD environment variable is not set!')
    print('------------------------------------')
    # --- End Debugging ---

    # Test basic database connectivity
    conn = psycopg2.connect(
        dbname=os.environ['POSTGRES_DATABASE'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        host=os.environ['POSTGRES_HOST'],
        port=os.environ.get('POSTGRES_PORT', 5432)
    )

    print('Database connection test successful')
    conn.close()

    # Note: Complex async migrations are handled by the application startup
    # This script focuses on basic connectivity verification

except Exception as e:
    print(f'Database connection test failed: {e}')
    # Don't exit - let the application handle initialization

EOF
}

# Main startup sequence
main() {
    echo "Starting LightRAG Production Server..."
    echo "Environment: ${NODE_ENV}"
    echo "Python Environment: ${PYTHON_ENV}"

    # Wait for dependencies
    wait_for_dependencies

    # Initialize database
    initialize_database

    # Perform health checks
    health_check

    echo "Starting Gunicorn server..."

    # Set environment variable to signal we're running under gunicorn
    export GUNICORN_CMD_ARGS="true"

    # Start the application server
    exec gunicorn \
        --bind "0.0.0.0:9621" \
        --workers 4 \
        --worker-class "uvicorn.workers.UvicornWorker" \
        --timeout 300 \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --preload \
        --access-logfile - \
        --error-logfile - \
        --log-level INFO \
        "lightrag.api.lightrag_server:app"
}

# Handle signals gracefully
trap 'echo "Received shutdown signal, stopping gracefully..."; exit 0' TERM INT

# Run main function
main "$@"
