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

    # Run database migrations
    python -c "
import asyncio
import sys
import os
import psycopg2

sys.path.insert(0, '/app')

async def main():
    try:
        from lightrag.api.migrations.auth_phase1_migration import AuthPhase1Migration

        # Create a database connection
        conn = psycopg2.connect(
            dbname=os.environ['POSTGRES_DATABASE'],
            user=os.environ['POSTGRES_USER'],
            password=os.environ['POSTGRES_PASSWORD'],
            host=os.environ['POSTGRES_HOST'],
            port=os.environ.get('POSTGRES_PORT', 5432)
        )

        migration = AuthPhase1Migration(db_connection=conn)
        await migration.migrate()  # Corrected method call
        print('Database migration completed successfully')

        conn.close()

    except Exception as e:
        print(f'Database migration failed: {e}')
        # Don't exit - continue startup

asyncio.run(main())
"
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

    # Start the application server
    exec gunicorn \
        --config /app/gunicorn_config.py \
        --bind 0.0.0.0:${PORT:-9621} \
        --workers ${WORKERS:-4} \
        --worker-class uvicorn.workers.UvicornWorker \
        --worker-timeout ${WORKER_TIMEOUT:-300} \
        --max-requests ${MAX_REQUESTS:-1000} \
        --max-requests-jitter ${MAX_REQUESTS_JITTER:-50} \
        --preload \
        --access-logfile - \
        --error-logfile - \
        --log-level ${LOG_LEVEL:-info} \
        "lightrag.api.app:app"
}

# Handle signals gracefully
trap 'echo "Received shutdown signal, stopping gracefully..."; exit 0' TERM INT

# Run main function
main "$@"
