#!/bin/bash
# Docker entrypoint script for LightRAG
# This script initializes demo tenants and starts the server

set -e

# Source environment variables
if [ -f /app/.env ]; then
    export $(cat /app/.env | grep -v '#' | xargs)
fi

# Function to check if server is running
wait_for_server() {
    echo "Waiting for server to be ready..."
    max_attempts=30
    attempt=1

    while [ $attempt -le $max_attempts ]; do
        # Check if the port is listening (simpler check that doesn't require auth)
        if nc -z localhost ${PORT:-9621} 2>/dev/null || timeout 1 bash -c "cat < /dev/null > /dev/tcp/localhost/${PORT:-9621}" 2>/dev/null; then
            echo "✓ Server port is listening!"
            sleep 2  # Give server a bit more time to fully initialize
            return 0
        fi
        echo "  Attempt $attempt/$max_attempts: Server not ready yet, waiting..."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo "⚠ Server startup check timed out after $max_attempts seconds"
    return 0  # Continue anyway
}

# Initialize demo tenants if environment variable is set
if [ "${INIT_DEMO_TENANTS}" = "true" ] || [ "${INIT_DEMO_TENANTS}" = "1" ]; then
    echo "Demo tenant initialization enabled"
    # Note: We'll initialize tenants after server starts
fi

echo "🚀 Starting LightRAG Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start the server in background
python -m lightrag.api.lightrag_server &
SERVER_PID=$!

# Wait for server to be ready
wait_for_server

# Initialize demo tenants if enabled
if [ "${INIT_DEMO_TENANTS}" = "true" ] || [ "${INIT_DEMO_TENANTS}" = "1" ]; then
    echo ""
    echo "📚 Initializing demo tenants..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f /app/scripts/init_demo_tenants.py ]; then
        python /app/scripts/init_demo_tenants.py || echo "⚠ Demo tenant initialization completed with warnings"
        echo ""
    else
        echo "⚠ Demo tenant script not found at /app/scripts/init_demo_tenants.py"
    fi
fi

echo ""
echo "✅ LightRAG is ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait for server process
wait $SERVER_PID
