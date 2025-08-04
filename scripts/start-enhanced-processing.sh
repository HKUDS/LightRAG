#!/bin/bash

# LightRAG Enhanced Processing Startup Script
# This script helps manage the startup of LightRAG with Docling service integration

set -e

echo "🚀 Starting LightRAG with Enhanced Document Processing"
echo "=================================================="

# Function to check if a container is running
check_container() {
    local container_name=$1
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        echo "✅ $container_name is running"
        return 0
    else
        echo "❌ $container_name is not running"
        return 1
    fi
}

# Function to wait for a service to be healthy
wait_for_health() {
    local container_name=$1
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $container_name to be healthy..."

    while [ $attempt -le $max_attempts ]; do
        if docker inspect --format="{{json .State.Health.Status}}" "$container_name" 2>/dev/null | grep -q "healthy"; then
            echo "✅ $container_name is healthy"
            return 0
        fi

        echo "   Attempt $attempt/$max_attempts..."
        sleep 5
        attempt=$((attempt + 1))
    done

    echo "❌ $container_name failed to become healthy within timeout"
    return 1
}

# Step 1: Start PostgreSQL
echo "📁 Step 1: Starting PostgreSQL..."
if ! check_container "lightrag_postgres_dev"; then
    docker compose up postgres -d
    wait_for_health "lightrag_postgres_dev"
fi

# Step 2: Build and start Docling service
echo "🤖 Step 2: Building and starting Docling service..."
echo "   Note: This may take several minutes due to ML dependencies..."

docker compose -f docker-compose.yml -f docker-compose.enhanced.yml --profile enhanced-processing build docling-service
docker compose -f docker-compose.yml -f docker-compose.enhanced.yml --profile enhanced-processing up docling-service -d

echo "⏳ Waiting for Docling service to start..."
sleep 10

if ! wait_for_health "lightrag_docling_dev"; then
    echo "⚠️  Docling service health check failed, but continuing..."
fi

# Step 3: Start LightRAG with enhanced processing
echo "⚡ Step 3: Starting LightRAG with enhanced processing..."
docker compose -f docker-compose.yml -f docker-compose.enhanced.yml --profile enhanced-processing up lightrag -d

echo "⏳ Waiting for LightRAG to start..."
sleep 10

# Step 4: Check status
echo "📊 Final Status Check..."
echo "======================"

check_container "lightrag_postgres_dev"
check_container "lightrag_docling_dev"
check_container "lightrag"

echo ""
echo "🌐 Service URLs:"
echo "   LightRAG API: http://localhost:9621"
echo "   Docling Service: http://localhost:8080"
echo "   PostgreSQL: localhost:5432"
echo ""
echo "📋 To check logs:"
echo "   docker compose logs -f lightrag"
echo "   docker compose logs -f docling-service"
echo ""
echo "🧪 To test integration:"
echo "   python examples/docling_service_demo.py"
echo ""
echo "✨ Enhanced processing setup complete!"
