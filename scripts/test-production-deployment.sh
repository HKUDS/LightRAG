#!/bin/bash

# ====================================================================
# LightRAG Production Deployment Test Script
# ====================================================================
# This script performs comprehensive testing of the LightRAG production
# deployment including all components: PostgreSQL, Redis, xAI, Ollama,
# knowledge graphs, and RAG querying.
#
# Usage: ./test-production-deployment.sh
# Requirements: curl, jq, docker-compose
# ====================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LIGHTRAG_URL="http://localhost:9621"
COMPOSE_FILE="docker-compose.production.yml"
TEST_DOCUMENT_PATH="/tmp/lightrag_test_document.txt"
POSTGRES_USER="${POSTGRES_USER:-lightrag_user}"
POSTGRES_DB="${POSTGRES_DATABASE:-lightrag_db}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-36Eae9j8bNPqgo}"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Test counter
TESTS_PASSED=0
TESTS_TOTAL=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    log "Running test: $test_name"

    if eval "$test_command"; then
        success "$test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        error "$test_name"
        return 1
    fi
}

# Wait for service to be ready
wait_for_service() {
    local url="$1"
    local service_name="$2"
    local max_attempts=30
    local attempt=1

    log "Waiting for $service_name to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            success "$service_name is ready"
            return 0
        fi

        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    error "$service_name failed to start within $(($max_attempts * 2)) seconds"
}

# Create test document
create_test_document() {
    log "Creating test document..."

    cat > "$TEST_DOCUMENT_PATH" << 'EOF'
# LightRAG System Overview

LightRAG is an advanced Retrieval-Augmented Generation (RAG) system that combines knowledge graphs with vector retrieval for enhanced document processing and querying capabilities.

## Key Components

### Core Architecture
LightRAG integrates multiple storage backends including PostgreSQL for robust data persistence, Redis for high-performance caching, and specialized vector databases for embedding storage.

### Language Models
The system supports multiple LLM providers including OpenAI, xAI's Grok models, and local Ollama deployments. For embeddings, it can utilize various models such as BGE-M3 for multilingual support.

### Knowledge Graph Processing
LightRAG uses Apache AGE extension for PostgreSQL to create and manage knowledge graphs. This enables sophisticated relationship modeling between entities extracted from documents.

## Production Features
- Enterprise security with JWT authentication
- Rate limiting and audit logging
- Docker containerization with production hardening
- Multi-storage backend support
- Comprehensive monitoring with Prometheus and Grafana

## Use Cases
LightRAG is ideal for organizations that need to process large volumes of documents and generate intelligent responses based on the content. It excels in scenarios requiring both semantic search and relationship understanding between concepts.

## Advanced Capabilities
The system provides hybrid query modes combining local context-dependent retrieval with global knowledge graph queries. This dual approach ensures comprehensive coverage of both specific document content and broader conceptual relationships.

## Integration Benefits
By leveraging both vector similarity search and graph-based reasoning, LightRAG delivers superior performance in complex question-answering scenarios where traditional RAG systems might fall short.
EOF

    success "Test document created at $TEST_DOCUMENT_PATH"
}

# Test Functions
test_container_status() {
    log "Checking container status..."
    local containers=(
        "lightrag_app:up"
        "lightrag_postgres:up"
        "lightrag_redis:up"
        "lightrag_grafana:up"
        "lightrag_prometheus:up"
        "lightrag_jaeger:up"
    )

    for container_check in "${containers[@]}"; do
        IFS=':' read -r container expected_status <<< "$container_check"
        local status
        status=$(docker compose -f "$COMPOSE_FILE" ps --format "{{.State}}" "$container" 2>/dev/null | head -1)

        if ([[ "$status" == *"Up"* ]] || [[ "$status" == *"running"* ]]) && [[ "$expected_status" == "up" ]]; then
            success "Container $container is running"
        else
            error "Container $container is not running (status: $status)"
        fi
    done
}

test_health_endpoint() {
    local response
    response=$(curl -sf "$LIGHTRAG_URL/health")

    if echo "$response" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

test_postgresql_connection() {
    log "Testing PostgreSQL connection and extensions..."

    # Test connection
    docker compose -f "$COMPOSE_FILE" exec -u postgres postgres \
        env PGPASSWORD="$POSTGRES_PASSWORD" \
        psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        -c "SELECT version();" > /dev/null 2>&1

    # Test extensions
    local extensions=("vector" "age" "pg_stat_statements")
    for ext in "${extensions[@]}"; do
        docker compose -f "$COMPOSE_FILE" exec -u postgres postgres \
            env PGPASSWORD="$POSTGRES_PASSWORD" \
            psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
            -c "SELECT * FROM pg_extension WHERE extname='$ext';" | grep -q "$ext"

        if [ $? -eq 0 ]; then
            success "Extension $ext is installed"
        else
            error "Extension $ext is not installed"
        fi
    done
}

test_document_upload() {
    log "Testing document upload..."

    local response
    response=$(curl -sf -X POST "$LIGHTRAG_URL/documents/upload" \
        -F "file=@$TEST_DOCUMENT_PATH" \
        -H "Content-Type: multipart/form-data")

    if echo "$response" | jq -e '.status == "success"' > /dev/null 2>&1; then
        success "Document uploaded successfully"
        # Extract track_id for monitoring
        TRACK_ID=$(echo "$response" | jq -r '.track_id')
        log "Track ID: $TRACK_ID"
        return 0
    else
        error "Document upload failed: $response"
        return 1
    fi
}

test_document_processing() {
    log "Monitoring document processing..."

    local max_attempts=60  # 2 minutes
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        local status_response
        status_response=$(curl -sf "$LIGHTRAG_URL/documents/pipeline_status")

        local busy
        busy=$(echo "$status_response" | jq -r '.busy')

        if [ "$busy" = "false" ]; then
            local latest_message
            latest_message=$(echo "$status_response" | jq -r '.latest_message')

            if [[ "$latest_message" == *"completed"* ]]; then
                success "Document processing completed"

                # Display processing stats
                local history
                history=$(echo "$status_response" | jq -r '.history_messages[]' | grep -E "(entities|relations|Ent|Rel)")
                if [ -n "$history" ]; then
                    log "Processing stats: $history"
                fi

                return 0
            else
                error "Document processing failed: $latest_message"
                return 1
            fi
        fi

        if [ $((attempt % 10)) -eq 0 ]; then
            local current_message
            current_message=$(echo "$status_response" | jq -r '.latest_message')
            log "Processing status: $current_message"
        fi

        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    error "Document processing timeout after $(($max_attempts * 2)) seconds"
}

test_document_status() {
    log "Checking processed document status..."

    local response
    response=$(curl -sf "$LIGHTRAG_URL/documents")

    local processed_count
    processed_count=$(echo "$response" | jq -r '.statuses.processed | length')

    if [ "$processed_count" -gt 0 ]; then
        success "Found $processed_count processed document(s)"

        # Display document details
        echo "$response" | jq -r '.statuses.processed[] | "Document: \(.file_path), Status: \(.status), Chunks: \(.chunks_count)"'

        return 0
    else
        error "No processed documents found"
        return 1
    fi
}

test_postgresql_data() {
    log "Verifying data storage in PostgreSQL..."

    # Check entities
    local entity_count
    entity_count=$(docker compose -f "$COMPOSE_FILE" exec -u postgres postgres \
        env PGPASSWORD="$POSTGRES_PASSWORD" \
        psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        -t -c "SELECT COUNT(*) FROM lightrag_vdb_entity;" | tr -d ' ')

    if [ "$entity_count" -gt 0 ]; then
        success "Found $entity_count entities in PostgreSQL"
    else
        error "No entities found in PostgreSQL"
    fi

    # Check relations
    local relation_count
    relation_count=$(docker compose -f "$COMPOSE_FILE" exec -u postgres postgres \
        env PGPASSWORD="$POSTGRES_PASSWORD" \
        psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        -t -c "SELECT COUNT(*) FROM lightrag_vdb_relation;" | tr -d ' ')

    if [ "$relation_count" -gt 0 ]; then
        success "Found $relation_count relations in PostgreSQL"
    else
        error "No relations found in PostgreSQL"
    fi

    # Check document status
    local doc_count
    doc_count=$(docker compose -f "$COMPOSE_FILE" exec -u postgres postgres \
        env PGPASSWORD="$POSTGRES_PASSWORD" \
        psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
        -t -c "SELECT COUNT(*) FROM lightrag_doc_status;" | tr -d ' ')

    if [ "$doc_count" -gt 0 ]; then
        success "Found $doc_count document(s) in status table"
        return 0
    else
        error "No documents found in status table"
        return 1
    fi
}

test_rag_query() {
    local query_text="$1"
    local query_mode="$2"

    log "Testing RAG query (mode: $query_mode): '$query_text'"

    local response
    response=$(curl -sf -X POST "$LIGHTRAG_URL/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query_text\", \"mode\": \"$query_mode\"}")

    if echo "$response" | jq -e '.response' > /dev/null 2>&1; then
        local answer
        answer=$(echo "$response" | jq -r '.response')

        if [ ${#answer} -gt 100 ]; then
            success "RAG query returned comprehensive answer (${#answer} characters)"
            log "Answer preview: ${answer:0:200}..."

            # Check for references
            if echo "$answer" | grep -q "\[KG\]\|\[DC\]"; then
                success "Answer includes proper citations"
            else
                warning "Answer lacks citations"
            fi

            return 0
        else
            error "RAG query returned short answer: $answer"
            return 1
        fi
    else
        error "RAG query failed: $response"
        return 1
    fi
}

test_monitoring_endpoints() {
    log "Testing monitoring endpoints..."

    local endpoints=(
        "http://localhost:3000:Grafana"
        "http://localhost:9091:Prometheus"
        "http://localhost:16686:Jaeger"
    )

    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r url service <<< "$endpoint_info"

        if curl -sf "$url" > /dev/null 2>&1; then
            success "$service monitoring is accessible"
        else
            warning "$service monitoring is not accessible (this is expected if services are starting)"
        fi
    done
}

cleanup_test_files() {
    log "Cleaning up test files..."
    [ -f "$TEST_DOCUMENT_PATH" ] && rm -f "$TEST_DOCUMENT_PATH"
    success "Test files cleaned up"
}

# Main test execution
main() {
    echo -e "${BLUE}"
    echo "======================================================================"
    echo "              LightRAG Production Deployment Test Suite"
    echo "======================================================================"
    echo -e "${NC}"

    log "Starting comprehensive production deployment tests..."

    # Prerequisites check
    log "Checking prerequisites..."
    command -v curl >/dev/null 2>&1 || error "curl is required but not installed"
    command -v jq >/dev/null 2>&1 || error "jq is required but not installed"
    command -v docker >/dev/null 2>&1 || error "docker is required but not installed"
    success "Prerequisites check passed"

    # Create test document
    create_test_document

    # Wait for services
    wait_for_service "$LIGHTRAG_URL/health" "LightRAG API"

    # Run tests
    echo -e "\n${YELLOW}=== INFRASTRUCTURE TESTS ===${NC}"
    run_test "Container Status Check" "test_container_status"
    run_test "LightRAG Health Check" "test_health_endpoint"
    run_test "PostgreSQL Connection & Extensions" "test_postgresql_connection"
    run_test "Monitoring Endpoints" "test_monitoring_endpoints"

    echo -e "\n${YELLOW}=== DOCUMENT PROCESSING TESTS ===${NC}"
    run_test "Document Upload" "test_document_upload"
    run_test "Document Processing" "test_document_processing"
    run_test "Document Status Verification" "test_document_status"
    run_test "PostgreSQL Data Verification" "test_postgresql_data"

    echo -e "\n${YELLOW}=== RAG QUERY TESTS ===${NC}"
    run_test "Hybrid Query Test" "test_rag_query 'What is LightRAG and what are its key features?' 'hybrid'"
    run_test "Local Query Test" "test_rag_query 'What storage backends does LightRAG support?' 'local'"
    run_test "Global Query Test" "test_rag_query 'How does LightRAG handle knowledge graphs?' 'global'"

    # Cleanup
    cleanup_test_files

    # Results summary
    echo -e "\n${BLUE}======================================================================"
    echo "                            TEST RESULTS"
    echo "======================================================================${NC}"

    if [ $TESTS_PASSED -eq $TESTS_TOTAL ]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED! ($TESTS_PASSED/$TESTS_TOTAL)${NC}"
        echo -e "${GREEN}✅ Production deployment is fully operational and ready for use!${NC}"

        echo -e "\n${BLUE}Available Services:${NC}"
        echo -e "• LightRAG API: $LIGHTRAG_URL"
        echo -e "• Grafana Dashboard: http://localhost:3000 (admin/admin)"
        echo -e "• Prometheus Metrics: http://localhost:9091"
        echo -e "• Jaeger Tracing: http://localhost:16686"

        exit 0
    else
        echo -e "${RED}❌ SOME TESTS FAILED! ($TESTS_PASSED/$TESTS_TOTAL passed)${NC}"
        echo -e "${RED}Please check the logs above for details.${NC}"
        exit 1
    fi
}

# Handle script interruption
trap 'echo -e "\n${YELLOW}Test script interrupted. Cleaning up...${NC}"; cleanup_test_files; exit 130' INT TERM

# Run main function
main "$@"
