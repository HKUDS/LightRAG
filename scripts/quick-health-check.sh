#!/bin/bash

# ====================================================================
# LightRAG Quick Health Check Script
# ====================================================================
# This script performs a quick health check of the LightRAG production
# deployment to verify all services are running properly.
#
# Usage: ./quick-health-check.sh
# ====================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

LIGHTRAG_URL="http://localhost:9621"
COMPOSE_FILE="docker-compose.production.yml"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Quick health checks
check_containers() {
    log "Checking container status..."

    local services=("lightrag" "postgres" "redis")
    local all_healthy=true

    for service in "${services[@]}"; do
        local status
        status=$(docker compose -f "$COMPOSE_FILE" ps --format "{{.State}}" "$service" 2>/dev/null | head -1)

        if [[ "$status" == *"Up"* ]] || [[ "$status" == *"running"* ]]; then
            success "$service is running"
        else
            error "$service is not running (status: $status)"
            all_healthy=false
        fi
    done

    return $([ "$all_healthy" = true ])
}

check_api_health() {
    log "Checking LightRAG API health..."

    local response
    if response=$(curl -sf "$LIGHTRAG_URL/health" 2>/dev/null); then
        if echo "$response" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
            success "LightRAG API is healthy"

            # Show configuration summary
            local llm_model
            local embedding_model
            llm_model=$(echo "$response" | jq -r '.configuration.llm_model // "unknown"')
            embedding_model=$(echo "$response" | jq -r '.configuration.embedding_model // "unknown"')

            log "LLM Model: $llm_model"
            log "Embedding Model: $embedding_model"

            return 0
        else
            error "LightRAG API reports unhealthy status"
            return 1
        fi
    else
        error "LightRAG API is not responding"
        return 1
    fi
}

check_database() {
    log "Checking database connectivity..."

    local postgres_user="${POSTGRES_USER:-lightrag_user}"
    local postgres_db="${POSTGRES_DATABASE:-lightrag_db}"
    local postgres_password="${POSTGRES_PASSWORD:-36Eae9j8bNPqgo}"

    if docker compose -f "$COMPOSE_FILE" exec -u postgres postgres \
        env PGPASSWORD="$postgres_password" \
        psql -h localhost -U "$postgres_user" -d "$postgres_db" \
        -c "SELECT 1;" > /dev/null 2>&1; then
        success "PostgreSQL database is accessible"
        return 0
    else
        error "PostgreSQL database is not accessible"
        return 1
    fi
}

check_document_count() {
    log "Checking processed documents..."

    local response
    if response=$(curl -sf "$LIGHTRAG_URL/documents" 2>/dev/null); then
        local processed_count
        processed_count=$(echo "$response" | jq -r '.statuses.processed | length' 2>/dev/null || echo "0")

        if [ "$processed_count" -gt 0 ]; then
            success "Found $processed_count processed document(s)"
        else
            warning "No processed documents found"
        fi
        return 0
    else
        warning "Could not retrieve document status"
        return 1
    fi
}

test_simple_query() {
    log "Testing simple query..."

    local response
    if response=$(curl -sf -X POST "$LIGHTRAG_URL/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "mode": "local"}' 2>/dev/null); then

        if echo "$response" | jq -e '.response' > /dev/null 2>&1; then
            success "Query functionality is working"
            return 0
        else
            warning "Query returned unexpected response"
            return 1
        fi
    else
        warning "Query test failed (may be due to no documents)"
        return 1
    fi
}

show_service_urls() {
    echo -e "\n${BLUE}📋 Service URLs:${NC}"
    echo -e "• LightRAG API: $LIGHTRAG_URL"
    echo -e "• Health Check: $LIGHTRAG_URL/health"
    echo -e "• Grafana: http://localhost:3000"
    echo -e "• Prometheus: http://localhost:9091"
    echo -e "• Jaeger: http://localhost:16686"
}

main() {
    echo -e "${BLUE}🔍 LightRAG Quick Health Check${NC}\n"

    local all_passed=true

    # Run checks
    check_containers || all_passed=false
    check_api_health || all_passed=false
    check_database || all_passed=false
    check_document_count || true  # Non-critical
    test_simple_query || true    # Non-critical

    echo ""
    if [ "$all_passed" = true ]; then
        success "All critical health checks passed! 🎉"
        show_service_urls
        exit 0
    else
        error "Some health checks failed!"
        echo -e "${YELLOW}Run the full test suite for detailed diagnostics:${NC}"
        echo -e "  ./scripts/test-production-deployment.sh"
        exit 1
    fi
}

main "$@"
