#!/bin/bash

# ============================================================================
# LightRAG Multi-Tenant Testing Script
#
# This script runs all three testing scenarios sequentially:
#   1. Backward Compatibility Mode (MULTITENANT_MODE=off)
#   2. Single-Tenant Multi-KB Mode (MULTITENANT_MODE=on)
#   3. Full Multi-Tenant Demo Mode (MULTITENANT_MODE=demo)
#
# Usage: ./run_all_tests.sh
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="lightrag-multitenant"
DOCKER_COMPOSE_FILE="docker-compose.yml"
LOG_DIR="test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$LOG_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

log_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

preflight_checks() {
    log_header "PRE-FLIGHT CHECKS"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_success "Docker is installed"

    # Check Docker Compose
    if ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    log_success "Docker Compose is installed"

    # Check pytest
    if ! command -v pytest &> /dev/null; then
        log_error "pytest is not installed"
        exit 1
    fi
    log_success "pytest is installed"

    # Check psql for database connections
    if ! command -v psql &> /dev/null; then
        log_warning "psql is not installed (optional, for manual DB inspection)"
    fi

    echo ""
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_environment() {
    local mode=$1
    log_header "SETTING UP ENVIRONMENT FOR MODE: $mode"

    # Create .env file
    cp env.example .env

    # Add mode-specific configuration
    echo "MULTITENANT_MODE=$mode" >> .env

    if [ "$mode" = "on" ]; then
        echo "DEFAULT_TENANT=tenant-1" >> .env
        echo "CREATE_DEFAULT_KB=kb-default,kb-secondary,kb-experimental" >> .env
    fi

    log_success "Environment configured for mode: $mode"
    echo ""
}

# ============================================================================
# Docker Service Management
# ============================================================================

start_services() {
    local mode=$1
    log_header "STARTING SERVICES (Mode: $mode)"

    log_info "Starting Docker Compose services..."
    docker compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME up -d

    log_info "Waiting for services to be healthy..."
    sleep 15

    # Check if services are running
    if ! docker compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME ps | grep -q "healthy"; then
        log_warning "Some services may not be healthy yet, but proceeding..."
    fi

    log_success "Services started"
    echo ""
}

stop_services() {
    log_info "Stopping services..."
    docker compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME down
    log_success "Services stopped"
}

init_database() {
    local mode=$1
    log_header "INITIALIZING DATABASE (Mode: $mode)"

    log_info "Creating database schema..."
    docker compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME exec -T postgres \
        psql -U lightrag -d postgres \
        -c "CREATE DATABASE lightrag_multitenant;" 2>/dev/null || true

    log_info "Applying initialization script..."
    docker compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME exec -T postgres \
        psql -U lightrag -d lightrag_multitenant \
        -f /docker-entrypoint-initdb.d/01-init.sql

    log_success "Database initialized"
    echo ""
}

# ============================================================================
# Test Execution
# ============================================================================

run_tests() {
    local mode=$1
    local log_file="$LOG_DIR/test_${mode}_${TIMESTAMP}.log"

    log_header "RUNNING TESTS (Mode: $mode)"

    log_info "Executing pytest..."

    export MULTITENANT_MODE=$mode

    # Select test files based on mode
    local test_files=""
    if [ "$mode" = "off" ]; then
        test_files="tests/test_backward_compatibility.py"
    elif [ "$mode" = "on" ]; then
        test_files="tests/test_multi_tenant_backends.py::TestTenantIsolation"
    elif [ "$mode" = "demo" ]; then
        test_files="tests/test_multi_tenant_backends.py tests/test_tenant_security.py"
    fi

    if pytest $test_files -v --tb=short 2>&1 | tee "$log_file"; then
        log_success "All tests passed for mode: $mode"
        echo "$mode: PASSED" >> "$LOG_DIR/summary_${TIMESTAMP}.txt"
    else
        log_error "Tests failed for mode: $mode"
        echo "$mode: FAILED" >> "$LOG_DIR/summary_${TIMESTAMP}.txt"
        return 1
    fi

    echo ""
}

# ============================================================================
# Test Scenarios
# ============================================================================

test_scenario_1() {
    log_header "SCENARIO 1: BACKWARD COMPATIBILITY MODE (MULTITENANT_MODE=off)"

    setup_environment "off"
    start_services "off"
    init_database "off"

    if ! run_tests "off"; then
        log_error "Scenario 1 tests failed"
        stop_services
        return 1
    fi

    stop_services
    log_success "Scenario 1 completed successfully"
    echo ""
}

test_scenario_2() {
    log_header "SCENARIO 2: SINGLE-TENANT MULTI-KB MODE (MULTITENANT_MODE=on)"

    setup_environment "on"
    start_services "on"
    init_database "on"

    if ! run_tests "on"; then
        log_error "Scenario 2 tests failed"
        stop_services
        return 1
    fi

    stop_services
    log_success "Scenario 2 completed successfully"
    echo ""
}

test_scenario_3() {
    log_header "SCENARIO 3: FULL MULTI-TENANT DEMO MODE (MULTITENANT_MODE=demo)"

    setup_environment "demo"
    start_services "demo"
    init_database "demo"

    if ! run_tests "demo"; then
        log_error "Scenario 3 tests failed"
        stop_services
        return 1
    fi

    stop_services
    log_success "Scenario 3 completed successfully"
    echo ""
}

# ============================================================================
# Health Checks
# ============================================================================

health_check() {
    local mode=$1
    log_header "HEALTH CHECK (Mode: $mode)"

    log_info "Checking API health..."
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "API is healthy"
    else
        log_warning "API health check failed"
    fi

    log_info "Checking database..."
    if docker compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME exec -T postgres \
        pg_isready -U lightrag -d lightrag_multitenant > /dev/null; then
        log_success "Database is ready"
    else
        log_warning "Database not ready"
    fi

    echo ""
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    log_header "CLEANUP"

    log_info "Removing Docker containers..."
    docker compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME down -v 2>/dev/null || true

    log_info "Removing temporary .env file..."
    rm -f .env

    log_success "Cleanup completed"
    echo ""
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log_header "LIGHTRAG MULTI-TENANT TESTING SUITE"
    echo ""
    echo "Testing Strategy:"
    echo "  • Scenario 1: Backward Compatibility (no multi-tenant)"
    echo "  • Scenario 2: Single-Tenant Multi-KB"
    echo "  • Scenario 3: Full Multi-Tenant"
    echo ""
    echo "Results will be saved to: $LOG_DIR/"
    echo ""

    # Create summary file
    > "$LOG_DIR/summary_${TIMESTAMP}.txt"
    echo "Test Summary - $TIMESTAMP" > "$LOG_DIR/summary_${TIMESTAMP}.txt"
    echo "==========================================" >> "$LOG_DIR/summary_${TIMESTAMP}.txt"
    echo "" >> "$LOG_DIR/summary_${TIMESTAMP}.txt"

    # Run preflight checks
    preflight_checks

    local failed_tests=0

    # Run all scenarios
    if ! test_scenario_1; then
        ((failed_tests++))
        log_error "Scenario 1 FAILED"
    fi

    if ! test_scenario_2; then
        ((failed_tests++))
        log_error "Scenario 2 FAILED"
    fi

    if ! test_scenario_3; then
        ((failed_tests++))
        log_error "Scenario 3 FAILED"
    fi

    # Cleanup
    cleanup

    # Final Report
    log_header "TEST SUMMARY"

    if [ $failed_tests -eq 0 ]; then
        log_success "ALL TESTS PASSED"
        echo ""
        echo -e "${GREEN}✓ Backward Compatibility: PASSED${NC}"
        echo -e "${GREEN}✓ Single-Tenant Multi-KB: PASSED${NC}"
        echo -e "${GREEN}✓ Full Multi-Tenant: PASSED${NC}"
        echo ""
        cat "$LOG_DIR/summary_${TIMESTAMP}.txt"
        exit 0
    else
        log_error "$failed_tests SCENARIO(S) FAILED"
        echo ""
        cat "$LOG_DIR/summary_${TIMESTAMP}.txt"
        exit 1
    fi
}

# Handle signals
trap cleanup EXIT

# Run main function
main "$@"
