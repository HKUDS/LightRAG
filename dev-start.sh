#!/bin/bash
# ============================================================================
# LightRAG Hybrid Development Stack
# ============================================================================
# Runs PostgreSQL + Redis in Docker, API + WebUI natively on host
# Uses configuration from .env file at project root
# ============================================================================

set -e

# Colors for beautiful output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
ENV_FILE="$PROJECT_ROOT/.env"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.dev-db.yml"

# PID files for background processes
API_PID_FILE="/tmp/lightrag-dev-api.pid"
WEBUI_PID_FILE="/tmp/lightrag-dev-webui.pid"

# Log files
API_LOG="/tmp/lightrag-dev-api.log"
WEBUI_LOG="/tmp/lightrag-dev-webui.log"

# ============================================================================
# Helper Functions
# ============================================================================

print_banner() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${CYAN}${BOLD}⚡ LightRAG Hybrid Development Stack${NC}                                    ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  ${DIM}PostgreSQL + Redis in Docker │ API + WebUI on Host${NC}                     ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  $1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_step() {
    echo -e "  ${CYAN}▶${NC} $1"
}

print_success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "  ${RED}✗${NC} $1"
}

print_warning() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

# Load environment variables from .env file
load_env() {
    if [ -f "$ENV_FILE" ]; then
        # Export variables from .env, handling comments and empty lines
        # Use while-read loop for broader shell compatibility (macOS, etc.)
        while IFS='=' read -r key value; do
            # Skip empty lines and comments
            [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
            # Trim leading/trailing whitespace from key
            key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            # Skip if key is empty after trimming
            [ -z "$key" ] && continue
            # Remove surrounding quotes from value and carriage return
            value=$(echo "$value" | sed "s/^['\"]//;s/['\"]$//;s/\r$//")
            # Export the variable
            export "$key=$value"
        done < "$ENV_FILE"
        
        # Extract Redis port from REDIS_URI if set
        if [ -n "$REDIS_URI" ]; then
            REDIS_PORT=$(echo "$REDIS_URI" | sed -n 's/.*:\([0-9]*\)$/\1/p')
        fi
        REDIS_PORT=${REDIS_PORT:-16379}
        export REDIS_PORT
    fi
}

# Check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "'$1' is not installed or not in PATH"
        return 1
    fi
    return 0
}

# Check if a port is in use
check_port() {
    local port=$1
    local name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $port ($name) is already in use"
        return 1
    fi
    return 0
}

# Wait for a service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-60}
    local attempt=1
    local log_file=${4:-""}
    
    printf "  ${DIM}Waiting for $name"
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${NC}"
            print_success "$name is ready"
            return 0
        fi
        
        # Check if process is still running (for API server)
        if [ -n "$log_file" ] && [ -f "$log_file" ]; then
            if grep -q "ERROR:" "$log_file" 2>/dev/null; then
                echo -e "${NC}"
                print_error "$name encountered an error during startup"
                echo ""
                echo -e "  ${YELLOW}Recent log entries:${NC}"
                tail -20 "$log_file" | head -15
                echo ""
                return 1
            fi
        fi
        
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    echo -e "${NC}"
    print_warning "$name may take longer to start"
    return 1
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo -e "\n${RED}Script interrupted or failed (exit code: $exit_code)${NC}"
        echo -e "${YELLOW}Run './dev-stop.sh' to clean up any started services${NC}"
    fi
}

trap cleanup ERR SIGINT SIGTERM

# ============================================================================
# Main Script
# ============================================================================

### Parse args
AUTO_CONFIRM=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -y|--yes)
            AUTO_CONFIRM=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

print_banner

# Check for .env file
if [ ! -f "$ENV_FILE" ]; then
    print_error ".env file not found at project root!"
    echo ""
    echo -e "  ${YELLOW}Create one from the example:${NC}"
    echo -e "    cp env.example .env"
    echo -e "    # Then edit .env with your configuration"
    echo ""
    exit 1
fi

# Load environment
load_env

# Display key runtime configuration (models / providers) and show partial view of keys
print_section "Configuration Summary"
print_step "LLM (text generation/extraction): ${LLM_BINDING:-<not-set>} / ${LLM_MODEL:-<not-set>}"
print_step "Embeddings (vector/embedding): ${EMBEDDING_BINDING:-<not-set>} / ${EMBEDDING_MODEL:-<not-set>} (${EMBEDDING_DIM:-?} dim)"

mask_key() {
    key="$1"
    if [ -z "$key" ]; then
        echo "<not-set>"
        return
    fi
    # Show first 6 and last 4 characters (if long enough)
    if [ ${#key} -le 12 ]; then
        echo "${key:0:3}****${key: -3}"
    else
        echo "${key:0:6}****${key: -4}"
    fi
}

print_step "LLM API key: $(mask_key "$LLM_BINDING_API_KEY")"
print_step "Embedding API key: $(mask_key "$EMBEDDING_BINDING_API_KEY")"

# Check ports early and optionally stop/kill conflicting processes or containers
assess_and_handle_ports() {
    local ports_to_check=()
    # Postgres and Redis will be started by docker compose but still inspect host occupancy
    ports_to_check+=("${POSTGRES_PORT:-15432}:postgres")
    ports_to_check+=("${REDIS_PORT:-16379}:redis")
    ports_to_check+=("${PORT:-9621}:api")
    ports_to_check+=("5173:webui")

    # Our own dev containers - safe to restart without prompting
    local OUR_CONTAINERS="lightrag-dev-postgres|lightrag-dev-redis"

    for p in "${ports_to_check[@]}"; do
        hostport=${p%%:*}
        service=${p##*:}

        # Check docker containers that publish this port (macOS compatible - no mapfile)
        container_using=""
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            name=$(echo "$line" | awk '{print $1}')
            ports=$(echo "$line" | cut -d' ' -f2-)
            if echo "$ports" | grep -qE ":${hostport}->|0\.0\.0\.0:${hostport}->"; then
                container_using="$name"
                break
            fi
        done < <(docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null || true)

        if [ -n "$container_using" ]; then
            # Check if it's one of our own dev containers - safe to auto-restart
            if echo "$container_using" | grep -qE "^($OUR_CONTAINERS)$"; then
                print_info "Port $hostport ($service): Our dev container '$container_using' is running (will be restarted by docker-compose)"
                continue
            fi
            
            print_warning "Port $hostport for $service is used by Docker container: $container_using"
            if [ "$AUTO_CONFIRM" = true ]; then
                print_step "Stopping container $container_using..."
                docker stop "$container_using" >/dev/null 2>&1 || true
                print_success "Stopped container $container_using"
            elif [ -t 1 ] && [ -t 0 ]; then
                # Only prompt if both stdout and stdin are terminals (truly interactive)
                read -r -p "Stop Docker container '$container_using' using port $hostport? [y/N] " answer </dev/tty
                if echo "$answer" | grep -iq "^y"; then
                    print_step "Stopping container $container_using..."
                    docker stop "$container_using" >/dev/null 2>&1 || true
                    print_success "Stopped container $container_using"
                else
                    print_warning "Left container running: $container_using. This may prevent the dev stack from starting correctly."
                fi
            else
                # Non-interactive mode: warn but don't block
                print_warning "Non-interactive mode: skipping prompt. Use --yes to auto-stop foreign containers."
            fi
            continue
        fi

        # Check native process using port
        pid=$(lsof -ti :$hostport -sTCP:LISTEN -P -n 2>/dev/null || true)
        if [ -n "$pid" ]; then
            pname=$(ps -p "$pid" -o comm= 2>/dev/null || echo "<unknown>")
            
            # Check if this is likely our own dev server (Python process on api/webui ports)
            is_our_process=false
            if [ "$service" = "api" ] || [ "$service" = "webui" ]; then
                # Check if process command line contains lightrag or uvicorn
                cmdline=$(ps -p "$pid" -o args= 2>/dev/null || echo "")
                if echo "$cmdline" | grep -qiE "lightrag|uvicorn|bun.*dev|vite"; then
                    is_our_process=true
                fi
            fi
            
            if [ "$is_our_process" = true ]; then
                print_info "Port $hostport ($service): Our previous dev process PID:$pid - will restart"
                kill -9 "$pid" 2>/dev/null || true
                sleep 1
                print_success "Stopped previous $service process"
            elif [ "$AUTO_CONFIRM" = true ]; then
                print_warning "Port $hostport for $service is used by process PID:$pid ($pname)"
                print_step "Killing process $pid on port $hostport..."
                kill -9 "$pid" 2>/dev/null || true
                print_success "Killed PID $pid"
            elif [ -t 1 ] && [ -t 0 ]; then
                # Only prompt if both stdout and stdin are terminals (truly interactive)
                print_warning "Port $hostport for $service is used by process PID:$pid ($pname)"
                read -r -p "Kill process PID $pid ($pname) on port $hostport? [y/N] " answer </dev/tty
                if echo "$answer" | grep -iq "^y"; then
                    print_step "Killing process $pid on port $hostport..."
                    kill -9 "$pid" 2>/dev/null || true
                    print_success "Killed PID $pid"
                else
                    print_warning "Left native process $pid running. It may block starting the dev stack."
                fi
            else
                # Non-interactive mode: warn but don't block
                print_warning "Port $hostport for $service is used by process PID:$pid ($pname)"
                print_warning "Non-interactive mode: skipping prompt. Use --yes to auto-kill processes."
            fi
        else
            print_success "Port $hostport ($service) is free"
        fi
    done
}

if [ "$AUTO_CONFIRM" = true ]; then
    print_section "Startup Policy"
    print_info "Auto-confirmation is enabled (will stop conflicting containers/processes)"
fi

# Assess ports before starting services
print_section "Port assessment"
assess_and_handle_ports

# ============================================================================
# Step 1: Prerequisites Check
# ============================================================================
print_section "Step 1/5: Checking Prerequisites"

MISSING_DEPS=0

print_step "Checking required tools..."

if check_command docker; then
    print_success "Docker installed"
else
    MISSING_DEPS=1
fi

if check_command python3 || check_command python; then
    PYTHON_CMD=$(command -v python3 || command -v python)
    print_success "Python installed ($PYTHON_CMD)"
else
    MISSING_DEPS=1
fi

# Check for bun or npm for WebUI
if check_command bun; then
    WEBUI_CMD="bun"
    print_success "Bun installed (will use for WebUI)"
elif check_command npm; then
    WEBUI_CMD="npm"
    print_success "npm installed (will use for WebUI)"
else
    print_error "Neither bun nor npm found - required for WebUI"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    print_error "Missing dependencies. Please install them and try again."
    exit 1
fi

# ============================================================================
# Step 2: Start Docker Services (PostgreSQL + Redis)
# ============================================================================
print_section "Step 2/5: Starting Docker Services (PostgreSQL + Redis)"

print_step "Stopping any existing dev containers..."
docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true

print_step "Starting PostgreSQL and Redis..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d

# Wait for PostgreSQL
print_step "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker exec lightrag-dev-postgres pg_isready -U "${POSTGRES_USER:-lightrag}" -d "${POSTGRES_DATABASE:-lightrag_multitenant}" > /dev/null 2>&1; then
        print_success "PostgreSQL is ready (port ${POSTGRES_PORT:-15432})"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "PostgreSQL failed to start after 30 seconds"
        echo ""
        echo -e "  ${YELLOW}Check logs with:${NC} docker logs lightrag-dev-postgres"
        exit 1
    fi
    sleep 1
done

# Wait for Redis
print_step "Waiting for Redis to be ready..."
for i in {1..15}; do
    if docker exec lightrag-dev-redis redis-cli ping > /dev/null 2>&1; then
        print_success "Redis is ready (port ${REDIS_PORT:-16379})"
        break
    fi
    if [ $i -eq 15 ]; then
        print_error "Redis failed to start"
        exit 1
    fi
    sleep 1
done

# Extra delay to ensure databases are fully initialized
print_step "Ensuring databases are fully initialized..."
sleep 3
print_success "Database services ready"

# ============================================================================
# Step 3: Start API Server
# ============================================================================
print_section "Step 3/5: Starting LightRAG API Server"

API_PORT=${PORT:-9621}

# Check if API port is available
if ! check_port $API_PORT "API Server"; then
    print_warning "Stopping existing process on port $API_PORT..."
    if [ -f "$API_PID_FILE" ]; then
        kill $(cat "$API_PID_FILE") 2>/dev/null || true
        rm -f "$API_PID_FILE"
    fi
    # Also try to kill any process on that port
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

print_step "Starting API server on port $API_PORT..."

# Ensure AUTH credentials are set
export AUTH_USER=${AUTH_USER:-admin}
export AUTH_PASS=${AUTH_PASS:-admin123}

# Start API server in background
cd "$PROJECT_ROOT"
$PYTHON_CMD -m lightrag.api.lightrag_server \
    --host 0.0.0.0 \
    --port $API_PORT \
    > "$API_LOG" 2>&1 &
API_PID=$!
echo $API_PID > "$API_PID_FILE"

print_info "API server starting (PID: $API_PID)"

# Wait for API (with log file monitoring)
if ! wait_for_service "http://localhost:$API_PORT/health" "API Server" 60 "$API_LOG"; then
    print_error "API Server failed to start!"
    echo ""
    echo -e "  ${YELLOW}Full logs:${NC} tail -f $API_LOG"
    echo ""
    echo -e "  ${YELLOW}Common issues:${NC}"
    echo -e "    • Database connection error - wait a few seconds and try again"
    echo -e "    • Missing LLM_BINDING_API_KEY in .env"
    echo -e "    • Port 9621 already in use"
    echo ""
    exit 1
fi

# ============================================================================
# Step 4: Start WebUI
# ============================================================================
print_section "Step 4/5: Starting WebUI Development Server"

WEBUI_PORT=5173
WEBUI_DIR="$PROJECT_ROOT/lightrag_webui"

if [ ! -d "$WEBUI_DIR" ]; then
    print_error "WebUI directory not found: $WEBUI_DIR"
    exit 1
fi

# Check if WebUI port is available
if ! check_port $WEBUI_PORT "WebUI"; then
    print_warning "Stopping existing process on port $WEBUI_PORT..."
    if [ -f "$WEBUI_PID_FILE" ]; then
        kill $(cat "$WEBUI_PID_FILE") 2>/dev/null || true
        rm -f "$WEBUI_PID_FILE"
    fi
    lsof -ti:$WEBUI_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

cd "$WEBUI_DIR"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    print_step "Installing WebUI dependencies..."
    if [ "$WEBUI_CMD" = "bun" ]; then
        bun install
    else
        npm install
    fi
fi

print_step "Starting WebUI dev server on port $WEBUI_PORT..."

# Set API base URL for the frontend
export VITE_API_BASE_URL="http://localhost:$API_PORT"

# Start WebUI in background
if [ "$WEBUI_CMD" = "bun" ]; then
    bun run dev > "$WEBUI_LOG" 2>&1 &
else
    npm run dev-no-bun > "$WEBUI_LOG" 2>&1 &
fi
WEBUI_PID=$!
echo $WEBUI_PID > "$WEBUI_PID_FILE"

print_info "WebUI server starting (PID: $WEBUI_PID)"

# Wait for WebUI
wait_for_service "http://localhost:$WEBUI_PORT" "WebUI" 30

# ============================================================================
# Step 5: Display Summary
# ============================================================================
print_section "Step 5/5: Stack Ready!"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${BOLD}🎉 LightRAG Development Stack is Running!${NC}                               ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}${BOLD}📡 Service URLs:${NC}"
echo -e "  ┌─────────────────────────────────────────────────────────────────────┐"
echo -e "  │  ${BOLD}WebUI${NC}          ${BLUE}http://localhost:$WEBUI_PORT${NC}                           │"
echo -e "  │  ${BOLD}API Server${NC}     ${BLUE}http://localhost:$API_PORT${NC}                            │"
echo -e "  │  ${BOLD}API Docs${NC}       ${BLUE}http://localhost:$API_PORT/docs${NC}                       │"
echo -e "  │  ${BOLD}Health Check${NC}   ${BLUE}http://localhost:$API_PORT/health${NC}                     │"
echo -e "  └─────────────────────────────────────────────────────────────────────┘"
echo ""

echo -e "${CYAN}${BOLD}🗄️  Database Services (Docker):${NC}"
echo -e "  ┌─────────────────────────────────────────────────────────────────────┐"
echo -e "  │  ${BOLD}PostgreSQL${NC}     localhost:${POSTGRES_PORT:-15432}                                  │"
echo -e "  │                 User: ${POSTGRES_USER:-lightrag} │ DB: ${POSTGRES_DATABASE:-lightrag_multitenant}           │"
echo -e "  │  ${BOLD}Redis${NC}          localhost:${REDIS_PORT:-16379}                                   │"
echo -e "  └─────────────────────────────────────────────────────────────────────┘"
echo ""

echo -e "${CYAN}${BOLD}🔐 Login Credentials:${NC}"
echo -e "  ┌─────────────────────────────────────────────────────────────────────┐"
echo -e "  │  ${BOLD}Username${NC}       ${YELLOW}${AUTH_USER:-admin}${NC}                                           │"
echo -e "  │  ${BOLD}Password${NC}       ${YELLOW}${AUTH_PASS:-admin123}${NC}                                        │"
echo -e "  └─────────────────────────────────────────────────────────────────────┘"
echo ""

echo -e "${CYAN}${BOLD}📋 Process Info:${NC}"
echo -e "  • API Server PID:  $API_PID"
echo -e "  • WebUI PID:       $WEBUI_PID"
echo ""

echo -e "${CYAN}${BOLD}📄 Log Files:${NC}"
echo -e "  • API Log:    ${DIM}tail -f $API_LOG${NC}"
echo -e "  • WebUI Log:  ${DIM}tail -f $WEBUI_LOG${NC}"
echo ""

echo -e "${CYAN}${BOLD}🛑 To Stop:${NC}"
echo -e "  Run: ${YELLOW}./dev-stop.sh${NC}  or  ${YELLOW}make dev-stop${NC}"
echo ""

echo -e "${DIM}Configuration loaded from: $ENV_FILE${NC}"
echo ""
