#!/bin/bash
# Deployment script for LightRAG stack configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from env.example..."
        if [ -f env.example ]; then
            cp env.example .env
            print_warning "Please update .env with your API keys before deployment!"
        else
            print_error "env.example file not found!"
            exit 1
        fi
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data/inputs
    mkdir -p data/rag_storage
    mkdir -p data/dev-storage
    print_success "Directories created"
}

# Deploy specific stack
deploy_stack() {
    local stack=$1
    local compose_file="docker-compose.${stack}.yml"

    if [ ! -f "$compose_file" ]; then
        print_error "Compose file $compose_file not found!"
        return 1
    fi

    print_status "Deploying $stack stack..."

    # Stop any existing containers
    docker-compose -f "$compose_file" down 2>/dev/null || true

    # Start the stack
    docker-compose -f "$compose_file" up -d

    if [ $? -eq 0 ]; then
        print_success "$stack stack deployed successfully!"

        # Show running services
        echo ""
        print_status "Running services:"
        docker-compose -f "$compose_file" ps

        # Wait a bit for services to start
        sleep 5

        # Check LightRAG health
        print_status "Checking LightRAG health..."
        for i in {1..30}; do
            if curl -s http://localhost:9621/health > /dev/null 2>&1; then
                print_success "LightRAG is healthy and ready!"
                echo ""
                echo "🌐 Web UI: http://localhost:9621/webui"
                echo "📖 API Docs: http://localhost:9621/docs"
                echo "💚 Health Check: http://localhost:9621/health"
                return 0
            fi
            echo -n "."
            sleep 2
        done

        print_warning "LightRAG health check timed out. Check logs with:"
        echo "docker-compose -f $compose_file logs lightrag"
    else
        print_error "Failed to deploy $stack stack!"
        return 1
    fi
}

# Stop stack
stop_stack() {
    local stack=$1
    local compose_file="docker-compose.${stack}.yml"

    if [ ! -f "$compose_file" ]; then
        print_error "Compose file $compose_file not found!"
        return 1
    fi

    print_status "Stopping $stack stack..."
    docker-compose -f "$compose_file" down
    print_success "$stack stack stopped"
}

# Clean up stack (stop and remove volumes)
cleanup_stack() {
    local stack=$1
    local compose_file="docker-compose.${stack}.yml"

    if [ ! -f "$compose_file" ]; then
        print_error "Compose file $compose_file not found!"
        return 1
    fi

    print_warning "This will remove all data for $stack stack. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up $stack stack..."
        docker-compose -f "$compose_file" down -v --remove-orphans
        print_success "$stack stack cleaned up"
    else
        print_status "Cleanup cancelled"
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 <command> [stack]"
    echo ""
    echo "Commands:"
    echo "  deploy <stack>    Deploy a specific stack"
    echo "  stop <stack>      Stop a specific stack"
    echo "  cleanup <stack>   Stop and remove all data for a stack"
    echo "  list              List available stacks"
    echo ""
    echo "Available stacks:"
    echo "  development       File-based storage (NetworkX + NanoVector + JSON)"
    echo "  minimal           File-based storage (NetworkX + PostgreSQL)"
    echo "  balanced          Mixed storage (NetworkX + Qdrant + Redis + PostgreSQL)"
    echo "  high-performance  Specialized storage (Neo4j + Milvus + Redis + PostgreSQL)"
    echo "  all-in-one       Cloud-native (Neo4j + Qdrant + Redis + MongoDB)"
    echo ""
    echo "Examples:"
    echo "  $0 deploy development"
    echo "  $0 stop Minimal"
    echo "  $0 cleanup high-performance"
}

# List available stacks
list_stacks() {
    print_status "Available LightRAG stack configurations:"
    echo ""
    echo "📚 development       - File-based storage, perfect for development"
    echo "💰 minimal           - PostgreSQL-based, single database"
    echo "🎯 balanced          - Mixed storage, good performance/complexity balance"
    echo "🏆 high-performance  - Specialized databases, maximum performance"
    echo "🐳 all-in-one       - Cloud-native, containerized services"
    echo ""
    echo "Use '$0 deploy <stack>' to deploy a configuration"
}

# Main script logic
main() {
    case "$1" in
        deploy)
            if [ -z "$2" ]; then
                print_error "Please specify a stack to deploy"
                show_usage
                exit 1
            fi
            check_env_file
            create_directories
            deploy_stack "$2"
            ;;
        stop)
            if [ -z "$2" ]; then
                print_error "Please specify a stack to stop"
                show_usage
                exit 1
            fi
            stop_stack "$2"
            ;;
        cleanup)
            if [ -z "$2" ]; then
                print_error "Please specify a stack to cleanup"
                show_usage
                exit 1
            fi
            cleanup_stack "$2"
            ;;
        list)
            list_stacks
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
