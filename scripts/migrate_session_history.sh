#!/bin/bash
# Migration script for Session History integration
# This script helps migrate from standalone service/ folder to integrated session history

set -e  # Exit on error

echo "=========================================="
echo "LightRAG Session History Migration Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating .env from env.example..."
    cp env.example .env
    echo -e "${GREEN}Created .env file. Please update it with your configuration.${NC}"
    echo ""
fi

# Check if session history config exists in .env
# Session history is now always enabled - no configuration needed!
echo -e "${GREEN}Session history is always enabled by default${NC}"
echo -e "${GREEN}Uses existing POSTGRES_* settings automatically${NC}"
echo ""

# Check if old service folder exists
if [ -d "service" ]; then
    echo -e "${YELLOW}Found old service/ folder${NC}"
    echo "Options:"
    echo "  1) Backup and remove"
    echo "  2) Keep as-is"
    echo "  3) Exit"
    read -p "Choose option (1-3): " choice
    
    case $choice in
        1)
            backup_name="service.backup.$(date +%Y%m%d_%H%M%S)"
            echo "Creating backup: $backup_name"
            mv service "$backup_name"
            echo -e "${GREEN}Old service folder backed up to $backup_name${NC}"
            ;;
        2)
            echo "Keeping service/ folder as-is"
            ;;
        3)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            exit 1
            ;;
    esac
    echo ""
fi

# Check if dependencies are installed
echo "Checking Python dependencies..."
python -c "import sqlalchemy" 2>/dev/null || {
    echo -e "${YELLOW}SQLAlchemy not found. Installing...${NC}"
    pip install sqlalchemy psycopg2-binary
}
echo -e "${GREEN}Dependencies OK${NC}"
echo ""

# Test database connection (optional)
echo "Would you like to test the PostgreSQL connection? (y/n)"
read -p "Test connection: " test_conn

if [ "$test_conn" = "y" ] || [ "$test_conn" = "Y" ]; then
    # Source .env file to get variables
    source .env
    
    # Use POSTGRES_* variables
    PG_HOST=${POSTGRES_HOST:-localhost}
    PG_PORT=${POSTGRES_PORT:-5432}
    PG_USER=${POSTGRES_USER:-postgres}
    PG_PASSWORD=${POSTGRES_PASSWORD:-password}
    PG_DB=${POSTGRES_DATABASE:-lightrag}
    
    echo "Testing connection to PostgreSQL..."
    PGPASSWORD=$PG_PASSWORD psql -h $PG_HOST -p $PG_PORT -U $PG_USER -d postgres -c '\q' 2>/dev/null && {
        echo -e "${GREEN}PostgreSQL connection successful${NC}"
        
        # Check if database exists, create if not
        PGPASSWORD=$PG_PASSWORD psql -h $PG_HOST -p $PG_PORT -U $PG_USER -d postgres -lqt | cut -d \| -f 1 | grep -qw $PG_DB
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Database '$PG_DB' exists${NC}"
        else
            echo -e "${YELLOW}Database '$PG_DB' does not exist${NC}"
            read -p "Create database? (y/n): " create_db
            if [ "$create_db" = "y" ] || [ "$create_db" = "Y" ]; then
                PGPASSWORD=$PG_PASSWORD psql -h $PG_HOST -p $PG_PORT -U $PG_USER -d postgres -c "CREATE DATABASE $PG_DB;"
                echo -e "${GREEN}Database created${NC}"
            fi
        fi
    } || {
        echo -e "${RED}Failed to connect to PostgreSQL${NC}"
        echo "Please check your database configuration in .env"
    }
    echo ""
fi

# Docker-specific instructions
if [ -f "docker-compose.yml" ]; then
    echo -e "${GREEN}Docker Compose detected${NC}"
    echo "To start all services including session database:"
    echo "  docker compose up -d"
    echo ""
    echo "To view logs:"
    echo "  docker compose logs -f lightrag session-db"
    echo ""
fi

echo "=========================================="
echo "Migration Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review and update .env configuration"
echo "2. Start LightRAG server: lightrag-server"
echo "3. Test session endpoints at: http://localhost:9621/docs"
echo "4. Review migration guide: docs/SessionHistoryMigration.md"
echo ""
echo -e "${GREEN}Happy LightRAGging! 🚀${NC}"

