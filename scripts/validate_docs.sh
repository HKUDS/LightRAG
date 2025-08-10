#!/bin/bash
# Documentation Link Validation Script
# Validates internal markdown links in the LightRAG documentation

echo "🔍 LightRAG Documentation Link Validation"
echo "========================================"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0
CHECKED=0

# Function to check if a file exists
check_file() {
    local file_path="$1"
    local referencing_file="$2"
    local line_num="$3"

    if [ -f "$file_path" ]; then
        echo -e "${GREEN}✅${NC} $file_path"
        return 0
    else
        echo -e "${RED}❌${NC} Missing: $file_path (referenced in $referencing_file:$line_num)"
        ((ERRORS++))
        return 1
    fi
}

# Function to check markdown files for internal links
check_markdown_file() {
    local file="$1"
    echo ""
    echo "📄 Checking: $file"
    echo "----------------------------------------"

    # Extract markdown links [text](path) - only relative paths, not URLs
    grep -n "\[.*\](.*\.md)" "$file" | grep -v "http" | while IFS=: read -r line_num link_line; do
        # Extract the path from [text](path)
        path=$(echo "$link_line" | sed -E 's/.*\[.*\]\(([^)]+)\).*/\1/' | sed 's/#.*$//')

        # Skip if path is empty or is a URL
        if [[ -z "$path" || "$path" =~ ^https?:// ]]; then
            continue
        fi

        # Resolve relative path
        dir=$(dirname "$file")
        if [[ "$path" =~ ^/ ]]; then
            # Absolute path from repo root
            resolved_path="${path#/}"
        elif [[ "$path" =~ ^\.\. ]]; then
            # Relative path with ../
            resolved_path="$dir/$path"
        else
            # Relative path in same dir
            resolved_path="$dir/$path"
        fi

        # Normalize path (resolve .. and .)
        resolved_path=$(realpath -m "$resolved_path" 2>/dev/null || echo "$resolved_path")

        ((CHECKED++))
        check_file "$resolved_path" "$file" "$line_num"
    done
}

# Change to repo root
cd "$(dirname "$0")/.." || exit 1

echo "Starting documentation validation..."
echo ""

# Key files to check
KEY_FILES=(
    "docs/README.md"
    "docs/DOCUMENTATION_INDEX.md"
    "docs/production/PRODUCTION_DEPLOYMENT_COMPLETE.md"
    "docs/architecture/SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md"
    "docs/architecture/Algorithm.md"
    "docs/integration_guides/README.md"
    "docs/integration_guides/MCP_IMPLEMENTATION_SUMMARY.md"
    "docs/security/SECURITY_HARDENING.md"
)

# Check each key file
for file in "${KEY_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_markdown_file "$file"
    else
        echo -e "${RED}❌ Missing key file: $file${NC}"
        ((ERRORS++))
    fi
done

echo ""
echo "========================================"
echo "📊 VALIDATION SUMMARY"
echo "========================================"
echo "Files checked: ${#KEY_FILES[@]}"
echo "Links validated: $CHECKED"
echo -e "Errors: ${RED}$ERRORS${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All documentation links are valid!${NC}"
    exit 0
else
    echo -e "${RED}❌ Found $ERRORS broken links. Please fix them.${NC}"
    exit 1
fi
