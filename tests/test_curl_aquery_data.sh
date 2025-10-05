#!/bin/bash

# LightRAG aquery_data endpoint test script
# Use curl command to test the new /query/data endpoint and validate the new data format

echo "🚀 LightRAG aquery_data Endpoint Test (New Data Format Validation)"
echo "=================================================="

# Base URL (adjust according to actual deployment)
BASE_URL="http://localhost:9621"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test result statistics
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to validate success response format
validate_success_response() {
    local response="$1"
    local test_name="$2"
    local expected_mode="$3"
    
    echo -e "${BLUE}Validating $test_name response format...${NC}"
    
    # Check if valid JSON
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        echo -e "${RED}❌ Response is not valid JSON format${NC}"
        return 1
    fi
    
    # Validate required fields
    local status=$(echo "$response" | jq -r '.status // "missing"')
    local message=$(echo "$response" | jq -r '.message // "missing"')
    local data_exists=$(echo "$response" | jq 'has("data")')
    local metadata_exists=$(echo "$response" | jq 'has("metadata")')
    
    echo "  Status: $status"
    echo "  Message: $message"
    
    # Validate data structure
    if [[ "$data_exists" == "true" ]]; then
        local entities_count=$(echo "$response" | jq '.data.entities | length // 0')
        local relationships_count=$(echo "$response" | jq '.data.relationships | length // 0')
        local chunks_count=$(echo "$response" | jq '.data.chunks | length // 0')
        local references_count=$(echo "$response" | jq '.data.references | length // 0')
        
        echo "  Data.entities: $entities_count"
        echo "  Data.relationships: $relationships_count"
        echo "  Data.chunks: $chunks_count"
        echo "  Data.references: $references_count"
    else
        echo -e "${RED}  ❌ Missing 'data' field${NC}"
        return 1
    fi
    
    # Validate metadata
    if [[ "$metadata_exists" == "true" ]]; then
        local query_mode=$(echo "$response" | jq -r '.metadata.query_mode // "missing"')
        local keywords_exists=$(echo "$response" | jq 'has("metadata") and (.metadata | has("keywords"))')
        local processing_info_exists=$(echo "$response" | jq 'has("metadata") and (.metadata | has("processing_info"))')
        
        echo "  Metadata.query_mode: $query_mode"
        echo "  Metadata.keywords: $keywords_exists"
        echo "  Metadata.processing_info: $processing_info_exists"
        
        # Validate if query mode matches
        if [[ "$expected_mode" != "" && "$query_mode" != "$expected_mode" ]]; then
            echo -e "${YELLOW}  ⚠️  Query mode mismatch: expected '$expected_mode', actual '$query_mode'${NC}"
        fi
    else
        echo -e "${RED}  ❌ Missing 'metadata' field${NC}"
        return 1
    fi
    
    # Validate status
    if [[ "$status" == "success" ]]; then
        echo -e "${GREEN}  ✅ Response format validation passed${NC}"
        return 0
    else
        echo -e "${RED}  ❌ Status is not 'success': $status${NC}"
        return 1
    fi
}

# Function to validate error response format
validate_error_response() {
    local response="$1"
    local test_name="$2"
    
    echo -e "${BLUE}Validating $test_name response format...${NC}"
    
    # Check if valid JSON
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        echo -e "${RED}❌ Response is not valid JSON format${NC}"
        return 1
    fi
    
    # Validate required fields
    local status=$(echo "$response" | jq -r '.status // "missing"')
    local message=$(echo "$response" | jq -r '.message // "missing"')
    local data_exists=$(echo "$response" | jq 'has("data")')
    local metadata_exists=$(echo "$response" | jq 'has("metadata")')
    
    echo "  Status: $status"
    echo "  Message: $message"
    
    # Validate basic structure exists
    if [[ "$data_exists" != "true" ]]; then
        echo -e "${RED}  ❌ Missing 'data' field${NC}"
        return 1
    fi
    
    if [[ "$metadata_exists" != "true" ]]; then
        echo -e "${RED}  ❌ Missing 'metadata' field${NC}"
        return 1
    fi
    
    echo "  Data: {}"
    echo "  Metadata: {}"
    
    # Validate status should be failure
    if [[ "$status" == "failure" ]]; then
        echo -e "${GREEN}  ✅ Error response format validation passed${NC}"
        return 0
    else
        echo -e "${RED}  ❌ Status is not 'failure': $status${NC}"
        return 1
    fi
}

# Function to run success test
run_success_test() {
    local test_name="$1"
    local query_data="$2"
    local expected_mode="$3"
    local print_json="${4:-false}"  # Optional parameter: whether to print JSON response (default: false)
    
    echo ""
    echo "=================================="
    echo -e "${BLUE}$test_name${NC}"
    echo "=================================="
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Send request
    echo "Sending request..."
    local response=$(curl -s -X POST "${BASE_URL}/query/data" \
      -H "Content-Type: application/json" \
      -H "X-API-Key: your-secure-api-key-here-123" \
      -d "$query_data")
    
    # Check if curl succeeded
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}❌ Request failed - cannot connect to server${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    
    # Print JSON response if requested
    if [[ "$print_json" == "true" ]]; then
        echo ""
        echo "Response JSON:"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
        echo ""
    fi
    
    # Validate response
    if validate_success_response "$response" "$test_name" "$expected_mode"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✅ $test_name test passed${NC}"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}❌ $test_name test failed${NC}"
        echo "Raw response:"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    fi
}

# Function to run error test
run_error_test() {
    local test_name="$1"
    local query_data="$2"
    
    echo ""
    echo "=================================="
    echo -e "${BLUE}$test_name${NC}"
    echo "=================================="
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Send request
    echo "Sending request..."
    local response=$(curl -s -X POST "${BASE_URL}/query/data" \
      -H "Content-Type: application/json" \
      -H "X-API-Key: your-secure-api-key-here-123" \
      -d "$query_data")
    
    # Check if curl succeeded
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}❌ Request failed - cannot connect to server${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    
    # Validate response
    if validate_error_response "$response" "$test_name"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✅ $test_name test passed${NC}"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}❌ $test_name test failed${NC}"
        echo "Raw response:"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    fi
}

# Start tests
echo "Starting tests for new /query/data endpoint data format..."
echo ""

# Test 1: Basic query test (mix mode)
run_success_test "1. Basic Query Test (mix mode)" '{
    "query": "What is GraphRAG",
    "mode": "mix",
    "top_k": 5
}' "mix" "true" # Output full JSON

# Test 2: Detailed parameter query test (hybrid mode)
run_success_test "2. Detailed Parameter Query Test (hybrid mode)" '{
    "query": "What is GraphRAG",
    "mode": "hybrid",
    "top_k": 5,
    "chunk_top_k": 8,
    "max_entity_tokens": 4000,
    "max_relation_tokens": 4000,
    "max_total_tokens": 16000,
    "enable_rerank": true,
    "response_type": "Multiple Paragraphs"
}' "hybrid"

# Output test result statistics
echo ""
echo "=================================================="
echo -e "${BLUE}Test Result Statistics${NC}"
echo "=================================================="
echo -e "Total tests: ${BLUE}$TOTAL_TESTS${NC}"
echo -e "Passed tests: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed tests: ${RED}$FAILED_TESTS${NC}"

if [[ $FAILED_TESTS -eq 0 ]]; then
    echo -e "${GREEN}🎉 All tests passed! New data format adaptation successful!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  $FAILED_TESTS test(s) failed, please check the issues${NC}"
    exit 1
fi

echo ""
echo "💡 Usage Instructions:"
echo "1. Ensure LightRAG API service is running (python -m lightrag.api.lightrag_server)"
echo "2. Adjust BASE_URL as needed"
echo "3. If authentication is required, add -H \"Authorization: Bearer your-token\""
echo "4. Install jq for better JSON formatting output: brew install jq (macOS) or apt install jq (Ubuntu)"
echo "5. Script will automatically validate new data format structure: status, message, data, metadata"
