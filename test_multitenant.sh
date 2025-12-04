#!/bin/bash
# Comprehensive test script for LightRAG multi-tenant functionality

set -e

API_URL="http://localhost:9621"
ADMIN_USER="admin"
ADMIN_PASS="admin123"

echo "🧪 Testing LightRAG Multi-Tenant System"
echo "========================================"
echo ""

# Test 1: Check server is running
echo "✓ Test 1: Server Health Check"
if curl -s "$API_URL/docs" > /dev/null; then
    echo "  ✅ Server is responding"
else
    echo "  ❌ Server is not responding"
    exit 1
fi
echo ""

# Test 2: Login as Admin
echo "✓ Test 2: Login as Admin"
LOGIN_RESPONSE=$(curl -s -X POST "$API_URL/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=$ADMIN_USER&password=$ADMIN_PASS")

TOKEN=$(echo "$LOGIN_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null || echo "")

if [ ! -z "$TOKEN" ]; then
    echo "  ✅ Login successful, got token"
else
    echo "  ❌ Login failed"
    echo "$LOGIN_RESPONSE"
    exit 1
fi
echo ""

# Test 3: Verify Public Tenant List (Now Allowed with Auth)
echo "✓ Test 3: Verify Public Tenant List (Auth Required)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $TOKEN" "$API_URL/api/v1/tenants")
if [ "$HTTP_CODE" == "200" ]; then
    echo "  ✅ Public tenant list accessible with auth (HTTP $HTTP_CODE)"
else
    echo "  ❌ Public tenant list failed (HTTP $HTTP_CODE)"
fi
echo ""

# Test 4: List All Tenants (Admin)
echo "✓ Test 4: List All Tenants (Admin)"
TENANTS_RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" "$API_URL/api/v1/admin/tenants?page=1&page_size=10")
TENANT_COUNT=$(echo "$TENANTS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total', 0))")
echo "  Found $TENANT_COUNT tenants"
echo "  Response:"
echo "$TENANTS_RESPONSE" | python3 -m json.tool | head -15
echo ""

# Test 5: Get First Tenant Details (using X-Tenant-ID)
echo "✓ Test 5: Get First Tenant Details"
FIRST_TENANT=$(echo "$TENANTS_RESPONSE" | python3 -c "import sys, json; items = json.load(sys.stdin).get('items', []); print(items[0]['tenant_id'] if items else '')")

if [ ! -z "$FIRST_TENANT" ]; then
    echo "  Found tenant: $FIRST_TENANT"
    # Use X-Tenant-ID header to switch context
    TENANT_DETAILS=$(curl -s -H "Authorization: Bearer $TOKEN" -H "X-Tenant-ID: $FIRST_TENANT" "$API_URL/api/v1/tenants/me")
    echo "  Tenant details:"
    echo "$TENANT_DETAILS" | python3 -m json.tool | head -10
else
    echo "  No tenants found, skipping tenant details"
fi
echo ""

# Test 6: List Knowledge Bases for Tenant
echo "✓ Test 6: List Knowledge Bases for Tenant"
if [ ! -z "$FIRST_TENANT" ]; then
    KBS_RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" -H "X-Tenant-ID: $FIRST_TENANT" "$API_URL/api/v1/knowledge-bases?page=1&page_size=10")
    KB_COUNT=$(echo "$KBS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total', 0))")
    echo "  Found $KB_COUNT knowledge bases in tenant $FIRST_TENANT"
    echo "  Response:"
    echo "$KBS_RESPONSE" | python3 -m json.tool | head -15
else
    echo "  Skipping KB list (no tenants found)"
fi
echo ""

# Test 7: Verify Demo Tenants (Admin Search)
echo "✓ Test 7: Verify Demo Tenants (Finance, Marketing Team, Engineering Team)"
DEMO_TENANTS=("Finance" "Marketing Team" "Engineering Team")
for tenant_name in "${DEMO_TENANTS[@]}"; do
    SEARCH=$(curl -s -G -H "Authorization: Bearer $TOKEN" "$API_URL/api/v1/admin/tenants" --data-urlencode "search=$tenant_name" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total', 0))" 2>/dev/null || echo "0")
    if [ "$SEARCH" -gt 0 ]; then
        echo "  ✅ Found '$tenant_name' tenant"
    else
        echo "  ❌ Missing '$tenant_name' tenant"
    fi
done
echo ""

# Test 8: Verify Default Tenant Resolution
echo "✓ Test 8: Verify Default Tenant Resolution"
DEFAULT_TENANT_RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" -H "X-Tenant-ID: default" "$API_URL/api/v1/tenants/me")
RESOLVED_ID=$(echo "$DEFAULT_TENANT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('tenant_id', ''))" 2>/dev/null || echo "")

if [ ! -z "$RESOLVED_ID" ]; then
    echo "  ✅ Resolved 'default' to $RESOLVED_ID"
else
    echo "  ❌ Failed to resolve 'default' tenant"
    echo "$DEFAULT_TENANT_RESPONSE"
fi
echo ""

# Test 9: Verify Legacy KB List Route
echo "✓ Test 9: Verify Legacy KB List Route (/tenants/default/knowledge-bases)"
LEGACY_KB_RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" "$API_URL/api/v1/tenants/default/knowledge-bases?page=1&page_size=5")
LEGACY_KB_COUNT=$(echo "$LEGACY_KB_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total', 0))" 2>/dev/null || echo "-1")

if [ "$LEGACY_KB_COUNT" -ge 0 ]; then
    echo "  ✅ Legacy KB list route working (Found $LEGACY_KB_COUNT KBs)"
else
    echo "  ❌ Legacy KB list route failed"
    echo "$LEGACY_KB_RESPONSE"
fi
echo ""

echo "========================================"
echo "✅ All tests completed!"
