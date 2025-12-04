#!/usr/bin/env python3
"""
Multi-Tenant API Test Script

This script tests the multi-tenant REST API endpoints without needing
the Web UI or Docker for the API server.

Requirements:
- PostgreSQL running on localhost:5433
- Redis running on localhost:6380  
- API server to be started separately

Usage:
    python test_api_isolation.py
"""

import httpx
import asyncio
import json
from typing import Optional
from dataclasses import dataclass
import sys

# Configuration
API_BASE_URL = "http://localhost:9621"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # Default admin password

@dataclass
class TestResult:
    test_name: str
    passed: bool
    message: str
    details: Optional[dict] = None

class MultiTenantAPITest:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.results: list[TestResult] = []
        self.tenant_a_id: Optional[str] = None
        self.tenant_b_id: Optional[str] = None
        self.kb_a_id: Optional[str] = None
        self.kb_b_id: Optional[str] = None
        
    def get_auth_header(self, username: str = ADMIN_USERNAME) -> dict:
        """Get authorization header."""
        return {
            "Authorization": f"Basic {username}:password",
            "Content-Type": "application/json"
        }
    
    def get_tenant_headers(self, tenant_id: str, kb_id: str) -> dict:
        """Get headers with tenant context."""
        headers = self.get_auth_header()
        headers["X-Tenant-ID"] = tenant_id
        headers["X-KB-ID"] = kb_id
        return headers
        
    async def check_api_health(self) -> bool:
        """Check if API is running."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            print(f"API not reachable: {e}")
            return False
            
    async def test_list_tenants(self) -> TestResult:
        """Test listing tenants (public endpoint)."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/tenants",
                    timeout=10.0
                )
                if response.status_code == 200:
                    tenants = response.json()
                    return TestResult(
                        test_name="List Tenants",
                        passed=True,
                        message=f"Found {len(tenants)} tenants",
                        details={"tenants": tenants}
                    )
                else:
                    return TestResult(
                        test_name="List Tenants",
                        passed=False,
                        message=f"Status {response.status_code}: {response.text}"
                    )
            except Exception as e:
                return TestResult(
                    test_name="List Tenants",
                    passed=False,
                    message=str(e)
                )
                
    async def test_create_tenant(self, name: str, description: str) -> TestResult:
        """Test creating a tenant."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/tenants",
                    headers=self.get_auth_header(),
                    json={
                        "name": name,
                        "description": description,
                        "config": {}
                    },
                    timeout=10.0
                )
                if response.status_code in [200, 201]:
                    tenant = response.json()
                    return TestResult(
                        test_name=f"Create Tenant: {name}",
                        passed=True,
                        message=f"Created tenant: {tenant.get('tenant_id')}",
                        details=tenant
                    )
                else:
                    return TestResult(
                        test_name=f"Create Tenant: {name}",
                        passed=False,
                        message=f"Status {response.status_code}: {response.text}"
                    )
            except Exception as e:
                return TestResult(
                    test_name=f"Create Tenant: {name}",
                    passed=False,
                    message=str(e)
                )

    async def test_create_kb(self, tenant_id: str, kb_name: str) -> TestResult:
        """Test creating a knowledge base for a tenant."""
        async with httpx.AsyncClient() as client:
            try:
                headers = self.get_auth_header()
                headers["X-Tenant-ID"] = tenant_id
                
                response = await client.post(
                    f"{self.base_url}/knowledge-bases",
                    headers=headers,
                    json={
                        "name": kb_name,
                        "description": f"Test KB for {tenant_id}"
                    },
                    timeout=10.0
                )
                if response.status_code in [200, 201]:
                    kb = response.json()
                    return TestResult(
                        test_name=f"Create KB: {kb_name}",
                        passed=True,
                        message=f"Created KB: {kb.get('kb_id')}",
                        details=kb
                    )
                else:
                    return TestResult(
                        test_name=f"Create KB: {kb_name}",
                        passed=False,
                        message=f"Status {response.status_code}: {response.text}"
                    )
            except Exception as e:
                return TestResult(
                    test_name=f"Create KB: {kb_name}",
                    passed=False,
                    message=str(e)
                )

    async def test_document_isolation(self) -> TestResult:
        """Test that documents are isolated between tenants."""
        if not all([self.tenant_a_id, self.tenant_b_id, self.kb_a_id, self.kb_b_id]):
            return TestResult(
                test_name="Document Isolation",
                passed=False,
                message="Tenants/KBs not set up yet"
            )
            
        async with httpx.AsyncClient() as client:
            try:
                # Upload document to Tenant A
                headers_a = self.get_tenant_headers(self.tenant_a_id, self.kb_a_id)
                
                response_a = await client.post(
                    f"{self.base_url}/documents/text",
                    headers=headers_a,
                    json={
                        "text": "This is a secret document for Tenant A only.",
                        "description": "Tenant A Secret Doc"
                    },
                    timeout=30.0
                )
                
                if response_a.status_code not in [200, 201]:
                    return TestResult(
                        test_name="Document Isolation",
                        passed=False,
                        message=f"Failed to upload doc to Tenant A: {response_a.text}"
                    )
                
                # List documents from Tenant A - should see the document
                list_a = await client.get(
                    f"{self.base_url}/documents",
                    headers=headers_a,
                    timeout=10.0
                )
                docs_a = list_a.json() if list_a.status_code == 200 else []
                
                # List documents from Tenant B - should NOT see the document
                headers_b = self.get_tenant_headers(self.tenant_b_id, self.kb_b_id)
                list_b = await client.get(
                    f"{self.base_url}/documents",
                    headers=headers_b,
                    timeout=10.0
                )
                docs_b = list_b.json() if list_b.status_code == 200 else []
                
                # Check isolation
                tenant_a_has_doc = isinstance(docs_a, dict) and docs_a.get("total_count", 0) > 0
                tenant_b_has_doc = isinstance(docs_b, dict) and docs_b.get("total_count", 0) > 0
                
                if tenant_a_has_doc and not tenant_b_has_doc:
                    return TestResult(
                        test_name="Document Isolation",
                        passed=True,
                        message="✅ Tenant A can see its doc, Tenant B cannot",
                        details={
                            "tenant_a_docs": docs_a.get("total_count", 0) if isinstance(docs_a, dict) else 0,
                            "tenant_b_docs": docs_b.get("total_count", 0) if isinstance(docs_b, dict) else 0
                        }
                    )
                else:
                    return TestResult(
                        test_name="Document Isolation",
                        passed=False,
                        message=f"Isolation failed: A has {tenant_a_has_doc}, B has {tenant_b_has_doc}",
                        details={
                            "docs_a": docs_a,
                            "docs_b": docs_b
                        }
                    )
                    
            except Exception as e:
                return TestResult(
                    test_name="Document Isolation",
                    passed=False,
                    message=str(e)
                )

    async def test_missing_tenant_header(self) -> TestResult:
        """Test behavior when tenant header is missing."""
        async with httpx.AsyncClient() as client:
            try:
                # Try to list documents without tenant headers
                response = await client.get(
                    f"{self.base_url}/documents",
                    headers=self.get_auth_header(),
                    timeout=10.0
                )
                
                # Should either fail with 400/401 or return empty/global data
                if response.status_code in [400, 401, 403]:
                    return TestResult(
                        test_name="Missing Tenant Header",
                        passed=True,
                        message=f"Correctly rejected: {response.status_code}"
                    )
                elif response.status_code == 200:
                    return TestResult(
                        test_name="Missing Tenant Header",
                        passed=False,
                        message="⚠️ Request succeeded without tenant header - potential security issue",
                        details={"response": response.json()}
                    )
                else:
                    return TestResult(
                        test_name="Missing Tenant Header",
                        passed=False,
                        message=f"Unexpected status: {response.status_code}"
                    )
            except Exception as e:
                return TestResult(
                    test_name="Missing Tenant Header",
                    passed=False,
                    message=str(e)
                )

    async def run_all_tests(self):
        """Run all tests."""
        print("=" * 60)
        print("Multi-Tenant API Isolation Tests")
        print("=" * 60)
        
        # Check API health
        print("\n🔍 Checking API health...")
        if not await self.check_api_health():
            print("❌ API is not running. Please start the API server first.")
            print(f"   Expected at: {self.base_url}")
            return
        print("✅ API is healthy")
        
        # Run tests
        print("\n📝 Running tests...\n")
        
        # Test 1: List tenants
        result = await self.test_list_tenants()
        self.results.append(result)
        self.print_result(result)
        
        # Test 2: Create tenant A
        result = await self.test_create_tenant("Test Tenant A", "First test tenant")
        self.results.append(result)
        self.print_result(result)
        if result.passed and result.details:
            self.tenant_a_id = result.details.get("tenant_id")
        
        # Test 3: Create tenant B
        result = await self.test_create_tenant("Test Tenant B", "Second test tenant")
        self.results.append(result)
        self.print_result(result)
        if result.passed and result.details:
            self.tenant_b_id = result.details.get("tenant_id")
            
        # Test 4: Create KB for tenant A
        if self.tenant_a_id:
            result = await self.test_create_kb(self.tenant_a_id, "Tenant A KB")
            self.results.append(result)
            self.print_result(result)
            if result.passed and result.details:
                self.kb_a_id = result.details.get("kb_id")
        
        # Test 5: Create KB for tenant B
        if self.tenant_b_id:
            result = await self.test_create_kb(self.tenant_b_id, "Tenant B KB")
            self.results.append(result)
            self.print_result(result)
            if result.passed and result.details:
                self.kb_b_id = result.details.get("kb_id")
                
        # Test 6: Document isolation
        result = await self.test_document_isolation()
        self.results.append(result)
        self.print_result(result)
        
        # Test 7: Missing tenant header
        result = await self.test_missing_tenant_header()
        self.results.append(result)
        self.print_result(result)
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total:  {len(self.results)}")
        
    def print_result(self, result: TestResult):
        """Print a single test result."""
        status = "✅" if result.passed else "❌"
        print(f"{status} {result.test_name}")
        print(f"   {result.message}")
        if result.details and not result.passed:
            print(f"   Details: {json.dumps(result.details, indent=2)[:200]}")


async def main():
    tester = MultiTenantAPITest()
    await tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if any(not r.passed for r in tester.results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
