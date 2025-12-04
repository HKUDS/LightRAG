#!/usr/bin/env python3
"""
Initialize demo tenants for LightRAG multi-tenant system using HTTP API.

This script creates sample tenants and knowledge bases for demonstration purposes.
It calls the REST API after the server starts to populate the system with test data.

Usage:
    python3 scripts/init_demo_tenants.py
"""

import requests
import logging
import time
import os
import sys
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:9621/api/v1")
LOGIN_URL = os.getenv("LOGIN_URL", "http://localhost:9621/login")
AUTH_STATUS_URL = os.getenv("AUTH_STATUS_URL", "http://localhost:9621/auth-status")
AUTH_USER = os.getenv("AUTH_USER", "admin")
AUTH_PASS = os.getenv("AUTH_PASS", "admin123")
MAX_RETRIES = 10
RETRY_DELAY = 2

# Global auth token
auth_token = None

# Sample tenants configuration
DEMO_TENANTS = [
    {
        "tenant_name": "Engineering Team",
        "description": "Knowledge base for the engineering department",
        "knowledge_bases": [
            {
                "kb_name": "Architecture Docs",
                "description": "System architecture and design documentation"
            },
            {
                "kb_name": "API Reference",
                "description": "API endpoints and integration guides"
            }
        ]
    },
    {
        "tenant_name": "Product Team",
        "description": "Product requirements and roadmap",
        "knowledge_bases": [
            {
                "kb_name": "Product Requirements",
                "description": "Feature requirements and specifications"
            },
            {
                "kb_name": "User Stories",
                "description": "User stories and acceptance criteria"
            }
        ]
    },
    {
        "tenant_name": "Marketing Team",
        "description": "Marketing materials and campaign information",
        "knowledge_bases": [
            {
                "kb_name": "Campaign Materials",
                "description": "Marketing campaign documents and assets"
            },
            {
                "kb_name": "Brand Guidelines",
                "description": "Brand standards and guidelines"
            }
        ]
    },
    {
        "tenant_name": "Finance",
        "description": "Financial reports and budget information",
        "knowledge_bases": [
            {
                "kb_name": "Budget Reports",
                "description": "Monthly and annual budget reports"
            },
            {
                "kb_name": "Financial Analysis",
                "description": "Financial analysis and forecasting"
            }
        ]
    }
]


def wait_for_api(max_retries=MAX_RETRIES):
    """Wait for API to be available."""
    logger.info(f"Waiting for API to be available at {API_BASE_URL}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            # Try to connect to auth-status endpoint (no auth required)
            response = requests.get(f"{AUTH_STATUS_URL}", timeout=5)
            if response.status_code in [200, 401, 403]:
                logger.info("✓ API is available!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_retries:
            logger.info(f"  Attempt {attempt}/{max_retries}: API not available yet, retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
    
    logger.warning(f"⚠ API did not become available after {max_retries} retries")
    return False


def get_auth_token(session):
    """Get authentication token using credentials."""
    global auth_token
    
    if auth_token:
        return auth_token
    
    logger.info("Attempting to authenticate...")
    
    try:
        # Check if auth is configured
        response = session.get(AUTH_STATUS_URL, timeout=10)
        if response.status_code == 200:
            auth_status = response.json()
            
            if not auth_status.get("auth_configured"):
                # Auth not configured, use guest token
                logger.info("✓ Authentication not configured, using guest access")
                auth_token = auth_status.get("access_token")
                return auth_token
        
        # Try to login with provided credentials
        login_data = {
            "username": AUTH_USER,
            "password": AUTH_PASS
        }
        
        response = session.post(LOGIN_URL, data=login_data, timeout=10)
        
        if response.status_code == 200:
            login_response = response.json()
            auth_token = login_response.get("access_token")
            
            if auth_token:
                logger.info(f"✓ Successfully authenticated as {AUTH_USER}")
                return auth_token
            else:
                logger.error("No access token in login response")
                return None
        else:
            logger.error(f"Login failed with status {response.status_code}: {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during authentication: {str(e)}")
        return None


def get_headers():
    """Get headers with authentication if available."""
    headers = {
        "Content-Type": "application/json"
    }
    
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    return headers


def create_tenants_and_kbs():
    """Create demo tenants and their knowledge bases via API."""
    
    if not wait_for_api():
        logger.error("Failed to connect to API")
        return False
    
    logger.info("\nInitializing demo tenants...")
    logger.info("━" * 50)
    
    session = requests.Session()
    
    # Get authentication token
    token = get_auth_token(session)
    if not token:
        logger.warning("⚠ Could not obtain authentication token, attempting public endpoints only")
    
    created_tenants = []
    
    for tenant_config in DEMO_TENANTS:
        try:
            # Create tenant
            tenant_data = {
                "name": tenant_config["tenant_name"],
                "description": tenant_config["description"]
            }
            
            response = session.post(
                f"{API_BASE_URL}/tenants",
                json=tenant_data,
                headers=get_headers(),
                timeout=10
            )
            
            if response.status_code == 201:
                tenant = response.json()
                tenant_id = tenant.get("tenant_id")
                logger.info(f"✓ Created tenant: {tenant_id} - {tenant_config['tenant_name']}")
                created_tenants.append((tenant_id, tenant_config))
            else:
                logger.warning(f"Failed to create tenant {tenant_config['tenant_name']}: {response.status_code}")
                if response.status_code == 401:
                    logger.info("  Hint: Authentication required. Set AUTH_USER and AUTH_PASS environment variables.")
                continue
            
            # Create knowledge bases for this tenant
            for kb_config in tenant_config["knowledge_bases"]:
                try:
                    kb_data = {
                        "name": kb_config["kb_name"],
                        "description": kb_config["description"]
                    }
                    
                    # Include X-Tenant-ID header for KB creation
                    headers = get_headers()
                    headers["X-Tenant-ID"] = tenant_id
                    
                    response = session.post(
                        f"{API_BASE_URL}/knowledge-bases",
                        json=kb_data,
                        headers=headers,
                        timeout=10
                    )
                    
                    if response.status_code == 201:
                        kb = response.json()
                        kb_id = kb.get("kb_id")
                        logger.info(f"  ├─ Created KB: {kb_id} - {kb_config['kb_name']}")
                    else:
                        logger.warning(f"  ├─ Failed to create KB {kb_config['kb_name']}: {response.status_code}")
                
                except requests.exceptions.RequestException as e:
                    logger.error(f"  ├─ Error creating KB {kb_config['kb_name']}: {str(e)}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating tenant {tenant_config['tenant_name']}: {str(e)}")
    
    # List all created tenants
    logger.info("\n" + "=" * 50)
    logger.info("Summary of Tenants and Knowledge Bases:")
    logger.info("=" * 50)
    
    try:
        response = session.get(
            f"{API_BASE_URL}/tenants?page=1&page_size=100",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            tenants = data.get("items", [])
            
            if not tenants:
                logger.info("No tenants created yet")
            else:
                for tenant in tenants:
                    logger.info(f"\n🏢 {tenant.get('tenant_id')} - {tenant.get('name')}")
                    logger.info(f"   Description: {tenant.get('description')}")
                    
                    # List KBs for this tenant
                    try:
                        headers = get_headers()
                        headers["X-Tenant-ID"] = tenant.get('tenant_id')
                        
                        kb_response = session.get(
                            f"{API_BASE_URL}/knowledge-bases?page=1&page_size=100",
                            headers=headers,
                            timeout=10
                        )
                        
                        if kb_response.status_code == 200:
                            kb_data = kb_response.json()
                            kbs = kb_data.get("items", [])
                            if kbs:
                                for kb in kbs:
                                    logger.info(f"   📚 {kb.get('kb_id')} - {kb.get('name')}")
                                    logger.info(f"      {kb.get('description')}")
                            else:
                                logger.info("   📚 No knowledge bases")
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"   ⚠ Could not fetch KBs: {str(e)}")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching tenants: {str(e)}")
    
    logger.info("\n" + "=" * 50)
    logger.info("✓ Initialization complete")
    logger.info("=" * 50)
    
    return True


if __name__ == "__main__":
    try:
        success = create_tenants_and_kbs()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
