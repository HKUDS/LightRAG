"""
Service discovery for Docling service.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from urllib.parse import urljoin
import httpx
from lightrag.utils import get_env_value

logger = logging.getLogger(__name__)


class ServiceDiscovery:
    """Service discovery for Docling service."""
    
    def __init__(self):
        self.service_url: Optional[str] = None
        self.service_available: Optional[bool] = None
        self.last_health_check: float = 0.0
        self.health_check_interval: int = int(get_env_value("DOCLING_HEALTH_CHECK_INTERVAL", 30))
        self.service_timeout: int = int(get_env_value("DOCLING_SERVICE_TIMEOUT", 10))
        
    def get_service_url(self) -> Optional[str]:
        """Get Docling service URL from environment."""
        if self.service_url is None:
            # Try multiple environment variable names for flexibility
            self.service_url = (
                get_env_value("DOCLING_SERVICE_URL", None) or
                get_env_value("LIGHTRAG_DOCLING_SERVICE_URL", None) or 
                get_env_value("DOCLING_HOST", None)
            )
            
            # If we got a host without protocol, add it
            if self.service_url and not self.service_url.startswith(('http://', 'https://')):
                port = get_env_value("DOCLING_PORT", "8080")
                self.service_url = f"http://{self.service_url}:{port}"
        
        return self.service_url
    
    async def is_service_available(self, force_check: bool = False) -> bool:
        """Check if Docling service is available."""
        import time
        
        # Use cached result if recent and not forced
        if (not force_check and 
            self.service_available is not None and 
            time.time() - self.last_health_check < self.health_check_interval):
            return self.service_available
        
        service_url = self.get_service_url()
        if not service_url:
            logger.debug("Docling service URL not configured")
            self.service_available = False
            return False
        
        try:
            health_url = urljoin(service_url, "/health")
            async with httpx.AsyncClient(timeout=self.service_timeout) as client:
                response = await client.get(health_url)
                
                if response.status_code == 200:
                    health_data = response.json()
                    service_healthy = health_data.get("status") in ["healthy", "degraded"]
                    
                    if service_healthy:
                        logger.debug(f"Docling service is available at {service_url} "
                                   f"(status: {health_data.get('status')})")
                    else:
                        logger.warning(f"Docling service reports unhealthy status at {service_url} "
                                     f"(status: {health_data.get('status')})")
                    
                    self.service_available = service_healthy
                else:
                    logger.warning(f"Docling service health check failed at {health_url} "
                                 f"(status code: {response.status_code})")
                    self.service_available = False
                    
        except httpx.TimeoutException:
            logger.warning(f"Docling service health check timed out: {service_url}")
            self.service_available = False
        except httpx.ConnectError:
            logger.debug(f"Docling service connection failed: {service_url}")
            self.service_available = False
        except Exception as e:
            logger.warning(f"Docling service health check error for {service_url}: {e}")
            self.service_available = False
        
        self.last_health_check = time.time()
        return self.service_available or False
    
    async def get_service_config(self) -> Optional[Dict[str, Any]]:
        """Get service configuration."""
        service_url = self.get_service_url()
        if not service_url or not await self.is_service_available():
            return None
        
        try:
            config_url = urljoin(service_url, "/config")
            async with httpx.AsyncClient(timeout=self.service_timeout) as client:
                response = await client.get(config_url)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Failed to get service config "
                                 f"(status code: {response.status_code})")
                    return None
                    
        except Exception as e:
            logger.warning(f"Error getting service config: {e}")
            return None
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information."""
        info = {
            "configured": self.get_service_url() is not None,
            "url": self.get_service_url(),
            "available": False,
            "config": None,
            "error": None
        }
        
        try:
            info["available"] = await self.is_service_available(force_check=True)
            if info["available"]:
                info["config"] = await self.get_service_config()
        except Exception as e:
            info["error"] = str(e)
        
        return info


# Global service discovery instance
service_discovery = ServiceDiscovery()