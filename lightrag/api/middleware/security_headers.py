"""
Security headers middleware for LightRAG.

Provides comprehensive security headers to protect against common
web vulnerabilities and attacks.
"""

import os
from typing import Dict, Optional, List
from dataclasses import dataclass
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import logging

logger = logging.getLogger("lightrag.auth.security_headers")


@dataclass
class SecurityHeadersConfig:
    """Security headers configuration."""
    
    # Content Security Policy
    csp_default_src: str = "'self'"
    csp_script_src: str = "'self' 'unsafe-inline'"
    csp_style_src: str = "'self' 'unsafe-inline'"
    csp_img_src: str = "'self' data: blob:"
    csp_font_src: str = "'self'"
    csp_connect_src: str = "'self'"
    csp_media_src: str = "'self'"
    csp_object_src: str = "'none'"
    csp_frame_src: str = "'none'"
    csp_frame_ancestors: str = "'none'"
    csp_base_uri: str = "'self'"
    csp_form_action: str = "'self'"
    
    # Strict Transport Security
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True
    
    # Other security headers
    x_content_type_options: str = "nosniff"
    x_frame_options: str = "DENY"
    x_xss_protection: str = "1; mode=block"
    referrer_policy: str = "strict-origin-when-cross-origin"
    permissions_policy: str = "camera=(), microphone=(), geolocation=()"
    
    # Feature control
    enable_csp: bool = True
    enable_hsts: bool = True
    enable_x_headers: bool = True
    enable_permissions_policy: bool = True
    
    # Custom headers
    custom_headers: Dict[str, str] = None
    
    # Server header
    hide_server_header: bool = True
    custom_server_header: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "SecurityHeadersConfig":
        """Create configuration from environment variables."""
        
        def get_env_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, "").lower()
            return value in ("true", "1", "yes", "on") if value else default
        
        # Parse custom headers from environment
        custom_headers = {}
        custom_headers_env = os.getenv("SECURITY_CUSTOM_HEADERS", "")
        if custom_headers_env:
            try:
                import json
                custom_headers = json.loads(custom_headers_env)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in SECURITY_CUSTOM_HEADERS")
        
        return cls(
            # CSP configuration
            csp_default_src=os.getenv("CSP_DEFAULT_SRC", "'self'"),
            csp_script_src=os.getenv("CSP_SCRIPT_SRC", "'self' 'unsafe-inline'"),
            csp_style_src=os.getenv("CSP_STYLE_SRC", "'self' 'unsafe-inline'"),
            csp_img_src=os.getenv("CSP_IMG_SRC", "'self' data: blob:"),
            csp_font_src=os.getenv("CSP_FONT_SRC", "'self'"),
            csp_connect_src=os.getenv("CSP_CONNECT_SRC", "'self'"),
            csp_media_src=os.getenv("CSP_MEDIA_SRC", "'self'"),
            csp_object_src=os.getenv("CSP_OBJECT_SRC", "'none'"),
            csp_frame_src=os.getenv("CSP_FRAME_SRC", "'none'"),
            csp_frame_ancestors=os.getenv("CSP_FRAME_ANCESTORS", "'none'"),
            csp_base_uri=os.getenv("CSP_BASE_URI", "'self'"),
            csp_form_action=os.getenv("CSP_FORM_ACTION", "'self'"),
            
            # HSTS configuration
            hsts_max_age=int(os.getenv("HSTS_MAX_AGE", "31536000")),
            hsts_include_subdomains=get_env_bool("HSTS_INCLUDE_SUBDOMAINS", True),
            hsts_preload=get_env_bool("HSTS_PRELOAD", True),
            
            # Other headers
            x_content_type_options=os.getenv("X_CONTENT_TYPE_OPTIONS", "nosniff"),
            x_frame_options=os.getenv("X_FRAME_OPTIONS", "DENY"),
            x_xss_protection=os.getenv("X_XSS_PROTECTION", "1; mode=block"),
            referrer_policy=os.getenv("REFERRER_POLICY", "strict-origin-when-cross-origin"),
            permissions_policy=os.getenv("PERMISSIONS_POLICY", "camera=(), microphone=(), geolocation=()"),
            
            # Feature toggles
            enable_csp=get_env_bool("SECURITY_ENABLE_CSP", True),
            enable_hsts=get_env_bool("SECURITY_ENABLE_HSTS", True),
            enable_x_headers=get_env_bool("SECURITY_ENABLE_X_HEADERS", True),
            enable_permissions_policy=get_env_bool("SECURITY_ENABLE_PERMISSIONS_POLICY", True),
            
            # Custom configuration
            custom_headers=custom_headers,
            hide_server_header=get_env_bool("SECURITY_HIDE_SERVER_HEADER", True),
            custom_server_header=os.getenv("SECURITY_CUSTOM_SERVER_HEADER")
        )
    
    def build_csp_header(self) -> str:
        """Build Content Security Policy header value."""
        if not self.enable_csp:
            return ""
        
        csp_parts = [
            f"default-src {self.csp_default_src}",
            f"script-src {self.csp_script_src}",
            f"style-src {self.csp_style_src}",
            f"img-src {self.csp_img_src}",
            f"font-src {self.csp_font_src}",
            f"connect-src {self.csp_connect_src}",
            f"media-src {self.csp_media_src}",
            f"object-src {self.csp_object_src}",
            f"frame-src {self.csp_frame_src}",
            f"frame-ancestors {self.csp_frame_ancestors}",
            f"base-uri {self.csp_base_uri}",
            f"form-action {self.csp_form_action}"
        ]
        
        return "; ".join(csp_parts)
    
    def build_hsts_header(self) -> str:
        """Build Strict Transport Security header value."""
        if not self.enable_hsts:
            return ""
        
        hsts_parts = [f"max-age={self.hsts_max_age}"]
        
        if self.hsts_include_subdomains:
            hsts_parts.append("includeSubDomains")
        
        if self.hsts_preload:
            hsts_parts.append("preload")
        
        return "; ".join(hsts_parts)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Security headers middleware.
    
    Adds comprehensive security headers to all responses to protect
    against common web vulnerabilities.
    """
    
    def __init__(self, app, config: Optional[SecurityHeadersConfig] = None):
        super().__init__(app)
        self.config = config or SecurityHeadersConfig.from_env()
        self._setup_headers()
    
    def _setup_headers(self):
        """Setup headers dictionary based on configuration."""
        self.headers = {}
        
        # Content Security Policy
        if self.config.enable_csp:
            csp_header = self.config.build_csp_header()
            if csp_header:
                self.headers["Content-Security-Policy"] = csp_header
        
        # Strict Transport Security (only over HTTPS)
        if self.config.enable_hsts:
            hsts_header = self.config.build_hsts_header()
            if hsts_header:
                self.headers["Strict-Transport-Security"] = hsts_header
        
        # X-Headers for legacy browser support
        if self.config.enable_x_headers:
            self.headers.update({
                "X-Content-Type-Options": self.config.x_content_type_options,
                "X-Frame-Options": self.config.x_frame_options,
                "X-XSS-Protection": self.config.x_xss_protection,
            })
        
        # Referrer Policy
        self.headers["Referrer-Policy"] = self.config.referrer_policy
        
        # Permissions Policy
        if self.config.enable_permissions_policy:
            self.headers["Permissions-Policy"] = self.config.permissions_policy
        
        # Custom headers
        if self.config.custom_headers:
            self.headers.update(self.config.custom_headers)
        
        logger.info(f"Security headers configured: {len(self.headers)} headers")
    
    async def dispatch(self, request: Request, call_next):
        """Process request and add security headers to response."""
        
        # Check if HTTPS is being used for HSTS
        is_https = (
            request.url.scheme == "https" or
            request.headers.get("x-forwarded-proto") == "https" or
            request.headers.get("x-forwarded-ssl") == "on"
        )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header_name, header_value in self.headers.items():
            # Only add HSTS over HTTPS
            if header_name == "Strict-Transport-Security" and not is_https:
                continue
            
            response.headers[header_name] = header_value
        
        # Handle server header
        if self.config.hide_server_header:
            if "server" in response.headers:
                if self.config.custom_server_header:
                    response.headers["server"] = self.config.custom_server_header
                else:
                    del response.headers["server"]
        
        # Add security-related headers for specific content types
        self._add_content_specific_headers(request, response)
        
        return response
    
    def _add_content_specific_headers(self, request: Request, response: Response):
        """Add content-specific security headers."""
        content_type = response.headers.get("content-type", "").lower()
        
        # Cache control for sensitive content
        if any(path in request.url.path.lower() for path in ["/auth", "/login", "/admin"]):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        # Additional headers for HTML content
        if "text/html" in content_type:
            response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # Additional headers for JSON API responses
        if "application/json" in content_type:
            response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Headers for file downloads
        if "application/octet-stream" in content_type:
            response.headers["X-Download-Options"] = "noopen"


class CSPViolationReporter:
    """Content Security Policy violation reporter."""
    
    def __init__(self, report_endpoint: str = "/security/csp-report"):
        self.report_endpoint = report_endpoint
        self.violations = []
    
    async def handle_csp_report(self, request: Request) -> Response:
        """Handle CSP violation reports."""
        try:
            report_data = await request.json()
            violation = {
                "timestamp": request.headers.get("date"),
                "user_agent": request.headers.get("user-agent"),
                "ip_address": request.client.host,
                "report": report_data
            }
            
            self.violations.append(violation)
            logger.warning(f"CSP Violation: {report_data}")
            
            # In production, you might want to:
            # - Store violations in database
            # - Send alerts for critical violations
            # - Analyze patterns for security threats
            
            return Response(status_code=204)  # No Content
            
        except Exception as e:
            logger.error(f"Error processing CSP report: {e}")
            return Response(status_code=400)
    
    def get_violations(self, limit: int = 100) -> List[Dict]:
        """Get recent CSP violations."""
        return self.violations[-limit:]
    
    def clear_violations(self):
        """Clear stored violations."""
        self.violations.clear()


class SecurityAnalyzer:
    """Security analysis and recommendations."""
    
    def __init__(self, config: SecurityHeadersConfig):
        self.config = config
    
    def analyze_security_posture(self) -> Dict[str, any]:
        """Analyze current security configuration."""
        analysis = {
            "score": 0,
            "max_score": 100,
            "recommendations": [],
            "warnings": [],
            "good_practices": []
        }
        
        # CSP Analysis
        if self.config.enable_csp:
            analysis["score"] += 25
            analysis["good_practices"].append("Content Security Policy enabled")
            
            # Check for unsafe CSP directives
            if "'unsafe-inline'" in self.config.csp_script_src:
                analysis["warnings"].append("CSP allows unsafe-inline scripts")
                analysis["recommendations"].append("Remove 'unsafe-inline' from script-src for better security")
            
            if "'unsafe-eval'" in self.config.csp_script_src:
                analysis["warnings"].append("CSP allows unsafe-eval")
                analysis["recommendations"].append("Remove 'unsafe-eval' from script-src")
        else:
            analysis["recommendations"].append("Enable Content Security Policy")
        
        # HSTS Analysis
        if self.config.enable_hsts:
            analysis["score"] += 20
            analysis["good_practices"].append("Strict Transport Security enabled")
            
            if self.config.hsts_max_age < 31536000:  # 1 year
                analysis["recommendations"].append("Increase HSTS max-age to at least 1 year")
            
            if not self.config.hsts_include_subdomains:
                analysis["recommendations"].append("Enable HSTS for subdomains")
        else:
            analysis["recommendations"].append("Enable Strict Transport Security")
        
        # X-Headers Analysis
        if self.config.enable_x_headers:
            analysis["score"] += 15
            analysis["good_practices"].append("Legacy security headers enabled")
        else:
            analysis["recommendations"].append("Enable X-Security headers for legacy browser support")
        
        # Frame Options Analysis
        if self.config.x_frame_options == "DENY":
            analysis["score"] += 10
            analysis["good_practices"].append("Clickjacking protection enabled")
        elif self.config.x_frame_options == "SAMEORIGIN":
            analysis["score"] += 5
            analysis["recommendations"].append("Consider using DENY for X-Frame-Options for stronger protection")
        
        # Permissions Policy Analysis
        if self.config.enable_permissions_policy:
            analysis["score"] += 10
            analysis["good_practices"].append("Permissions Policy configured")
        else:
            analysis["recommendations"].append("Configure Permissions Policy to control browser features")
        
        # Server Header Analysis
        if self.config.hide_server_header:
            analysis["score"] += 5
            analysis["good_practices"].append("Server information hidden")
        else:
            analysis["recommendations"].append("Hide server header to reduce information disclosure")
        
        # Additional security checks
        if self.config.referrer_policy == "strict-origin-when-cross-origin":
            analysis["score"] += 10
            analysis["good_practices"].append("Secure referrer policy configured")
        
        # Calculate final score
        analysis["security_level"] = self._get_security_level(analysis["score"])
        
        return analysis
    
    def _get_security_level(self, score: int) -> str:
        """Get security level based on score."""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Weak"
    
    def generate_security_report(self) -> str:
        """Generate security report as formatted string."""
        analysis = self.analyze_security_posture()
        
        report = f"""
LightRAG Security Headers Analysis
==================================

Security Score: {analysis['score']}/{analysis['max_score']} ({analysis['security_level']})

Good Practices:
{chr(10).join('✓ ' + practice for practice in analysis['good_practices'])}

Warnings:
{chr(10).join('⚠ ' + warning for warning in analysis['warnings'])}

Recommendations:
{chr(10).join('• ' + rec for rec in analysis['recommendations'])}
"""
        return report


# Global instances
security_config = SecurityHeadersConfig.from_env()
csp_reporter = CSPViolationReporter()
security_analyzer = SecurityAnalyzer(security_config)