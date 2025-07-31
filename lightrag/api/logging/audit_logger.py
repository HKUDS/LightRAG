"""
Comprehensive audit logging system for LightRAG.

Provides structured security event logging, analysis, and monitoring
capabilities for authentication and system operations.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import os
import threading
from collections import defaultdict, deque

logger = logging.getLogger("lightrag.audit")


class AuditEventType(Enum):
    """Audit event types for categorization."""

    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    TOKEN_EXPIRED = "auth.token.expired"
    TOKEN_INVALID = "auth.token.invalid"
    PASSWORD_CHANGE = "auth.password.change"
    PASSWORD_RESET = "auth.password.reset"
    ACCOUNT_LOCKED = "auth.account.locked"
    ACCOUNT_UNLOCKED = "auth.account.unlocked"

    # Multi-factor authentication
    MFA_SETUP = "auth.mfa.setup"
    MFA_ENABLED = "auth.mfa.enabled"
    MFA_DISABLED = "auth.mfa.disabled"
    MFA_SUCCESS = "auth.mfa.success"
    MFA_FAILURE = "auth.mfa.failure"
    MFA_BACKUP_USED = "auth.mfa.backup_used"

    # API access events
    API_ACCESS = "api.access"
    API_ERROR = "api.error"
    API_FORBIDDEN = "api.forbidden"
    API_UNAUTHORIZED = "api.unauthorized"
    API_RATE_LIMITED = "api.rate_limited"
    API_KEY_CREATED = "api.key.created"
    API_KEY_REVOKED = "api.key.revoked"
    API_KEY_USED = "api.key.used"

    # Document events
    DOCUMENT_UPLOAD = "document.upload"
    DOCUMENT_DELETE = "document.delete"
    DOCUMENT_ACCESS = "document.access"
    DOCUMENT_SHARE = "document.share"
    DOCUMENT_EXPORT = "document.export"

    # Query events
    QUERY_EXECUTE = "query.execute"
    QUERY_STREAM = "query.stream"
    QUERY_ERROR = "query.error"
    QUERY_TIMEOUT = "query.timeout"

    # Graph events
    GRAPH_ACCESS = "graph.access"
    GRAPH_MODIFY = "graph.modify"
    GRAPH_EXPORT = "graph.export"
    ENTITY_CREATE = "graph.entity.create"
    ENTITY_UPDATE = "graph.entity.update"
    ENTITY_DELETE = "graph.entity.delete"

    # Security events
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    IP_BLOCKED = "security.ip_blocked"
    PRIVILEGE_ESCALATION = "security.privilege_escalation"
    BRUTE_FORCE_DETECTED = "security.brute_force"
    ANOMALY_DETECTED = "security.anomaly"

    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_CONFIG_CHANGE = "system.config.change"
    SYSTEM_ERROR = "system.error"
    SYSTEM_MAINTENANCE = "system.maintenance"

    # Admin events
    USER_CREATED = "admin.user.created"
    USER_DELETED = "admin.user.deleted"
    USER_MODIFIED = "admin.user.modified"
    ROLE_ASSIGNED = "admin.role.assigned"
    ROLE_REVOKED = "admin.role.revoked"
    PERMISSION_GRANTED = "admin.permission.granted"
    PERMISSION_DENIED = "admin.permission.denied"


class AuditSeverity(Enum):
    """Audit event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""

    # Core event information
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity

    # User and session information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Request information
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None

    # Event details
    message: str = ""
    details: Dict[str, Any] = None
    success: bool = True
    error_message: Optional[str] = None

    # Additional context
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    action: Optional[str] = None

    # Metadata
    correlation_id: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.details is None:
            self.details = {}
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        data = asdict(self)

        # Convert enum values to strings
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        data["timestamp"] = self.timestamp.isoformat()

        # Remove None values for cleaner logs
        return {k: v for k, v in data.items() if v is not None}

    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


class AuditLogFormatter(logging.Formatter):
    """Custom formatter for audit logs."""

    def format(self, record):
        """Format audit log record."""
        if hasattr(record, "audit_event"):
            # Structured audit event
            return record.audit_event.to_json()
        else:
            # Standard log message
            return super().format(record)


@dataclass
class AuditLogConfig:
    """Audit logging configuration."""

    # File settings
    log_file: str = "logs/audit.log"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 10
    compress_backups: bool = True

    # Logging settings
    log_level: str = "INFO"
    structured_logging: bool = True
    async_logging: bool = True
    buffer_size: int = 1000
    flush_interval: float = 5.0

    # Security settings
    encrypt_logs: bool = False
    encryption_key: Optional[str] = None

    # Retention settings
    retention_days: int = 365
    auto_cleanup: bool = True

    # Analysis settings
    enable_analytics: bool = True
    anomaly_detection: bool = True
    real_time_alerts: bool = True

    # Export settings
    export_format: str = "json"  # json, csv, xml
    enable_syslog: bool = False
    syslog_server: Optional[str] = None
    syslog_port: int = 514

    @classmethod
    def from_env(cls) -> "AuditLogConfig":
        """Create configuration from environment variables."""

        def get_env_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, "").lower()
            return value in ("true", "1", "yes", "on") if value else default

        return cls(
            log_file=os.getenv("AUDIT_LOG_FILE", "logs/audit.log"),
            max_file_size=int(os.getenv("AUDIT_MAX_FILE_SIZE", str(100 * 1024 * 1024))),
            backup_count=int(os.getenv("AUDIT_BACKUP_COUNT", "10")),
            compress_backups=get_env_bool("AUDIT_COMPRESS_BACKUPS", True),
            log_level=os.getenv("AUDIT_LOG_LEVEL", "INFO").upper(),
            structured_logging=get_env_bool("AUDIT_STRUCTURED_LOGGING", True),
            async_logging=get_env_bool("AUDIT_ASYNC_LOGGING", True),
            buffer_size=int(os.getenv("AUDIT_BUFFER_SIZE", "1000")),
            flush_interval=float(os.getenv("AUDIT_FLUSH_INTERVAL", "5.0")),
            encrypt_logs=get_env_bool("AUDIT_ENCRYPT_LOGS", False),
            encryption_key=os.getenv("AUDIT_ENCRYPTION_KEY"),
            retention_days=int(os.getenv("AUDIT_RETENTION_DAYS", "365")),
            auto_cleanup=get_env_bool("AUDIT_AUTO_CLEANUP", True),
            enable_analytics=get_env_bool("AUDIT_ENABLE_ANALYTICS", True),
            anomaly_detection=get_env_bool("AUDIT_ANOMALY_DETECTION", True),
            real_time_alerts=get_env_bool("AUDIT_REAL_TIME_ALERTS", True),
            export_format=os.getenv("AUDIT_EXPORT_FORMAT", "json"),
            enable_syslog=get_env_bool("AUDIT_ENABLE_SYSLOG", False),
            syslog_server=os.getenv("AUDIT_SYSLOG_SERVER"),
            syslog_port=int(os.getenv("AUDIT_SYSLOG_PORT", "514")),
        )


class AuditLogger:
    """Comprehensive audit logging system."""

    def __init__(self, config: Optional[AuditLogConfig] = None):
        self.config = config or AuditLogConfig.from_env()
        self._logger = None
        self._queue = None
        self._worker_task = None
        self._analytics = AuditAnalytics()
        self._event_buffer = deque(maxlen=self.config.buffer_size)
        self._last_flush = datetime.now()
        self._lock = threading.Lock()

        self._setup_logger()

    def _setup_logger(self):
        """Setup audit logger."""
        self._logger = logging.getLogger("lightrag.audit.events")
        self._logger.setLevel(getattr(logging, self.config.log_level))

        # Clear existing handlers
        self._logger.handlers.clear()

        # Create log directory
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup file handler with rotation
        from logging.handlers import RotatingFileHandler

        handler = RotatingFileHandler(
            filename=self.config.log_file,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
        )

        # Setup formatter
        if self.config.structured_logging:
            formatter = AuditLogFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        # Setup console handler for development
        if os.getenv("AUDIT_CONSOLE_OUTPUT", "").lower() == "true":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    async def start(self):
        """Start async logging."""
        if self.config.async_logging:
            self._queue = asyncio.Queue(maxsize=self.config.buffer_size * 2)
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        """Stop async logging and flush remaining events."""
        if self._worker_task:
            # Flush remaining events
            await self._flush_buffer()

            # Stop worker
            await self._queue.put(None)  # Sentinel value
            await self._worker_task

    async def _worker(self):
        """Async worker for processing log events."""
        while True:
            try:
                event = await self._queue.get()
                if event is None:  # Sentinel value to stop
                    break

                self._write_event(event)

                # Update analytics
                if self.config.enable_analytics:
                    self._analytics.process_event(event)

                # Check for real-time alerts
                if self.config.real_time_alerts:
                    await self._check_alerts(event)

            except Exception as e:
                logger.error(f"Audit logging worker error: {e}")

    def _write_event(self, event: AuditEvent):
        """Write event to log file."""
        try:
            log_record = logging.LogRecord(
                name="lightrag.audit.events",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="",
                args=(),
                exc_info=None,
            )

            log_record.audit_event = event
            self._logger.handle(log_record)

        except Exception as e:
            logger.error(f"Error writing audit event: {e}")

    async def log_event(self, event: AuditEvent):
        """Log audit event."""
        if self.config.async_logging and self._queue:
            try:
                await self._queue.put(event)
            except asyncio.QueueFull:
                # Fallback to synchronous logging
                self._write_event(event)
        else:
            self._write_event(event)

    async def log_auth_event(
        self,
        event_type: AuditEventType,
        user_id: str = None,
        success: bool = True,
        details: Dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None,
        **kwargs,
    ):
        """Log authentication event."""
        severity = AuditSeverity.LOW if success else AuditSeverity.MEDIUM

        if event_type in [AuditEventType.LOGIN_FAILURE, AuditEventType.ACCOUNT_LOCKED]:
            severity = AuditSeverity.HIGH

        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {},
            **kwargs,
        )

        await self.log_event(event)

    async def log_api_access(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        user_id: str = None,
        ip_address: str = None,
        response_time: float = None,
        **kwargs,
    ):
        """Log API access event."""
        success = 200 <= status_code < 400

        # Determine severity based on status code
        if status_code >= 500:
            severity = AuditSeverity.HIGH
        elif status_code >= 400:
            severity = AuditSeverity.MEDIUM
        else:
            severity = AuditSeverity.LOW

        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.API_ACCESS,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            success=success,
            **kwargs,
        )

        await self.log_event(event)

    async def log_security_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        message: str = "",
        details: Dict[str, Any] = None,
        **kwargs,
    ):
        """Log security event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            message=message,
            details=details or {},
            success=False,  # Security events are typically incidents
            **kwargs,
        )

        await self.log_event(event)

    async def _flush_buffer(self):
        """Flush buffered events."""
        with self._lock:
            while self._event_buffer:
                event = self._event_buffer.popleft()
                self._write_event(event)

    async def _check_alerts(self, event: AuditEvent):
        """Check if event should trigger real-time alerts."""
        # Critical events always trigger alerts
        if event.severity == AuditSeverity.CRITICAL:
            await self._send_alert(event, "Critical security event detected")

        # Multiple failed logins
        if event.event_type == AuditEventType.LOGIN_FAILURE:
            recent_failures = self._analytics.get_recent_failures(
                event.user_id or event.ip_address, minutes=15
            )
            if recent_failures >= 5:
                await self._send_alert(
                    event, f"Multiple login failures: {recent_failures}"
                )

        # Privilege escalation attempts
        if event.event_type == AuditEventType.PRIVILEGE_ESCALATION:
            await self._send_alert(event, "Privilege escalation attempt detected")

    async def _send_alert(self, event: AuditEvent, message: str):
        """Send real-time alert."""
        # In production, implement actual alerting (email, Slack, etc.)
        logger.critical(f"SECURITY ALERT: {message} - Event ID: {event.event_id}")

    def get_analytics(self) -> "AuditAnalytics":
        """Get audit analytics instance."""
        return self._analytics


class AuditAnalytics:
    """Audit log analytics and reporting."""

    def __init__(self):
        self._event_counts = defaultdict(int)
        self._user_activity = defaultdict(list)
        self._ip_activity = defaultdict(list)
        self._recent_events = deque(maxlen=10000)
        self._anomaly_detector = AnomalyDetector()

    def process_event(self, event: AuditEvent):
        """Process event for analytics."""
        self._event_counts[event.event_type.value] += 1
        self._recent_events.append(event)

        if event.user_id:
            self._user_activity[event.user_id].append(event)

        if event.ip_address:
            self._ip_activity[event.ip_address].append(event)

        # Anomaly detection
        self._anomaly_detector.analyze_event(event)

    def get_event_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get event summary for specified time period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        recent_events = [
            event for event in self._recent_events if event.timestamp > cutoff
        ]

        summary = {
            "total_events": len(recent_events),
            "time_period_hours": hours,
            "event_types": defaultdict(int),
            "severity_breakdown": defaultdict(int),
            "success_rate": 0,
            "top_users": defaultdict(int),
            "top_ips": defaultdict(int),
        }

        successful_events = 0

        for event in recent_events:
            summary["event_types"][event.event_type.value] += 1
            summary["severity_breakdown"][event.severity.value] += 1

            if event.success:
                successful_events += 1

            if event.user_id:
                summary["top_users"][event.user_id] += 1

            if event.ip_address:
                summary["top_ips"][event.ip_address] += 1

        if recent_events:
            summary["success_rate"] = successful_events / len(recent_events)

        # Convert to regular dicts and sort
        summary["top_users"] = dict(
            sorted(summary["top_users"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        summary["top_ips"] = dict(
            sorted(summary["top_ips"].items(), key=lambda x: x[1], reverse=True)[:10]
        )

        return summary

    def get_recent_failures(self, identifier: str, minutes: int = 15) -> int:
        """Get count of recent failures for user or IP."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        failure_events = [
            AuditEventType.LOGIN_FAILURE,
            AuditEventType.MFA_FAILURE,
            AuditEventType.API_UNAUTHORIZED,
            AuditEventType.API_FORBIDDEN,
        ]

        count = 0
        for event in self._recent_events:
            if (
                event.timestamp > cutoff
                and event.event_type in failure_events
                and (event.user_id == identifier or event.ip_address == identifier)
            ):
                count += 1

        return count

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous patterns in audit logs."""
        return self._anomaly_detector.get_anomalies()


class AnomalyDetector:
    """Simple anomaly detection for audit events."""

    def __init__(self):
        self._baseline_patterns = defaultdict(list)
        self._anomalies = []

    def analyze_event(self, event: AuditEvent):
        """Analyze event for anomalies."""
        # Simple pattern-based detection
        current_hour = event.timestamp.hour

        # Track hourly patterns for each event type
        pattern_key = f"{event.event_type.value}:{current_hour}"
        self._baseline_patterns[pattern_key].append(event.timestamp)

        # Keep only last 30 days of data
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        self._baseline_patterns[pattern_key] = [
            ts for ts in self._baseline_patterns[pattern_key] if ts > cutoff
        ]

    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get detected anomalies."""
        return self._anomalies[-100:]  # Return last 100 anomalies


# Global audit logger instance
audit_logger = AuditLogger()
