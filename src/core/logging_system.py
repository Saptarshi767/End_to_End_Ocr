"""
Comprehensive logging and monitoring system

Provides structured logging, performance monitoring, metrics collection,
and audit trails for all system operations.
"""

import logging
import logging.handlers
import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict, deque
import uuid
import os


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    component: str
    operation: str
    message: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    document_id: Optional[str] = None
    table_id: Optional[str] = None
    duration_ms: Optional[float] = None
    status: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class AuditEvent:
    """Audit trail event"""
    event_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class StructuredLogger:
    """Structured logging with JSON format"""
    
    def __init__(self, name: str, log_file: str = None, log_level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Create formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # JSON handler for structured logs
        json_file = log_file.replace('.log', '_structured.jsonl') if log_file else None
        if json_file:
            self.json_handler = logging.handlers.RotatingFileHandler(
                json_file, maxBytes=10*1024*1024, backupCount=5
            )
            self.json_handler.setFormatter(logging.Formatter('%(message)s'))
            
            # Create separate logger for JSON
            self.json_logger = logging.getLogger(f"{name}_json")
            self.json_logger.setLevel(logging.INFO)
            self.json_logger.addHandler(self.json_handler)
            self.json_logger.propagate = False
    
    def log_structured(self, entry: LogEntry):
        """Log structured entry as JSON"""
        
        # Log to standard logger
        self.logger.log(
            getattr(logging, entry.level),
            f"[{entry.component}:{entry.operation}] {entry.message}",
            extra={
                'user_id': entry.user_id,
                'request_id': entry.request_id,
                'document_id': entry.document_id,
                'duration_ms': entry.duration_ms
            }
        )
        
        # Log to JSON logger if available
        if hasattr(self, 'json_logger'):
            self.json_logger.info(json.dumps(entry.to_dict()))
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        entry = self._create_entry(LogLevel.DEBUG.value, message, **kwargs)
        self.log_structured(entry)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        entry = self._create_entry(LogLevel.INFO.value, message, **kwargs)
        self.log_structured(entry)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        entry = self._create_entry(LogLevel.WARNING.value, message, **kwargs)
        self.log_structured(entry)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        entry = self._create_entry(LogLevel.ERROR.value, message, **kwargs)
        self.log_structured(entry)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        entry = self._create_entry(LogLevel.CRITICAL.value, message, **kwargs)
        self.log_structured(entry)
    
    def _create_entry(self, level: str, message: str, **kwargs) -> LogEntry:
        """Create log entry from parameters"""
        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            component=kwargs.get('component', 'unknown'),
            operation=kwargs.get('operation', 'unknown'),
            message=message,
            user_id=kwargs.get('user_id'),
            request_id=kwargs.get('request_id'),
            document_id=kwargs.get('document_id'),
            table_id=kwargs.get('table_id'),
            duration_ms=kwargs.get('duration_ms'),
            status=kwargs.get('status'),
            error_code=kwargs.get('error_code'),
            metadata=kwargs.get('metadata', {})
        )


class MetricsCollector:
    """Performance metrics collection and aggregation"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics = deque(maxlen=max_metrics)
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        self.lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        with self.lock:
            key = self._make_key(name, tags)
            self.counters[key] += value
            
            metric = PerformanceMetric(
                name=name,
                metric_type=MetricType.COUNTER,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric value"""
        with self.lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value
            
            metric = PerformanceMetric(
                name=name,
                metric_type=MetricType.GAUGE,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        with self.lock:
            key = self._make_key(name, tags)
            self.histograms[key].append(value)
            
            # Keep only last 1000 values per histogram
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            metric = PerformanceMetric(
                name=name,
                metric_type=MetricType.HISTOGRAM,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def record_timer(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a timer value"""
        with self.lock:
            key = self._make_key(name, tags)
            self.timers[key].append(duration_ms)
            
            # Keep only last 1000 values per timer
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]
            
            metric = PerformanceMetric(
                name=name,
                metric_type=MetricType.TIMER,
                value=duration_ms,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def get_counter_value(self, name: str, tags: Dict[str, str] = None) -> float:
        """Get current counter value"""
        key = self._make_key(name, tags)
        return self.counters.get(key, 0.0)
    
    def get_gauge_value(self, name: str, tags: Dict[str, str] = None) -> float:
        """Get current gauge value"""
        key = self._make_key(name, tags)
        return self.gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        key = self._make_key(name, tags)
        values = self.histograms.get(key, [])
        
        if not values:
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_timer_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get timer statistics"""
        return self.get_histogram_stats(name, tags)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histogram_stats": {
                    key: self.get_histogram_stats(key.split(":")[0], 
                                                 self._parse_tags(key))
                    for key in self.histograms.keys()
                },
                "timer_stats": {
                    key: self.get_timer_stats(key.split(":")[0], 
                                            self._parse_tags(key))
                    for key in self.timers.keys()
                }
            }
    
    def _make_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Create metric key from name and tags"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"
    
    def _parse_tags(self, key: str) -> Dict[str, str]:
        """Parse tags from metric key"""
        if ":" not in key:
            return {}
        
        tag_str = key.split(":", 1)[1]
        tags = {}
        
        for tag_pair in tag_str.split(","):
            if "=" in tag_pair:
                k, v = tag_pair.split("=", 1)
                tags[k] = v
        
        return tags


class AuditLogger:
    """Audit trail logging for compliance and security"""
    
    def __init__(self, audit_file: str = "logs/audit.jsonl"):
        self.audit_file = audit_file
        os.makedirs(os.path.dirname(audit_file), exist_ok=True)
        
        # Setup audit logger
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.handlers.RotatingFileHandler(
            audit_file, maxBytes=50*1024*1024, backupCount=10
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.propagate = False
    
    def log_event(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        old_values: Dict[str, Any] = None,
        new_values: Dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True,
        error_message: str = None
    ):
        """Log an audit event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
        
        self.logger.info(json.dumps(event.to_dict()))
    
    def log_document_upload(self, user_id: str, document_id: str, filename: str, 
                           ip_address: str = None, success: bool = True):
        """Log document upload event"""
        self.log_event(
            user_id=user_id,
            action="document_upload",
            resource_type="document",
            resource_id=document_id,
            new_values={"filename": filename},
            ip_address=ip_address,
            success=success
        )
    
    def log_data_export(self, user_id: str, resource_id: str, export_format: str,
                       ip_address: str = None, success: bool = True):
        """Log data export event"""
        self.log_event(
            user_id=user_id,
            action="data_export",
            resource_type="table",
            resource_id=resource_id,
            new_values={"format": export_format},
            ip_address=ip_address,
            success=success
        )
    
    def log_dashboard_share(self, user_id: str, dashboard_id: str, share_type: str,
                           permissions: List[str], ip_address: str = None):
        """Log dashboard sharing event"""
        self.log_event(
            user_id=user_id,
            action="dashboard_share",
            resource_type="dashboard",
            resource_id=dashboard_id,
            new_values={"share_type": share_type, "permissions": permissions},
            ip_address=ip_address
        )
    
    def log_user_login(self, user_id: str, ip_address: str = None, 
                      user_agent: str = None, success: bool = True):
        """Log user login event"""
        self.log_event(
            user_id=user_id,
            action="user_login",
            resource_type="user",
            resource_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )


class PerformanceMonitor:
    """Performance monitoring with timing decorators and context managers"""
    
    def __init__(self, metrics_collector: MetricsCollector, logger: StructuredLogger):
        self.metrics = metrics_collector
        self.logger = logger
    
    @contextmanager
    def timer(self, operation: str, component: str = "unknown", 
              user_id: str = None, request_id: str = None, **tags):
        """Context manager for timing operations"""
        
        start_time = time.time()
        start_timestamp = datetime.now()
        
        try:
            yield
            
            # Success case
            duration_ms = (time.time() - start_time) * 1000
            
            self.metrics.record_timer(
                f"{component}.{operation}.duration",
                duration_ms,
                tags
            )
            
            self.logger.info(
                f"Operation completed successfully",
                component=component,
                operation=operation,
                user_id=user_id,
                request_id=request_id,
                duration_ms=duration_ms,
                status="success"
            )
            
        except Exception as e:
            # Error case
            duration_ms = (time.time() - start_time) * 1000
            
            self.metrics.increment_counter(
                f"{component}.{operation}.errors",
                tags={**tags, "error_type": type(e).__name__}
            )
            
            self.logger.error(
                f"Operation failed: {str(e)}",
                component=component,
                operation=operation,
                user_id=user_id,
                request_id=request_id,
                duration_ms=duration_ms,
                status="error",
                error_code=type(e).__name__
            )
            
            raise
    
    def time_function(self, component: str = None):
        """Decorator for timing function execution"""
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                comp = component or func.__module__
                op = func.__name__
                
                with self.timer(op, comp):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def monitor_resource_usage(self):
        """Monitor system resource usage"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.set_gauge("system.cpu.usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.set_gauge("system.memory.usage_percent", memory.percent)
            self.metrics.set_gauge("system.memory.available_mb", memory.available / 1024 / 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics.set_gauge("system.disk.usage_percent", 
                                 (disk.used / disk.total) * 100)
            
        except ImportError:
            self.logger.warning("psutil not available for resource monitoring")
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")


class LoggingSystem:
    """Main logging system that coordinates all logging components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.logger = StructuredLogger(
            name="ocr_analytics",
            log_file=self.config.get("log_file", "logs/application.log"),
            log_level=self.config.get("log_level", "INFO")
        )
        
        self.metrics = MetricsCollector(
            max_metrics=self.config.get("max_metrics", 10000)
        )
        
        self.audit = AuditLogger(
            audit_file=self.config.get("audit_file", "logs/audit.jsonl")
        )
        
        self.monitor = PerformanceMonitor(self.metrics, self.logger)
        
        # Start background monitoring if enabled
        if self.config.get("enable_resource_monitoring", True):
            self._start_resource_monitoring()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default logging configuration"""
        return {
            "log_file": "logs/application.log",
            "audit_file": "logs/audit.jsonl",
            "log_level": "INFO",
            "max_metrics": 10000,
            "enable_resource_monitoring": True,
            "resource_monitoring_interval": 60
        }
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring"""
        def monitor_loop():
            while True:
                try:
                    self.monitor.monitor_resource_usage()
                    time.sleep(self.config.get("resource_monitoring_interval", 60))
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
    
    def get_logger(self) -> StructuredLogger:
        """Get the structured logger"""
        return self.logger
    
    def get_metrics(self) -> MetricsCollector:
        """Get the metrics collector"""
        return self.metrics
    
    def get_audit_logger(self) -> AuditLogger:
        """Get the audit logger"""
        return self.audit
    
    def get_monitor(self) -> PerformanceMonitor:
        """Get the performance monitor"""
        return self.monitor
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "metrics": self.metrics.get_all_metrics(),
            "timestamp": datetime.now().isoformat(),
            "config": self.config
        }


# Global logging system instance
logging_system = LoggingSystem()


def get_logger() -> StructuredLogger:
    """Get the global structured logger"""
    return logging_system.get_logger()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector"""
    return logging_system.get_metrics()


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger"""
    return logging_system.get_audit_logger()


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor"""
    return logging_system.get_monitor()