"""
System monitoring and health checks

Provides comprehensive monitoring capabilities including health checks,
performance metrics, and system status reporting.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .logging_system import get_logger, get_metrics, MetricsCollector
from .error_handler import ErrorHandler, ErrorContext


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SystemMonitor:
    """System monitoring and health checks"""
    
    def __init__(self):
        self.logger = get_logger()
        self.metrics = get_metrics()
        self.error_handler = ErrorHandler()
        self.health_checks = {}
        self.monitoring_thread = None
        self.monitoring_active = False
        self.check_interval = 60  # seconds
    
    def register_health_check(self, name: str, check_function, interval: int = 60):
        """
        Register a health check function
        
        Args:
            name: Name of the health check
            check_function: Function that returns HealthCheck result
            interval: Check interval in seconds
        """
        self.health_checks[name] = {
            'function': check_function,
            'interval': interval,
            'last_check': None,
            'last_result': None
        }
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("System monitoring started", 
                        component="monitor", operation="start_monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("System monitoring stopped", 
                        component="monitor", operation="stop_monitoring")
    
    def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_config in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_config['function']()
                duration_ms = (time.time() - start_time) * 1000
                
                if isinstance(result, HealthCheck):
                    result.duration_ms = duration_ms
                    results[name] = result
                else:
                    # Convert simple result to HealthCheck
                    results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.HEALTHY if result else HealthStatus.CRITICAL,
                        message="Check completed" if result else "Check failed",
                        timestamp=datetime.now(),
                        duration_ms=duration_ms
                    )
                
                check_config['last_check'] = datetime.now()
                check_config['last_result'] = results[name]
                
                # Record metrics
                self.metrics.record_timer(f"health_check.{name}.duration", duration_ms)
                self.metrics.set_gauge(f"health_check.{name}.status", 
                                     1 if results[name].status == HealthStatus.HEALTHY else 0)
                
            except Exception as e:
                error_context = ErrorContext(
                    operation="health_check",
                    component="monitor",
                    additional_data={"check_name": name}
                )
                
                error_response = self.error_handler.handle_error(e, error_context)
                
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    duration_ms=0,
                    metadata={"error": error_response.to_dict()}
                )
                
                self.metrics.increment_counter(f"health_check.{name}.errors")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        health_results = self.run_health_checks()
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        for result in health_results.values():
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif result.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        # Get metrics summary
        all_metrics = self.metrics.get_all_metrics()
        
        # Get error statistics
        error_stats = self.error_handler.get_error_statistics()
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "health_checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "timestamp": result.timestamp.isoformat(),
                    "duration_ms": result.duration_ms,
                    "metadata": result.metadata
                }
                for name, result in health_results.items()
            },
            "metrics_summary": {
                "total_counters": len(all_metrics.get("counters", {})),
                "total_gauges": len(all_metrics.get("gauges", {})),
                "total_histograms": len(all_metrics.get("histogram_stats", {})),
                "total_timers": len(all_metrics.get("timer_stats", {}))
            },
            "error_summary": {
                "total_errors": error_stats.get("total_errors", 0),
                "most_common_errors": error_stats.get("most_common_errors", [])[:5]
            },
            "system_info": {
                "monitoring_active": self.monitoring_active,
                "registered_checks": list(self.health_checks.keys()),
                "check_interval": self.check_interval
            }
        }
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Run health checks
                self.run_health_checks()
                
                # Log system status periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    status = self.get_system_status()
                    self.logger.info(
                        f"System status: {status['overall_status']}",
                        component="monitor",
                        operation="periodic_check",
                        metadata=status
                    )
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(
                    f"Monitoring loop error: {str(e)}",
                    component="monitor",
                    operation="monitoring_loop"
                )
                time.sleep(60)  # Wait longer on error


# Default health checks
def database_health_check() -> HealthCheck:
    """Check database connectivity"""
    try:
        from .database import get_database_connection
        
        start_time = time.time()
        conn = get_database_connection()
        
        # Simple query to test connection
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheck(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection successful",
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            metadata={"query_result": result}
        )
        
    except Exception as e:
        return HealthCheck(
            name="database",
            status=HealthStatus.CRITICAL,
            message=f"Database connection failed: {str(e)}",
            timestamp=datetime.now(),
            duration_ms=0
        )


def ocr_engines_health_check() -> HealthCheck:
    """Check OCR engines availability"""
    try:
        from ..ocr.engine_factory import OCREngineFactory
        
        factory = OCREngineFactory()
        available_engines = []
        failed_engines = []
        
        for engine_name in ['tesseract', 'easyocr']:
            try:
                engine = factory.create_engine(engine_name)
                if engine:
                    available_engines.append(engine_name)
                else:
                    failed_engines.append(engine_name)
            except Exception:
                failed_engines.append(engine_name)
        
        if available_engines:
            status = HealthStatus.HEALTHY if not failed_engines else HealthStatus.WARNING
            message = f"OCR engines available: {', '.join(available_engines)}"
        else:
            status = HealthStatus.CRITICAL
            message = "No OCR engines available"
        
        return HealthCheck(
            name="ocr_engines",
            status=status,
            message=message,
            timestamp=datetime.now(),
            duration_ms=0,
            metadata={
                "available_engines": available_engines,
                "failed_engines": failed_engines
            }
        )
        
    except Exception as e:
        return HealthCheck(
            name="ocr_engines",
            status=HealthStatus.CRITICAL,
            message=f"OCR engine check failed: {str(e)}",
            timestamp=datetime.now(),
            duration_ms=0
        )


def llm_service_health_check() -> HealthCheck:
    """Check LLM service availability"""
    try:
        from ..ai.llm_provider import LLMProvider
        
        provider = LLMProvider()
        
        if provider.is_available():
            return HealthCheck(
                name="llm_service",
                status=HealthStatus.HEALTHY,
                message="LLM service is available",
                timestamp=datetime.now(),
                duration_ms=0
            )
        else:
            return HealthCheck(
                name="llm_service",
                status=HealthStatus.WARNING,
                message="LLM service is not available",
                timestamp=datetime.now(),
                duration_ms=0
            )
            
    except Exception as e:
        return HealthCheck(
            name="llm_service",
            status=HealthStatus.CRITICAL,
            message=f"LLM service check failed: {str(e)}",
            timestamp=datetime.now(),
            duration_ms=0
        )


def disk_space_health_check() -> HealthCheck:
    """Check available disk space"""
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        free_percent = (free / total) * 100
        
        if free_percent > 20:
            status = HealthStatus.HEALTHY
            message = f"Disk space OK: {free_percent:.1f}% free"
        elif free_percent > 10:
            status = HealthStatus.WARNING
            message = f"Disk space low: {free_percent:.1f}% free"
        else:
            status = HealthStatus.CRITICAL
            message = f"Disk space critical: {free_percent:.1f}% free"
        
        return HealthCheck(
            name="disk_space",
            status=status,
            message=message,
            timestamp=datetime.now(),
            duration_ms=0,
            metadata={
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "free_percent": free_percent
            }
        )
        
    except Exception as e:
        return HealthCheck(
            name="disk_space",
            status=HealthStatus.UNKNOWN,
            message=f"Disk space check failed: {str(e)}",
            timestamp=datetime.now(),
            duration_ms=0
        )


# Global system monitor instance
system_monitor = SystemMonitor()

# Register default health checks
system_monitor.register_health_check("database", database_health_check, 60)
system_monitor.register_health_check("ocr_engines", ocr_engines_health_check, 300)
system_monitor.register_health_check("llm_service", llm_service_health_check, 120)
system_monitor.register_health_check("disk_space", disk_space_health_check, 180)


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor"""
    return system_monitor


def start_system_monitoring():
    """Start system monitoring"""
    system_monitor.start_monitoring()


def stop_system_monitoring():
    """Stop system monitoring"""
    system_monitor.stop_monitoring()


def get_health_status() -> Dict[str, Any]:
    """Get current system health status"""
    return system_monitor.get_system_status()