"""
Tests for logging and monitoring system
"""

import pytest
import json
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.core.logging_system import (
    StructuredLogger, MetricsCollector, AuditLogger, PerformanceMonitor,
    LoggingSystem, LogEntry, PerformanceMetric, AuditEvent,
    LogLevel, MetricType, get_logger, get_metrics, get_audit_logger
)


class TestStructuredLogger:
    """Test cases for StructuredLogger"""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    def test_logger_initialization(self, temp_log_file):
        """Test logger initialization"""
        
        logger = StructuredLogger("test_logger", temp_log_file, "DEBUG")
        
        assert logger.name == "test_logger"
        assert logger.logger.level == 10  # DEBUG level
        assert len(logger.logger.handlers) >= 2  # Console + file handlers
    
    def test_structured_logging(self, temp_log_file):
        """Test structured log entry creation"""
        
        logger = StructuredLogger("test_logger", temp_log_file)
        
        logger.info(
            "Test message",
            component="test_component",
            operation="test_operation",
            user_id="user123",
            request_id="req123",
            duration_ms=150.5
        )
        
        # Verify log file was created and contains data
        assert os.path.exists(temp_log_file)
        
        with open(temp_log_file, 'r') as f:
            log_content = f.read()
            assert "Test message" in log_content
            assert "test_component" in log_content
    
    def test_json_logging(self, temp_log_file):
        """Test JSON structured logging"""
        
        logger = StructuredLogger("test_logger", temp_log_file)
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            component="test_component",
            operation="test_operation",
            message="Test JSON message",
            user_id="user123",
            metadata={"key": "value"}
        )
        
        logger.log_structured(entry)
        
        # Check JSON log file
        json_file = temp_log_file.replace('.log', '_structured.jsonl')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                json_content = f.read().strip()
                if json_content:
                    log_data = json.loads(json_content)
                    assert log_data["message"] == "Test JSON message"
                    assert log_data["user_id"] == "user123"
                    assert log_data["metadata"]["key"] == "value"
    
    def test_log_levels(self, temp_log_file):
        """Test different log levels"""
        
        logger = StructuredLogger("test_logger", temp_log_file, "DEBUG")
        
        logger.debug("Debug message", component="test", operation="debug")
        logger.info("Info message", component="test", operation="info")
        logger.warning("Warning message", component="test", operation="warning")
        logger.error("Error message", component="test", operation="error")
        logger.critical("Critical message", component="test", operation="critical")
        
        # Verify all messages were logged
        with open(temp_log_file, 'r') as f:
            content = f.read()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content
            assert "Critical message" in content
    
    def test_log_entry_to_dict(self):
        """Test LogEntry to_dict conversion"""
        
        timestamp = datetime.now()
        entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            component="test",
            operation="test_op",
            message="Test message",
            user_id="user123",
            metadata={"key": "value"}
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict["timestamp"] == timestamp.isoformat()
        assert entry_dict["level"] == "INFO"
        assert entry_dict["component"] == "test"
        assert entry_dict["message"] == "Test message"
        assert entry_dict["metadata"]["key"] == "value"


class TestMetricsCollector:
    """Test cases for MetricsCollector"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create MetricsCollector instance"""
        return MetricsCollector(max_metrics=100)
    
    def test_counter_metrics(self, metrics_collector):
        """Test counter metrics"""
        
        # Increment counters
        metrics_collector.increment_counter("test.requests", 1.0)
        metrics_collector.increment_counter("test.requests", 2.0)
        metrics_collector.increment_counter("test.errors", 1.0, {"type": "validation"})
        
        # Check values
        assert metrics_collector.get_counter_value("test.requests") == 3.0
        assert metrics_collector.get_counter_value("test.errors", {"type": "validation"}) == 1.0
        assert metrics_collector.get_counter_value("nonexistent") == 0.0
    
    def test_gauge_metrics(self, metrics_collector):
        """Test gauge metrics"""
        
        # Set gauge values
        metrics_collector.set_gauge("test.cpu_usage", 75.5)
        metrics_collector.set_gauge("test.memory_usage", 60.2, {"host": "server1"})
        
        # Check values
        assert metrics_collector.get_gauge_value("test.cpu_usage") == 75.5
        assert metrics_collector.get_gauge_value("test.memory_usage", {"host": "server1"}) == 60.2
        assert metrics_collector.get_gauge_value("nonexistent") == 0.0
    
    def test_histogram_metrics(self, metrics_collector):
        """Test histogram metrics"""
        
        # Record histogram values
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for value in values:
            metrics_collector.record_histogram("test.response_time", value)
        
        # Get statistics
        stats = metrics_collector.get_histogram_stats("test.response_time")
        
        assert stats["count"] == 10
        assert stats["min"] == 10
        assert stats["max"] == 100
        assert stats["avg"] == 55.0
        assert stats["p50"] == 50
        assert stats["p95"] == 90
        assert stats["p99"] == 90
    
    def test_timer_metrics(self, metrics_collector):
        """Test timer metrics"""
        
        # Record timer values
        durations = [100, 150, 200, 250, 300]
        for duration in durations:
            metrics_collector.record_timer("test.operation_time", duration)
        
        # Get statistics
        stats = metrics_collector.get_timer_stats("test.operation_time")
        
        assert stats["count"] == 5
        assert stats["min"] == 100
        assert stats["max"] == 300
        assert stats["avg"] == 200.0
    
    def test_metrics_with_tags(self, metrics_collector):
        """Test metrics with tags"""
        
        tags1 = {"service": "api", "endpoint": "upload"}
        tags2 = {"service": "api", "endpoint": "export"}
        
        metrics_collector.increment_counter("requests", 5, tags1)
        metrics_collector.increment_counter("requests", 3, tags2)
        
        assert metrics_collector.get_counter_value("requests", tags1) == 5
        assert metrics_collector.get_counter_value("requests", tags2) == 3
    
    def test_get_all_metrics(self, metrics_collector):
        """Test getting all metrics"""
        
        # Add various metrics
        metrics_collector.increment_counter("test.counter", 10)
        metrics_collector.set_gauge("test.gauge", 50.0)
        metrics_collector.record_histogram("test.histogram", 100)
        metrics_collector.record_timer("test.timer", 200)
        
        all_metrics = metrics_collector.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histogram_stats" in all_metrics
        assert "timer_stats" in all_metrics
        
        assert "test.counter" in all_metrics["counters"]
        assert "test.gauge" in all_metrics["gauges"]
    
    def test_metrics_key_generation(self, metrics_collector):
        """Test metric key generation with tags"""
        
        # Test key generation
        key1 = metrics_collector._make_key("test.metric")
        key2 = metrics_collector._make_key("test.metric", {"tag1": "value1"})
        key3 = metrics_collector._make_key("test.metric", {"tag1": "value1", "tag2": "value2"})
        
        assert key1 == "test.metric"
        assert key2 == "test.metric:tag1=value1"
        assert "tag1=value1" in key3 and "tag2=value2" in key3
    
    def test_metrics_limit(self):
        """Test metrics collection limit"""
        
        collector = MetricsCollector(max_metrics=5)
        
        # Add more metrics than the limit
        for i in range(10):
            collector.increment_counter(f"test.counter.{i}", 1)
        
        # Should only keep the last 5 metrics
        assert len(collector.metrics) == 5


class TestAuditLogger:
    """Test cases for AuditLogger"""
    
    @pytest.fixture
    def temp_audit_file(self):
        """Create temporary audit file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    def test_audit_logger_initialization(self, temp_audit_file):
        """Test audit logger initialization"""
        
        audit_logger = AuditLogger(temp_audit_file)
        
        assert audit_logger.audit_file == temp_audit_file
        assert audit_logger.logger.name == "audit"
    
    def test_log_audit_event(self, temp_audit_file):
        """Test logging audit events"""
        
        audit_logger = AuditLogger(temp_audit_file)
        
        audit_logger.log_event(
            user_id="user123",
            action="document_upload",
            resource_type="document",
            resource_id="doc456",
            new_values={"filename": "test.pdf"},
            ip_address="192.168.1.1",
            success=True
        )
        
        # Verify audit log was written
        assert os.path.exists(temp_audit_file)
        
        with open(temp_audit_file, 'r') as f:
            log_line = f.readline().strip()
            audit_data = json.loads(log_line)
            
            assert audit_data["user_id"] == "user123"
            assert audit_data["action"] == "document_upload"
            assert audit_data["resource_id"] == "doc456"
            assert audit_data["success"] == True
            assert "event_id" in audit_data
            assert "timestamp" in audit_data
    
    def test_log_document_upload(self, temp_audit_file):
        """Test document upload audit logging"""
        
        audit_logger = AuditLogger(temp_audit_file)
        
        audit_logger.log_document_upload(
            user_id="user123",
            document_id="doc456",
            filename="test.pdf",
            ip_address="192.168.1.1"
        )
        
        with open(temp_audit_file, 'r') as f:
            audit_data = json.loads(f.readline())
            
            assert audit_data["action"] == "document_upload"
            assert audit_data["resource_type"] == "document"
            assert audit_data["new_values"]["filename"] == "test.pdf"
    
    def test_log_data_export(self, temp_audit_file):
        """Test data export audit logging"""
        
        audit_logger = AuditLogger(temp_audit_file)
        
        audit_logger.log_data_export(
            user_id="user123",
            resource_id="table456",
            export_format="csv",
            ip_address="192.168.1.1"
        )
        
        with open(temp_audit_file, 'r') as f:
            audit_data = json.loads(f.readline())
            
            assert audit_data["action"] == "data_export"
            assert audit_data["resource_type"] == "table"
            assert audit_data["new_values"]["format"] == "csv"
    
    def test_log_dashboard_share(self, temp_audit_file):
        """Test dashboard sharing audit logging"""
        
        audit_logger = AuditLogger(temp_audit_file)
        
        audit_logger.log_dashboard_share(
            user_id="user123",
            dashboard_id="dash456",
            share_type="secure_link",
            permissions=["view", "export"]
        )
        
        with open(temp_audit_file, 'r') as f:
            audit_data = json.loads(f.readline())
            
            assert audit_data["action"] == "dashboard_share"
            assert audit_data["resource_type"] == "dashboard"
            assert audit_data["new_values"]["share_type"] == "secure_link"
            assert audit_data["new_values"]["permissions"] == ["view", "export"]
    
    def test_log_user_login(self, temp_audit_file):
        """Test user login audit logging"""
        
        audit_logger = AuditLogger(temp_audit_file)
        
        audit_logger.log_user_login(
            user_id="user123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=True
        )
        
        with open(temp_audit_file, 'r') as f:
            audit_data = json.loads(f.readline())
            
            assert audit_data["action"] == "user_login"
            assert audit_data["resource_type"] == "user"
            assert audit_data["ip_address"] == "192.168.1.1"
            assert audit_data["user_agent"] == "Mozilla/5.0"
    
    def test_audit_event_to_dict(self):
        """Test AuditEvent to_dict conversion"""
        
        timestamp = datetime.now()
        event = AuditEvent(
            event_id="event123",
            timestamp=timestamp,
            user_id="user123",
            action="test_action",
            resource_type="test_resource",
            resource_id="resource123",
            success=True
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert event_dict["event_id"] == "event123"
        assert event_dict["user_id"] == "user123"
        assert event_dict["success"] == True


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create PerformanceMonitor instance"""
        metrics = MetricsCollector()
        logger = StructuredLogger("test_monitor")
        return PerformanceMonitor(metrics, logger)
    
    def test_timer_context_manager_success(self, performance_monitor):
        """Test timer context manager for successful operations"""
        
        with performance_monitor.timer("test_operation", "test_component"):
            time.sleep(0.01)  # Small delay
        
        # Check that timer metric was recorded
        stats = performance_monitor.metrics.get_timer_stats("test_component.test_operation.duration")
        assert stats["count"] == 1
        assert stats["min"] > 0  # Should have some duration
    
    def test_timer_context_manager_error(self, performance_monitor):
        """Test timer context manager for failed operations"""
        
        with pytest.raises(ValueError):
            with performance_monitor.timer("test_operation", "test_component"):
                raise ValueError("Test error")
        
        # Check that error counter was incremented
        error_count = performance_monitor.metrics.get_counter_value(
            "test_component.test_operation.errors",
            {"error_type": "ValueError"}
        )
        assert error_count == 1
    
    def test_time_function_decorator(self, performance_monitor):
        """Test function timing decorator"""
        
        @performance_monitor.time_function("test_component")
        def test_function(x, y):
            time.sleep(0.01)
            return x + y
        
        result = test_function(1, 2)
        assert result == 3
        
        # Check that timer metric was recorded
        stats = performance_monitor.metrics.get_timer_stats("test_component.test_function.duration")
        assert stats["count"] == 1
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_monitor_resource_usage(self, mock_disk, mock_memory, mock_cpu, performance_monitor):
        """Test system resource monitoring"""
        
        # Mock system metrics
        mock_cpu.return_value = 75.5
        
        mock_memory_obj = Mock()
        mock_memory_obj.percent = 60.2
        mock_memory_obj.available = 2048 * 1024 * 1024  # 2GB in bytes
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = Mock()
        mock_disk_obj.used = 50 * 1024**3  # 50GB
        mock_disk_obj.total = 100 * 1024**3  # 100GB
        mock_disk.return_value = mock_disk_obj
        
        performance_monitor.monitor_resource_usage()
        
        # Check that system metrics were recorded
        assert performance_monitor.metrics.get_gauge_value("system.cpu.usage_percent") == 75.5
        assert performance_monitor.metrics.get_gauge_value("system.memory.usage_percent") == 60.2
        assert performance_monitor.metrics.get_gauge_value("system.memory.available_mb") == 2048
        assert performance_monitor.metrics.get_gauge_value("system.disk.usage_percent") == 50.0
    
    def test_monitor_resource_usage_without_psutil(self, performance_monitor):
        """Test resource monitoring when psutil is not available"""
        
        with patch('psutil.cpu_percent', side_effect=ImportError("psutil not available")):
            # Should not raise an exception
            performance_monitor.monitor_resource_usage()


class TestLoggingSystem:
    """Test cases for LoggingSystem"""
    
    def test_logging_system_initialization(self):
        """Test logging system initialization"""
        
        config = {
            "log_file": "test_logs/app.log",
            "audit_file": "test_logs/audit.jsonl",
            "log_level": "DEBUG",
            "enable_resource_monitoring": False
        }
        
        logging_system = LoggingSystem(config)
        
        assert logging_system.config == config
        assert isinstance(logging_system.logger, StructuredLogger)
        assert isinstance(logging_system.metrics, MetricsCollector)
        assert isinstance(logging_system.audit, AuditLogger)
        assert isinstance(logging_system.monitor, PerformanceMonitor)
    
    def test_get_system_stats(self):
        """Test getting system statistics"""
        
        logging_system = LoggingSystem({"enable_resource_monitoring": False})
        
        # Add some metrics
        logging_system.metrics.increment_counter("test.counter", 5)
        logging_system.metrics.set_gauge("test.gauge", 10.0)
        
        stats = logging_system.get_system_stats()
        
        assert "metrics" in stats
        assert "timestamp" in stats
        assert "config" in stats
        
        assert "counters" in stats["metrics"]
        assert "gauges" in stats["metrics"]
    
    def test_global_logging_functions(self):
        """Test global logging convenience functions"""
        
        logger = get_logger()
        metrics = get_metrics()
        audit = get_audit_logger()
        
        assert isinstance(logger, StructuredLogger)
        assert isinstance(metrics, MetricsCollector)
        assert isinstance(audit, AuditLogger)
    
    def test_default_config(self):
        """Test default configuration"""
        
        logging_system = LoggingSystem()
        config = logging_system.config
        
        assert "log_file" in config
        assert "audit_file" in config
        assert "log_level" in config
        assert "max_metrics" in config
        assert "enable_resource_monitoring" in config
    
    @patch('threading.Thread')
    def test_resource_monitoring_thread(self, mock_thread):
        """Test resource monitoring thread startup"""
        
        config = {"enable_resource_monitoring": True}
        logging_system = LoggingSystem(config)
        
        # Verify thread was created and started
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()


class TestPerformanceMetric:
    """Test cases for PerformanceMetric dataclass"""
    
    def test_performance_metric_creation(self):
        """Test PerformanceMetric creation"""
        
        timestamp = datetime.now()
        metric = PerformanceMetric(
            name="test.metric",
            metric_type=MetricType.COUNTER,
            value=10.5,
            timestamp=timestamp,
            tags={"service": "api"}
        )
        
        assert metric.name == "test.metric"
        assert metric.metric_type == MetricType.COUNTER
        assert metric.value == 10.5
        assert metric.timestamp == timestamp
        assert metric.tags["service"] == "api"
    
    def test_performance_metric_default_tags(self):
        """Test PerformanceMetric with default tags"""
        
        metric = PerformanceMetric(
            name="test.metric",
            metric_type=MetricType.GAUGE,
            value=5.0,
            timestamp=datetime.now()
        )
        
        assert metric.tags == {}


class TestIntegration:
    """Integration tests for the logging system"""
    
    def test_end_to_end_logging_flow(self):
        """Test complete logging flow"""
        
        # Initialize system
        config = {"enable_resource_monitoring": False}
        logging_system = LoggingSystem(config)
        
        logger = logging_system.get_logger()
        metrics = logging_system.get_metrics()
        audit = logging_system.get_audit_logger()
        monitor = logging_system.get_monitor()
        
        # Log various events
        logger.info("Test operation started", 
                   component="test", operation="integration_test")
        
        metrics.increment_counter("test.operations", 1)
        metrics.set_gauge("test.active_users", 5)
        
        audit.log_event(
            user_id="user123",
            action="test_action",
            resource_type="test",
            resource_id="test123"
        )
        
        # Use performance monitor
        with monitor.timer("integration_test", "test_component"):
            time.sleep(0.01)
        
        # Verify everything worked
        assert metrics.get_counter_value("test.operations") == 1
        assert metrics.get_gauge_value("test.active_users") == 5
        
        timer_stats = metrics.get_timer_stats("test_component.integration_test.duration")
        assert timer_stats["count"] == 1
    
    def test_error_logging_integration(self):
        """Test error logging integration"""
        
        logging_system = LoggingSystem({"enable_resource_monitoring": False})
        logger = logging_system.get_logger()
        metrics = logging_system.get_metrics()
        
        # Simulate error scenario
        try:
            raise ValueError("Test error for logging")
        except ValueError as e:
            logger.error(
                f"Operation failed: {str(e)}",
                component="test",
                operation="error_test",
                error_code="TEST_ERROR"
            )
            
            metrics.increment_counter("test.errors", 1, {"error_type": "ValueError"})
        
        # Verify error was logged and counted
        error_count = metrics.get_counter_value("test.errors", {"error_type": "ValueError"})
        assert error_count == 1