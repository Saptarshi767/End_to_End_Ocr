# Error Handling and Logging System

This document describes the comprehensive error handling framework and logging system implemented for the OCR Table Analytics project.

## 🚀 Features Implemented

### 1. Centralized Error Handling Framework (`src/core/error_handler.py`)

A sophisticated error handling system that provides consistent error management across all components.

#### Key Features
- **Custom Exception Types**: Specific exceptions for different error scenarios
- **Recovery Strategies**: Automated recovery suggestions for each error type
- **User-Friendly Messages**: Clear, actionable error messages for end users
- **Error Statistics**: Tracking and analysis of error patterns
- **Contextual Information**: Rich context data for debugging and analysis

#### Error Types Handled
- **OCR Errors**: Low confidence, no text detected, timeouts
- **Table Extraction Errors**: No tables found, malformed structures
- **Data Processing Errors**: Data cleaning and validation failures
- **Authentication/Authorization Errors**: Access control issues
- **Rate Limiting Errors**: API usage limits exceeded
- **Export Errors**: Data export failures
- **Sharing Errors**: Dashboard sharing issues

#### Usage Examples

```python
from src.core.error_handler import handle_error, create_error_context
from src.core.exceptions import OCRError

# Create error context
context = create_error_context(
    operation="document_processing",
    component="ocr_engine",
    user_id="user123",
    document_id="doc456"
)

# Handle error with recovery suggestions
try:
    # Some OCR operation
    pass
except OCRError as e:
    error_response = handle_error(e, context)
    
    # Error response contains:
    # - User-friendly message
    # - Recovery suggestions
    # - Severity level
    # - Technical details (optional)
    print(error_response.user_message)
    print(error_response.suggestions)
```

#### Error Response Structure

```json
{
    "error": "OCRError",
    "code": "OCR_LOW_CONFIDENCE", 
    "message": "The document quality is too low for accurate text recognition.",
    "severity": "medium",
    "suggestions": [
        "Try uploading a higher quality scan",
        "Ensure the document is well-lit and in focus",
        "Consider using a different OCR engine"
    ],
    "retry_after": null,
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456"
}
```

### 2. Comprehensive Logging System (`src/core/logging_system.py`)

A multi-faceted logging system providing structured logging, performance monitoring, and audit trails.

#### Components

##### Structured Logger
- **JSON Format**: Machine-readable structured logs
- **Multiple Handlers**: Console, file, and rotating file handlers
- **Contextual Data**: Rich metadata with each log entry
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

##### Metrics Collector
- **Counter Metrics**: Increment-only values (requests, errors)
- **Gauge Metrics**: Current state values (CPU usage, active users)
- **Histogram Metrics**: Distribution of values (response times)
- **Timer Metrics**: Duration measurements with statistics

##### Audit Logger
- **Compliance Logging**: Detailed audit trails for security
- **User Actions**: Track all user operations
- **Data Changes**: Before/after values for modifications
- **Access Tracking**: IP addresses, user agents, timestamps

##### Performance Monitor
- **Timing Decorators**: Automatic function timing
- **Context Managers**: Easy operation timing
- **Resource Monitoring**: CPU, memory, disk usage
- **Error Tracking**: Automatic error counting and categorization

#### Usage Examples

```python
from src.core.logging_system import get_logger, get_metrics, get_audit_logger, get_monitor

# Structured logging
logger = get_logger()
logger.info(
    "Document processing completed",
    component="document_processor",
    operation="extract_tables",
    user_id="user123",
    document_id="doc456",
    duration_ms=1500,
    metadata={"tables_found": 3}
)

# Metrics collection
metrics = get_metrics()
metrics.increment_counter("documents.processed", 1, {"format": "pdf"})
metrics.set_gauge("active_users", 25)
metrics.record_timer("ocr.processing_time", 2500)

# Audit logging
audit = get_audit_logger()
audit.log_document_upload(
    user_id="user123",
    document_id="doc456", 
    filename="report.pdf",
    ip_address="192.168.1.100"
)

# Performance monitoring
monitor = get_monitor()

# Using context manager
with monitor.timer("table_extraction", "ocr_engine", user_id="user123"):
    # Extract tables from document
    pass

# Using decorator
@monitor.time_function("data_processing")
def clean_data(dataframe):
    # Data cleaning operations
    return cleaned_dataframe
```

### 3. System Monitoring (`src/core/monitoring.py`)

Comprehensive system health monitoring with automated checks and status reporting.

#### Health Checks
- **Database Connectivity**: Connection and query testing
- **OCR Engine Availability**: Engine status and functionality
- **LLM Service Status**: AI service availability
- **Disk Space Monitoring**: Storage capacity tracking
- **Custom Health Checks**: Extensible framework for additional checks

#### Monitoring Features
- **Background Monitoring**: Continuous health check execution
- **Status Aggregation**: Overall system health determination
- **Alerting Integration**: Status change notifications
- **Performance Tracking**: Health check duration and success rates

#### Usage Examples

```python
from src.core.monitoring import get_system_monitor, get_health_status

# Get current system status
status = get_health_status()
print(f"Overall Status: {status['overall_status']}")

# Register custom health check
monitor = get_system_monitor()

def custom_service_check():
    # Check external service
    return HealthCheck(
        name="external_api",
        status=HealthStatus.HEALTHY,
        message="Service is responding",
        timestamp=datetime.now(),
        duration_ms=150
    )

monitor.register_health_check("external_api", custom_service_check, interval=120)
```

## 🧪 Testing

Comprehensive test suites ensure reliability and accuracy of the error handling and logging systems.

### Test Coverage

#### Error Handling Tests (`tests/test_error_handling.py`)
- **Error Type Handling**: Verify correct handling of each error type
- **Recovery Strategies**: Test recovery action generation
- **Error Statistics**: Validate error counting and analysis
- **Context Management**: Test error context creation and usage
- **Message Generation**: Verify user-friendly message creation

#### Logging System Tests (`tests/test_logging_system.py`)
- **Structured Logging**: Test log entry creation and formatting
- **Metrics Collection**: Validate counter, gauge, histogram, and timer metrics
- **Audit Logging**: Test audit trail generation and storage
- **Performance Monitoring**: Test timing and resource monitoring
- **Integration Testing**: End-to-end logging workflow validation

### Running Tests

```bash
# Run error handling tests
pytest tests/test_error_handling.py -v

# Run logging system tests  
pytest tests/test_logging_system.py -v

# Run all error handling and logging tests
pytest tests/test_error_handling.py tests/test_logging_system.py -v

# Generate coverage report
pytest --cov=src.core.error_handler --cov=src.core.logging_system --cov-report=html
```

## 📊 Monitoring and Analytics

### Error Analytics
- **Error Frequency**: Track most common errors by type and component
- **Error Trends**: Monitor error rates over time
- **Recovery Success**: Track effectiveness of recovery strategies
- **User Impact**: Analyze errors by user and operation type

### Performance Metrics
- **Response Times**: API endpoint and operation duration tracking
- **Throughput**: Request rates and processing capacity
- **Resource Usage**: CPU, memory, and disk utilization
- **Success Rates**: Operation success vs. failure ratios

### Audit Analytics
- **User Activity**: Track user actions and access patterns
- **Data Access**: Monitor data export and sharing activities
- **Security Events**: Authentication and authorization tracking
- **Compliance Reporting**: Generate audit reports for compliance

## 🔧 Configuration

### Environment Variables

```bash
# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/application.log
AUDIT_FILE=logs/audit.jsonl
ENABLE_JSON_LOGGING=true

# Monitoring Configuration
ENABLE_RESOURCE_MONITORING=true
RESOURCE_MONITORING_INTERVAL=60
HEALTH_CHECK_INTERVAL=60

# Error Handling Configuration
INCLUDE_TECHNICAL_DETAILS=false
ERROR_STATISTICS_RETENTION=7d
MAX_ERROR_HISTORY=10000

# Metrics Configuration
MAX_METRICS_HISTORY=10000
METRICS_RETENTION_DAYS=30
ENABLE_SYSTEM_METRICS=true
```

### Logging Configuration

```python
# Custom logging configuration
logging_config = {
    "log_file": "logs/custom_app.log",
    "audit_file": "logs/custom_audit.jsonl", 
    "log_level": "DEBUG",
    "max_metrics": 50000,
    "enable_resource_monitoring": True,
    "resource_monitoring_interval": 30
}

from src.core.logging_system import LoggingSystem
logging_system = LoggingSystem(logging_config)
```

## 🚀 Integration with API

The error handling and logging systems are fully integrated with the FastAPI application:

### Middleware Integration
- **Global Error Handling**: Automatic error catching and response formatting
- **Request Logging**: All API requests logged with context
- **Performance Tracking**: Automatic timing of all endpoints
- **Audit Logging**: User actions automatically audited

### Enhanced Health Endpoint
```bash
GET /health
```

Returns comprehensive system status including:
- Overall health status
- Individual component health checks
- Metrics summary
- Error statistics
- System information

### Error Response Format
All API errors return consistent format:
```json
{
    "error": "ValidationError",
    "code": "VALIDATION_FAILED", 
    "message": "The provided data failed validation checks.",
    "severity": "low",
    "suggestions": [
        "Check the input data format",
        "Ensure all required fields are provided"
    ],
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456"
}
```

## 📈 Metrics and Dashboards

### Key Metrics Tracked

#### API Metrics
- `api.requests.total` - Total API requests by method and path
- `api.requests.success` - Successful requests by status code
- `api.requests.errors` - Failed requests by error type
- `api.request.duration` - Request processing time distribution

#### System Metrics
- `system.cpu.usage_percent` - CPU utilization percentage
- `system.memory.usage_percent` - Memory utilization percentage
- `system.disk.usage_percent` - Disk space utilization
- `health_check.{name}.duration` - Health check execution time

#### Business Metrics
- `documents.uploaded` - Document upload count
- `tables.extracted` - Table extraction count
- `dashboards.generated` - Dashboard creation count
- `exports.completed` - Data export count

### Dashboard Integration
The metrics can be integrated with monitoring dashboards like:
- **Grafana**: For real-time monitoring and alerting
- **Prometheus**: For metrics collection and storage
- **DataDog**: For comprehensive application monitoring
- **New Relic**: For application performance monitoring

## 🔒 Security and Compliance

### Audit Trail Features
- **Immutable Logs**: Audit logs cannot be modified after creation
- **Comprehensive Tracking**: All user actions and data changes logged
- **Compliance Ready**: Meets requirements for SOX, GDPR, HIPAA
- **Retention Policies**: Configurable log retention periods

### Security Monitoring
- **Authentication Events**: Login attempts and failures
- **Authorization Violations**: Access denied events
- **Data Access Patterns**: Unusual data access detection
- **Rate Limiting Events**: API abuse detection

### Privacy Protection
- **PII Redaction**: Automatic removal of sensitive data from logs
- **Data Anonymization**: User data anonymization in analytics
- **Access Controls**: Role-based access to logs and metrics
- **Encryption**: Log data encrypted at rest and in transit

## 🎯 Benefits

### For Developers
- **Faster Debugging**: Rich error context and structured logs
- **Performance Insights**: Detailed metrics and timing data
- **Quality Assurance**: Comprehensive error tracking and analysis
- **Operational Visibility**: Real-time system health monitoring

### For Operations
- **Proactive Monitoring**: Early detection of system issues
- **Automated Alerting**: Notification of critical errors and outages
- **Capacity Planning**: Resource usage trends and forecasting
- **Incident Response**: Detailed logs for troubleshooting

### For Business
- **User Experience**: Better error messages and recovery guidance
- **Compliance**: Comprehensive audit trails for regulatory requirements
- **Analytics**: Business metrics and usage patterns
- **Reliability**: Improved system stability and uptime

## 🔄 Future Enhancements

### Planned Features
1. **Machine Learning**: Anomaly detection in error patterns
2. **Predictive Analytics**: Forecasting system issues before they occur
3. **Advanced Alerting**: Smart alerting based on error trends
4. **Custom Dashboards**: User-configurable monitoring dashboards
5. **Integration APIs**: Export metrics to external monitoring systems

### Extensibility
The system is designed for easy extension:
- **Custom Error Types**: Add new error categories and handlers
- **Additional Metrics**: Define business-specific metrics
- **Health Check Plugins**: Create custom health monitoring
- **Log Processors**: Add custom log analysis and enrichment

This comprehensive error handling and logging system provides the foundation for reliable, observable, and maintainable OCR Table Analytics operations, fulfilling requirements 8.1, 8.2, 8.3, 8.4, 8.5, and 8.6.