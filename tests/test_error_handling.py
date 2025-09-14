"""
Tests for error handling framework
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.core.error_handler import (
    ErrorHandler, ErrorContext, ErrorResponse, ErrorSeverity, 
    RecoveryAction, handle_error, create_error_context
)
from src.core.exceptions import (
    OCRError, TableExtractionError, DataProcessingError,
    ValidationError, ExportError, AuthenticationError,
    AuthorizationError, RateLimitError, SharingError
)


class TestErrorHandler:
    """Test cases for ErrorHandler"""
    
    @pytest.fixture
    def error_handler(self):
        """Create ErrorHandler instance"""
        return ErrorHandler()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample error context"""
        return ErrorContext(
            operation="test_operation",
            component="test_component",
            user_id="user123",
            document_id="doc123",
            request_id="req123"
        )
    
    def test_handle_ocr_low_confidence_error(self, error_handler, sample_context):
        """Test handling OCR low confidence error"""
        
        error = OCRError("Low confidence score: 0.3")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_type == "OCRError"
        assert response.error_code == "OCR_LOW_CONFIDENCE"
        assert response.severity == ErrorSeverity.MEDIUM
        assert response.recovery_action == RecoveryAction.RETRY_WITH_DIFFERENT_ENGINE
        assert "higher quality scan" in response.suggestions[0]
        assert response.context == sample_context
    
    def test_handle_ocr_no_text_error(self, error_handler, sample_context):
        """Test handling OCR no text detected error"""
        
        error = OCRError("No text detected in document")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "OCR_NO_TEXT"
        assert response.severity == ErrorSeverity.HIGH
        assert response.recovery_action == RecoveryAction.SUGGEST_ALTERNATIVE
        assert "Verify the document contains readable text" in response.suggestions[0]
    
    def test_handle_ocr_timeout_error(self, error_handler, sample_context):
        """Test handling OCR timeout error"""
        
        error = OCRError("OCR processing timeout after 300 seconds")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "OCR_TIMEOUT"
        assert response.severity == ErrorSeverity.MEDIUM
        assert response.recovery_action == RecoveryAction.RETRY
        assert response.retry_after == 60
        assert "smaller document" in response.suggestions[0]
    
    def test_handle_table_extraction_no_tables(self, error_handler, sample_context):
        """Test handling no tables found error"""
        
        error = TableExtractionError("No tables found in document")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "NO_TABLES_FOUND"
        assert response.severity == ErrorSeverity.MEDIUM
        assert response.recovery_action == RecoveryAction.ENABLE_MANUAL_SELECTION
        assert "Verify the document contains table structures" in response.suggestions[0]
    
    def test_handle_table_extraction_malformed(self, error_handler, sample_context):
        """Test handling malformed table error"""
        
        error = TableExtractionError("Malformed table structure detected")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "MALFORMED_TABLE"
        assert response.recovery_action == RecoveryAction.PROVIDE_CORRECTION_TOOLS
        assert "correction interface" in response.suggestions[0]
    
    def test_handle_data_processing_error(self, error_handler, sample_context):
        """Test handling data processing error"""
        
        error = DataProcessingError("Failed to process extracted data")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_type == "DataProcessingError"
        assert response.error_code == "DATA_PROCESSING_FAILED"
        assert response.severity == ErrorSeverity.MEDIUM
        assert response.recovery_action == RecoveryAction.LOG_AND_NOTIFY
    
    def test_handle_validation_error(self, error_handler, sample_context):
        """Test handling validation error"""
        
        error = ValidationError("Invalid data format")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "VALIDATION_FAILED"
        assert response.severity == ErrorSeverity.LOW
        assert response.recovery_action == RecoveryAction.LOG_AND_CONTINUE
        assert "input data format" in response.suggestions[0]
    
    def test_handle_export_error(self, error_handler, sample_context):
        """Test handling export error"""
        
        error = ExportError("Failed to export to PDF format")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "EXPORT_FAILED"
        assert response.severity == ErrorSeverity.MEDIUM
        assert response.recovery_action == RecoveryAction.SUGGEST_ALTERNATIVE
        assert "different export format" in response.suggestions[0]
    
    def test_handle_authentication_error(self, error_handler, sample_context):
        """Test handling authentication error"""
        
        error = AuthenticationError("Invalid credentials")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "AUTH_FAILED"
        assert response.severity == ErrorSeverity.HIGH
        assert response.recovery_action == RecoveryAction.ABORT_OPERATION
        assert "username and password" in response.suggestions[0]
    
    def test_handle_authorization_error(self, error_handler, sample_context):
        """Test handling authorization error"""
        
        error = AuthorizationError("Access denied to resource")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "ACCESS_DENIED"
        assert response.severity == ErrorSeverity.HIGH
        assert response.recovery_action == RecoveryAction.ABORT_OPERATION
        assert "required permissions" in response.suggestions[0]
    
    def test_handle_rate_limit_error(self, error_handler, sample_context):
        """Test handling rate limit error"""
        
        error = RateLimitError("Rate limit exceeded")
        error.retry_after = 120
        
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "RATE_LIMIT_EXCEEDED"
        assert response.severity == ErrorSeverity.MEDIUM
        assert response.recovery_action == RecoveryAction.RETRY
        assert response.retry_after == 120
        assert "120 seconds" in response.suggestions[0]
    
    def test_handle_sharing_error(self, error_handler, sample_context):
        """Test handling sharing error"""
        
        error = SharingError("Failed to create share link")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_code == "SHARING_FAILED"
        assert response.severity == ErrorSeverity.MEDIUM
        assert response.recovery_action == RecoveryAction.LOG_AND_NOTIFY
        assert "resource exists" in response.suggestions[0]
    
    def test_handle_generic_error(self, error_handler, sample_context):
        """Test handling unknown/generic error"""
        
        error = ValueError("Some unexpected error")
        response = error_handler.handle_error(error, sample_context)
        
        assert response.error_type == "ValueError"
        assert response.error_code == "UNKNOWN_ERROR"
        assert response.severity == ErrorSeverity.HIGH
        assert response.recovery_action == RecoveryAction.LOG_AND_NOTIFY
        assert "unexpected error occurred" in response.user_message
    
    def test_error_response_to_dict(self, error_handler, sample_context):
        """Test ErrorResponse to_dict conversion"""
        
        error = ValidationError("Test error")
        response = error_handler.handle_error(error, sample_context)
        
        response_dict = response.to_dict()
        
        assert "error" in response_dict
        assert "code" in response_dict
        assert "message" in response_dict
        assert "severity" in response_dict
        assert "suggestions" in response_dict
        assert "timestamp" in response_dict
        assert "request_id" in response_dict
        
        assert response_dict["request_id"] == "req123"
    
    def test_include_technical_details(self, error_handler, sample_context):
        """Test including technical details in error response"""
        
        error = ValueError("Test error with details")
        response = error_handler.handle_error(error, sample_context, include_technical_details=True)
        
        assert response.technical_details is not None
        assert "exception_type" in response.technical_details
        assert "exception_message" in response.technical_details
        assert "traceback" in response.technical_details
    
    def test_error_statistics(self, error_handler, sample_context):
        """Test error statistics collection"""
        
        # Generate some errors
        error1 = OCRError("Error 1")
        error2 = ValidationError("Error 2")
        error3 = OCRError("Error 3")
        
        error_handler.handle_error(error1, sample_context)
        error_handler.handle_error(error2, sample_context)
        error_handler.handle_error(error3, sample_context)
        
        stats = error_handler.get_error_statistics()
        
        assert "error_counts" in stats
        assert "total_errors" in stats
        assert "most_common_errors" in stats
        
        assert stats["total_errors"] == 3
        assert "test_component:OCRError" in stats["error_counts"]
        assert stats["error_counts"]["test_component:OCRError"] == 2
    
    @patch('src.core.error_handler.logging.getLogger')
    def test_error_logging(self, mock_get_logger, error_handler, sample_context):
        """Test error logging functionality"""
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        error = OCRError("Test error for logging")
        error_handler.handle_error(error, sample_context)
        
        # Verify logger was called
        mock_logger.warning.assert_called_once()
    
    def test_create_error_context_convenience_function(self):
        """Test create_error_context convenience function"""
        
        context = create_error_context(
            operation="test_op",
            component="test_comp",
            user_id="user123",
            custom_field="custom_value"
        )
        
        assert context.operation == "test_op"
        assert context.component == "test_comp"
        assert context.user_id == "user123"
        assert context.additional_data["custom_field"] == "custom_value"
        assert isinstance(context.timestamp, datetime)
    
    def test_handle_error_convenience_function(self, sample_context):
        """Test handle_error convenience function"""
        
        error = ValidationError("Test validation error")
        response = handle_error(error, sample_context)
        
        assert isinstance(response, ErrorResponse)
        assert response.error_type == "ValidationError"
    
    def test_recovery_strategy_execution(self, error_handler):
        """Test recovery strategy execution"""
        
        context = ErrorContext(operation="test", component="test")
        
        # Test retry strategy
        result = error_handler._execute_retry(context)
        assert result["action"] == "retry"
        assert "delay" in result
        
        # Test retry with different engine
        result = error_handler._execute_retry_different_engine(context)
        assert result["action"] == "retry"
        assert result["change_engine"] == True
        
        # Test manual fallback
        result = error_handler._execute_manual_fallback(context)
        assert result["action"] == "manual_fallback"
        assert "ui_component" in result
    
    def test_error_context_post_init(self):
        """Test ErrorContext post-initialization"""
        
        context = ErrorContext(
            operation="test",
            component="test"
        )
        
        assert context.timestamp is not None
        assert isinstance(context.timestamp, datetime)
        assert context.additional_data == {}
        
        # Test with provided values
        timestamp = datetime.now()
        additional_data = {"key": "value"}
        
        context2 = ErrorContext(
            operation="test",
            component="test",
            timestamp=timestamp,
            additional_data=additional_data
        )
        
        assert context2.timestamp == timestamp
        assert context2.additional_data == additional_data
    
    def test_error_severity_levels(self, error_handler, sample_context):
        """Test different error severity levels"""
        
        # Low severity
        error1 = ValidationError("Low severity error")
        response1 = error_handler.handle_error(error1, sample_context)
        assert response1.severity == ErrorSeverity.LOW
        
        # Medium severity
        error2 = OCRError("Medium severity error")
        response2 = error_handler.handle_error(error2, sample_context)
        assert response2.severity == ErrorSeverity.MEDIUM
        
        # High severity
        error3 = AuthenticationError("High severity error")
        response3 = error_handler.handle_error(error3, sample_context)
        assert response3.severity == ErrorSeverity.HIGH
    
    def test_error_message_customization(self, error_handler):
        """Test error message customization"""
        
        # Verify error messages are user-friendly
        messages = error_handler.error_messages
        
        assert "OCR_LOW_CONFIDENCE" in messages
        assert "document quality" in messages["OCR_LOW_CONFIDENCE"].lower()
        
        assert "NO_TABLES_FOUND" in messages
        assert "tables were detected" in messages["NO_TABLES_FOUND"].lower()
        
        assert "AUTH_FAILED" in messages
        assert "credentials" in messages["AUTH_FAILED"].lower()
    
    def test_multiple_error_handling(self, error_handler, sample_context):
        """Test handling multiple errors in sequence"""
        
        errors = [
            OCRError("First error"),
            ValidationError("Second error"),
            ExportError("Third error")
        ]
        
        responses = []
        for error in errors:
            response = error_handler.handle_error(error, sample_context)
            responses.append(response)
        
        assert len(responses) == 3
        assert responses[0].error_type == "OCRError"
        assert responses[1].error_type == "ValidationError"
        assert responses[2].error_type == "ExportError"
        
        # Check statistics updated correctly
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 3