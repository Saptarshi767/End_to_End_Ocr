"""
Comprehensive error handling framework

Provides centralized error handling with custom exception types,
recovery strategies, and user-friendly error messages.
"""

import traceback
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from .exceptions import (
    OCRError, TableExtractionError, DataProcessingError, 
    ValidationError, ExportError, AuthenticationError,
    AuthorizationError, RateLimitError, SharingError
)
from .models import Document, Table, TableRegion, Query


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Available recovery actions"""
    RETRY = "retry"
    RETRY_WITH_DIFFERENT_ENGINE = "retry_with_different_engine"
    RETRY_WITH_PREPROCESSING = "retry_with_preprocessing"
    FALLBACK_TO_MANUAL = "fallback_to_manual"
    SUGGEST_ALTERNATIVE = "suggest_alternative"
    LOG_AND_CONTINUE = "log_and_continue"
    LOG_AND_NOTIFY = "log_and_notify"
    ABORT_OPERATION = "abort_operation"
    ENABLE_MANUAL_SELECTION = "enable_manual_selection"
    PROVIDE_CORRECTION_TOOLS = "provide_correction_tools"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    component: str
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    table_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = None
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.additional_data is None:
            self.additional_data = {}


@dataclass
class ErrorResponse:
    """Structured error response"""
    error_type: str
    error_code: str
    message: str
    user_message: str
    severity: ErrorSeverity
    recovery_action: RecoveryAction
    suggestions: List[str]
    context: ErrorContext
    technical_details: Optional[str] = None
    retry_after: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "error": self.error_type,
            "code": self.error_code,
            "message": self.user_message,
            "severity": self.severity.value,
            "suggestions": self.suggestions,
            "retry_after": self.retry_after,
            "timestamp": self.context.timestamp.isoformat(),
            "request_id": self.context.request_id
        }


class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.error_messages = self._initialize_error_messages()
    
    def handle_error(
        self, 
        error: Exception, 
        context: ErrorContext,
        include_technical_details: bool = False
    ) -> ErrorResponse:
        """
        Handle any error with appropriate recovery strategy
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            include_technical_details: Whether to include technical details
            
        Returns:
            Structured error response with recovery suggestions
        """
        
        # Determine error type and get handler
        error_type = type(error).__name__
        handler_method = getattr(self, f'_handle_{error_type.lower()}', self._handle_generic_error)
        
        # Get error response
        error_response = handler_method(error, context)
        
        # Add technical details if requested
        if include_technical_details:
            error_response.technical_details = self._get_technical_details(error)
        
        # Log the error
        self._log_error(error, error_response, context)
        
        # Update error statistics
        self._update_error_stats(error_type, context)
        
        return error_response
    
    def _handle_ocrerror(self, error: OCRError, context: ErrorContext) -> ErrorResponse:
        """Handle OCR-specific errors"""
        
        if "confidence" in str(error).lower():
            return ErrorResponse(
                error_type="OCRError",
                error_code="OCR_LOW_CONFIDENCE",
                message=str(error),
                user_message="The document quality is too low for accurate text recognition.",
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY_WITH_DIFFERENT_ENGINE,
                suggestions=[
                    "Try uploading a higher quality scan",
                    "Ensure the document is well-lit and in focus",
                    "Consider using a different OCR engine",
                    "Try preprocessing the image to improve quality"
                ],
                context=context
            )
        
        elif "no text detected" in str(error).lower():
            return ErrorResponse(
                error_type="OCRError",
                error_code="OCR_NO_TEXT",
                message=str(error),
                user_message="No text was detected in the document.",
                severity=ErrorSeverity.HIGH,
                recovery_action=RecoveryAction.SUGGEST_ALTERNATIVE,
                suggestions=[
                    "Verify the document contains readable text",
                    "Check if the document is rotated or upside down",
                    "Try a different file format",
                    "Manually enter the data if OCR continues to fail"
                ],
                context=context
            )
        
        elif "timeout" in str(error).lower():
            return ErrorResponse(
                error_type="OCRError",
                error_code="OCR_TIMEOUT",
                message=str(error),
                user_message="OCR processing took too long and was cancelled.",
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY,
                suggestions=[
                    "Try again with a smaller document",
                    "Split large documents into smaller sections",
                    "Use a faster OCR engine for large documents"
                ],
                context=context,
                retry_after=60
            )
        
        else:
            return self._handle_generic_ocr_error(error, context)
    
    def _handle_tableextractionerror(self, error: TableExtractionError, context: ErrorContext) -> ErrorResponse:
        """Handle table extraction errors"""
        
        if "no tables found" in str(error).lower():
            return ErrorResponse(
                error_type="TableExtractionError",
                error_code="NO_TABLES_FOUND",
                message=str(error),
                user_message="No tables were detected in the document.",
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.ENABLE_MANUAL_SELECTION,
                suggestions=[
                    "Verify the document contains table structures",
                    "Try adjusting table detection sensitivity",
                    "Manually select table regions if available",
                    "Check if tables are in a supported format"
                ],
                context=context
            )
        
        elif "malformed table" in str(error).lower():
            return ErrorResponse(
                error_type="TableExtractionError",
                error_code="MALFORMED_TABLE",
                message=str(error),
                user_message="The table structure could not be properly extracted.",
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.PROVIDE_CORRECTION_TOOLS,
                suggestions=[
                    "Use the table correction interface to fix issues",
                    "Verify table borders are clear and complete",
                    "Try preprocessing to enhance table structure",
                    "Consider manual table entry for complex layouts"
                ],
                context=context
            )
        
        else:
            return self._handle_generic_table_error(error, context)
    
    def _handle_dataprocessingerror(self, error: DataProcessingError, context: ErrorContext) -> ErrorResponse:
        """Handle data processing errors"""
        
        return ErrorResponse(
            error_type="DataProcessingError",
            error_code="DATA_PROCESSING_FAILED",
            message=str(error),
            user_message="There was an issue processing the extracted data.",
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.LOG_AND_NOTIFY,
            suggestions=[
                "Check data format and structure",
                "Verify all required fields are present",
                "Try processing with different settings",
                "Contact support if the issue persists"
            ],
            context=context
        )
    
    def _handle_validationerror(self, error: ValidationError, context: ErrorContext) -> ErrorResponse:
        """Handle validation errors"""
        
        return ErrorResponse(
            error_type="ValidationError",
            error_code="VALIDATION_FAILED",
            message=str(error),
            user_message="The provided data failed validation checks.",
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.LOG_AND_CONTINUE,
            suggestions=[
                "Check the input data format",
                "Ensure all required fields are provided",
                "Verify data types match expected formats",
                "Review the API documentation for correct usage"
            ],
            context=context
        )
    
    def _handle_exporterror(self, error: ExportError, context: ErrorContext) -> ErrorResponse:
        """Handle export errors"""
        
        return ErrorResponse(
            error_type="ExportError",
            error_code="EXPORT_FAILED",
            message=str(error),
            user_message="Failed to export data in the requested format.",
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.SUGGEST_ALTERNATIVE,
            suggestions=[
                "Try a different export format",
                "Check if the data is too large for the format",
                "Verify you have permission to export",
                "Try exporting smaller data subsets"
            ],
            context=context
        )
    
    def _handle_authenticationerror(self, error: AuthenticationError, context: ErrorContext) -> ErrorResponse:
        """Handle authentication errors"""
        
        return ErrorResponse(
            error_type="AuthenticationError",
            error_code="AUTH_FAILED",
            message=str(error),
            user_message="Authentication failed. Please check your credentials.",
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.ABORT_OPERATION,
            suggestions=[
                "Verify your username and password",
                "Check if your API key is valid and not expired",
                "Ensure you're using the correct authentication method",
                "Contact support if credentials should be valid"
            ],
            context=context
        )
    
    def _handle_authorizationerror(self, error: AuthorizationError, context: ErrorContext) -> ErrorResponse:
        """Handle authorization errors"""
        
        return ErrorResponse(
            error_type="AuthorizationError",
            error_code="ACCESS_DENIED",
            message=str(error),
            user_message="You don't have permission to perform this action.",
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.ABORT_OPERATION,
            suggestions=[
                "Check if you have the required permissions",
                "Verify you own the resource you're trying to access",
                "Contact the resource owner for access",
                "Review your account permissions with an administrator"
            ],
            context=context
        )
    
    def _handle_ratelimiterror(self, error: RateLimitError, context: ErrorContext) -> ErrorResponse:
        """Handle rate limit errors"""
        
        retry_after = getattr(error, 'retry_after', 60)
        
        return ErrorResponse(
            error_type="RateLimitError",
            error_code="RATE_LIMIT_EXCEEDED",
            message=str(error),
            user_message="You've exceeded the rate limit for this operation.",
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.RETRY,
            suggestions=[
                f"Wait {retry_after} seconds before trying again",
                "Consider upgrading your plan for higher limits",
                "Batch multiple operations to reduce request count",
                "Implement exponential backoff in your client"
            ],
            context=context,
            retry_after=retry_after
        )
    
    def _handle_sharingerror(self, error: SharingError, context: ErrorContext) -> ErrorResponse:
        """Handle sharing errors"""
        
        return ErrorResponse(
            error_type="SharingError",
            error_code="SHARING_FAILED",
            message=str(error),
            user_message="Failed to create or access the shared resource.",
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.LOG_AND_NOTIFY,
            suggestions=[
                "Check if the resource exists and is accessible",
                "Verify sharing permissions are correctly set",
                "Ensure the share link hasn't expired",
                "Try creating a new share link"
            ],
            context=context
        )
    
    def _handle_generic_error(self, error: Exception, context: ErrorContext) -> ErrorResponse:
        """Handle generic/unknown errors"""
        
        return ErrorResponse(
            error_type=type(error).__name__,
            error_code="UNKNOWN_ERROR",
            message=str(error),
            user_message="An unexpected error occurred. Please try again.",
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.LOG_AND_NOTIFY,
            suggestions=[
                "Try the operation again",
                "Check if the issue persists",
                "Contact support with the error details",
                "Try using different input parameters"
            ],
            context=context
        )
    
    def _handle_generic_ocr_error(self, error: OCRError, context: ErrorContext) -> ErrorResponse:
        """Handle generic OCR errors"""
        
        return ErrorResponse(
            error_type="OCRError",
            error_code="OCR_GENERIC_ERROR",
            message=str(error),
            user_message="OCR processing failed. Please try again with a different approach.",
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.RETRY_WITH_DIFFERENT_ENGINE,
            suggestions=[
                "Try a different OCR engine",
                "Improve document image quality",
                "Check document format compatibility",
                "Contact support if the issue persists"
            ],
            context=context
        )
    
    def _handle_generic_table_error(self, error: TableExtractionError, context: ErrorContext) -> ErrorResponse:
        """Handle generic table extraction errors"""
        
        return ErrorResponse(
            error_type="TableExtractionError",
            error_code="TABLE_EXTRACTION_ERROR",
            message=str(error),
            user_message="Table extraction failed. The document may have complex formatting.",
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.PROVIDE_CORRECTION_TOOLS,
            suggestions=[
                "Use manual table selection tools",
                "Try different table detection settings",
                "Ensure tables have clear borders",
                "Consider reformatting the source document"
            ],
            context=context
        )
    
    def _get_technical_details(self, error: Exception) -> str:
        """Get technical details for debugging"""
        
        return {
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "error_args": getattr(error, 'args', [])
        }
    
    def _log_error(self, error: Exception, error_response: ErrorResponse, context: ErrorContext):
        """Log error with appropriate level"""
        
        log_data = {
            "error_type": error_response.error_type,
            "error_code": error_response.error_code,
            "severity": error_response.severity.value,
            "operation": context.operation,
            "component": context.component,
            "user_id": context.user_id,
            "document_id": context.document_id,
            "request_id": context.request_id,
            "recovery_action": error_response.recovery_action.value
        }
        
        if error_response.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", extra=log_data, exc_info=error)
        elif error_response.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error", extra=log_data, exc_info=error)
        elif error_response.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error", extra=log_data)
        else:
            self.logger.info("Low severity error", extra=log_data)
    
    def _update_error_stats(self, error_type: str, context: ErrorContext):
        """Update error statistics for monitoring"""
        
        key = f"{context.component}:{error_type}"
        if key not in self.error_counts:
            self.error_counts[key] = 0
        self.error_counts[key] += 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "most_common_errors": sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategy mappings"""
        
        return {
            RecoveryAction.RETRY: self._execute_retry,
            RecoveryAction.RETRY_WITH_DIFFERENT_ENGINE: self._execute_retry_different_engine,
            RecoveryAction.RETRY_WITH_PREPROCESSING: self._execute_retry_preprocessing,
            RecoveryAction.FALLBACK_TO_MANUAL: self._execute_manual_fallback,
            RecoveryAction.SUGGEST_ALTERNATIVE: self._execute_suggest_alternative,
            RecoveryAction.LOG_AND_CONTINUE: self._execute_log_continue,
            RecoveryAction.LOG_AND_NOTIFY: self._execute_log_notify,
            RecoveryAction.ABORT_OPERATION: self._execute_abort,
            RecoveryAction.ENABLE_MANUAL_SELECTION: self._execute_enable_manual,
            RecoveryAction.PROVIDE_CORRECTION_TOOLS: self._execute_correction_tools
        }
    
    def _initialize_error_messages(self) -> Dict[str, str]:
        """Initialize user-friendly error messages"""
        
        return {
            "OCR_LOW_CONFIDENCE": "The document quality is too low for accurate text recognition.",
            "OCR_NO_TEXT": "No text was detected in the document.",
            "OCR_TIMEOUT": "OCR processing took too long and was cancelled.",
            "NO_TABLES_FOUND": "No tables were detected in the document.",
            "MALFORMED_TABLE": "The table structure could not be properly extracted.",
            "DATA_PROCESSING_FAILED": "There was an issue processing the extracted data.",
            "VALIDATION_FAILED": "The provided data failed validation checks.",
            "EXPORT_FAILED": "Failed to export data in the requested format.",
            "AUTH_FAILED": "Authentication failed. Please check your credentials.",
            "ACCESS_DENIED": "You don't have permission to perform this action.",
            "RATE_LIMIT_EXCEEDED": "You've exceeded the rate limit for this operation.",
            "SHARING_FAILED": "Failed to create or access the shared resource."
        }
    
    # Recovery strategy implementations
    def _execute_retry(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute retry recovery strategy"""
        return {"action": "retry", "delay": 1}
    
    def _execute_retry_different_engine(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute retry with different engine strategy"""
        return {"action": "retry", "change_engine": True}
    
    def _execute_retry_preprocessing(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute retry with preprocessing strategy"""
        return {"action": "retry", "enable_preprocessing": True}
    
    def _execute_manual_fallback(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute manual fallback strategy"""
        return {"action": "manual_fallback", "ui_component": "manual_entry"}
    
    def _execute_suggest_alternative(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute suggest alternative strategy"""
        return {"action": "suggest_alternatives", "show_options": True}
    
    def _execute_log_continue(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute log and continue strategy"""
        return {"action": "continue", "log_error": True}
    
    def _execute_log_notify(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute log and notify strategy"""
        return {"action": "notify_user", "log_error": True}
    
    def _execute_abort(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute abort operation strategy"""
        return {"action": "abort", "cleanup": True}
    
    def _execute_enable_manual(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute enable manual selection strategy"""
        return {"action": "enable_manual_selection", "ui_component": "table_selector"}
    
    def _execute_correction_tools(self, context: ErrorContext) -> Dict[str, Any]:
        """Execute provide correction tools strategy"""
        return {"action": "show_correction_tools", "ui_component": "table_editor"}


# Global error handler instance
error_handler = ErrorHandler()


def handle_error(error: Exception, context: ErrorContext, include_technical: bool = False) -> ErrorResponse:
    """
    Convenience function for error handling
    
    Args:
        error: The exception that occurred
        context: Context information about the error
        include_technical: Whether to include technical details
        
    Returns:
        Structured error response
    """
    return error_handler.handle_error(error, context, include_technical)


def create_error_context(
    operation: str,
    component: str,
    user_id: str = None,
    document_id: str = None,
    table_id: str = None,
    request_id: str = None,
    **kwargs
) -> ErrorContext:
    """
    Convenience function for creating error context
    
    Args:
        operation: The operation being performed
        component: The component where error occurred
        user_id: User ID if applicable
        document_id: Document ID if applicable
        table_id: Table ID if applicable
        request_id: Request ID for tracking
        **kwargs: Additional context data
        
    Returns:
        ErrorContext instance
    """
    return ErrorContext(
        operation=operation,
        component=component,
        user_id=user_id,
        document_id=document_id,
        table_id=table_id,
        request_id=request_id,
        additional_data=kwargs
    )