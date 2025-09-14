"""
Custom exception classes for the OCR Table Analytics system.
"""


class OCRAnalyticsException(Exception):
    """Base exception for OCR Analytics system."""
    
    def __init__(self, message: str, error_code: str = None, context: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


class DocumentProcessingError(OCRAnalyticsException):
    """Exception raised during document processing."""
    pass


class OCRError(OCRAnalyticsException):
    """Exception raised during OCR processing."""
    pass


class OCREngineError(OCRAnalyticsException):
    """Exception raised for OCR engine specific issues."""
    pass


class TableExtractionError(OCRAnalyticsException):
    """Exception raised during table extraction."""
    pass


class DataProcessingError(OCRAnalyticsException):
    """Exception raised during data processing."""
    pass


class DataCleaningError(OCRAnalyticsException):
    """Exception raised during data cleaning."""
    pass


class VisualizationError(OCRAnalyticsException):
    """Exception raised during visualization generation."""
    pass


class ConversationalAIError(OCRAnalyticsException):
    """Exception raised during conversational AI processing."""
    pass


class ConfigurationError(OCRAnalyticsException):
    """Exception raised for configuration issues."""
    pass


class DatabaseError(OCRAnalyticsException):
    """Exception raised for database operations."""
    pass


class ExportError(OCRAnalyticsException):
    """Exception raised during data export."""
    pass


class ValidationError(OCRAnalyticsException):
    """Exception raised during data validation."""
    pass