"""
Tests for core functionality.
"""

import pytest
import os
from pathlib import Path
import sys

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.models import (
    ProcessingOptions, OCREngine, DataType, Table, Document, 
    ProcessingStatus, ValidationResult
)
from core.config import ConfigurationManager
from core.exceptions import ConfigurationError
from core.utils import sanitize_column_name, validate_api_key


class TestCoreModels:
    """Test core data models."""
    
    def test_processing_options_defaults(self):
        """Test ProcessingOptions default values."""
        options = ProcessingOptions()
        assert options.ocr_engine == OCREngine.AUTO
        assert options.preprocessing is True
        assert options.table_detection is True
        assert options.confidence_threshold == 0.8
    
    def test_table_creation(self):
        """Test Table model creation."""
        table = Table(
            headers=["Name", "Age", "City"],
            rows=[["John", "25", "NYC"], ["Jane", "30", "LA"]],
            confidence=0.95
        )
        assert len(table.headers) == 3
        assert len(table.rows) == 2
        assert table.confidence == 0.95
        assert table.id is not None
    
    def test_document_status(self):
        """Test Document processing status."""
        doc = Document(filename="test.pdf")
        assert doc.processing_status == ProcessingStatus.PENDING
        assert doc.filename == "test.pdf"
        assert doc.id is not None


class TestConfigurationManager:
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test configuration manager initialization."""
        config_manager = ConfigurationManager()
        assert config_manager.config is not None
        assert config_manager.config.ocr.confidence_threshold == 0.8
    
    def test_ocr_config_retrieval(self):
        """Test OCR configuration retrieval."""
        config_manager = ConfigurationManager()
        tesseract_config = config_manager.get_ocr_config('tesseract')
        assert 'confidence_threshold' in tesseract_config
        assert 'tesseract_path' in tesseract_config
    
    def test_llm_config_retrieval(self):
        """Test LLM configuration retrieval."""
        config_manager = ConfigurationManager()
        openai_config = config_manager.get_llm_config('openai')
        assert 'timeout_seconds' in openai_config
        assert 'model' in openai_config
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigurationManager()
        result = config_manager.validate_config()
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)


class TestUtilities:
    """Test utility functions."""
    
    def test_sanitize_column_name(self):
        """Test column name sanitization."""
        assert sanitize_column_name("First Name") == "first_name"
        assert sanitize_column_name("Age (years)") == "age_years"
        assert sanitize_column_name("123abc") == "col_123abc"
        assert sanitize_column_name("") == "unnamed_column"
    
    def test_validate_api_key(self):
        """Test API key validation."""
        # Valid OpenAI key format
        valid_key = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
        assert validate_api_key(valid_key, "openai") is True
        
        # Invalid key
        assert validate_api_key("invalid", "openai") is False
        assert validate_api_key("", "openai") is False


class TestExceptions:
    """Test custom exceptions."""
    
    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test error", "CONFIG_001")


if __name__ == "__main__":
    pytest.main([__file__])