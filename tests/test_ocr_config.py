"""
Unit tests for OCR configuration management.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open

from src.ocr.config import (
    OCREngineConfig, TableDetectionConfig, ProcessingConfig,
    OCRConfigurationManager, ConfigurationSource,
    get_global_config_manager, create_config_manager
)
from src.core.exceptions import ConfigurationError


class TestOCREngineConfig:
    """Test cases for OCREngineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OCREngineConfig(name="test")
        
        assert config.name == "test"
        assert config.enabled is True
        assert config.confidence_threshold == 0.8
        assert config.language == "eng"
        assert config.dpi == 300
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.parameters == {}
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = OCREngineConfig(
            name="test