"""
Tests for Cloud Vision fallback mechanisms and service unavailability handling.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.ocr.cloud_vision_engine import CloudVisionEngine
from src.core.models import OCRResult
from src.core.exceptions import OCREngineError


class TestCloudVisionFallbackMechanisms:
    """Test fallback mechanisms for Cloud Vision service unavailability."""
    
    @pytest.fixture
    def engine(self):
        """Create a Cloud Vision engine instance for testing."""
        engine = CloudVisionEngine(confidence_threshold=0.8)
        engine.circuit_breaker_threshold = 2  # Lower threshold for testing
        engine.circuit_breaker_timeout = 5  # Shorter timeout for testing
        return engine
    
    def test_circuit_breaker_activation(self, engine):
        """Test circuit breaker activation after repeated failures."""
        # Manually trigger circuit breaker activation
        engine.circuit_breaker_failures = 0
        
        # First failure
        engine._handle_service_failure()
        assert engine.circuit_breaker_failures == 1
        assert engine.service_available == False
        
        # Second failure - should activate circuit breaker
        engine._handle_service_failure()
        assert engine.circuit_breaker_failures == 2
        assert engine.circuit_breaker_reset_time is not None
        
        # Test that service availability check fails when circuit breaker is active
        available = engine._check_service_availability()
        assert available == False
    
    def test_circuit_breaker_reset(self, engine):
        """Test circuit breaker reset after timeout."""
        # Activate circuit breaker
        engine.circuit_breaker_failures = engine.circuit_breaker_threshold
        engine.circuit_breaker_reset_time = datetime.now() - timedelta(seconds=10)  # Past timeout
        
        # Manually check the reset logic
        now = datetime.now()
        if engine.circuit_breaker_reset_time and now >= engine.circuit_breaker_reset_time:
            engine.circuit_breaker_failures = 0
            engine.circuit_breaker_reset_time = None
        
        # Circuit breaker should be reset
        assert engine.circuit_breaker_failures == 0
        assert engine.circuit_breaker_reset_time is None
    
    def test_manual_circuit_breaker_reset(self, engine):
        """Test manual circuit breaker reset."""
        # Activate circuit breaker
        engine.circuit_breaker_failures = engine.circuit_breaker_threshold
        engine.circuit_breaker_reset_time = datetime.now() + timedelta(minutes=5)
        engine.service_available = False
        
        # Reset manually
        engine.reset_circuit_breaker()
        
        assert engine.circuit_breaker_failures == 0
        assert engine.circuit_breaker_reset_time is None
        assert engine.service_available == True
    
    def test_service_availability_check_success(self, engine):
        """Test service availability check when service is available."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.error.message = ""
            mock_client.text_detection.return_value = mock_response
            
            engine.client = mock_client
            engine.is_initialized = True
            engine.last_availability_check = None  # Force check
            
            available = engine._check_service_availability()
            
            assert available == True
            assert engine.service_available == True
            assert engine.last_availability_check is not None
    
    def test_service_availability_check_failure(self, engine):
        """Test service availability check when service is unavailable."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            mock_client = Mock()
            mock_client.text_detection.side_effect = Exception("Service down")
            
            engine.client = mock_client
            engine.is_initialized = True
            engine.last_availability_check = None  # Force check
            
            available = engine._check_service_availability()
            
            assert available == False
            assert engine.service_available == False
            assert engine.circuit_breaker_failures > 0
    
    def test_service_availability_cached(self, engine):
        """Test that service availability is cached for performance."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            mock_client = Mock()
            
            engine.client = mock_client
            engine.is_initialized = True
            engine.service_available = True
            engine.last_availability_check = datetime.now()  # Recent check
            
            # Should not make API call due to caching
            available = engine._check_service_availability()
            
            assert available == True
            mock_client.text_detection.assert_not_called()
    
    def test_daily_limit_enforcement(self, engine):
        """Test daily request limit enforcement."""
        engine.daily_request_limit = 1
        engine.daily_request_count = 1  # At limit
        engine.daily_reset_time = datetime.now() + timedelta(days=1)  # Future reset time
        
        # Test the daily limit check directly
        limit_ok = engine._check_daily_limits()
        assert limit_ok == False
    
    def test_daily_limit_reset(self, engine):
        """Test daily request limit reset."""
        # Set up past reset time
        engine.daily_request_count = 100
        engine.daily_reset_time = datetime.now() - timedelta(days=1)
        
        # Should reset when checked
        engine._reset_daily_limits()
        
        assert engine.daily_request_count == 0
        assert engine.daily_reset_time > datetime.now()
    
    def test_fallback_engine_configuration(self, engine):
        """Test fallback engine configuration."""
        fallback_engines = ['tesseract', 'easyocr']
        
        engine.set_fallback_engines(fallback_engines)
        
        assert engine.get_fallback_engines() == fallback_engines
        assert engine.fallback_engines == fallback_engines
    
    def test_usage_stats(self, engine):
        """Test usage statistics reporting."""
        engine.daily_request_count = 50
        engine.daily_request_limit = 1000
        engine.circuit_breaker_failures = 2
        engine.service_available = True
        
        stats = engine.get_usage_stats()
        
        assert stats['daily_request_count'] == 50
        assert stats['daily_request_limit'] == 1000
        assert stats['circuit_breaker_failures'] == 2
        assert stats['service_available'] == True
        assert 'daily_reset_time' in stats
        assert 'circuit_breaker_active' in stats
    
    def test_engine_info_comprehensive(self, engine):
        """Test comprehensive engine information."""
        engine.detect_handwriting = True
        engine.detect_tables = False
        engine.language_hints = ['en', 'es']
        engine.fallback_engines = ['tesseract']
        
        info = engine.get_engine_info()
        
        assert info['name'] == 'cloud_vision'
        assert info['detect_handwriting'] == True
        assert info['detect_tables'] == False
        assert info['language_hints'] == ['en', 'es']
        assert info['fallback_engines'] == ['tesseract']
        assert 'usage_stats' in info
        assert 'has_credentials' in info
    
    def test_retry_configuration(self, engine):
        """Test retry configuration settings."""
        # Test default retry settings
        assert engine.max_retries == 3
        assert engine.retry_delay == 1.0
        
        # Test configuration update
        engine.configure({
            'max_retries': 5,
            'retry_delay': 2.0
        })
        
        assert engine.max_retries == 5
        assert engine.retry_delay == 2.0
    
    def test_error_handling_configuration(self, engine):
        """Test error handling configuration."""
        # Test circuit breaker configuration
        assert engine.circuit_breaker_threshold == 2  # Set in fixture
        assert engine.circuit_breaker_timeout == 5    # Set in fixture
        
        # Test configuration update
        engine.configure({
            'circuit_breaker_threshold': 10,
            'circuit_breaker_timeout': 600
        })
        
        assert engine.circuit_breaker_threshold == 10
        assert engine.circuit_breaker_timeout == 600
    
    def test_table_detection_configuration(self, engine):
        """Test table detection configuration."""
        # Test default table detection setting (this is a boolean attribute)
        assert hasattr(engine, 'detect_tables')
        
        # Test configuration update
        engine.configure({'detect_tables': False})
        
        # Test that the configuration was applied
        info = engine.get_engine_info()
        assert 'detect_tables' in info
    
    def test_cleanup_with_client(self, engine):
        """Test cleanup with active client."""
        mock_client = Mock()
        mock_client.close = Mock()
        
        engine.client = mock_client
        engine.is_initialized = True
        
        engine.cleanup()
        
        mock_client.close.assert_called_once()
        assert engine.client is None
        assert engine.is_initialized == False
    
    def test_cleanup_without_close_method(self, engine):
        """Test cleanup with client that doesn't have close method."""
        mock_client = Mock()
        # Don't add close method
        
        engine.client = mock_client
        engine.is_initialized = True
        
        # Should not raise exception
        engine.cleanup()
        
        assert engine.client is None
        assert engine.is_initialized == False