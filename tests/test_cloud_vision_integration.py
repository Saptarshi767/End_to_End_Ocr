"""
Integration tests for Cloud Vision OCR engine with API mocking.
"""

import pytest
import numpy as np
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.ocr.cloud_vision_engine import CloudVisionEngine, RateLimiter
from src.core.models import OCRResult, BoundingBox, WordData, TableRegion
from src.core.exceptions import OCREngineError


class TestRateLimiter:
    """Test the rate limiter functionality."""
    
    def test_rate_limiter_allows_requests_within_limit(self):
        """Test that rate limiter allows requests within the limit."""
        limiter = RateLimiter(requests_per_minute=60)
        
        # Should allow requests within limit
        for _ in range(10):
            limiter.wait_if_needed()  # Should not block
    
    def test_rate_limiter_blocks_when_limit_exceeded(self):
        """Test that rate limiter blocks when limit is exceeded."""
        limiter = RateLimiter(requests_per_minute=2)
        
        # First two requests should be allowed
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        
        # Third request should be blocked (but we won't wait in test)
        start_time = time.time()
        with patch('time.sleep') as mock_sleep:
            limiter.wait_if_needed()
            mock_sleep.assert_called_once()
            # Verify sleep was called with a positive duration
            sleep_duration = mock_sleep.call_args[0][0]
            assert sleep_duration > 0


class TestCloudVisionEngine:
    """Test the Cloud Vision OCR engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a Cloud Vision engine instance for testing."""
        return CloudVisionEngine(confidence_threshold=0.8)
    
    @pytest.fixture
    def mock_vision_client(self):
        """Create a mock Google Cloud Vision client."""
        mock_client = Mock()
        
        # Mock successful text detection response
        mock_response = Mock()
        mock_response.error.message = ""
        # Create mock vertices
        mock_vertices_1 = [
            Mock(x=10, y=10), Mock(x=60, y=10), Mock(x=60, y=30), Mock(x=10, y=30)
        ]
        mock_vertices_2 = [
            Mock(x=70, y=10), Mock(x=100, y=10), Mock(x=100, y=30), Mock(x=70, y=30)
        ]
        
        mock_response.text_annotations = [
            Mock(description="Sample text from image"),
            Mock(description="Sample", bounding_poly=Mock(vertices=mock_vertices_1)),
            Mock(description="text", bounding_poly=Mock(vertices=mock_vertices_2))
        ]
        
        mock_client.text_detection.return_value = mock_response
        mock_client.document_text_detection.return_value = mock_response
        
        return mock_client
    
    @pytest.fixture
    def mock_document_response(self):
        """Create a mock document text detection response."""
        mock_response = Mock()
        mock_response.error.message = ""
        
        # Mock full text annotation
        mock_full_text = Mock()
        mock_full_text.text = "Sample document text"
        
        # Mock page structure
        mock_page = Mock()
        mock_block = Mock()
        mock_block.confidence = 0.95
        mock_block.bounding_box.vertices = [
            Mock(x=10, y=10), Mock(x=200, y=10), Mock(x=200, y=50), Mock(x=10, y=50)
        ]
        
        mock_paragraph = Mock()
        mock_paragraph.confidence = 0.93
        
        mock_word = Mock()
        mock_word.confidence = 0.91
        mock_word.bounding_box.vertices = [
            Mock(x=10, y=10), Mock(x=60, y=10), Mock(x=60, y=30), Mock(x=10, y=30)
        ]
        mock_word.symbols = [Mock(text="S"), Mock(text="a"), Mock(text="m"), Mock(text="p"), Mock(text="l"), Mock(text="e")]
        
        mock_paragraph.words = [mock_word]
        mock_block.paragraphs = [mock_paragraph]
        mock_page.blocks = [mock_block]
        
        mock_full_text.pages = [mock_page]
        mock_response.full_text_annotation = mock_full_text
        
        return mock_response
    
    def test_engine_initialization_with_credentials(self, engine):
        """Test engine initialization with various credential sources."""
        # Test initialization without credentials (should fail gracefully)
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                # Mock successful connection test
                mock_response = Mock()
                mock_response.error.message = ""
                mock_client.text_detection.return_value = mock_response
                
                engine.initialize()
                assert engine.is_available()
                assert engine.client is not None
    
    def test_engine_initialization_with_credentials_path(self, engine):
        """Test engine initialization with credentials file path."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch('google.oauth2.service_account.Credentials.from_service_account_file') as mock_creds:
                with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client_class:
                    with patch('os.path.exists', return_value=True):
                        mock_credentials = Mock()
                        mock_creds.return_value = mock_credentials
                        
                        mock_client = Mock()
                        mock_client_class.return_value = mock_client
                        
                        # Mock successful connection test
                        mock_response = Mock()
                        mock_response.error.message = ""
                        mock_client.text_detection.return_value = mock_response
                        
                        engine.configure({'credentials_path': '/path/to/credentials.json'})
                        engine.initialize()
                        
                        assert engine.is_available()
                        mock_creds.assert_called_once_with('/path/to/credentials.json')
    
    def test_engine_initialization_with_credentials_json(self, engine):
        """Test engine initialization with credentials JSON string."""
        credentials_json = json.dumps({
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456789",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        })
        
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch('google.oauth2.service_account.Credentials.from_service_account_info') as mock_creds:
                with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client_class:
                    mock_credentials = Mock()
                    mock_creds.return_value = mock_credentials
                    
                    mock_client = Mock()
                    mock_client_class.return_value = mock_client
                    
                    # Mock successful connection test
                    mock_response = Mock()
                    mock_response.error.message = ""
                    mock_client.text_detection.return_value = mock_response
                    
                    engine.configure({'credentials_json': credentials_json})
                    engine.initialize()
                    
                    assert engine.is_available()
                    mock_creds.assert_called_once()
    
    def test_engine_initialization_failure(self, engine):
        """Test engine initialization failure handling."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', False):
            with pytest.raises(OCREngineError, match="Google Cloud Vision not available"):
                engine.initialize()
    
    def test_text_extraction_success(self, engine, mock_vision_client):
        """Test successful text extraction."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            engine.client = mock_vision_client
            engine.is_initialized = True
            engine.service_available = True
            engine.detect_handwriting = False  # Use regular text detection
            engine.last_availability_check = datetime.now()  # Skip availability check
            
            # Create test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            result = engine.extract_text(test_image)
            
            assert isinstance(result, OCRResult)
            assert result.text == "Sample text from image"
            assert result.confidence > 0
            assert len(result.word_level_data) > 0
            assert mock_vision_client.text_detection.call_count >= 1
    
    def test_document_text_extraction_success(self, engine, mock_document_response):
        """Test successful document text extraction with handwriting detection."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            mock_client = Mock()
            mock_client.document_text_detection.return_value = mock_document_response
            
            engine.client = mock_client
            engine.is_initialized = True
            engine.service_available = True
            engine.detect_handwriting = True
            
            # Create test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            result = engine.extract_text(test_image)
            
            assert isinstance(result, OCRResult)
            assert result.text == "Sample document text"
            assert result.confidence > 0
            mock_client.document_text_detection.assert_called_once()
    
    def test_text_extraction_with_retry_on_quota_exceeded(self, engine):
        """Test text extraction with retry on quota exceeded error."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch('google.api_core.exceptions') as mock_exceptions:
                mock_client = Mock()
                
                # First call raises quota exceeded, second succeeds
                quota_error = Exception("Quota exceeded")
                quota_error.__class__ = type('ResourceExhausted', (Exception,), {})
                mock_exceptions.ResourceExhausted = quota_error.__class__
                
                mock_response = Mock()
                mock_response.error.message = ""
                mock_response.text_annotations = [Mock(description="Success")]
                
                mock_client.text_detection.side_effect = [quota_error, mock_response]
                
                engine.client = mock_client
                engine.is_initialized = True
                engine.service_available = True
                engine.max_retries = 1
                
                # Create test image
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                
                with patch('time.sleep'):  # Mock sleep to speed up test
                    result = engine.extract_text(test_image)
                
                assert isinstance(result, OCRResult)
                assert mock_client.text_detection.call_count == 2
    
    def test_text_extraction_failure_after_retries(self, engine):
        """Test text extraction failure after all retries exhausted."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch('google.api_core.exceptions') as mock_exceptions:
                mock_client = Mock()
                
                # All calls raise quota exceeded
                quota_error = Exception("Quota exceeded")
                quota_error.__class__ = type('ResourceExhausted', (Exception,), {})
                mock_exceptions.ResourceExhausted = quota_error.__class__
                
                mock_client.text_detection.side_effect = quota_error
                
                engine.client = mock_client
                engine.is_initialized = True
                engine.service_available = True
                engine.max_retries = 2
                
                # Create test image
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                
                with patch('time.sleep'):  # Mock sleep to speed up test
                    with pytest.raises(OCREngineError, match="Cloud Vision OCR failed after"):
                        engine.extract_text(test_image)
                
                assert mock_client.text_detection.call_count == 3  # Initial + 2 retries
    
    def test_authentication_error_no_retry(self, engine):
        """Test that authentication errors are not retried."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch('google.api_core.exceptions') as mock_exceptions:
                mock_client = Mock()
                
                # Authentication error should not be retried
                auth_error = Exception("Authentication failed")
                auth_error.__class__ = type('Unauthenticated', (Exception,), {})
                mock_exceptions.Unauthenticated = auth_error.__class__
                
                mock_client.text_detection.side_effect = auth_error
                
                engine.client = mock_client
                engine.is_initialized = True
                engine.service_available = True
                engine.max_retries = 2
                
                # Create test image
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                
                with pytest.raises(OCREngineError, match="Cloud Vision OCR failed after"):
                    engine.extract_text(test_image)
                
                # Should only be called once (no retries for auth errors)
                assert mock_client.text_detection.call_count == 1
    
    def test_rate_limiting_integration(self, engine):
        """Test rate limiting integration with OCR requests."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.error.message = ""
            mock_response.text_annotations = [Mock(description="Test")]
            mock_client.text_detection.return_value = mock_response
            
            engine.client = mock_client
            engine.is_initialized = True
            engine.service_available = True
            
            # Set very low rate limit for testing
            engine.rate_limiter = RateLimiter(requests_per_minute=1)
            
            # Create test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            # First request should succeed
            result1 = engine.extract_text(test_image)
            assert isinstance(result1, OCRResult)
            
            # Second request should be rate limited
            with patch('time.sleep') as mock_sleep:
                result2 = engine.extract_text(test_image)
                assert isinstance(result2, OCRResult)
                mock_sleep.assert_called_once()
    
    def test_daily_limit_enforcement(self, engine):
        """Test daily request limit enforcement."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            engine.client = Mock()
            engine.is_initialized = True
            engine.service_available = True
            engine.daily_request_limit = 1
            engine.daily_request_count = 1  # Already at limit
            
            # Create test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            with pytest.raises(OCREngineError, match="Daily request limit exceeded"):
                engine.extract_text(test_image)
    
    def test_circuit_breaker_activation(self, engine):
        """Test circuit breaker activation after repeated failures."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            engine.client = Mock()
            engine.is_initialized = True
            engine.service_available = True
            engine.circuit_breaker_threshold = 2
            engine.circuit_breaker_failures = 2  # At threshold
            
            # Create test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            # Trigger circuit breaker
            engine._handle_service_failure()
            
            with pytest.raises(OCREngineError, match="Cloud Vision service is not available"):
                engine.extract_text(test_image)
    
    def test_table_detection_success(self, engine, mock_document_response):
        """Test successful table detection."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            mock_client = Mock()
            mock_client.document_text_detection.return_value = mock_document_response
            
            engine.client = mock_client
            engine.is_initialized = True
            engine.service_available = True
            engine.detect_tables = True
            
            # Create test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            table_regions = engine.detect_tables(test_image)
            
            assert isinstance(table_regions, list)
            mock_client.document_text_detection.assert_called_once()
    
    def test_table_detection_disabled(self, engine):
        """Test table detection when disabled."""
        engine.detect_tables = False
        
        # Create test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        table_regions = engine.detect_tables(test_image)
        
        assert table_regions == []
    
    def test_table_detection_failure_returns_empty_list(self, engine):
        """Test that table detection failures return empty list instead of raising."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            mock_client = Mock()
            mock_client.document_text_detection.side_effect = Exception("API Error")
            
            engine.client = mock_client
            engine.is_initialized = True
            engine.service_available = True
            engine.detect_tables = True
            engine.max_retries = 0  # No retries for faster test
            
            # Create test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            table_regions = engine.detect_tables(test_image)
            
            assert table_regions == []
    
    def test_configuration_update(self, engine):
        """Test engine configuration updates."""
        config = {
            'confidence_threshold': 0.95,
            'detect_handwriting': False,
            'detect_tables': False,
            'language_hints': ['es', 'fr'],
            'max_results': 100,
            'requests_per_minute': 300,
            'daily_request_limit': 500,
            'max_retries': 5,
            'retry_delay': 2.0,
            'circuit_breaker_threshold': 10,
            'circuit_breaker_timeout': 600
        }
        
        engine.configure(config)
        
        assert engine.confidence_threshold == 0.95
        assert engine.detect_handwriting == False
        assert engine.detect_tables == False
        assert engine.language_hints == ['es', 'fr']
        assert engine.max_results == 100
        assert engine.requests_per_minute == 300
        assert engine.daily_request_limit == 500
        assert engine.max_retries == 5
        assert engine.retry_delay == 2.0
        assert engine.circuit_breaker_threshold == 10
        assert engine.circuit_breaker_timeout == 600
    
    def test_service_availability_check(self, engine):
        """Test service availability checking mechanism."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.error.message = ""
            mock_client.text_detection.return_value = mock_response
            
            engine.client = mock_client
            engine.is_initialized = True
            engine.last_availability_check = None  # Force availability check
            
            # Should perform availability check
            available = engine._check_service_availability()
            
            assert available == True
            assert engine.service_available == True
            assert engine.last_availability_check is not None
            mock_client.text_detection.assert_called_once()
    
    def test_credentials_from_environment_variable(self, engine):
        """Test loading credentials from environment variable."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch.dict('os.environ', {'GOOGLE_CLOUD_VISION_CREDENTIALS': '/path/to/creds.json'}):
                with patch('os.path.exists', return_value=True):
                    with patch('google.oauth2.service_account.Credentials.from_service_account_file') as mock_creds:
                        mock_credentials = Mock()
                        mock_creds.return_value = mock_credentials
                        
                        credentials = engine._get_credentials()
                        
                        assert credentials == mock_credentials
                        mock_creds.assert_called_once_with('/path/to/creds.json')
    
    def test_language_support(self, engine):
        """Test language support checking."""
        # Test supported languages
        assert engine.supports_language('en')
        assert engine.supports_language('es')
        assert engine.supports_language('fr')
        assert engine.supports_language('zh')
        
        # Test unsupported language (should still return True for Cloud Vision)
        # Cloud Vision supports many languages, so most should be supported
        supported_langs = engine.get_supported_languages()
        assert len(supported_langs) > 50  # Cloud Vision supports many languages
    
    def test_engine_info(self, engine):
        """Test getting engine information."""
        info = engine.get_engine_info()
        
        assert info['name'] == 'cloud_vision'
        assert 'confidence_threshold' in info
        assert 'supported_languages' in info
        assert 'is_initialized' in info
        assert 'config' in info
    
    def test_cleanup(self, engine):
        """Test engine cleanup."""
        engine.is_initialized = True
        engine.cleanup()
        
        assert engine.is_initialized == False


class TestCloudVisionEngineIntegration:
    """Integration tests for Cloud Vision engine with real-world scenarios."""
    
    def test_end_to_end_ocr_workflow(self):
        """Test complete OCR workflow with mocked API responses."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client_class:
                # Setup mock client
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                # Mock connection test response
                test_response = Mock()
                test_response.error.message = ""
                
                # Mock OCR response
                ocr_response = Mock()
                ocr_response.error.message = ""
                ocr_response.text_annotations = [
                    Mock(description="Invoice\nDate: 2024-01-15\nAmount: $1,234.56"),
                    Mock(description="Invoice", bounding_poly=Mock(vertices=[
                        Mock(x=10, y=10), Mock(x=100, y=10), Mock(x=100, y=40), Mock(x=10, y=40)
                    ])),
                    Mock(description="Date:", bounding_poly=Mock(vertices=[
                        Mock(x=10, y=50), Mock(x=60, y=50), Mock(x=60, y=70), Mock(x=10, y=70)
                    ])),
                    Mock(description="2024-01-15", bounding_poly=Mock(vertices=[
                        Mock(x=70, y=50), Mock(x=150, y=50), Mock(x=150, y=70), Mock(x=70, y=70)
                    ]))
                ]
                
                mock_client.text_detection.side_effect = [test_response, ocr_response]
                
                # Create and initialize engine
                engine = CloudVisionEngine(confidence_threshold=0.8)
                engine.initialize()
                
                # Create test image (simulating a document)
                test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
                
                # Perform OCR
                result = engine.extract_text(test_image)
                
                # Verify results
                assert isinstance(result, OCRResult)
                assert "Invoice" in result.text
                assert "2024-01-15" in result.text
                assert "$1,234.56" in result.text
                assert len(result.word_level_data) > 0
                assert result.confidence > 0
    
    def test_batch_processing_simulation(self):
        """Test batch processing of multiple documents."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                # Mock responses for multiple documents
                responses = []
                for i in range(3):
                    response = Mock()
                    response.error.message = ""
                    response.text_annotations = [
                        Mock(description=f"Document {i+1} content")
                    ]
                    responses.append(response)
                
                # Connection test response
                test_response = Mock()
                test_response.error.message = ""
                
                mock_client.text_detection.side_effect = [test_response] + responses
                
                # Create and initialize engine
                engine = CloudVisionEngine()
                engine.initialize()
                
                # Process multiple documents
                results = []
                for i in range(3):
                    test_image = np.ones((100, 100, 3), dtype=np.uint8) * (50 + i * 50)
                    result = engine.extract_text(test_image)
                    results.append(result)
                
                # Verify all results
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert isinstance(result, OCRResult)
                    assert f"Document {i+1}" in result.text
    
    def test_error_recovery_workflow(self):
        """Test error recovery and fallback mechanisms."""
        with patch('src.ocr.cloud_vision_engine.CLOUD_VISION_AVAILABLE', True):
            with patch('google.api_core.exceptions') as mock_exceptions:
                with patch('google.cloud.vision.ImageAnnotatorClient') as mock_client_class:
                    mock_client = Mock()
                    mock_client_class.return_value = mock_client
                    
                    # Setup exception types
                    quota_error = Exception("Quota exceeded")
                    quota_error.__class__ = type('ResourceExhausted', (Exception,), {})
                    mock_exceptions.ResourceExhausted = quota_error.__class__
                    
                    # Connection test succeeds
                    test_response = Mock()
                    test_response.error.message = ""
                    
                    # First OCR call fails, second succeeds
                    success_response = Mock()
                    success_response.error.message = ""
                    success_response.text_annotations = [Mock(description="Recovered text")]
                    
                    mock_client.text_detection.side_effect = [
                        test_response,  # Connection test
                        quota_error,    # First OCR attempt fails
                        success_response  # Second OCR attempt succeeds
                    ]
                    
                    # Create and initialize engine
                    engine = CloudVisionEngine()
                    engine.max_retries = 1
                    engine.initialize()
                    
                    # Create test image
                    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                    
                    # Should recover from the error
                    with patch('time.sleep'):  # Speed up test
                        result = engine.extract_text(test_image)
                    
                    assert isinstance(result, OCRResult)
                    assert result.text == "Recovered text"
                    assert mock_client.text_detection.call_count == 3  # Test + 2 OCR attempts
   