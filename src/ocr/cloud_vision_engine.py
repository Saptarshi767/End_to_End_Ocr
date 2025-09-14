"""
Google Cloud Vision OCR engine implementation with enhanced API key management,
rate limiting, and fallback mechanisms.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
import cv2
import os
import base64
import time
import json
from datetime import datetime, timedelta
from threading import Lock
from pathlib import Path

from .base_engine import BaseOCREngine
from ..core.models import OCRResult, BoundingBox, WordData, TableRegion
from ..core.exceptions import OCREngineError

logger = logging.getLogger(__name__)

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    from google.api_core import exceptions as gcp_exceptions
    CLOUD_VISION_AVAILABLE = True
except ImportError:
    CLOUD_VISION_AVAILABLE = False
    vision = None
    service_account = None
    gcp_exceptions = None
    logger.warning("Google Cloud Vision not available. Install with: pip install google-cloud-vision")


class RateLimiter:
    """Thread-safe rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 600):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = Lock()
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = datetime.now()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < timedelta(minutes=1)]
            
            if len(self.requests) >= self.requests_per_minute:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_until = oldest_request + timedelta(minutes=1)
                wait_seconds = (wait_until - now).total_seconds()
                
                if wait_seconds > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_seconds:.1f} seconds")
                    time.sleep(wait_seconds)
                    # Clean up old requests after waiting
                    now = datetime.now()
                    self.requests = [req_time for req_time in self.requests 
                                   if now - req_time < timedelta(minutes=1)]
            
            # Record this request
            self.requests.append(now)


class CloudVisionEngine(BaseOCREngine):
    """
    Enhanced Google Cloud Vision OCR engine implementation with improved
    API key management, rate limiting, and fallback mechanisms.
    """
    
    def __init__(self, confidence_threshold: float = 0.9):
        super().__init__("cloud_vision", confidence_threshold)
        
        # Cloud Vision client
        self.client = None
        
        # API configuration with multiple credential sources
        self.api_key = None
        self.credentials_path = None
        self.credentials_json = None
        self.project_id = None
        
        # Enhanced rate limiting
        self.rate_limiter = RateLimiter(requests_per_minute=600)
        self.requests_per_minute = 600
        self.daily_request_limit = 1000  # Default free tier limit
        self.daily_request_count = 0
        self.daily_reset_time = None
        
        # Fallback configuration
        self.fallback_engines = []  # List of fallback engine names
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_time = None
        self.circuit_breaker_timeout = 300  # 5 minutes
        
        # Cloud Vision specific settings
        self.detect_handwriting = True
        self.detect_tables = True
        self.language_hints = ['en']
        self.max_results = 50
        self.batch_size = 16  # Maximum batch size for Cloud Vision
        
        # Service availability tracking
        self.service_available = True
        self.last_availability_check = None
        self.availability_check_interval = 300  # 5 minutes
        
        # Extensive language support
        self.supported_languages = [
            'af', 'sq', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca', 'ceb',
            'ny', 'zh', 'zh-cn', 'zh-tw', 'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo',
            'et', 'tl', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha',
            'haw', 'iw', 'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jw',
            'kn', 'kk', 'km', 'ko', 'ku', 'ky', 'lo', 'la', 'lv', 'lt', 'lb', 'mk',
            'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mn', 'my', 'ne', 'no', 'ps', 'fa',
            'pl', 'pt', 'pa', 'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd', 'si',
            'sk', 'sl', 'so', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 'te', 'th', 'tr',
            'uk', 'ur', 'uz', 'vi', 'cy', 'xh', 'yi', 'yo', 'zu'
        ]
        
    def initialize(self) -> None:
        """Initialize Google Cloud Vision client with enhanced credential management."""
        if not CLOUD_VISION_AVAILABLE:
            raise OCREngineError("Google Cloud Vision not available. Install with: pip install google-cloud-vision")
            
        try:
            # Initialize client with multiple credential sources
            credentials = self._get_credentials()
            
            if credentials:
                self.client = vision.ImageAnnotatorClient(credentials=credentials)
                logger.info("Cloud Vision initialized with service account credentials")
            elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                self.client = vision.ImageAnnotatorClient()
                logger.info("Cloud Vision initialized with default credentials from GOOGLE_APPLICATION_CREDENTIALS")
            elif 'GOOGLE_CLOUD_PROJECT' in os.environ:
                # Try to use default credentials with project ID
                self.project_id = os.environ['GOOGLE_CLOUD_PROJECT']
                self.client = vision.ImageAnnotatorClient()
                logger.info(f"Cloud Vision initialized with default credentials for project: {self.project_id}")
            else:
                # Try to initialize without explicit credentials (will use default)
                try:
                    self.client = vision.ImageAnnotatorClient()
                    logger.info("Cloud Vision initialized with default credentials")
                except Exception:
                    raise OCREngineError(
                        "No Google Cloud credentials found. Please set one of:\n"
                        "1. GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                        "2. Provide credentials_path in configuration\n"
                        "3. Provide credentials_json in configuration\n"
                        "4. Set up default credentials with 'gcloud auth application-default login'"
                    )
                
            # Test the connection and service availability
            self._test_connection()
            self._reset_daily_limits()
            
            super().initialize()
            
        except Exception as e:
            self.service_available = False
            raise OCREngineError(f"Failed to initialize Google Cloud Vision: {str(e)}")
    
    def _get_credentials(self) -> Optional[Any]:
        """Get credentials from various sources."""
        # Try credentials from JSON string
        if self.credentials_json:
            try:
                credentials_info = json.loads(self.credentials_json)
                return service_account.Credentials.from_service_account_info(credentials_info)
            except Exception as e:
                logger.warning(f"Failed to load credentials from JSON: {e}")
        
        # Try credentials from file path
        if self.credentials_path and os.path.exists(self.credentials_path):
            try:
                return service_account.Credentials.from_service_account_file(self.credentials_path)
            except Exception as e:
                logger.warning(f"Failed to load credentials from file {self.credentials_path}: {e}")
        
        # Try credentials from environment variable file path
        env_creds_path = os.environ.get('GOOGLE_CLOUD_VISION_CREDENTIALS')
        if env_creds_path and os.path.exists(env_creds_path):
            try:
                return service_account.Credentials.from_service_account_file(env_creds_path)
            except Exception as e:
                logger.warning(f"Failed to load credentials from env file {env_creds_path}: {e}")
        
        return None
    
    def _reset_daily_limits(self) -> None:
        """Reset daily request limits if needed."""
        now = datetime.now()
        if self.daily_reset_time is None or now >= self.daily_reset_time:
            self.daily_request_count = 0
            # Reset at midnight
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.daily_reset_time = tomorrow
            logger.info("Reset daily request limits for Cloud Vision API")
            
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure Cloud Vision specific parameters with enhanced options."""
        super().configure(config)
        
        # API credentials - multiple sources
        if 'credentials_path' in config:
            self.credentials_path = config['credentials_path']
            
        if 'credentials_json' in config:
            self.credentials_json = config['credentials_json']
            
        if 'project_id' in config:
            self.project_id = config['project_id']
            
        # Detection settings
        if 'detect_handwriting' in config:
            self.detect_handwriting = config['detect_handwriting']
            
        if 'detect_tables' in config:
            self.detect_tables = config['detect_tables']
            
        if 'language_hints' in config:
            self.language_hints = config['language_hints']
            
        if 'max_results' in config:
            self.max_results = config['max_results']
            
        if 'batch_size' in config:
            self.batch_size = min(config['batch_size'], 16)  # Cloud Vision limit
            
        # Enhanced rate limiting
        if 'requests_per_minute' in config:
            self.requests_per_minute = config['requests_per_minute']
            self.rate_limiter = RateLimiter(self.requests_per_minute)
            
        if 'daily_request_limit' in config:
            self.daily_request_limit = config['daily_request_limit']
            
        # Fallback configuration
        if 'fallback_engines' in config:
            self.fallback_engines = config['fallback_engines']
            
        if 'max_retries' in config:
            self.max_retries = config['max_retries']
            
        if 'retry_delay' in config:
            self.retry_delay = config['retry_delay']
            
        # Circuit breaker settings
        if 'circuit_breaker_threshold' in config:
            self.circuit_breaker_threshold = config['circuit_breaker_threshold']
            
        if 'circuit_breaker_timeout' in config:
            self.circuit_breaker_timeout = config['circuit_breaker_timeout']
            
    def _test_connection(self) -> None:
        """Test the Cloud Vision API connection with enhanced error handling."""
        try:
            # Create a small test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            _, encoded_image = cv2.imencode('.png', test_image)
            
            image = vision.Image(content=encoded_image.tobytes())
            
            # Make a simple text detection request with timeout
            response = self.client.text_detection(image=image, timeout=10.0)
            
            if response.error.message:
                raise OCREngineError(f"Cloud Vision API error: {response.error.message}")
                
            self.service_available = True
            self.circuit_breaker_failures = 0
            self.last_availability_check = datetime.now()
            logger.info("Cloud Vision API connection test successful")
            
        except gcp_exceptions.Unauthenticated as e:
            self.service_available = False
            raise OCREngineError(f"Cloud Vision authentication failed: {str(e)}")
        except gcp_exceptions.PermissionDenied as e:
            self.service_available = False
            raise OCREngineError(f"Cloud Vision permission denied: {str(e)}")
        except gcp_exceptions.ResourceExhausted as e:
            self.service_available = False
            raise OCREngineError(f"Cloud Vision quota exceeded: {str(e)}")
        except Exception as e:
            self.service_available = False
            raise OCREngineError(f"Cloud Vision API connection test failed: {str(e)}")
    
    def _check_service_availability(self) -> bool:
        """Check if the Cloud Vision service is available."""
        now = datetime.now()
        
        # Check circuit breaker
        if self.circuit_breaker_reset_time and now < self.circuit_breaker_reset_time:
            logger.warning("Cloud Vision service in circuit breaker state")
            return False
        
        # Reset circuit breaker if timeout has passed
        if self.circuit_breaker_reset_time and now >= self.circuit_breaker_reset_time:
            self.circuit_breaker_failures = 0
            self.circuit_breaker_reset_time = None
            logger.info("Cloud Vision circuit breaker reset")
        
        # Check if we need to test availability
        if (self.last_availability_check is None or 
            now - self.last_availability_check > timedelta(seconds=self.availability_check_interval)):
            try:
                self._test_connection()
                return True
            except Exception as e:
                logger.warning(f"Cloud Vision availability check failed: {e}")
                self._handle_service_failure()
                return False
        
        return self.service_available
    
    def _handle_service_failure(self) -> None:
        """Handle service failure and update circuit breaker."""
        self.circuit_breaker_failures += 1
        self.service_available = False
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_reset_time = datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)
            logger.warning(f"Cloud Vision circuit breaker activated for {self.circuit_breaker_timeout} seconds")
    
    def _check_daily_limits(self) -> bool:
        """Check if daily request limits are exceeded."""
        self._reset_daily_limits()
        
        if self.daily_request_count >= self.daily_request_limit:
            logger.warning(f"Daily request limit exceeded: {self.daily_request_count}/{self.daily_request_limit}")
            return False
        
        return True
            
    def _pre_request_checks(self) -> None:
        """Perform all pre-request checks."""
        # Check service availability
        if not self._check_service_availability():
            raise OCREngineError("Cloud Vision service is not available")
        
        # Check daily limits
        if not self._check_daily_limits():
            raise OCREngineError("Daily request limit exceeded")
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Increment daily counter
        self.daily_request_count += 1
        
    def _extract_text_impl(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Extract text using Google Cloud Vision OCR with enhanced error handling and retries.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional parameters
            
        Returns:
            OCR result with extracted text and metadata
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Perform pre-request checks
                self._pre_request_checks()
                
                # Encode image
                _, encoded_image = cv2.imencode('.png', image)
                vision_image = vision.Image(content=encoded_image.tobytes())
                
                # Configure image context
                image_context = vision.ImageContext()
                
                # Set language hints
                language_hints = kwargs.get('language_hints', self.language_hints)
                if language_hints:
                    image_context.language_hints = language_hints
                
                # Perform text detection with timeout
                timeout = kwargs.get('timeout', 30.0)
                
                if self.detect_handwriting:
                    # Use document text detection for better handwriting support
                    response = self.client.document_text_detection(
                        image=vision_image,
                        image_context=image_context,
                        timeout=timeout
                    )
                else:
                    # Use regular text detection
                    response = self.client.text_detection(
                        image=vision_image,
                        image_context=image_context,
                        timeout=timeout
                    )
                
                if response.error.message:
                    raise OCREngineError(f"Cloud Vision API error: {response.error.message}")
                
                # Process response
                if self.detect_handwriting and response.full_text_annotation:
                    result = self._process_document_response(response)
                else:
                    result = self._process_text_response(response)
                
                # Reset failure count on success
                self.circuit_breaker_failures = 0
                return result
                
            except gcp_exceptions.ResourceExhausted as e:
                last_exception = e
                logger.warning(f"Cloud Vision quota exceeded (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self._handle_service_failure()
                    
            except gcp_exceptions.DeadlineExceeded as e:
                last_exception = e
                logger.warning(f"Cloud Vision timeout (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    
            except gcp_exceptions.ServiceUnavailable as e:
                last_exception = e
                logger.warning(f"Cloud Vision service unavailable (attempt {attempt + 1}): {e}")
                self._handle_service_failure()
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    
            except gcp_exceptions.Unauthenticated as e:
                last_exception = e
                logger.error(f"Cloud Vision authentication failed: {e}")
                self._handle_service_failure()
                break  # Don't retry authentication errors
                
            except gcp_exceptions.PermissionDenied as e:
                last_exception = e
                logger.error(f"Cloud Vision permission denied: {e}")
                self._handle_service_failure()
                break  # Don't retry permission errors
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Cloud Vision OCR failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # All retries failed
        self._handle_service_failure()
        raise OCREngineError(f"Cloud Vision OCR failed after {self.max_retries + 1} attempts: {last_exception}")
            
    def _process_document_response(self, response) -> OCRResult:
        """Process document text detection response."""
        full_text = response.full_text_annotation.text
        
        word_data = []
        bounding_boxes = []
        confidences = []
        
        # Process pages
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                block_confidence = block.confidence
                
                # Create block bounding box
                vertices = block.bounding_box.vertices
                block_bbox = self._vertices_to_bbox(vertices, block_confidence)
                bounding_boxes.append(block_bbox)
                
                # Process paragraphs
                for paragraph in block.paragraphs:
                    para_confidence = paragraph.confidence
                    confidences.append(para_confidence)
                    
                    # Process words
                    for word in paragraph.words:
                        word_confidence = word.confidence
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        
                        # Create word bounding box
                        word_vertices = word.bounding_box.vertices
                        word_bbox = self._vertices_to_bbox(word_vertices, word_confidence)
                        
                        word_data.append(WordData(
                            text=word_text,
                            confidence=word_confidence,
                            bounding_box=word_bbox
                        ))
                        
        # Calculate overall confidence
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        return OCRResult(
            text=full_text,
            confidence=overall_confidence,
            bounding_boxes=bounding_boxes,
            word_level_data=word_data
        )
        
    def _process_text_response(self, response) -> OCRResult:
        """Process regular text detection response."""
        texts = response.text_annotations
        
        if not texts:
            return OCRResult(
                text="",
                confidence=0.0,
                bounding_boxes=[],
                word_level_data=[]
            )
            
        # First annotation contains the full text
        full_text = texts[0].description
        
        # Process individual text annotations
        word_data = []
        bounding_boxes = []
        confidences = []
        
        for text in texts[1:]:  # Skip the first one (full text)
            confidence = 0.9  # Cloud Vision doesn't provide confidence for text detection
            confidences.append(confidence)
            
            # Create bounding box
            vertices = text.bounding_poly.vertices
            bbox = self._vertices_to_bbox(vertices, confidence)
            bounding_boxes.append(bbox)
            
            # Create word data
            word_data.append(WordData(
                text=text.description,
                confidence=confidence,
                bounding_box=bbox
            ))
            
        # Calculate overall confidence
        overall_confidence = np.mean(confidences) if confidences else 0.9
        
        return OCRResult(
            text=full_text,
            confidence=overall_confidence,
            bounding_boxes=bounding_boxes,
            word_level_data=word_data
        )
        
    def _vertices_to_bbox(self, vertices, confidence: float) -> BoundingBox:
        """Convert Cloud Vision vertices to BoundingBox."""
        x_coords = [vertex.x for vertex in vertices]
        y_coords = [vertex.y for vertex in vertices]
        
        x = min(x_coords)
        y = min(y_coords)
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        return BoundingBox(x=x, y=y, width=width, height=height, confidence=confidence)
        
    def detect_tables(self, image: np.ndarray, **kwargs) -> List[TableRegion]:
        """
        Detect table regions using Cloud Vision with enhanced error handling.
        """
        if not self.detect_tables:
            return []
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Perform pre-request checks
                self._pre_request_checks()
                
                # Encode image
                _, encoded_image = cv2.imencode('.png', image)
                vision_image = vision.Image(content=encoded_image.tobytes())
                
                # Use document text detection for table structure
                timeout = kwargs.get('timeout', 30.0)
                response = self.client.document_text_detection(
                    image=vision_image,
                    timeout=timeout
                )
                
                if response.error.message:
                    raise OCREngineError(f"Cloud Vision API error: {response.error.message}")
                
                # Extract table regions from document structure
                table_regions = self._extract_table_regions(response)
                
                # Reset failure count on success
                self.circuit_breaker_failures = 0
                return table_regions
                
            except gcp_exceptions.ResourceExhausted as e:
                last_exception = e
                logger.warning(f"Cloud Vision quota exceeded for table detection (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    self._handle_service_failure()
                    
            except gcp_exceptions.ServiceUnavailable as e:
                last_exception = e
                logger.warning(f"Cloud Vision service unavailable for table detection (attempt {attempt + 1}): {e}")
                self._handle_service_failure()
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"Cloud Vision table detection failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # All retries failed, return empty list instead of raising exception
        logger.error(f"Cloud Vision table detection failed after {self.max_retries + 1} attempts: {last_exception}")
        return []
            
    def _extract_table_regions(self, response) -> List[TableRegion]:
        """Extract table regions from document text detection response."""
        table_regions = []
        
        if not response.full_text_annotation:
            return table_regions
            
        # Analyze block structure to identify tables
        for page in response.full_text_annotation.pages:
            blocks = []
            
            for block in page.blocks:
                vertices = block.bounding_box.vertices
                bbox = self._vertices_to_bbox(vertices, block.confidence)
                
                blocks.append({
                    'bbox': bbox,
                    'confidence': block.confidence,
                    'paragraphs': len(block.paragraphs)
                })
                
            # Look for regular patterns that might indicate tables
            if len(blocks) >= 4:  # Minimum for a table
                # Sort blocks by position
                blocks.sort(key=lambda b: (b['bbox'].y, b['bbox'].x))
                
                # Group blocks into potential table regions
                table_candidates = self._group_blocks_into_tables(blocks)
                
                for candidate in table_candidates:
                    table_regions.append(candidate)
                    
        return table_regions
        
    def _group_blocks_into_tables(self, blocks: List[Dict]) -> List[TableRegion]:
        """Group blocks into potential table regions."""
        table_regions = []
        
        if len(blocks) < 4:
            return table_regions
            
        # Simple heuristic: look for aligned blocks
        rows = []
        current_row = []
        current_y = None
        row_threshold = 20  # Pixels
        
        for block in blocks:
            block_y = block['bbox'].y
            
            if current_y is None or abs(block_y - current_y) <= row_threshold:
                current_row.append(block)
                current_y = block_y
            else:
                if len(current_row) >= 2:  # At least 2 columns
                    rows.append(current_row)
                current_row = [block]
                current_y = block_y
                
        # Add last row
        if len(current_row) >= 2:
            rows.append(current_row)
            
        # If we have multiple rows, consider it a table
        if len(rows) >= 2:
            all_blocks = [block for row in rows for block in row]
            
            min_x = min(block['bbox'].x for block in all_blocks)
            max_x = max(block['bbox'].x + block['bbox'].width for block in all_blocks)
            min_y = min(block['bbox'].y for block in all_blocks)
            max_y = max(block['bbox'].y + block['bbox'].height for block in all_blocks)
            
            avg_confidence = np.mean([block['confidence'] for block in all_blocks])
            
            table_regions.append(TableRegion(
                bounding_box=BoundingBox(
                    x=min_x,
                    y=min_y,
                    width=max_x - min_x,
                    height=max_y - min_y,
                    confidence=avg_confidence
                ),
                confidence=avg_confidence,
                page_number=1
            ))
            
        return table_regions
        
    def set_fallback_engines(self, fallback_engines: List[str]) -> None:
        """Set list of fallback engine names."""
        self.fallback_engines = fallback_engines
        logger.info(f"Set fallback engines for Cloud Vision: {fallback_engines}")
    
    def get_fallback_engines(self) -> List[str]:
        """Get list of fallback engine names."""
        return self.fallback_engines.copy()
    
    def is_service_available(self) -> bool:
        """Check if the Cloud Vision service is currently available."""
        return self._check_service_availability()
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        self.circuit_breaker_failures = 0
        self.circuit_breaker_reset_time = None
        self.service_available = True
        logger.info("Cloud Vision circuit breaker manually reset")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            'daily_request_count': self.daily_request_count,
            'daily_request_limit': self.daily_request_limit,
            'daily_reset_time': self.daily_reset_time.isoformat() if self.daily_reset_time else None,
            'circuit_breaker_failures': self.circuit_breaker_failures,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'circuit_breaker_active': self.circuit_breaker_reset_time is not None,
            'circuit_breaker_reset_time': self.circuit_breaker_reset_time.isoformat() if self.circuit_breaker_reset_time else None,
            'service_available': self.service_available,
            'last_availability_check': self.last_availability_check.isoformat() if self.last_availability_check else None
        }
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get comprehensive information about Cloud Vision engine."""
        info = super().get_engine_info()
        info.update({
            'detect_handwriting': self.detect_handwriting,
            'detect_tables': self.detect_tables,
            'language_hints': self.language_hints,
            'requests_per_minute': self.requests_per_minute,
            'daily_request_limit': self.daily_request_limit,
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'fallback_engines': self.fallback_engines,
            'has_credentials': self.client is not None,
            'has_credentials_path': bool(self.credentials_path),
            'has_credentials_json': bool(self.credentials_json),
            'project_id': self.project_id,
            'usage_stats': self.get_usage_stats()
        })
        return info
    
    def cleanup(self) -> None:
        """Cleanup resources and close client connection."""
        if self.client:
            try:
                # Close the client if it has a close method
                if hasattr(self.client, 'close'):
                    self.client.close()
            except Exception as e:
                logger.warning(f"Error closing Cloud Vision client: {e}")
            finally:
                self.client = None
        
        super().cleanup()