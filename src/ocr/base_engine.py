"""
Base OCR engine implementation and abstract classes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import time
import numpy as np

from ..core.interfaces import OCREngineInterface
from ..core.models import OCRResult, BoundingBox, WordData, TableRegion, OCREngine
from ..core.exceptions import OCREngineError


logger = logging.getLogger(__name__)


class BaseOCREngine(OCREngineInterface):
    """
    Base implementation for OCR engines with common functionality.
    """
    
    def __init__(self, name: str, confidence_threshold: float = 0.8):
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.supported_languages = ['eng']  # Default to English
        self.is_initialized = False
        self.config = {}
        
    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for this engine."""
        return self.confidence_threshold
        
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for this engine."""
        if not 0.0 <= threshold <= 1.0:
            raise OCREngineError(f"Confidence threshold must be between 0.0 and 1.0, got {threshold}")
        self.confidence_threshold = threshold
        logger.info(f"Set confidence threshold for {self.name}: {threshold}")
        
    def supports_language(self, language: str) -> bool:
        """Check if engine supports specified language."""
        return language in self.supported_languages
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
        
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the OCR engine with parameters."""
        self.config.update(config)
        
        # Update confidence threshold if provided
        if 'confidence_threshold' in config:
            self.set_confidence_threshold(config['confidence_threshold'])
            
        # Update supported languages if provided
        if 'languages' in config:
            self.supported_languages = config['languages']
            
        logger.info(f"Configured {self.name} engine with {len(config)} parameters")
        
    def is_available(self) -> bool:
        """Check if the OCR engine is available and ready to use."""
        return self.is_initialized
        
    def initialize(self) -> None:
        """Initialize the OCR engine. Override in subclasses."""
        self.is_initialized = True
        logger.info(f"Initialized {self.name} OCR engine")
        
    def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses if needed."""
        self.is_initialized = False
        logger.info(f"Cleaned up {self.name} OCR engine")
        
    @abstractmethod
    def _extract_text_impl(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Internal implementation of text extraction.
        Must be implemented by subclasses.
        """
        pass
        
    def extract_text(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Extract text from image using OCR.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional parameters for OCR processing
            
        Returns:
            OCR result with extracted text and metadata
            
        Raises:
            OCREngineError: If OCR processing fails
        """
        if not self.is_available():
            raise OCREngineError(f"OCR engine {self.name} is not initialized")
            
        if image is None or image.size == 0:
            raise OCREngineError("Input image is empty or None")
            
        start_time = time.time()
        
        try:
            # Validate image format
            if len(image.shape) not in [2, 3]:
                raise OCREngineError(f"Invalid image shape: {image.shape}")
                
            # Call implementation
            result = self._extract_text_impl(image, **kwargs)
            
            # Set processing time
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time
            
            # Set engine used (try to match enum, fallback to AUTO if not found)
            try:
                result.engine_used = OCREngine(self.name)
            except ValueError:
                result.engine_used = OCREngine.AUTO
            
            # Validate confidence threshold
            if result.confidence < self.confidence_threshold:
                logger.warning(f"OCR confidence {result.confidence:.2f} below threshold {self.confidence_threshold}")
                
            logger.debug(f"OCR extraction completed in {processing_time}ms with confidence {result.confidence:.2f}")
            
            return result
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"OCR extraction failed after {processing_time}ms: {str(e)}")
            raise OCREngineError(f"OCR extraction failed: {str(e)}")
            
    def detect_tables(self, image: np.ndarray, **kwargs) -> List[TableRegion]:
        """
        Detect table regions in document.
        Base implementation returns empty list - override in subclasses.
        """
        logger.warning(f"Table detection not implemented for {self.name} engine")
        return []
        
    def preprocess_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        Base implementation returns image unchanged - override in subclasses.
        """
        return image
        
    def validate_result(self, result: OCRResult) -> bool:
        """
        Validate OCR result quality.
        
        Args:
            result: OCR result to validate
            
        Returns:
            True if result meets quality criteria
        """
        if not result.text or not result.text.strip():
            return False
            
        if result.confidence < self.confidence_threshold:
            return False
            
        # Check for reasonable text content
        if len(result.text.strip()) < 3:
            return False
            
        return True
        
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about this OCR engine."""
        return {
            'name': self.name,
            'confidence_threshold': self.confidence_threshold,
            'supported_languages': self.supported_languages,
            'is_initialized': self.is_initialized,
            'config': self.config.copy()
        }
        
    def __str__(self) -> str:
        return f"{self.name}OCREngine(threshold={self.confidence_threshold})"
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', threshold={self.confidence_threshold})"


class MockOCREngine(BaseOCREngine):
    """
    Mock OCR engine for testing purposes.
    """
    
    def __init__(self, name: str = "mock", confidence_threshold: float = 0.9):
        super().__init__(name, confidence_threshold)
        self.mock_text = "Sample extracted text from mock OCR engine"
        self.mock_confidence = 0.95
        
    def _extract_text_impl(self, image: np.ndarray, **kwargs) -> OCRResult:
        """Mock implementation that returns predefined text."""
        # Simulate processing time
        time.sleep(0.1)
        
        # Create mock bounding boxes
        height, width = image.shape[:2]
        bounding_boxes = [
            BoundingBox(x=10, y=10, width=width-20, height=30, confidence=self.mock_confidence)
        ]
        
        # Create mock word data
        words = self.mock_text.split()
        word_data = []
        x_offset = 10
        for word in words:
            word_width = len(word) * 8  # Approximate width
            word_data.append(WordData(
                text=word,
                confidence=self.mock_confidence,
                bounding_box=BoundingBox(x=x_offset, y=10, width=word_width, height=20, confidence=self.mock_confidence)
            ))
            x_offset += word_width + 5
            
        return OCRResult(
            text=self.mock_text,
            confidence=self.mock_confidence,
            bounding_boxes=bounding_boxes,
            word_level_data=word_data
        )
        
    def set_mock_result(self, text: str, confidence: float = 0.95) -> None:
        """Set the mock result for testing."""
        self.mock_text = text
        self.mock_confidence = confidence
        
    def detect_tables(self, image: np.ndarray, **kwargs) -> List[TableRegion]:
        """Mock table detection."""
        height, width = image.shape[:2]
        return [
            TableRegion(
                bounding_box=BoundingBox(x=50, y=50, width=width-100, height=height-100, confidence=0.9),
                confidence=0.9,
                page_number=1
            )
        ]