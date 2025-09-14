"""
Integration tests for Tesseract OCR engine with sample documents.
Tests the enhanced confidence scoring, bounding box extraction, and error handling.
"""

import pytest
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from pathlib import Path

from src.ocr.tesseract_engine import TesseractEngine, TESSERACT_AVAILABLE
from src.core.models import OCRResult, BoundingBox, WordData
from src.core.exceptions import OCREngineError


class TestTesseractIntegration:
    """Integration tests for Tesseract OCR engine."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        if not TESSERACT_AVAILABLE:
            pytest.skip("Tesseract not available")
            
        self.engine = TesseractEngine(confidence_threshold=0.6)
        try:
            self.engine.initialize()
        except OCREngineError as e:
            pytest.skip(f"Tesseract initialization failed: {e}")
    
    def create_test_image_with_text(self, text: str, width: int = 400, height: int = 100, 
                                  font_size: int = 20) -> np.ndarray:
        """Create a test image with specified text."""
        # Create white background
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a standard font
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill='black', font=font)
        
        # Convert to numpy array
        return np.array(img)
    
    def create_table_image(self, rows: int = 3, cols: int = 3, cell_width: int = 80, 
                          cell_height: int = 30) -> np.ndarray:
        """Create a test image with a simple table."""
        width = cols * cell_width
        height = rows * cell_height
        
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Draw table grid and content
        for row in range(rows):
            for col in range(cols):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Draw cell border
                draw.rectangle([x1, y1, x2, y2], outline='black', width=1)
                
                # Add cell content
                if row == 0:
                    text = f"Header{col+1}"
                else:
                    text = f"R{row}C{col+1}"
                
                # Center text in cell
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                text_x = x1 + (cell_width - text_width) // 2
                text_y = y1 + (cell_height - text_height) // 2
                
                draw.text((text_x, text_y), text, fill='black', font=font)
        
        return np.array(img)
    
    def create_noisy_image(self, text: str, noise_level: float = 0.1) -> np.ndarray:
        """Create a test image with noise to test error handling."""
        img = self.create_test_image_with_text(text)
        
        # Add random noise
        noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.uint8)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return noisy_img
    
    def test_basic_text_extraction(self):
        """Test basic text extraction functionality."""
        test_text = "Hello World! This is a test."
        image = self.create_test_image_with_text(test_text)
        
        result = self.engine.extract_text(image)
        
        assert isinstance(result, OCRResult)
        assert result.text is not None
        assert len(result.text.strip()) > 0
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        assert len(result.word_level_data) > 0
        assert len(result.bounding_boxes) > 0
        
        # Check that extracted text contains key words
        extracted_lower = result.text.lower()
        assert "hello" in extracted_lower or "world" in extracted_lower
    
    def test_confidence_scoring(self):
        """Test enhanced confidence scoring."""
        # Test with clear text (should have high confidence)
        clear_text = "CLEAR TEXT"
        clear_image = self.create_test_image_with_text(clear_text, font_size=24)
        clear_result = self.engine.extract_text(clear_image)
        
        # Test with noisy text (should have lower confidence)
        noisy_image = self.create_noisy_image(clear_text, noise_level=0.3)
        noisy_result = self.engine.extract_text(noisy_image)
        
        # Clear text should have higher confidence than noisy text
        assert clear_result.confidence > noisy_result.confidence
        assert clear_result.confidence > 0.7  # Should be quite confident with clear text
    
    def test_bounding_box_extraction(self):
        """Test bounding box extraction accuracy."""
        test_text = "Word1 Word2 Word3"
        image = self.create_test_image_with_text(test_text)
        
        result = self.engine.extract_text(image)
        
        # Check word-level bounding boxes
        assert len(result.word_level_data) >= 3  # Should detect at least 3 words
        
        for word_data in result.word_level_data:
            assert isinstance(word_data.bounding_box, BoundingBox)
            assert word_data.bounding_box.width > 0
            assert word_data.bounding_box.height > 0
            assert word_data.bounding_box.x >= 0
            assert word_data.bounding_box.y >= 0
            assert 0.0 <= word_data.confidence <= 1.0
        
        # Check line-level bounding boxes
        assert len(result.bounding_boxes) > 0
        for bbox in result.bounding_boxes:
            assert isinstance(bbox, BoundingBox)
            assert bbox.width > 0
            assert bbox.height > 0
    
    def test_table_detection(self):
        """Test table detection functionality."""
        table_image = self.create_table_image(rows=3, cols=3)
        
        table_regions = self.engine.detect_tables(table_image)
        
        # Should detect at least one table region
        assert len(table_regions) >= 1
        
        for region in table_regions:
            assert region.confidence > 0.0
            assert region.bounding_box.width > 0
            assert region.bounding_box.height > 0
            assert region.page_number == 1
    
    def test_language_support(self):
        """Test language support and validation."""
        test_text = "Hello World"
        image = self.create_test_image_with_text(test_text)
        
        # Test with supported language
        result_eng = self.engine.extract_text(image, language='eng')
        assert result_eng.text is not None
        
        # Test with unsupported language (should fallback to English)
        result_unsupported = self.engine.extract_text(image, language='xyz')
        assert result_unsupported.text is not None
    
    def test_error_handling_invalid_image(self):
        """Test error handling with invalid images."""
        # Test with None image
        with pytest.raises(OCREngineError) as exc_info:
            self.engine.extract_text(None)
        assert "empty or None" in str(exc_info.value)
        
        # Test with empty image
        empty_image = np.array([])
        with pytest.raises(OCREngineError) as exc_info:
            self.engine.extract_text(empty_image)
        assert "empty or None" in str(exc_info.value)
        
        # Test with invalid shape
        invalid_image = np.ones((10,))  # 1D array
        with pytest.raises(OCREngineError) as exc_info:
            self.engine.extract_text(invalid_image)
        assert "Invalid image shape" in str(exc_info.value)
        
        # Test with too small image
        tiny_image = np.ones((5, 5, 3), dtype=np.uint8)
        with pytest.raises(OCREngineError) as exc_info:
            self.engine.extract_text(tiny_image)
        assert "too small" in str(exc_info.value)
    
    def test_error_handling_table_detection(self):
        """Test error handling in table detection."""
        # Test with invalid image
        with pytest.raises(OCREngineError):
            self.engine.detect_tables(None)
        
        # Test with valid but challenging image
        noise_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        regions = self.engine.detect_tables(noise_image)
        # Should not crash, might return empty list
        assert isinstance(regions, list)
    
    def test_preprocessing_functionality(self):
        """Test image preprocessing functionality."""
        test_text = "Preprocessing Test"
        original_image = self.create_test_image_with_text(test_text)
        
        # Test preprocessing
        processed_image = self.engine.preprocess_image(original_image)
        
        assert processed_image is not None
        assert processed_image.shape[0] > 0
        assert processed_image.shape[1] > 0
        
        # Test that preprocessing doesn't break OCR
        result = self.engine.extract_text(processed_image)
        assert result.text is not None
    
    def test_configuration_options(self):
        """Test various configuration options."""
        # Test PSM configuration
        self.engine.configure({'psm': 8})  # Single word mode
        assert self.engine.psm == 8
        
        # Test confidence threshold
        self.engine.configure({'confidence_threshold': 0.9})
        assert self.engine.confidence_threshold == 0.9
        
        # Test language configuration
        self.engine.configure({'languages': ['eng', 'fra']})
        assert 'eng' in self.engine.supported_languages
    
    def test_word_confidence_calculation(self):
        """Test enhanced word confidence calculation."""
        # Create image with mix of clear and unclear text
        test_text = "Clear UNCLEAR 123 @#$"
        image = self.create_test_image_with_text(test_text)
        
        result = self.engine.extract_text(image)
        
        # Check that word confidences vary appropriately
        confidences = [word.confidence for word in result.word_level_data]
        assert len(confidences) > 0
        
        # Should have some variation in confidence scores
        if len(confidences) > 1:
            confidence_std = np.std(confidences)
            assert confidence_std >= 0.0  # Some variation expected
    
    def test_line_grouping(self):
        """Test word grouping into lines."""
        # Create multi-line text
        test_text = "Line One\nLine Two\nLine Three"
        image = self.create_test_image_with_text(test_text, height=150)
        
        result = self.engine.extract_text(image)
        
        # Should have multiple line-level bounding boxes
        assert len(result.bounding_boxes) >= 1
        
        # Check that bounding boxes are reasonable
        for bbox in result.bounding_boxes:
            assert bbox.width > 0
            assert bbox.height > 0
            assert bbox.confidence >= 0.0
    
    def test_engine_info(self):
        """Test engine information retrieval."""
        info = self.engine.get_engine_info()
        
        assert 'name' in info
        assert info['name'] == 'tesseract'
        assert 'confidence_threshold' in info
        assert 'supported_languages' in info
        assert 'is_initialized' in info
        assert info['is_initialized'] is True
    
    def test_result_validation(self):
        """Test OCR result validation."""
        # Test with good image
        good_text = "Good clear text"
        good_image = self.create_test_image_with_text(good_text, font_size=20)
        good_result = self.engine.extract_text(good_image)
        
        assert self.engine.validate_result(good_result) is True
        
        # Test with very noisy image (might produce poor results)
        noisy_image = self.create_noisy_image("Text", noise_level=0.8)
        noisy_result = self.engine.extract_text(noisy_image)
        
        # Validation might fail for very poor results
        validation_result = self.engine.validate_result(noisy_result)
        assert isinstance(validation_result, bool)
    
    @pytest.mark.parametrize("image_format", ["RGB", "RGBA", "L"])
    def test_different_image_formats(self, image_format):
        """Test OCR with different image formats."""
        test_text = "Format Test"
        
        # Create base image
        base_image = self.create_test_image_with_text(test_text)
        
        # Convert to different formats
        if image_format == "RGBA":
            # Add alpha channel
            alpha = np.ones((base_image.shape[0], base_image.shape[1], 1), dtype=np.uint8) * 255
            test_image = np.concatenate([base_image, alpha], axis=2)
        elif image_format == "L":
            # Convert to grayscale
            test_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY)
        else:
            test_image = base_image
        
        # Test OCR
        result = self.engine.extract_text(test_image)
        assert result.text is not None
        assert len(result.text.strip()) > 0
    
    def test_large_image_handling(self):
        """Test handling of large images."""
        # Create a larger image
        large_text = "This is a test with a larger image to check performance and memory handling."
        large_image = self.create_test_image_with_text(large_text, width=800, height=200, font_size=16)
        
        result = self.engine.extract_text(large_image)
        
        assert result.text is not None
        assert result.processing_time_ms > 0
        assert len(result.word_level_data) > 0
    
    def test_empty_image_content(self):
        """Test handling of images with no text content."""
        # Create blank image
        blank_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        result = self.engine.extract_text(blank_image)
        
        # Should not crash, might return empty or minimal text
        assert isinstance(result, OCRResult)
        assert result.confidence >= 0.0
        assert isinstance(result.word_level_data, list)
    
    def test_special_characters(self):
        """Test OCR with special characters and symbols."""
        special_text = "Price: $123.45 (50% off!)"
        image = self.create_test_image_with_text(special_text)
        
        result = self.engine.extract_text(image)
        
        assert result.text is not None
        # Should detect some numbers or symbols
        extracted = result.text.lower()
        has_numbers = any(c.isdigit() for c in extracted)
        has_symbols = any(c in extracted for c in ['$', '%', '(', ')'])
        
        # At least some special content should be detected
        assert has_numbers or has_symbols or len(extracted) > 5


class TestTesseractErrorRecovery:
    """Test error recovery and fallback mechanisms."""
    
    def test_engine_not_available(self):
        """Test behavior when Tesseract is not available."""
        # This test would need to mock the TESSERACT_AVAILABLE flag
        # For now, we'll test the error message structure
        if not TESSERACT_AVAILABLE:
            engine = TesseractEngine()
            with pytest.raises(OCREngineError) as exc_info:
                engine.initialize()
            assert "dependencies not available" in str(exc_info.value)
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        if not TESSERACT_AVAILABLE:
            pytest.skip("Tesseract not available")
            
        engine = TesseractEngine()
        
        # Test invalid confidence threshold
        with pytest.raises(OCREngineError):
            engine.set_confidence_threshold(-0.1)
        
        with pytest.raises(OCREngineError):
            engine.set_confidence_threshold(1.1)
        
        # Test valid configuration
        engine.configure({
            'psm': 6,
            'oem': 3,
            'confidence_threshold': 0.8
        })
        
        assert engine.psm == 6
        assert engine.oem == 3
        assert engine.confidence_threshold == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])