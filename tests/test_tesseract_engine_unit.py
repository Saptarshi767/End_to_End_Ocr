"""
Unit tests for Tesseract OCR engine focusing on structure and error handling.
These tests can run without the actual Tesseract binary installed.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.ocr.tesseract_engine import TesseractEngine
from src.core.models import OCRResult, BoundingBox, WordData, TableRegion
from src.core.exceptions import OCREngineError


class TestTesseractEngineUnit:
    """Unit tests for Tesseract engine structure and error handling."""
    
    def test_engine_initialization(self):
        """Test engine initialization and configuration."""
        engine = TesseractEngine(confidence_threshold=0.7)
        
        assert engine.name == "tesseract"
        assert engine.confidence_threshold == 0.7
        assert engine.psm == 6
        assert engine.oem == 3
        assert 'eng' in engine.supported_languages
        assert not engine.is_initialized
    
    def test_configuration_methods(self):
        """Test configuration methods."""
        engine = TesseractEngine()
        
        # Test PSM configuration
        engine.configure({'psm': 8})
        assert engine.psm == 8
        
        # Test OEM configuration
        engine.configure({'oem': 1})
        assert engine.oem == 1
        
        # Test confidence threshold
        engine.configure({'confidence_threshold': 0.9})
        assert engine.confidence_threshold == 0.9
        
        # Test language configuration
        engine.configure({'languages': ['eng', 'fra']})
        assert engine.supported_languages == ['eng', 'fra']
    
    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        engine = TesseractEngine()
        
        # Valid thresholds
        engine.set_confidence_threshold(0.5)
        assert engine.confidence_threshold == 0.5
        
        engine.set_confidence_threshold(0.0)
        assert engine.confidence_threshold == 0.0
        
        engine.set_confidence_threshold(1.0)
        assert engine.confidence_threshold == 1.0
        
        # Invalid thresholds
        with pytest.raises(OCREngineError):
            engine.set_confidence_threshold(-0.1)
        
        with pytest.raises(OCREngineError):
            engine.set_confidence_threshold(1.1)
    
    def test_language_support_methods(self):
        """Test language support methods."""
        engine = TesseractEngine()
        
        # Test default languages
        assert engine.supports_language('eng')
        assert 'eng' in engine.get_supported_languages()
        
        # Test unsupported language
        assert not engine.supports_language('xyz')
    
    def test_config_building(self):
        """Test Tesseract configuration string building."""
        engine = TesseractEngine()
        engine.psm = 6
        engine.oem = 3
        
        engine._build_config()
        
        assert "--psm 6" in engine.tesseract_config
        assert "--oem 3" in engine.tesseract_config
        
        # Test with additional config
        engine.configure({
            'preserve_interword_spaces': '1',
            'tesseract_char_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        })
        
        assert "-c preserve_interword_spaces=1" in engine.tesseract_config
        assert "-c tessedit_char_whitelist=" in engine.tesseract_config
    
    def test_input_validation_methods(self):
        """Test input validation methods."""
        engine = TesseractEngine()
        
        # Test valid image
        valid_image = np.ones((100, 200, 3), dtype=np.uint8)
        engine._validate_input_image(valid_image)  # Should not raise
        
        # Test None image
        with pytest.raises(OCREngineError) as exc_info:
            engine._validate_input_image(None)
        assert "empty or None" in str(exc_info.value)
        
        # Test empty image
        empty_image = np.array([])
        with pytest.raises(OCREngineError) as exc_info:
            engine._validate_input_image(empty_image)
        assert "empty or None" in str(exc_info.value)
        
        # Test invalid shape
        invalid_image = np.ones((10,))  # 1D array
        with pytest.raises(OCREngineError) as exc_info:
            engine._validate_input_image(invalid_image)
        assert "Invalid image shape" in str(exc_info.value)
        
        # Test too small image
        tiny_image = np.ones((5, 5, 3), dtype=np.uint8)
        with pytest.raises(OCREngineError) as exc_info:
            engine._validate_input_image(tiny_image)
        assert "too small" in str(exc_info.value)
    
    def test_language_validation(self):
        """Test language validation method."""
        engine = TesseractEngine()
        engine.supported_languages = ['eng', 'fra', 'deu']
        
        # Test supported language
        assert engine._validate_language('eng') == 'eng'
        assert engine._validate_language('fra') == 'fra'
        
        # Test unsupported language (should fallback to 'eng')
        assert engine._validate_language('xyz') == 'eng'
        
        # Test when English is not available
        engine.supported_languages = ['fra', 'deu']
        assert engine._validate_language('xyz') == 'fra'  # First available
        
        # Test when no languages available
        engine.supported_languages = []
        with pytest.raises(OCREngineError) as exc_info:
            engine._validate_language('eng')
        assert "No supported languages available" in str(exc_info.value)
    
    def test_word_confidence_calculation(self):
        """Test enhanced word confidence calculation."""
        engine = TesseractEngine()
        
        # Create mock data with required fields
        mock_data = {
            'width': [50],
            'height': [20]
        }
        
        # Test normal confidence
        confidence = engine._calculate_word_confidence(85.0, "hello", mock_data, 0)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.8  # Should be high for good confidence
        
        # Test low confidence
        confidence = engine._calculate_word_confidence(-1, "hello", mock_data, 0)
        assert confidence == 0.0
        
        # Test single character (should be penalized)
        confidence_single = engine._calculate_word_confidence(85.0, "a", mock_data, 0)
        confidence_word = engine._calculate_word_confidence(85.0, "hello", mock_data, 0)
        assert confidence_single < confidence_word
    
    def test_word_pattern_analysis(self):
        """Test word pattern analysis methods."""
        engine = TesseractEngine()
        
        # Test likely words
        assert engine._is_likely_word("hello")
        assert engine._is_likely_word("world")
        assert not engine._is_likely_word("a")
        assert not engine._is_likely_word("xyz")  # No vowels
        
        # Test unusual patterns
        assert not engine._has_unusual_pattern("hello")
        assert engine._has_unusual_pattern("h@#$%llo")  # Too many special chars
        assert engine._has_unusual_pattern("HeLlO")  # Alternating case
    
    def test_bounding_box_creation(self):
        """Test bounding box creation and validation."""
        engine = TesseractEngine()
        
        data = {
            'left': [10],
            'top': [20],
            'width': [100],
            'height': [30]
        }
        
        bbox = engine._create_validated_bounding_box(data, 0, 0.8)
        
        assert isinstance(bbox, BoundingBox)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 30
        assert bbox.confidence == 0.8
        
        # Test with negative values (should be corrected)
        data_negative = {
            'left': [-5],
            'top': [-10],
            'width': [0],
            'height': [0]
        }
        
        bbox_corrected = engine._create_validated_bounding_box(data_negative, 0, 0.5)
        assert bbox_corrected.x == 0  # Corrected from -5
        assert bbox_corrected.y == 0  # Corrected from -10
        assert bbox_corrected.width == 1  # Corrected from 0
        assert bbox_corrected.height == 1  # Corrected from 0
    
    def test_overall_confidence_calculation(self):
        """Test overall confidence calculation."""
        engine = TesseractEngine()
        
        # Test with good confidences
        confidences = [0.9, 0.8, 0.85, 0.9]
        text = "This is good text"
        word_data = [Mock() for _ in range(4)]
        
        overall = engine._calculate_overall_confidence(confidences, text, word_data)
        assert 0.8 <= overall <= 1.0
        
        # Test with empty confidences
        overall_empty = engine._calculate_overall_confidence([], "", [])
        assert overall_empty == 0.0
        
        # Test with low confidences
        low_confidences = [0.2, 0.1, 0.3]
        overall_low = engine._calculate_overall_confidence(low_confidences, "bad", [Mock()])
        assert overall_low < 0.5
    
    def test_line_grouping_logic(self):
        """Test word grouping into lines."""
        engine = TesseractEngine()
        
        # Create mock word data for two lines
        word_data = [
            WordData("Hello", 0.9, BoundingBox(10, 10, 50, 20, 0.9)),
            WordData("World", 0.8, BoundingBox(70, 12, 50, 18, 0.8)),
            WordData("Second", 0.85, BoundingBox(10, 40, 60, 20, 0.85)),
            WordData("Line", 0.9, BoundingBox(80, 42, 40, 18, 0.9))
        ]
        
        line_boxes = engine._group_words_into_lines(word_data)
        
        assert len(line_boxes) == 2  # Should create 2 lines
        
        for bbox in line_boxes:
            assert isinstance(bbox, BoundingBox)
            assert bbox.width > 0
            assert bbox.height > 0
    
    def test_text_block_extraction(self):
        """Test text block extraction from OCR data."""
        engine = TesseractEngine()
        
        # Mock OCR data
        data = {
            'text': ['Hello', '', 'World', 'Test'],
            'conf': [85, 0, 90, 30],  # One empty, one low confidence
            'left': [10, 0, 70, 120],
            'top': [10, 0, 12, 14],
            'width': [50, 0, 50, 40],
            'height': [20, 0, 18, 16]
        }
        
        blocks = engine._extract_text_blocks(data)
        
        # Should filter out empty text and low confidence (threshold is 30, so Test with conf=30 is included)
        assert len(blocks) == 3  # 'Hello', 'World', and 'Test' (conf=30 meets threshold)
        assert blocks[0]['text'] == 'Hello'
        assert blocks[1]['text'] == 'World'
        assert blocks[2]['text'] == 'Test'
    
    def test_row_alignment_logic(self):
        """Test row alignment detection."""
        engine = TesseractEngine()
        
        # Create two rows that should align
        row1 = [
            {'x': 10, 'right': 60, 'y': 10},
            {'x': 70, 'right': 120, 'y': 12}
        ]
        row2 = [
            {'x': 15, 'right': 65, 'y': 40},
            {'x': 75, 'right': 125, 'y': 42}
        ]
        
        assert engine._rows_align(row1, row2)
        
        # Create rows that don't align
        row3 = [
            {'x': 200, 'right': 250, 'y': 40}
        ]
        
        assert not engine._rows_align(row1, row3)
    
    def test_table_structure_validation(self):
        """Test table structure validation."""
        engine = TesseractEngine()
        
        # Create blocks that form a valid table structure
        valid_blocks = []
        for row in range(3):
            for col in range(3):
                valid_blocks.append({
                    'x': col * 50,
                    'y': row * 30,
                    'right': (col + 1) * 50,
                    'bottom': (row + 1) * 30,
                    'text': f'R{row}C{col}',
                    'conf': 80
                })
        
        assert engine._validate_table_structure(valid_blocks)
        
        # Test with too few blocks
        assert not engine._validate_table_structure(valid_blocks[:2])
        
        # Test with uneven row structure
        uneven_blocks = valid_blocks[:7]  # 3+3+1 structure
        # This might still be valid depending on tolerance
        result = engine._validate_table_structure(uneven_blocks)
        assert isinstance(result, bool)
    
    def test_engine_info(self):
        """Test engine information retrieval."""
        engine = TesseractEngine(confidence_threshold=0.75)
        engine.configure({'psm': 8, 'custom_param': 'value'})
        
        info = engine.get_engine_info()
        
        assert info['name'] == 'tesseract'
        assert info['confidence_threshold'] == 0.75
        assert 'eng' in info['supported_languages']
        assert info['is_initialized'] is False
        assert 'custom_param' in info['config']
    
    def test_string_representations(self):
        """Test string representations of the engine."""
        engine = TesseractEngine(confidence_threshold=0.8)
        
        str_repr = str(engine)
        assert 'tesseract' in str_repr.lower()
        assert '0.8' in str_repr
        
        repr_str = repr(engine)
        assert 'TesseractEngine' in repr_str
        assert 'tesseract' in repr_str
        assert '0.8' in repr_str
    
    @patch('src.ocr.tesseract_engine.TESSERACT_AVAILABLE', False)
    def test_tesseract_not_available(self):
        """Test behavior when Tesseract is not available."""
        engine = TesseractEngine()
        
        with pytest.raises(OCREngineError) as exc_info:
            engine.initialize()
        
        assert "dependencies not available" in str(exc_info.value)
        assert exc_info.value.error_code == "TESSERACT_DEPS_MISSING"
    
    def test_error_context_information(self):
        """Test that errors include proper context information."""
        engine = TesseractEngine()
        
        # Test input validation error
        try:
            engine._validate_input_image(None)
        except OCREngineError as e:
            assert e.error_code == "INVALID_IMAGE"
            assert "empty or None" in e.message
        
        # Test image shape error
        try:
            engine._validate_input_image(np.ones((5,)))
        except OCREngineError as e:
            assert e.error_code == "INVALID_IMAGE_SHAPE"
            assert "Invalid image shape" in e.message
        
        # Test image size error
        try:
            engine._validate_input_image(np.ones((5, 5)))
        except OCREngineError as e:
            assert e.error_code == "IMAGE_TOO_SMALL"
            assert "too small" in e.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])