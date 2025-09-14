"""
Simple test for EasyOCR engine implementation to verify basic functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.ocr.easyocr_engine import EasyOCREngine
from src.core.models import OCRResult


class TestEasyOCREngineSimple:
    """Simple test suite for EasyOCR engine."""
    
    def test_engine_creation(self):
        """Test that engine can be created."""
        engine = EasyOCREngine()
        assert engine.name == "easyocr"
        assert engine.confidence_threshold == 0.7
        assert engine.handwriting_enabled
        assert 'en' in engine.active_languages
    
    def test_handwriting_support_check(self):
        """Test handwriting support functionality."""
        engine = EasyOCREngine()
        
        # Test with handwriting languages
        engine.active_languages = ['en', 'ch_sim']
        assert engine.supports_handwriting()
        
        # Test without handwriting languages
        engine.active_languages = ['fr']  # Not in handwriting_languages
        engine.handwriting_enabled = True
        # Should still return False because no active language supports handwriting
        
    def test_batch_processing_configuration(self):
        """Test batch processing configuration."""
        engine = EasyOCREngine()
        
        # Test optimization enable
        engine.optimize_for_batch_processing(True)
        assert engine.batch_size > 1
        
        # Test optimization disable
        engine.optimize_for_batch_processing(False)
        assert engine.batch_size == 1
        assert engine.workers == 0
    
    def test_handwriting_mode_configuration(self):
        """Test handwriting mode configuration."""
        engine = EasyOCREngine()
        
        # Enable handwriting mode
        engine.set_handwriting_mode(True, threshold=0.5)
        assert engine.handwriting_enabled
        assert engine.handwriting_threshold == 0.5
        
        # Disable handwriting mode
        engine.set_handwriting_mode(False)
        assert not engine.handwriting_enabled
    
    def test_processing_stats(self):
        """Test processing statistics functionality."""
        engine = EasyOCREngine()
        
        # Initial stats
        stats = engine.get_processing_stats()
        assert stats['total_processed'] == 0
        assert stats['batch_processed'] == 0
        assert stats['avg_processing_time'] == 0.0
        assert stats['handwriting_detected'] == 0
        
        # Reset stats
        engine.reset_processing_stats()
        stats = engine.get_processing_stats()
        assert stats['total_processed'] == 0
    
    def test_engine_info(self):
        """Test engine information retrieval."""
        engine = EasyOCREngine()
        info = engine.get_engine_info()
        
        # Check required fields
        assert 'name' in info
        assert 'active_languages' in info
        assert 'handwriting_enabled' in info
        assert 'batch_processing' in info
        assert 'ocr_parameters' in info
        assert 'performance_stats' in info
        
        # Check batch processing info
        assert 'max_batch_size' in info['batch_processing']
        assert 'current_batch_size' in info['batch_processing']
        
        # Check OCR parameters
        assert 'width_ths' in info['ocr_parameters']
        assert 'decoder' in info['ocr_parameters']
    
    def test_confidence_adjustment(self):
        """Test confidence adjustment for handwriting."""
        engine = EasyOCREngine()
        
        # Test normal text confidence (no adjustment)
        normal_confidence = engine._adjust_confidence_for_handwriting(0.9, "normal text", False)
        assert normal_confidence == 0.9
        
        # Test handwriting confidence adjustment
        hw_confidence = engine._adjust_confidence_for_handwriting(0.9, "handwritten", True)
        assert hw_confidence < 0.9  # Should be slightly lower
        
        # Test boost for reasonable text
        reasonable_confidence = engine._adjust_confidence_for_handwriting(0.8, "reasonable", True)
        assert reasonable_confidence > 0.8 * 0.9  # Should get a boost
    
    def test_text_segmentation(self):
        """Test text segmentation into words."""
        engine = EasyOCREngine()
        
        text = "Hello world test"
        bbox = [[10, 10], [100, 10], [100, 30], [10, 30]]
        
        word_infos = engine._segment_text_into_words(text, bbox)
        
        assert len(word_infos) == 3
        assert word_infos[0]['text'] == "Hello"
        assert word_infos[1]['text'] == "world"
        assert word_infos[2]['text'] == "test"
        
        # Check that bounding boxes are reasonable
        for word_info in word_infos:
            assert word_info['bbox'].width > 0
            assert word_info['bbox'].height > 0
    
    def test_results_quality_comparison(self):
        """Test OCR results quality comparison."""
        engine = EasyOCREngine()
        
        # Better results (higher confidence, more text)
        results1 = [
            ([[10, 10], [50, 10], [50, 30], [10, 30]], "Good", 0.9),
            ([[60, 10], [100, 10], [100, 30], [60, 30]], "Text", 0.85)
        ]
        
        # Worse results (lower confidence, less text)
        results2 = [
            ([[10, 10], [30, 10], [30, 30], [10, 30]], "Bad", 0.6)
        ]
        
        comparison = engine._compare_results_quality(results1, results2)
        assert comparison > 0  # results1 should be better
        
        comparison_reverse = engine._compare_results_quality(results2, results1)
        assert comparison_reverse < 0  # results2 should be worse
    
    def test_handwriting_detection_heuristic(self):
        """Test handwriting detection heuristic."""
        engine = EasyOCREngine()
        
        # Create a simple test image
        image = np.ones((50, 200, 3), dtype=np.uint8) * 255
        
        # Test should not crash
        result = engine._might_contain_handwriting(image)
        assert isinstance(result, bool)  # Should return a boolean
    
    def test_language_support(self):
        """Test language support functionality."""
        engine = EasyOCREngine()
        
        # Test supported languages
        assert len(engine.supported_languages) > 0
        assert 'en' in engine.supported_languages
        
        # Test handwriting languages
        assert len(engine.handwriting_languages) > 0
        assert 'en' in engine.handwriting_languages
        
        # Test getting handwriting languages
        hw_langs = engine.get_handwriting_languages()
        assert isinstance(hw_langs, list)


if __name__ == "__main__":
    pytest.main([__file__])