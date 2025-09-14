"""
Comprehensive tests for enhanced EasyOCR engine implementation.
Tests multi-language support, handwriting recognition, batch processing, and accuracy comparison with Tesseract.
"""

import pytest
import numpy as np
import cv2
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.ocr.easyocr_engine import EasyOCREngine
from src.ocr.tesseract_engine import TesseractEngine
from src.core.models import OCRResult, BoundingBox, WordData
from src.core.exceptions import OCREngineError


class TestEasyOCREngineEnhanced:
    """Test suite for enhanced EasyOCR engine functionality."""
    
    @pytest.fixture
    def mock_easyocr_available(self):
        """Mock EasyOCR availability."""
        with patch('src.ocr.easyocr_engine.EASYOCR_AVAILABLE', True):
            with patch('easyocr.Reader') as mock_reader_class:
                with patch('torch.cuda.is_available') as mock_cuda:
                    mock_cuda.return_value = True
                    yield mock_reader_class, mock_cuda
    
    @pytest.fixture
    def engine(self, mock_easyocr_available):
        """Create EasyOCR engine instance for testing."""
        mock_reader_class, mock_cuda = mock_easyocr_available
        
        # Mock reader instance
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[[10, 10], [100, 10], [100, 30], [10, 30]]], "Sample text", 0.95)
        ]
        mock_reader_class.return_value = mock_reader
        
        engine = EasyOCREngine(confidence_threshold=0.7)
        engine.initialize()
        return engine
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(image, "Sample Text", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return image
    
    @pytest.fixture
    def handwriting_image(self):
        """Create a sample handwriting-like image."""
        image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        # Simulate handwriting with more irregular shapes
        cv2.putText(image, "Handwritten", (10, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)
        return image

    def test_initialization_with_handwriting_support(self, mock_easyocr_available):
        """Test engine initialization with handwriting support."""
        mock_reader_class, mock_cuda = mock_easyocr_available
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        
        engine = EasyOCREngine()
        engine.initialize()
        
        assert engine.is_initialized
        assert engine.handwriting_enabled
        assert 'en' in engine.active_languages
        assert engine.supports_handwriting()
    
    def test_multi_language_support(self, mock_easyocr_available):
        """Test multi-language support functionality."""
        mock_reader_class, mock_cuda = mock_easyocr_available
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        
        engine = EasyOCREngine()
        engine.configure({'languages': ['en', 'fr', 'de']})
        engine.initialize()
        
        assert set(engine.active_languages) == {'en', 'fr', 'de'}
        assert all(lang in engine.supported_languages for lang in engine.active_languages)
    
    def test_handwriting_detection_heuristic(self, engine, handwriting_image):
        """Test handwriting detection heuristic."""
        # Test with handwriting-like image
        might_be_handwriting = engine._might_contain_handwriting(handwriting_image)
        
        # Create a clean printed text image
        clean_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
        cv2.putText(clean_image, "CLEAN TEXT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        might_be_clean = engine._might_contain_handwriting(clean_image)
        
        # Handwriting should have higher edge ratio
        assert isinstance(might_be_handwriting, bool)
        assert isinstance(might_be_clean, bool)
    
    def test_handwriting_mode_configuration(self, engine):
        """Test handwriting mode configuration."""
        # Enable handwriting mode
        engine.set_handwriting_mode(True, threshold=0.5)
        assert engine.handwriting_enabled
        assert engine.handwriting_threshold == 0.5
        
        # Disable handwriting mode
        engine.set_handwriting_mode(False)
        assert not engine.handwriting_enabled
    
    def test_batch_processing_single_image(self, engine, sample_image):
        """Test batch processing with single image."""
        images = [sample_image]
        results = engine.extract_text_batch(images)
        
        assert len(results) == 1
        assert isinstance(results[0], OCRResult)
        assert results[0].text
        assert results[0].confidence > 0
    
    def test_batch_processing_multiple_images(self, engine, sample_image):
        """Test batch processing with multiple images."""
        # Create multiple test images
        images = []
        for i in range(5):
            img = sample_image.copy()
            cv2.putText(img, f"Text {i}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            images.append(img)
        
        results = engine.extract_text_batch(images, batch_size=3)
        
        assert len(results) == 5
        assert all(isinstance(result, OCRResult) for result in results)
        assert all(result.text for result in results)
    
    def test_batch_processing_optimization(self, engine):
        """Test batch processing optimization settings."""
        # Enable optimization
        engine.optimize_for_batch_processing(True)
        assert engine.batch_size > 1
        assert engine.workers >= 0
        
        # Disable optimization
        engine.optimize_for_batch_processing(False)
        assert engine.batch_size == 1
        assert engine.workers == 0
    
    def test_confidence_adjustment_for_handwriting(self, engine):
        """Test confidence adjustment for handwriting recognition."""
        # Test normal text confidence
        normal_confidence = engine._adjust_confidence_for_handwriting(0.9, "normal text", False)
        assert normal_confidence == 0.9
        
        # Test handwriting confidence adjustment
        hw_confidence = engine._adjust_confidence_for_handwriting(0.9, "handwritten", True)
        assert hw_confidence < 0.9  # Should be slightly lower
        
        # Test boost for reasonable text
        reasonable_confidence = engine._adjust_confidence_for_handwriting(0.8, "reasonable", True)
        assert reasonable_confidence > 0.8 * 0.9  # Should get a boost
    
    def test_text_segmentation_into_words(self, engine):
        """Test text segmentation into individual words."""
        text = "Hello world test"
        bbox = [[10, 10], [100, 10], [100, 30], [10, 30]]
        
        word_infos = engine._segment_text_into_words(text, bbox)
        
        assert len(word_infos) == 3
        assert word_infos[0]['text'] == "Hello"
        assert word_infos[1]['text'] == "world"
        assert word_infos[2]['text'] == "test"
        
        # Check that bounding boxes are reasonable
        for word_info in word_infos:
            assert isinstance(word_info['bbox'], BoundingBox)
            assert word_info['bbox'].width > 0
            assert word_info['bbox'].height > 0
    
    def test_results_quality_comparison(self, engine):
        """Test OCR results quality comparison."""
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
    
    def test_processing_statistics(self, engine, sample_image):
        """Test processing statistics tracking."""
        # Reset stats
        engine.reset_processing_stats()
        initial_stats = engine.get_processing_stats()
        assert initial_stats['total_processed'] == 0
        
        # Process some images
        engine.extract_text(sample_image)
        engine.extract_text_batch([sample_image, sample_image])
        
        stats = engine.get_processing_stats()
        assert stats['total_processed'] == 3  # 1 single + 2 batch
        assert stats['batch_processed'] == 2
        assert stats['avg_processing_time'] > 0
    
    def test_reader_management(self, engine):
        """Test reader creation and management for different languages."""
        # Test getting reader for default language
        reader1 = engine._get_or_create_reader(['en'])
        assert reader1 is not None
        
        # Test getting reader for different language
        reader2 = engine._get_or_create_reader(['fr'])
        assert reader2 is not None
        
        # Should have created separate readers
        assert len(engine.readers) >= 1
    
    def test_engine_info_comprehensive(self, engine):
        """Test comprehensive engine information."""
        info = engine.get_engine_info()
        
        # Check all expected fields are present
        expected_fields = [
            'name', 'confidence_threshold', 'supported_languages', 'is_initialized',
            'active_languages', 'all_supported_languages', 'handwriting_languages',
            'gpu_available', 'supports_handwriting', 'handwriting_enabled',
            'batch_processing', 'ocr_parameters', 'performance_stats'
        ]
        
        for field in expected_fields:
            assert field in info
        
        # Check batch processing info
        assert 'max_batch_size' in info['batch_processing']
        assert 'current_batch_size' in info['batch_processing']
        
        # Check OCR parameters
        assert 'width_ths' in info['ocr_parameters']
        assert 'decoder' in info['ocr_parameters']
    
    def test_error_handling_in_batch_processing(self, mock_easyocr_available):
        """Test error handling in batch processing."""
        mock_easyocr, mock_torch = mock_easyocr_available
        
        # Mock reader that raises exception
        mock_reader = Mock()
        mock_reader.readtext.side_effect = Exception("OCR failed")
        mock_easyocr.Reader.return_value = mock_reader
        
        engine = EasyOCREngine()
        engine.initialize()
        
        # Create test images
        images = [np.ones((50, 100, 3), dtype=np.uint8) * 255] * 3
        
        # Should handle errors gracefully
        results = engine.extract_text_batch(images)
        
        assert len(results) == 3
        # All results should be empty due to errors
        assert all(result.text == "" for result in results)
        assert all(result.confidence == 0.0 for result in results)
    
    def test_cleanup_resources(self, engine):
        """Test proper cleanup of engine resources."""
        # Ensure engine has some readers
        engine._get_or_create_reader(['en'])
        engine._get_or_create_reader(['fr'])
        
        assert len(engine.readers) >= 1
        
        # Cleanup
        engine.cleanup()
        
        assert not engine.is_initialized
        assert len(engine.readers) == 0


class TestEasyOCRVsTesseractAccuracy:
    """Test suite comparing EasyOCR and Tesseract accuracy."""
    
    @pytest.fixture
    def mock_engines_available(self):
        """Mock both engines being available."""
        with patch('src.ocr.easyocr_engine.EASYOCR_AVAILABLE', True):
            with patch('src.ocr.tesseract_engine.TESSERACT_AVAILABLE', True):
                with patch('src.ocr.easyocr_engine.easyocr') as mock_easyocr:
                    with patch('src.ocr.tesseract_engine.pytesseract') as mock_tesseract:
                        with patch('src.ocr.easyocr_engine.torch') as mock_torch:
                            mock_torch.cuda.is_available.return_value = False
                            yield mock_easyocr, mock_tesseract
    
    @pytest.fixture
    def test_images(self):
        """Create various test images for accuracy comparison."""
        images = {}
        
        # Clean printed text
        clean_img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        cv2.putText(clean_img, "Clean printed text", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        images['clean'] = clean_img
        
        # Noisy text
        noisy_img = clean_img.copy()
        noise = np.random.randint(0, 50, noisy_img.shape, dtype=np.uint8)
        noisy_img = cv2.add(noisy_img, noise)
        images['noisy'] = noisy_img
        
        # Small text
        small_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        cv2.putText(small_img, "Small text", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        images['small'] = small_img
        
        # Handwriting-like text
        hw_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(hw_img, "Handwritten", (10, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)
        images['handwriting'] = hw_img
        
        return images
    
    def test_accuracy_comparison_clean_text(self, mock_engines_available, test_images):
        """Compare accuracy on clean printed text."""
        mock_easyocr, mock_tesseract = mock_engines_available
        
        # Mock EasyOCR results
        mock_easyocr_reader = Mock()
        mock_easyocr_reader.readtext.return_value = [
            ([[[10, 10], [200, 10], [200, 30], [10, 30]]], "Clean printed text", 0.95)
        ]
        mock_easyocr.Reader.return_value = mock_easyocr_reader
        
        # Mock Tesseract results
        mock_tesseract.image_to_string.return_value = "Clean printed text"
        mock_tesseract.image_to_data.return_value = {
            'text': ['', '', '', '', '', 'Clean', 'printed', 'text'],
            'conf': [-1, -1, -1, -1, -1, 95, 93, 94],
            'left': [0, 0, 0, 0, 0, 10, 60, 120],
            'top': [0, 0, 0, 0, 0, 10, 10, 10],
            'width': [400, 400, 400, 400, 400, 45, 55, 40],
            'height': [100, 100, 100, 100, 100, 20, 20, 20]
        }
        mock_tesseract.get_tesseract_version.return_value = "5.0.0"
        mock_tesseract.get_languages.return_value = ['eng', 'fra', 'deu']
        
        # Initialize engines
        easyocr_engine = EasyOCREngine()
        easyocr_engine.initialize()
        
        tesseract_engine = TesseractEngine()
        tesseract_engine.initialize()
        
        # Test on clean image
        clean_image = test_images['clean']
        
        easyocr_result = easyocr_engine.extract_text(clean_image)
        tesseract_result = tesseract_engine.extract_text(clean_image)
        
        # Both should perform well on clean text
        assert easyocr_result.confidence > 0.8
        assert tesseract_result.confidence > 0.8
        assert "Clean" in easyocr_result.text
        assert "Clean" in tesseract_result.text
    
    def test_accuracy_comparison_handwriting(self, mock_engines_available, test_images):
        """Compare accuracy on handwriting-like text."""
        mock_easyocr, mock_tesseract = mock_engines_available
        
        # Mock EasyOCR results (should be better for handwriting)
        mock_easyocr_reader = Mock()
        mock_easyocr_reader.readtext.return_value = [
            ([[[10, 10], [150, 10], [150, 30], [10, 30]]], "Handwritten", 0.82)
        ]
        mock_easyocr.Reader.return_value = mock_easyocr_reader
        
        # Mock Tesseract results (typically worse for handwriting)
        mock_tesseract.image_to_string.return_value = "Handwrtten"  # Simulated OCR error
        mock_tesseract.image_to_data.return_value = {
            'text': ['', '', '', '', '', 'Handwrtten'],
            'conf': [-1, -1, -1, -1, -1, 65],
            'left': [0, 0, 0, 0, 0, 10],
            'top': [0, 0, 0, 0, 0, 10],
            'width': [300, 300, 300, 300, 300, 140],
            'height': [100, 100, 100, 100, 100, 20]
        }
        mock_tesseract.get_tesseract_version.return_value = "5.0.0"
        mock_tesseract.get_languages.return_value = ['eng']
        
        # Initialize engines
        easyocr_engine = EasyOCREngine()
        easyocr_engine.set_handwriting_mode(True)
        easyocr_engine.initialize()
        
        tesseract_engine = TesseractEngine()
        tesseract_engine.initialize()
        
        # Test on handwriting image
        hw_image = test_images['handwriting']
        
        easyocr_result = easyocr_engine.extract_text(hw_image, detect_handwriting=True)
        tesseract_result = tesseract_engine.extract_text(hw_image)
        
        # EasyOCR should perform better on handwriting
        assert easyocr_result.confidence >= tesseract_result.confidence
        # EasyOCR should have more accurate text
        assert "Handwritten" in easyocr_result.text
    
    def test_performance_comparison(self, mock_engines_available, test_images):
        """Compare processing performance between engines."""
        mock_easyocr, mock_tesseract = mock_engines_available
        
        # Setup mocks
        mock_easyocr_reader = Mock()
        mock_easyocr_reader.readtext.return_value = [
            ([[[10, 10], [100, 10], [100, 30], [10, 30]]], "Test", 0.9)
        ]
        mock_easyocr.Reader.return_value = mock_easyocr_reader
        
        mock_tesseract.image_to_string.return_value = "Test"
        mock_tesseract.image_to_data.return_value = {
            'text': ['', '', '', '', '', 'Test'],
            'conf': [-1, -1, -1, -1, -1, 90],
            'left': [0, 0, 0, 0, 0, 10],
            'top': [0, 0, 0, 0, 0, 10],
            'width': [100, 100, 100, 100, 100, 40],
            'height': [50, 50, 50, 50, 50, 20]
        }
        mock_tesseract.get_tesseract_version.return_value = "5.0.0"
        mock_tesseract.get_languages.return_value = ['eng']
        
        # Initialize engines
        easyocr_engine = EasyOCREngine()
        easyocr_engine.initialize()
        
        tesseract_engine = TesseractEngine()
        tesseract_engine.initialize()
        
        # Test batch processing performance
        test_images_list = [test_images['clean']] * 5
        
        # EasyOCR batch processing
        start_time = time.time()
        easyocr_results = easyocr_engine.extract_text_batch(test_images_list)
        easyocr_time = time.time() - start_time
        
        # Tesseract individual processing (no native batch support)
        start_time = time.time()
        tesseract_results = []
        for img in test_images_list:
            tesseract_results.append(tesseract_engine.extract_text(img))
        tesseract_time = time.time() - start_time
        
        # Check results
        assert len(easyocr_results) == 5
        assert len(tesseract_results) == 5
        
        # EasyOCR batch processing should be more efficient
        # (This is a mock test, so we're mainly testing the interface)
        assert all(isinstance(result, OCRResult) for result in easyocr_results)
        assert all(isinstance(result, OCRResult) for result in tesseract_results)
    
    def test_language_support_comparison(self, mock_engines_available):
        """Compare language support between engines."""
        mock_easyocr, mock_tesseract = mock_engines_available
        
        # Setup mocks
        mock_easyocr.Reader.return_value = Mock()
        mock_tesseract.get_tesseract_version.return_value = "5.0.0"
        mock_tesseract.get_languages.return_value = ['eng', 'fra', 'deu', 'spa']
        
        # Initialize engines
        easyocr_engine = EasyOCREngine()
        easyocr_engine.initialize()
        
        tesseract_engine = TesseractEngine()
        tesseract_engine.initialize()
        
        # Compare language support
        easyocr_languages = set(easyocr_engine.get_supported_languages())
        tesseract_languages = set(tesseract_engine.get_supported_languages())
        
        # EasyOCR should support more languages
        assert len(easyocr_languages) >= len(tesseract_languages)
        
        # Check handwriting support
        easyocr_hw_languages = set(easyocr_engine.get_handwriting_languages())
        assert len(easyocr_hw_languages) > 0  # EasyOCR should support handwriting
        
        # Tesseract typically doesn't have specialized handwriting support
        assert not hasattr(tesseract_engine, 'get_handwriting_languages')


if __name__ == "__main__":
    pytest.main([__file__])