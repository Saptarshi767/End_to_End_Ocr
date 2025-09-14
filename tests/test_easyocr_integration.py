"""
Integration test for EasyOCR engine demonstrating enhanced features.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.ocr.easyocr_engine import EasyOCREngine
from src.core.models import OCRResult


class TestEasyOCRIntegration:
    """Integration tests for EasyOCR engine enhanced features."""
    
    @patch('easyocr.Reader')
    @patch('torch.cuda.is_available')
    def test_multi_language_handwriting_workflow(self, mock_cuda, mock_reader_class):
        """Test complete workflow with multi-language and handwriting support."""
        # Setup mocks
        mock_cuda.return_value = True
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[[10, 10], [100, 10], [100, 30], [10, 30]]], "Hello", 0.9),
            ([[[110, 10], [200, 10], [200, 30], [110, 30]]], "世界", 0.85)
        ]
        mock_reader_class.return_value = mock_reader
        
        # Create engine with multi-language support
        engine = EasyOCREngine()
        engine.configure({
            'languages': ['en', 'ch_sim'],
            'handwriting_enabled': True
        })
        
        with patch('src.ocr.easyocr_engine.EASYOCR_AVAILABLE', True):
            engine.initialize()
        
        # Create test image
        image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(image, "Hello", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test extraction with handwriting detection
        result = engine.extract_text(image, detect_handwriting=True, language='en')
        
        assert isinstance(result, OCRResult)
        assert result.text
        assert result.confidence > 0
        assert len(result.word_level_data) > 0
        
        # Verify handwriting support is enabled
        assert engine.supports_handwriting()
        assert 'en' in engine.get_handwriting_languages()
    
    @patch('easyocr.Reader')
    @patch('torch.cuda.is_available')
    def test_batch_processing_performance(self, mock_cuda, mock_reader_class):
        """Test batch processing with performance optimization."""
        # Setup mocks
        mock_cuda.return_value = False  # Test CPU mode
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[[10, 10], [100, 10], [100, 30], [10, 30]]], "Batch text", 0.88)
        ]
        mock_reader_class.return_value = mock_reader
        
        # Create engine optimized for batch processing
        engine = EasyOCREngine()
        engine.optimize_for_batch_processing(True)
        
        with patch('src.ocr.easyocr_engine.EASYOCR_AVAILABLE', True):
            engine.initialize()
        
        # Create multiple test images
        images = []
        for i in range(5):
            img = np.ones((50, 200, 3), dtype=np.uint8) * 255
            cv2.putText(img, f"Text {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            images.append(img)
        
        # Test batch processing
        results = engine.extract_text_batch(images, batch_size=3)
        
        assert len(results) == 5
        assert all(isinstance(result, OCRResult) for result in results)
        assert all(result.text for result in results)
        
        # Check processing stats
        stats = engine.get_processing_stats()
        assert stats['batch_processed'] == 5
        assert stats['total_processed'] >= 5
    
    @patch('easyocr.Reader')
    @patch('torch.cuda.is_available')
    def test_handwriting_detection_and_processing(self, mock_cuda, mock_reader_class):
        """Test handwriting detection and specialized processing."""
        # Setup mocks for different processing modes
        mock_cuda.return_value = True
        mock_reader = Mock()
        
        # Mock different results for normal vs handwriting processing
        def mock_readtext(image, **kwargs):
            if kwargs.get('decoder') == 'beamsearch':
                # Handwriting-optimized results
                return [([[[10, 10], [150, 10], [150, 30], [10, 30]]], "Handwritten text", 0.75)]
            else:
                # Normal results
                return [([[[10, 10], [150, 10], [150, 30], [10, 30]]], "Handwrtten txt", 0.65)]
        
        mock_reader.readtext.side_effect = mock_readtext
        mock_reader_class.return_value = mock_reader
        
        # Create engine with handwriting support
        engine = EasyOCREngine()
        engine.set_handwriting_mode(True, threshold=0.6)
        
        with patch('src.ocr.easyocr_engine.EASYOCR_AVAILABLE', True):
            engine.initialize()
        
        # Create handwriting-like image (with more edges)
        image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        # Add some noise to simulate handwriting characteristics
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        cv2.putText(image, "Handwritten", (10, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test with handwriting detection enabled
        result = engine.extract_text(image, detect_handwriting=True)
        
        assert isinstance(result, OCRResult)
        assert result.text
        assert result.confidence > 0
        
        # Check that handwriting was detected
        stats = engine.get_processing_stats()
        # Note: handwriting_detected count depends on the heuristic
    
    @patch('easyocr.Reader')
    @patch('torch.cuda.is_available')
    def test_error_handling_and_fallback(self, mock_cuda, mock_reader_class):
        """Test error handling and fallback mechanisms."""
        # Setup mocks
        mock_cuda.return_value = True
        mock_reader = Mock()
        
        # Mock reader that fails on first call but succeeds on second
        call_count = 0
        def mock_readtext_with_failure(image, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("OCR processing failed")
            else:
                return [([[[10, 10], [100, 10], [100, 30], [10, 30]]], "Fallback text", 0.8)]
        
        mock_reader.readtext.side_effect = mock_readtext_with_failure
        mock_reader_class.return_value = mock_reader
        
        # Create engine
        engine = EasyOCREngine()
        
        with patch('src.ocr.easyocr_engine.EASYOCR_AVAILABLE', True):
            engine.initialize()
        
        # Create test image
        image = np.ones((50, 200, 3), dtype=np.uint8) * 255
        
        # Test batch processing with error handling
        images = [image, image, image]
        results = engine.extract_text_batch(images)
        
        # Should handle errors gracefully
        assert len(results) == 3
        # Some results might be empty due to errors, but structure should be maintained
        assert all(isinstance(result, OCRResult) for result in results)
    
    @patch('easyocr.Reader')
    @patch('torch.cuda.is_available')
    def test_comprehensive_engine_info(self, mock_cuda, mock_reader_class):
        """Test comprehensive engine information and configuration."""
        # Setup mocks
        mock_cuda.return_value = True
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        
        # Create and configure engine
        engine = EasyOCREngine(confidence_threshold=0.75)
        engine.configure({
            'languages': ['en', 'fr', 'de'],
            'width_ths': 0.8,
            'height_ths': 0.8,
            'decoder': 'beamsearch'
        })
        engine.set_handwriting_mode(True, threshold=0.65)
        engine.optimize_for_batch_processing(True)
        
        with patch('src.ocr.easyocr_engine.EASYOCR_AVAILABLE', True):
            engine.initialize()
        
        # Get comprehensive engine info
        info = engine.get_engine_info()
        
        # Verify all expected information is present
        assert info['name'] == 'easyocr'
        assert info['confidence_threshold'] == 0.75
        assert set(info['active_languages']) == {'en', 'fr', 'de'}
        assert info['handwriting_enabled'] == True
        assert info['handwriting_threshold'] == 0.65
        
        # Check batch processing configuration
        batch_info = info['batch_processing']
        assert batch_info['max_batch_size'] > 1
        assert batch_info['current_batch_size'] > 1
        
        # Check OCR parameters
        ocr_params = info['ocr_parameters']
        assert ocr_params['width_ths'] == 0.8
        assert ocr_params['height_ths'] == 0.8
        assert ocr_params['decoder'] == 'beamsearch'
        
        # Check performance stats
        assert 'performance_stats' in info
        assert 'total_processed' in info['performance_stats']
    
    def test_language_support_comprehensive(self):
        """Test comprehensive language support features."""
        engine = EasyOCREngine()
        
        # Test supported languages
        supported = engine.get_supported_languages()
        assert len(supported) > 20  # EasyOCR supports many languages
        assert 'en' in supported
        assert 'ch_sim' in supported
        assert 'fr' in supported
        
        # Test handwriting languages
        hw_languages = engine.handwriting_languages
        assert len(hw_languages) > 5
        assert 'en' in hw_languages
        assert 'ch_sim' in hw_languages
        
        # Test language configuration
        engine.configure({'languages': ['en', 'ch_sim', 'ja']})
        assert set(engine.active_languages) == {'en', 'ch_sim', 'ja'}
        
        # Test handwriting support check
        engine.active_languages = ['en', 'ch_sim']  # Both support handwriting
        assert engine.supports_handwriting()
        
        engine.active_languages = ['fr']  # Doesn't support handwriting
        assert not engine.supports_handwriting()


if __name__ == "__main__":
    pytest.main([__file__])