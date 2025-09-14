"""
Unit tests for OCR engine abstraction layer.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os

from src.core.models import OCRResult, BoundingBox, WordData, TableRegion, OCREngine
from src.core.exceptions import OCREngineError, ConfigurationError
from src.ocr.base_engine import BaseOCREngine, MockOCREngine
from src.ocr.engine_factory import OCREngineFactory, OCREngineManager, EngineSelectionStrategy


class TestBaseOCREngine(unittest.TestCase):
    """Test cases for BaseOCREngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MockOCREngine("test_engine", 0.8)
        
    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.name, "test_engine")
        self.assertEqual(self.engine.confidence_threshold, 0.8)
        self.assertFalse(self.engine.is_initialized)
        
        self.engine.initialize()
        self.assertTrue(self.engine.is_initialized)
        self.assertTrue(self.engine.is_available())
        
    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        # Valid thresholds
        self.engine.set_confidence_threshold(0.5)
        self.assertEqual(self.engine.confidence_threshold, 0.5)
        
        self.engine.set_confidence_threshold(1.0)
        self.assertEqual(self.engine.confidence_threshold, 1.0)
        
        # Invalid thresholds
        with self.assertRaises(OCREngineError):
            self.engine.set_confidence_threshold(-0.1)
            
        with self.assertRaises(OCREngineError):
            self.engine.set_confidence_threshold(1.1)
            
    def test_language_support(self):
        """Test language support functionality."""
        self.assertTrue(self.engine.supports_language('eng'))
        self.assertFalse(self.engine.supports_language('fra'))
        
        # Update supported languages
        self.engine.configure({'languages': ['eng', 'fra', 'deu']})
        self.assertTrue(self.engine.supports_language('fra'))
        self.assertTrue(self.engine.supports_language('deu'))
        
    def test_configuration(self):
        """Test engine configuration."""
        config = {
            'confidence_threshold': 0.9,
            'languages': ['eng', 'fra'],
            'custom_param': 'test_value'
        }
        
        self.engine.configure(config)
        
        self.assertEqual(self.engine.confidence_threshold, 0.9)
        self.assertEqual(self.engine.supported_languages, ['eng', 'fra'])
        self.assertEqual(self.engine.config['custom_param'], 'test_value')
        
    def test_extract_text_validation(self):
        """Test text extraction input validation."""
        self.engine.initialize()
        
        # Test with None image
        with self.assertRaises(OCREngineError):
            self.engine.extract_text(None)
            
        # Test with empty image
        empty_image = np.array([])
        with self.assertRaises(OCREngineError):
            self.engine.extract_text(empty_image)
            
        # Test with invalid shape
        invalid_image = np.ones((10,))  # 1D array
        with self.assertRaises(OCREngineError):
            self.engine.extract_text(invalid_image)
            
    def test_extract_text_success(self):
        """Test successful text extraction."""
        self.engine.initialize()
        
        # Create test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result = self.engine.extract_text(test_image)
        
        self.assertIsInstance(result, OCRResult)
        self.assertIsInstance(result.text, str)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsInstance(result.bounding_boxes, list)
        self.assertIsInstance(result.word_level_data, list)
        self.assertGreater(result.processing_time_ms, 0)
        
    def test_extract_text_without_initialization(self):
        """Test text extraction without initialization."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with self.assertRaises(OCREngineError):
            self.engine.extract_text(test_image)
            
    def test_result_validation(self):
        """Test OCR result validation."""
        # Valid result
        valid_result = OCRResult(
            text="Sample text",
            confidence=0.9,
            bounding_boxes=[],
            word_level_data=[]
        )
        self.assertTrue(self.engine.validate_result(valid_result))
        
        # Invalid results
        invalid_results = [
            OCRResult(text="", confidence=0.9, bounding_boxes=[], word_level_data=[]),  # Empty text
            OCRResult(text="   ", confidence=0.9, bounding_boxes=[], word_level_data=[]),  # Whitespace only
            OCRResult(text="Sample", confidence=0.5, bounding_boxes=[], word_level_data=[]),  # Low confidence
            OCRResult(text="Hi", confidence=0.9, bounding_boxes=[], word_level_data=[]),  # Too short
        ]
        
        for result in invalid_results:
            self.assertFalse(self.engine.validate_result(result))
            
    def test_engine_info(self):
        """Test engine information retrieval."""
        info = self.engine.get_engine_info()
        
        self.assertIn('name', info)
        self.assertIn('confidence_threshold', info)
        self.assertIn('supported_languages', info)
        self.assertIn('is_initialized', info)
        self.assertIn('config', info)
        
        self.assertEqual(info['name'], self.engine.name)
        self.assertEqual(info['confidence_threshold'], self.engine.confidence_threshold)
        
    def test_cleanup(self):
        """Test engine cleanup."""
        self.engine.initialize()
        self.assertTrue(self.engine.is_available())
        
        self.engine.cleanup()
        self.assertFalse(self.engine.is_available())


class TestMockOCREngine(unittest.TestCase):
    """Test cases for MockOCREngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MockOCREngine()
        self.engine.initialize()
        
    def test_mock_text_extraction(self):
        """Test mock text extraction."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result = self.engine.extract_text(test_image)
        
        self.assertEqual(result.text, self.engine.mock_text)
        self.assertEqual(result.confidence, self.engine.mock_confidence)
        self.assertGreater(len(result.bounding_boxes), 0)
        self.assertGreater(len(result.word_level_data), 0)
        
    def test_set_mock_result(self):
        """Test setting custom mock results."""
        custom_text = "Custom mock text"
        custom_confidence = 0.85
        
        self.engine.set_mock_result(custom_text, custom_confidence)
        
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = self.engine.extract_text(test_image)
        
        self.assertEqual(result.text, custom_text)
        self.assertEqual(result.confidence, custom_confidence)
        
    def test_mock_table_detection(self):
        """Test mock table detection."""
        test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
        
        tables = self.engine.detect_tables(test_image)
        
        self.assertGreater(len(tables), 0)
        self.assertIsInstance(tables[0], TableRegion)
        self.assertGreater(tables[0].confidence, 0.0)


class TestOCREngineFactory(unittest.TestCase):
    """Test cases for OCREngineFactory."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create factory without auto-registration for testing
        self.factory = OCREngineFactory()
        self.factory._engines.clear()  # Clear auto-registered engines
        self.factory._engine_configs.clear()
        
    def test_engine_registration(self):
        """Test engine registration."""
        mock_engine = MockOCREngine("test_mock")
        config = {'confidence_threshold': 0.9}
        
        self.factory.register_engine("test_mock", mock_engine, config)
        
        self.assertIn("test_mock", self.factory.get_available_engines())
        self.assertEqual(self.factory.get_engine("test_mock"), mock_engine)
        self.assertEqual(self.factory.get_engine_config("test_mock"), config)
        
    def test_engine_unregistration(self):
        """Test engine unregistration."""
        mock_engine = MockOCREngine("test_mock")
        self.factory.register_engine("test_mock", mock_engine)
        
        self.assertIn("test_mock", self.factory.get_available_engines())
        
        self.factory.unregister_engine("test_mock")
        
        self.assertNotIn("test_mock", self.factory.get_available_engines())
        self.assertIsNone(self.factory.get_engine("test_mock"))
        
    def test_invalid_engine_registration(self):
        """Test registration of invalid engine."""
        invalid_engine = "not_an_engine"
        
        with self.assertRaises(ConfigurationError):
            self.factory.register_engine("invalid", invalid_engine)
            
    def test_engine_creation(self):
        """Test engine creation by type."""
        mock_engine = MockOCREngine("mock")
        mock_engine.initialize()
        self.factory.register_engine("mock", mock_engine)
        
        created_engine = self.factory.create_engine(OCREngine.AUTO)  # Will fallback to available engine
        self.assertIsNotNone(created_engine)
        
    def test_engine_creation_not_found(self):
        """Test engine creation with non-existent type."""
        with self.assertRaises(OCREngineError):
            self.factory.create_engine(OCREngine.TESSERACT)  # Not registered
            
    def test_best_engine_selection(self):
        """Test automatic engine selection."""
        # Register multiple engines
        engines = [
            MockOCREngine("fast_engine"),
            MockOCREngine("accurate_engine"),
            MockOCREngine("balanced_engine")
        ]
        
        for engine in engines:
            engine.initialize()
            self.factory.register_engine(engine.name, engine)
            
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Test different strategies
        strategies = [
            EngineSelectionStrategy.FASTEST,
            EngineSelectionStrategy.MOST_ACCURATE,
            EngineSelectionStrategy.BALANCED,
            EngineSelectionStrategy.LANGUAGE_OPTIMIZED
        ]
        
        for strategy in strategies:
            selected = self.factory.select_best_engine(test_image, strategy=strategy)
            self.assertIn(selected, self.factory.get_available_engines())
            
    def test_engine_selection_no_engines(self):
        """Test engine selection with no available engines."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with self.assertRaises(OCREngineError):
            self.factory.select_best_engine(test_image)
            
    def test_engine_configuration_update(self):
        """Test updating engine configuration."""
        mock_engine = MockOCREngine("test_mock")
        self.factory.register_engine("test_mock", mock_engine, {'param1': 'value1'})
        
        new_config = {'param2': 'value2'}
        self.factory.update_engine_config("test_mock", new_config)
        
        updated_config = self.factory.get_engine_config("test_mock")
        self.assertIn('param1', updated_config)
        self.assertIn('param2', updated_config)
        self.assertEqual(updated_config['param2'], 'value2')
        
    def test_engine_configuration_update_not_found(self):
        """Test updating configuration for non-existent engine."""
        with self.assertRaises(ConfigurationError):
            self.factory.update_engine_config("non_existent", {'param': 'value'})
            
    def test_default_strategy(self):
        """Test default strategy setting."""
        self.assertEqual(self.factory._default_strategy, EngineSelectionStrategy.BALANCED)
        
        self.factory.set_default_strategy(EngineSelectionStrategy.FASTEST)
        self.assertEqual(self.factory._default_strategy, EngineSelectionStrategy.FASTEST)
        
    def test_engine_status(self):
        """Test engine status retrieval."""
        mock_engine = MockOCREngine("test_mock")
        mock_engine.initialize()
        self.factory.register_engine("test_mock", mock_engine, {'enabled': True})
        
        status = self.factory.get_engine_status("test_mock")
        
        self.assertIn('status', status)
        self.assertIn('confidence_threshold', status)
        self.assertIn('config', status)
        self.assertEqual(status['status'], 'available')
        
    def test_all_engine_status(self):
        """Test getting status for all engines."""
        engines = [MockOCREngine(f"engine_{i}") for i in range(3)]
        
        for engine in engines:
            engine.initialize()
            self.factory.register_engine(engine.name, engine)
            
        all_status = self.factory.get_all_engine_status()
        
        self.assertEqual(len(all_status), 3)
        for engine in engines:
            self.assertIn(engine.name, all_status)


class TestOCREngineManager(unittest.TestCase):
    """Test cases for OCREngineManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = OCREngineFactory()
        self.factory._engines.clear()  # Clear auto-registered engines
        self.factory._engine_configs.clear()
        
        # Register mock engines
        self.mock_engine = MockOCREngine("mock")
        self.mock_engine.initialize()
        self.factory.register_engine("mock", self.mock_engine)
        
        self.manager = OCREngineManager(self.factory)
        
    def test_text_extraction_auto_selection(self):
        """Test text extraction with automatic engine selection."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result = self.manager.extract_text(test_image, engine='auto')
        
        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.engine_used, OCREngine("mock"))
        
    def test_text_extraction_specific_engine(self):
        """Test text extraction with specific engine."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result = self.manager.extract_text(test_image, engine='mock')
        
        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.engine_used, OCREngine("mock"))
        
    def test_text_extraction_engine_not_found(self):
        """Test text extraction with non-existent engine."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with self.assertRaises(OCREngineError):
            self.manager.extract_text(test_image, engine='non_existent')
            
    def test_table_detection(self):
        """Test table detection."""
        test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
        
        tables = self.manager.detect_tables(test_image)
        
        self.assertIsInstance(tables, list)
        # Mock engine should return at least one table
        self.assertGreater(len(tables), 0)
        
    def test_table_detection_disabled(self):
        """Test table detection when disabled."""
        self.manager.enable_table_detection(False)
        
        test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
        tables = self.manager.detect_tables(test_image)
        
        self.assertEqual(len(tables), 0)
        
    def test_engine_registration_through_manager(self):
        """Test engine registration through manager."""
        new_engine = MockOCREngine("new_mock")
        config = {'test_param': 'test_value'}
        
        self.manager.register_engine("new_mock", new_engine, config)
        
        self.assertIn("new_mock", self.manager.get_available_engines())
        
    def test_get_available_engines(self):
        """Test getting available engines."""
        engines = self.manager.get_available_engines()
        
        self.assertIsInstance(engines, list)
        self.assertIn("mock", engines)


if __name__ == '__main__':
    unittest.main()