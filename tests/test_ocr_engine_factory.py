"""
Unit tests for OCR engine factory and abstraction layer.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List

from src.core.interfaces import OCREngineInterface
from src.core.models import OCRResult, OCREngine, BoundingBox, WordData, TableRegion
from src.core.exceptions import OCREngineError, ConfigurationError
from src.ocr.engine_factory import (
    OCREngineFactory, OCREngineManager, EngineSelectionStrategy,
    get_global_factory, create_engine_manager
)


class MockOCREngine(OCREngineInterface):
    """Mock OCR engine for testing."""
    
    def __init__(self, name: str, confidence_threshold: float = 0.8, languages: List[str] = None):
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.languages = languages or ['eng']
        self.configured = False
        self.config = {}
    
    def extract_text(self, image: np.ndarray, **kwargs) -> OCRResult:
        """Mock text extraction."""
        return OCRResult(
            text=f"Mock text from {self.name}",
            confidence=self.confidence_threshold,
            bounding_boxes=[BoundingBox(0, 0, 100, 20, self.confidence_threshold)],
            word_level_data=[WordData("Mock", self.confidence_threshold, BoundingBox(0, 0, 50, 20, self.confidence_threshold))],
            processing_time_ms=100,
            engine_used=OCREngine.TESSERACT
        )
    
    def get_confidence_threshold(self) -> float:
        """Get confidence threshold."""
        return self.confidence_threshold
    
    def supports_language(self, language: str) -> bool:
        """Check language support."""
        return language in self.languages
    
    def configure(self, config: dict) -> None:
        """Configure the engine."""
        self.configured = True
        self.config = config
    
    def detect_tables(self, image: np.ndarray, **kwargs) -> List[TableRegion]:
        """Mock table detection."""
        return [TableRegion(BoundingBox(10, 10, 200, 100, 0.9), 0.9, 1)]
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return self.languages


class TestOCREngineFactory:
    """Test cases for OCREngineFactory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = OCREngineFactory()
        self.mock_engine = MockOCREngine("test_engine")
    
    def test_register_engine(self):
        """Test engine registration."""
        config = {"param1": "value1"}
        self.factory.register_engine("test", self.mock_engine, config)
        
        assert "test" in self.factory.get_available_engines()
        assert self.factory.get_engine("test") == self.mock_engine
        assert self.factory.get_engine_config("test") == config
    
    def test_register_invalid_engine(self):
        """Test registration of invalid engine."""
        with pytest.raises(ConfigurationError):
            self.factory.register_engine("invalid", "not_an_engine")
    
    def test_unregister_engine(self):
        """Test engine unregistration."""
        self.factory.register_engine("test", self.mock_engine)
        assert "test" in self.factory.get_available_engines()
        
        self.factory.unregister_engine("test")
        assert "test" not in self.factory.get_available_engines()
        assert self.factory.get_engine("test") is None
    
    def test_unregister_nonexistent_engine(self):
        """Test unregistering non-existent engine."""
        # Should not raise exception
        self.factory.unregister_engine("nonexistent")
    
    def test_create_engine(self):
        """Test engine creation."""
        self.factory.register_engine("tesseract", self.mock_engine)
        
        engine = self.factory.create_engine(OCREngine.TESSERACT)
        assert engine == self.mock_engine
    
    def test_create_unregistered_engine(self):
        """Test creating unregistered engine."""
        with pytest.raises(OCREngineError):
            self.factory.create_engine(OCREngine.TESSERACT)
    
    def test_select_best_engine_no_engines(self):
        """Test engine selection with no registered engines."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(OCREngineError):
            self.factory.select_best_engine(image)
    
    def test_select_best_engine_fastest_strategy(self):
        """Test engine selection with fastest strategy."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        engine1 = MockOCREngine("engine1", 0.8, ['eng'])
        engine2 = MockOCREngine("engine2", 0.9, ['eng'])
        
        self.factory.register_engine("engine1", engine1)
        self.factory.register_engine("engine2", engine2)
        
        selected = self.factory.select_best_engine(
            image, strategy=EngineSelectionStrategy.FASTEST
        )
        
        # Should return first available engine
        assert selected in ["engine1", "engine2"]
    
    def test_select_best_engine_language_filter(self):
        """Test engine selection with language filtering."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        engine1 = MockOCREngine("engine1", 0.8, ['eng'])
        engine2 = MockOCREngine("engine2", 0.9, ['fra'])
        
        self.factory.register_engine("engine1", engine1)
        self.factory.register_engine("engine2", engine2)
        
        selected = self.factory.select_best_engine(image, language='fra')
        assert selected == "engine2"
    
    def test_select_best_engine_most_accurate_strategy(self):
        """Test engine selection with most accurate strategy."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        engine1 = MockOCREngine("tesseract_engine", 0.8, ['eng'])
        engine2 = MockOCREngine("cloud_vision_engine", 0.9, ['eng'])
        
        self.factory.register_engine("tesseract", engine1)
        self.factory.register_engine("cloud_vision", engine2)
        
        selected = self.factory.select_best_engine(
            image, strategy=EngineSelectionStrategy.MOST_ACCURATE
        )
        
        # Should prefer cloud engine
        assert selected == "cloud_vision"
    
    def test_update_engine_config(self):
        """Test updating engine configuration."""
        self.factory.register_engine("test", self.mock_engine, {"param1": "value1"})
        
        new_config = {"param2": "value2"}
        self.factory.update_engine_config("test", new_config)
        
        config = self.factory.get_engine_config("test")
        assert "param1" in config
        assert "param2" in config
        assert config["param2"] == "value2"
    
    def test_update_config_unregistered_engine(self):
        """Test updating configuration for unregistered engine."""
        with pytest.raises(ConfigurationError):
            self.factory.update_engine_config("nonexistent", {"param": "value"})
    
    def test_set_default_strategy(self):
        """Test setting default selection strategy."""
        self.factory.set_default_strategy(EngineSelectionStrategy.MOST_ACCURATE)
        assert self.factory._default_strategy == EngineSelectionStrategy.MOST_ACCURATE


class TestOCREngineManager:
    """Test cases for OCREngineManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = OCREngineFactory()
        self.manager = OCREngineManager(self.factory)
        self.mock_engine = MockOCREngine("test_engine")
    
    def test_extract_text_with_specific_engine(self):
        """Test text extraction with specific engine."""
        self.factory.register_engine("test", self.mock_engine)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = self.manager.extract_text(image, engine="test")
        
        assert result.text == "Mock text from test_engine"
        assert result.confidence == 0.8
        assert result.engine_used == OCREngine.TESSERACT
    
    def test_extract_text_with_auto_selection(self):
        """Test text extraction with automatic engine selection."""
        self.factory.register_engine("test", self.mock_engine)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = self.manager.extract_text(image, engine="auto")
        
        assert result.text == "Mock text from test_engine"
        assert result.confidence == 0.8
    
    def test_extract_text_with_nonexistent_engine(self):
        """Test text extraction with non-existent engine."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(OCREngineError):
            self.manager.extract_text(image, engine="nonexistent")
    
    def test_extract_text_with_engine_error(self):
        """Test text extraction when engine raises error."""
        failing_engine = Mock(spec=OCREngineInterface)
        failing_engine.extract_text.side_effect = Exception("Engine failed")
        
        self.factory.register_engine("failing", failing_engine)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(OCREngineError):
            self.manager.extract_text(image, engine="failing")
    
    def test_detect_tables(self):
        """Test table detection."""
        self.factory.register_engine("test", self.mock_engine)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        regions = self.manager.detect_tables(image)
        
        assert len(regions) == 1
        assert regions[0].confidence == 0.9
    
    def test_detect_tables_disabled(self):
        """Test table detection when disabled."""
        self.factory.register_engine("test", self.mock_engine)
        self.manager.enable_table_detection(False)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        regions = self.manager.detect_tables(image)
        
        assert len(regions) == 0
    
    def test_detect_tables_fallback(self):
        """Test table detection fallback."""
        engine_without_tables = MockOCREngine("basic_engine")
        # Remove detect_tables method to simulate basic engine
        delattr(engine_without_tables, 'detect_tables')
        
        self.factory.register_engine("basic", engine_without_tables)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        regions = self.manager.detect_tables(image)
        
        # Should use fallback (returns empty list in our implementation)
        assert len(regions) == 0
    
    def test_register_engine(self):
        """Test engine registration through manager."""
        config = {"param": "value"}
        self.manager.register_engine("test", self.mock_engine, config)
        
        assert "test" in self.manager.get_available_engines()
    
    def test_get_available_engines(self):
        """Test getting available engines."""
        self.factory.register_engine("engine1", self.mock_engine)
        self.factory.register_engine("engine2", MockOCREngine("engine2"))
        
        engines = self.manager.get_available_engines()
        
        assert "engine1" in engines
        assert "engine2" in engines
        assert len(engines) == 2
    
    def test_enable_disable_table_detection(self):
        """Test enabling and disabling table detection."""
        assert self.manager._table_detection_enabled is True
        
        self.manager.enable_table_detection(False)
        assert self.manager._table_detection_enabled is False
        
        self.manager.enable_table_detection(True)
        assert self.manager._table_detection_enabled is True


class TestGlobalFactory:
    """Test cases for global factory functions."""
    
    def test_get_global_factory(self):
        """Test getting global factory instance."""
        factory1 = get_global_factory()
        factory2 = get_global_factory()
        
        assert factory1 is factory2  # Should be same instance
        assert isinstance(factory1, OCREngineFactory)
    
    def test_create_engine_manager(self):
        """Test creating engine manager."""
        manager = create_engine_manager()
        
        assert isinstance(manager, OCREngineManager)
        assert manager.factory is get_global_factory()
    
    def test_create_engine_manager_with_custom_factory(self):
        """Test creating engine manager with custom factory."""
        custom_factory = OCREngineFactory()
        manager = create_engine_manager(custom_factory)
        
        assert isinstance(manager, OCREngineManager)
        assert manager.factory is custom_factory


class TestEngineSelectionStrategies:
    """Test cases for engine selection strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = OCREngineFactory()
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    def test_balanced_strategy_scoring(self):
        """Test balanced strategy scoring system."""
        # Create engines with different characteristics
        tesseract_engine = MockOCREngine("tesseract_engine", 0.8, ['eng'])
        cloud_engine = MockOCREngine("cloud_vision_engine", 0.9, ['eng'])
        easyocr_engine = MockOCREngine("easyocr_engine", 0.7, ['eng'])
        
        self.factory.register_engine("tesseract", tesseract_engine)
        self.factory.register_engine("cloud_vision", cloud_engine)
        self.factory.register_engine("easyocr", easyocr_engine)
        
        selected = self.factory.select_best_engine(
            self.image, strategy=EngineSelectionStrategy.BALANCED
        )
        
        # Cloud engine should score highest due to "cloud" in name and high confidence
        assert selected == "cloud_vision"
    
    def test_language_optimized_strategy(self):
        """Test language optimized strategy."""
        engine1 = MockOCREngine("engine1", 0.8, ['eng'])
        engine2 = MockOCREngine("engine2", 0.9, ['fra', 'eng'])
        
        self.factory.register_engine("engine1", engine1)
        self.factory.register_engine("engine2", engine2)
        
        selected = self.factory.select_best_engine(
            self.image, 
            strategy=EngineSelectionStrategy.LANGUAGE_OPTIMIZED,
            language='fra'
        )
        
        assert selected == "engine2"
    
    def test_fallback_when_no_language_support(self):
        """Test fallback when no engine supports requested language."""
        engine1 = MockOCREngine("engine1", 0.8, ['eng'])
        engine2 = MockOCREngine("engine2", 0.9, ['fra'])
        
        self.factory.register_engine("engine1", engine1)
        self.factory.register_engine("engine2", engine2)
        
        # Request unsupported language
        selected = self.factory.select_best_engine(
            self.image, language='deu'  # German
        )
        
        # Should fallback to any available engine
        assert selected in ["engine1", "engine2"]


if __name__ == "__main__":
    pytest.main([__file__])