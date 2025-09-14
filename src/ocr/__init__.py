# OCR processing components

from .base_engine import BaseOCREngine, MockOCREngine
from .tesseract_engine import TesseractEngine
from .easyocr_engine import EasyOCREngine
from .cloud_vision_engine import CloudVisionEngine
from .engine_factory import OCREngineFactory, OCREngineManager, EngineSelectionStrategy, get_global_factory, create_engine_manager
from .config import OCRConfigurationManager, OCREngineConfig, TableDetectionConfig, ProcessingConfig, get_global_config_manager, create_config_manager

__all__ = [
    # Base classes
    'BaseOCREngine',
    'MockOCREngine',
    
    # Engine implementations
    'TesseractEngine',
    'EasyOCREngine', 
    'CloudVisionEngine',
    
    # Factory and management
    'OCREngineFactory',
    'OCREngineManager',
    'EngineSelectionStrategy',
    'get_global_factory',
    'create_engine_manager',
    
    # Configuration
    'OCRConfigurationManager',
    'OCREngineConfig',
    'TableDetectionConfig',
    'ProcessingConfig',
    'get_global_config_manager',
    'create_config_manager'
]