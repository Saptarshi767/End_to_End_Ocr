"""
OCR Engine Factory for dynamic engine selection and management.
"""

from typing import Dict, Optional, List, Any
from enum import Enum
import logging

from ..core.interfaces import OCREngineInterface, OCREngineManagerInterface
from ..core.models import OCREngine, OCRResult, TableRegion, ProcessingOptions
from ..core.exceptions import OCREngineError, ConfigurationError
from .base_engine import BaseOCREngine, MockOCREngine
from .tesseract_engine import TesseractEngine
from .easyocr_engine import EasyOCREngine
from .cloud_vision_engine import CloudVisionEngine
import numpy as np


logger = logging.getLogger(__name__)


class EngineSelectionStrategy(Enum):
    """Strategy for automatic engine selection."""
    FASTEST = "fastest"
    MOST_ACCURATE = "most_accurate"
    BALANCED = "balanced"
    LANGUAGE_OPTIMIZED = "language_optimized"


class OCREngineFactory:
    """Factory class for creating and managing OCR engines."""
    
    def __init__(self):
        self._engines: Dict[str, OCREngineInterface] = {}
        self._engine_configs: Dict[str, Dict[str, Any]] = {}
        self._default_strategy = EngineSelectionStrategy.BALANCED
        
        # Register default engines
        self._register_default_engines()
        
    def register_engine(self, name: str, engine: OCREngineInterface, config: Dict[str, Any] = None) -> None:
        """
        Register a new OCR engine with optional configuration.
        
        Args:
            name: Engine identifier
            engine: OCR engine implementation
            config: Engine-specific configuration
        """
        if not isinstance(engine, OCREngineInterface):
            raise ConfigurationError(f"Engine {name} must implement OCREngineInterface")
            
        self._engines[name] = engine
        self._engine_configs[name] = config or {}
        logger.info(f"Registered OCR engine: {name}")
        
    def unregister_engine(self, name: str) -> None:
        """Remove an OCR engine from the factory."""
        if name in self._engines:
            del self._engines[name]
            del self._engine_configs[name]
            logger.info(f"Unregistered OCR engine: {name}")
        else:
            logger.warning(f"Attempted to unregister unknown engine: {name}")
            
    def get_engine(self, name: str) -> Optional[OCREngineInterface]:
        """Get a specific OCR engine by name."""
        return self._engines.get(name)
        
    def get_available_engines(self) -> List[str]:
        """Get list of available engine names."""
        return list(self._engines.keys())
        
    def create_engine(self, engine_type: OCREngine, **kwargs) -> OCREngineInterface:
        """
        Create an OCR engine instance based on type.
        
        Args:
            engine_type: Type of OCR engine to create
            **kwargs: Additional configuration parameters
            
        Returns:
            OCR engine instance
            
        Raises:
            OCREngineError: If engine type is not supported or creation fails
        """
        engine_name = engine_type.value
        
        # Handle AUTO selection
        if engine_name == "auto":
            if not self._engines:
                raise OCREngineError("No OCR engines are registered")
            # Select the first available engine for AUTO
            engine_name = next(iter(self._engines.keys()))
        
        if engine_name not in self._engines:
            raise OCREngineError(f"OCR engine '{engine_name}' is not registered")
            
        try:
            engine = self._engines[engine_name]
            # Apply any runtime configuration
            if hasattr(engine, 'configure'):
                config = self._engine_configs.get(engine_name, {})
                config.update(kwargs)
                engine.configure(config)
            return engine
        except Exception as e:
            raise OCREngineError(f"Failed to create OCR engine '{engine_name}': {str(e)}")
            
    def select_best_engine(self, 
                          image: np.ndarray, 
                          strategy: EngineSelectionStrategy = None,
                          language: str = "eng",
                          **kwargs) -> str:
        """
        Automatically select the best OCR engine for given image and requirements.
        
        Args:
            image: Input image for OCR processing
            strategy: Selection strategy to use
            language: Target language for OCR
            **kwargs: Additional selection criteria
            
        Returns:
            Name of selected engine
            
        Raises:
            OCREngineError: If no suitable engine is found
        """
        if not self._engines:
            raise OCREngineError("No OCR engines are registered")
            
        strategy = strategy or self._default_strategy
        available_engines = []
        
        # Filter engines by language support
        for name, engine in self._engines.items():
            if engine.supports_language(language):
                available_engines.append((name, engine))
                
        if not available_engines:
            # Fallback to any available engine
            available_engines = list(self._engines.items())
            logger.warning(f"No engines support language '{language}', using fallback")
            
        if not available_engines:
            raise OCREngineError("No suitable OCR engines available")
            
        # Apply selection strategy
        selected_engine = self._apply_selection_strategy(
            available_engines, strategy, image, **kwargs
        )
        
        logger.info(f"Selected OCR engine: {selected_engine} (strategy: {strategy.value})")
        return selected_engine
        
    def _apply_selection_strategy(self, 
                                engines: List[tuple], 
                                strategy: EngineSelectionStrategy,
                                image: np.ndarray,
                                **kwargs) -> str:
        """Apply the specified selection strategy to choose an engine."""
        
        if strategy == EngineSelectionStrategy.FASTEST:
            # Return first available engine (assuming fastest)
            return engines[0][0]
            
        elif strategy == EngineSelectionStrategy.MOST_ACCURATE:
            # Prefer cloud-based engines for accuracy
            for name, engine in engines:
                if 'cloud' in name.lower():
                    return name
            return engines[0][0]
            
        elif strategy == EngineSelectionStrategy.LANGUAGE_OPTIMIZED:
            # Prefer engines optimized for specific languages
            language = kwargs.get('language', 'eng')
            for name, engine in engines:
                if hasattr(engine, 'get_supported_languages'):
                    supported = engine.get_supported_languages()
                    if language in supported:
                        return name
            return engines[0][0]
            
        else:  # BALANCED strategy
            # Use a balanced approach considering various factors
            scores = {}
            for name, engine in engines:
                score = 0
                
                # Prefer engines with higher confidence thresholds (more reliable)
                threshold = engine.get_confidence_threshold()
                score += (1.0 - threshold) * 10
                
                # Prefer cloud engines for complex documents
                if 'cloud' in name.lower():
                    score += 5
                    
                # Prefer local engines for simple documents (faster)
                elif 'tesseract' in name.lower() or 'easyocr' in name.lower():
                    score += 3
                    
                scores[name] = score
                
            # Return engine with highest score
            best_engine = max(scores.items(), key=lambda x: x[1])
            return best_engine[0]
            
    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Get configuration for a specific engine."""
        return self._engine_configs.get(engine_name, {}).copy()
        
    def update_engine_config(self, engine_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific engine."""
        if engine_name not in self._engines:
            raise ConfigurationError(f"Engine '{engine_name}' is not registered")
            
        self._engine_configs[engine_name].update(config)
        logger.info(f"Updated configuration for engine: {engine_name}")
        
    def set_default_strategy(self, strategy: EngineSelectionStrategy) -> None:
        """Set the default engine selection strategy."""
        self._default_strategy = strategy
        logger.info(f"Set default engine selection strategy: {strategy.value}")
        
    def _register_default_engines(self) -> None:
        """Register all available OCR engines."""
        try:
            # Register Mock engine (always available for testing)
            mock_engine = MockOCREngine()
            self.register_engine("mock", mock_engine, {
                'confidence_threshold': 0.95,
                'enabled': True
            })
            
            # Register Tesseract engine
            try:
                tesseract_engine = TesseractEngine()
                tesseract_engine.initialize()
                self.register_engine("tesseract", tesseract_engine, {
                    'confidence_threshold': 0.8,
                    'psm': 6,
                    'oem': 3,
                    'enabled': True
                })
                logger.info("Tesseract engine registered successfully")
            except Exception as e:
                logger.warning(f"Failed to register Tesseract engine: {str(e)}")
                
            # Register EasyOCR engine
            try:
                easyocr_engine = EasyOCREngine()
                easyocr_engine.initialize()
                self.register_engine("easyocr", easyocr_engine, {
                    'confidence_threshold': 0.7,
                    'languages': ['en'],
                    'enabled': True
                })
                logger.info("EasyOCR engine registered successfully")
            except Exception as e:
                logger.warning(f"Failed to register EasyOCR engine: {str(e)}")
                
            # Register Cloud Vision engine
            try:
                from .cloud_vision_config import configure_cloud_vision_from_env
                
                cloud_vision_engine = CloudVisionEngine()
                cloud_vision_config = configure_cloud_vision_from_env()
                cloud_vision_engine.configure(cloud_vision_config)
                cloud_vision_engine.initialize()
                
                self.register_engine("cloud_vision", cloud_vision_engine, cloud_vision_config)
                logger.info("Cloud Vision engine registered successfully")
            except Exception as e:
                logger.warning(f"Failed to register Cloud Vision engine: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error during engine registration: {str(e)}")
            
    def initialize_engine(self, engine_name: str) -> bool:
        """
        Initialize a specific engine.
        
        Args:
            engine_name: Name of the engine to initialize
            
        Returns:
            True if initialization successful, False otherwise
        """
        if engine_name not in self._engines:
            logger.error(f"Engine '{engine_name}' not found")
            return False
            
        try:
            engine = self._engines[engine_name]
            if hasattr(engine, 'initialize'):
                engine.initialize()
            logger.info(f"Engine '{engine_name}' initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize engine '{engine_name}': {str(e)}")
            return False
            
    def cleanup_engine(self, engine_name: str) -> None:
        """Cleanup resources for a specific engine."""
        if engine_name in self._engines:
            try:
                engine = self._engines[engine_name]
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
                logger.info(f"Engine '{engine_name}' cleaned up successfully")
            except Exception as e:
                logger.error(f"Failed to cleanup engine '{engine_name}': {str(e)}")
                
    def cleanup_all_engines(self) -> None:
        """Cleanup resources for all engines."""
        for engine_name in list(self._engines.keys()):
            self.cleanup_engine(engine_name)
            
    def get_engine_status(self, engine_name: str) -> Dict[str, Any]:
        """Get status information for a specific engine."""
        if engine_name not in self._engines:
            return {'status': 'not_found'}
            
        engine = self._engines[engine_name]
        config = self._engine_configs.get(engine_name, {})
        
        status = {
            'status': 'available' if engine.is_available() else 'unavailable',
            'confidence_threshold': engine.get_confidence_threshold(),
            'supported_languages': engine.get_supported_languages() if hasattr(engine, 'get_supported_languages') else [],
            'config': config.copy()
        }
        
        if hasattr(engine, 'get_engine_info'):
            status.update(engine.get_engine_info())
            
        return status
        
    def get_all_engine_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all engines."""
        return {name: self.get_engine_status(name) for name in self._engines.keys()}


class OCREngineManager(OCREngineManagerInterface):
    """
    Main OCR engine manager that coordinates multiple OCR engines.
    """
    
    def __init__(self, factory: OCREngineFactory = None):
        self.factory = factory or OCREngineFactory()
        self._table_detection_enabled = True
        
    def extract_text(self, image: np.ndarray, engine: str = 'auto', **kwargs) -> OCRResult:
        """
        Extract text using specified or auto-selected OCR engine.
        
        Args:
            image: Input image for OCR processing
            engine: Engine name or 'auto' for automatic selection
            **kwargs: Additional parameters for OCR processing
            
        Returns:
            OCR result with extracted text and metadata
            
        Raises:
            OCREngineError: If OCR processing fails
        """
        try:
            # Select engine
            if engine == 'auto':
                language = kwargs.get('language', 'eng')
                strategy = kwargs.get('strategy', EngineSelectionStrategy.BALANCED)
                engine_name = self.factory.select_best_engine(
                    image, strategy=strategy, language=language, **kwargs
                )
            else:
                engine_name = engine
                
            # Get engine instance
            ocr_engine = self.factory.get_engine(engine_name)
            if not ocr_engine:
                raise OCREngineError(f"OCR engine '{engine_name}' not found")
                
            # Extract text
            result = ocr_engine.extract_text(image, **kwargs)
            
            # Set engine used (try to match enum, fallback to what the engine set)
            try:
                result.engine_used = OCREngine(engine_name)
            except ValueError:
                # Keep whatever the engine set, or default to AUTO
                if not hasattr(result, 'engine_used') or result.engine_used is None:
                    result.engine_used = OCREngine.AUTO
            
            logger.info(f"OCR extraction completed using {engine_name}, "
                       f"confidence: {result.confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            raise OCREngineError(f"OCR extraction failed: {str(e)}")
            
    def detect_tables(self, image: np.ndarray, **kwargs) -> List[TableRegion]:
        """
        Detect table regions in document.
        
        Args:
            image: Input image for table detection
            **kwargs: Additional parameters for table detection
            
        Returns:
            List of detected table regions
        """
        if not self._table_detection_enabled:
            return []
            
        try:
            # Use the best available engine for table detection
            engine_name = self.factory.select_best_engine(
                image, strategy=EngineSelectionStrategy.MOST_ACCURATE, **kwargs
            )
            
            ocr_engine = self.factory.get_engine(engine_name)
            if hasattr(ocr_engine, 'detect_tables'):
                return ocr_engine.detect_tables(image, **kwargs)
            else:
                # Fallback to basic table detection
                return self._basic_table_detection(image, **kwargs)
                
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}")
            return []
            
    def register_engine(self, name: str, engine: OCREngineInterface, config: Dict[str, Any] = None) -> None:
        """Register a new OCR engine."""
        self.factory.register_engine(name, engine, config)
        
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines."""
        return self.factory.get_available_engines()
        
    def enable_table_detection(self, enabled: bool = True) -> None:
        """Enable or disable table detection."""
        self._table_detection_enabled = enabled
        logger.info(f"Table detection {'enabled' if enabled else 'disabled'}")
        
    def _basic_table_detection(self, image: np.ndarray, **kwargs) -> List[TableRegion]:
        """
        Basic table detection fallback implementation.
        This is a placeholder for more sophisticated table detection logic.
        """
        # This would implement basic table detection using computer vision
        # For now, return empty list as placeholder
        logger.warning("Using basic table detection fallback")
        return []


# Global factory instance
_global_factory = OCREngineFactory()


def get_global_factory() -> OCREngineFactory:
    """Get the global OCR engine factory instance."""
    return _global_factory


def create_engine_manager(factory: OCREngineFactory = None) -> OCREngineManager:
    """Create a new OCR engine manager instance."""
    return OCREngineManager(factory or _global_factory)