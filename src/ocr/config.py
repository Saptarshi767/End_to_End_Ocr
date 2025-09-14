"""
Configuration management for OCR engines and processing parameters.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import os
import json
import logging
from pathlib import Path

from ..core.models import OCREngine, DataType
from ..core.exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class ConfigurationSource(Enum):
    """Sources for configuration data."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DEFAULT = "default"
    RUNTIME = "runtime"


@dataclass
class OCREngineConfig:
    """Configuration for a specific OCR engine."""
    name: str
    enabled: bool = True
    confidence_threshold: float = 0.8
    language: str = "eng"
    dpi: int = 300
    timeout_seconds: int = 30
    max_retries: int = 3
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'confidence_threshold': self.confidence_threshold,
            'language': self.language,
            'dpi': self.dpi,
            'timeout_seconds': self.timeout_seconds,
            'max_retries': self.max_retries,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCREngineConfig':
        """Create configuration from dictionary."""
        return cls(
            name=data['name'],
            enabled=data.get('enabled', True),
            confidence_threshold=data.get('confidence_threshold', 0.8),
            language=data.get('language', 'eng'),
            dpi=data.get('dpi', 300),
            timeout_seconds=data.get('timeout_seconds', 30),
            max_retries=data.get('max_retries', 3),
            parameters=data.get('parameters', {})
        )


@dataclass
class TableDetectionConfig:
    """Configuration for table detection."""
    enabled: bool = True
    min_table_area: int = 1000
    max_tables_per_page: int = 10
    confidence_threshold: float = 0.7
    merge_nearby_tables: bool = True
    merge_distance_threshold: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'min_table_area': self.min_table_area,
            'max_tables_per_page': self.max_tables_per_page,
            'confidence_threshold': self.confidence_threshold,
            'merge_nearby_tables': self.merge_nearby_tables,
            'merge_distance_threshold': self.merge_distance_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableDetectionConfig':
        """Create configuration from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            min_table_area=data.get('min_table_area', 1000),
            max_tables_per_page=data.get('max_tables_per_page', 10),
            confidence_threshold=data.get('confidence_threshold', 0.7),
            merge_nearby_tables=data.get('merge_nearby_tables', True),
            merge_distance_threshold=data.get('merge_distance_threshold', 50)
        )


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    preprocessing_enabled: bool = True
    noise_reduction: bool = True
    contrast_enhancement: bool = True
    deskew: bool = True
    resize_factor: float = 1.0
    max_image_size: int = 4096
    supported_formats: List[str] = field(default_factory=lambda: [
        'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'preprocessing_enabled': self.preprocessing_enabled,
            'noise_reduction': self.noise_reduction,
            'contrast_enhancement': self.contrast_enhancement,
            'deskew': self.deskew,
            'resize_factor': self.resize_factor,
            'max_image_size': self.max_image_size,
            'supported_formats': self.supported_formats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Create configuration from dictionary."""
        return cls(
            preprocessing_enabled=data.get('preprocessing_enabled', True),
            noise_reduction=data.get('noise_reduction', True),
            contrast_enhancement=data.get('contrast_enhancement', True),
            deskew=data.get('deskew', True),
            resize_factor=data.get('resize_factor', 1.0),
            max_image_size=data.get('max_image_size', 4096),
            supported_formats=data.get('supported_formats', [
                'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'
            ])
        )


class OCRConfigurationManager:
    """
    Manages configuration for OCR engines and processing parameters.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_file()
        self._engine_configs: Dict[str, OCREngineConfig] = {}
        self._table_detection_config = TableDetectionConfig()
        self._processing_config = ProcessingConfig()
        self._config_sources: Dict[str, ConfigurationSource] = {}
        
        # Load default configurations
        self._load_default_configs()
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            self.load_from_file(self.config_file)
        
        # Override with environment variables
        self._load_from_environment()
    
    def _get_default_config_file(self) -> str:
        """Get default configuration file path."""
        return os.path.join(os.getcwd(), 'config', 'ocr_config.json')
    
    def _load_default_configs(self) -> None:
        """Load default configurations for all engines."""
        # Tesseract default configuration
        self._engine_configs['tesseract'] = OCREngineConfig(
            name='tesseract',
            confidence_threshold=0.8,
            parameters={
                'psm': 6,  # Uniform block of text
                'oem': 3,  # Default OCR Engine Mode
                'preserve_interword_spaces': 1
            }
        )
        self._config_sources['tesseract'] = ConfigurationSource.DEFAULT
        
        # EasyOCR default configuration
        self._engine_configs['easyocr'] = OCREngineConfig(
            name='easyocr',
            confidence_threshold=0.7,
            parameters={
                'width_ths': 0.7,
                'height_ths': 0.7,
                'decoder': 'greedy',
                'beamWidth': 5,
                'batch_size': 1
            }
        )
        self._config_sources['easyocr'] = ConfigurationSource.DEFAULT
        
        # Cloud Vision default configuration
        self._engine_configs['cloud_vision'] = OCREngineConfig(
            name='cloud_vision',
            confidence_threshold=0.9,
            parameters={
                'detect_handwriting': True,
                'detect_tables': True,
                'language_hints': ['en'],
                'max_results': 50
            }
        )
        self._config_sources['cloud_vision'] = ConfigurationSource.DEFAULT
        
        # LayoutLM default configuration
        self._engine_configs['layoutlm'] = OCREngineConfig(
            name='layoutlm',
            confidence_threshold=0.85,
            parameters={
                'model_name': 'microsoft/layoutlm-base-uncased',
                'max_seq_length': 512,
                'doc_stride': 128
            }
        )
        self._config_sources['layoutlm'] = ConfigurationSource.DEFAULT
        
        logger.info("Loaded default OCR engine configurations")
    
    def _load_from_environment(self) -> None:
        """Load configuration overrides from environment variables."""
        # General OCR settings
        if 'OCR_DEFAULT_ENGINE' in os.environ:
            default_engine = os.environ['OCR_DEFAULT_ENGINE']
            if default_engine in self._engine_configs:
                logger.info(f"Set default OCR engine from environment: {default_engine}")
        
        # Engine-specific settings
        for engine_name in self._engine_configs.keys():
            env_prefix = f'OCR_{engine_name.upper()}_'
            
            # Check for confidence threshold override
            threshold_key = f'{env_prefix}CONFIDENCE_THRESHOLD'
            if threshold_key in os.environ:
                try:
                    threshold = float(os.environ[threshold_key])
                    self._engine_configs[engine_name].confidence_threshold = threshold
                    self._config_sources[f'{engine_name}_confidence'] = ConfigurationSource.ENVIRONMENT
                    logger.info(f"Set {engine_name} confidence threshold from environment: {threshold}")
                except ValueError:
                    logger.warning(f"Invalid confidence threshold in {threshold_key}")
            
            # Check for language override
            language_key = f'{env_prefix}LANGUAGE'
            if language_key in os.environ:
                language = os.environ[language_key]
                self._engine_configs[engine_name].language = language
                self._config_sources[f'{engine_name}_language'] = ConfigurationSource.ENVIRONMENT
                logger.info(f"Set {engine_name} language from environment: {language}")
            
            # Check for enabled/disabled override
            enabled_key = f'{env_prefix}ENABLED'
            if enabled_key in os.environ:
                enabled = os.environ[enabled_key].lower() in ('true', '1', 'yes', 'on')
                self._engine_configs[engine_name].enabled = enabled
                self._config_sources[f'{engine_name}_enabled'] = ConfigurationSource.ENVIRONMENT
                logger.info(f"Set {engine_name} enabled from environment: {enabled}")
    
    def get_engine_config(self, engine_name: str) -> Optional[OCREngineConfig]:
        """Get configuration for a specific OCR engine."""
        return self._engine_configs.get(engine_name)
    
    def set_engine_config(self, engine_name: str, config: OCREngineConfig) -> None:
        """Set configuration for a specific OCR engine."""
        self._engine_configs[engine_name] = config
        self._config_sources[engine_name] = ConfigurationSource.RUNTIME
        logger.info(f"Updated configuration for engine: {engine_name}")
    
    def update_engine_parameter(self, engine_name: str, parameter: str, value: Any) -> None:
        """Update a specific parameter for an OCR engine."""
        if engine_name not in self._engine_configs:
            raise ConfigurationError(f"Engine '{engine_name}' not found in configuration")
        
        if parameter in ['confidence_threshold', 'language', 'dpi', 'timeout_seconds', 'max_retries']:
            setattr(self._engine_configs[engine_name], parameter, value)
        else:
            self._engine_configs[engine_name].parameters[parameter] = value
        
        self._config_sources[f'{engine_name}_{parameter}'] = ConfigurationSource.RUNTIME
        logger.info(f"Updated {engine_name}.{parameter} = {value}")
    
    def get_table_detection_config(self) -> TableDetectionConfig:
        """Get table detection configuration."""
        return self._table_detection_config
    
    def set_table_detection_config(self, config: TableDetectionConfig) -> None:
        """Set table detection configuration."""
        self._table_detection_config = config
        logger.info("Updated table detection configuration")
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get document processing configuration."""
        return self._processing_config
    
    def set_processing_config(self, config: ProcessingConfig) -> None:
        """Set document processing configuration."""
        self._processing_config = config
        logger.info("Updated document processing configuration")
    
    def get_enabled_engines(self) -> List[str]:
        """Get list of enabled OCR engines."""
        return [name for name, config in self._engine_configs.items() if config.enabled]
    
    def enable_engine(self, engine_name: str, enabled: bool = True) -> None:
        """Enable or disable a specific OCR engine."""
        if engine_name not in self._engine_configs:
            raise ConfigurationError(f"Engine '{engine_name}' not found in configuration")
        
        self._engine_configs[engine_name].enabled = enabled
        self._config_sources[f'{engine_name}_enabled'] = ConfigurationSource.RUNTIME
        logger.info(f"{'Enabled' if enabled else 'Disabled'} OCR engine: {engine_name}")
    
    def load_from_file(self, file_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Load engine configurations
            if 'engines' in data:
                for engine_name, engine_data in data['engines'].items():
                    config = OCREngineConfig.from_dict(engine_data)
                    self._engine_configs[engine_name] = config
                    self._config_sources[engine_name] = ConfigurationSource.FILE
            
            # Load table detection configuration
            if 'table_detection' in data:
                self._table_detection_config = TableDetectionConfig.from_dict(data['table_detection'])
            
            # Load processing configuration
            if 'processing' in data:
                self._processing_config = ProcessingConfig.from_dict(data['processing'])
            
            logger.info(f"Loaded configuration from file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {str(e)}")
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def save_to_file(self, file_path: Optional[str] = None) -> None:
        """Save current configuration to JSON file."""
        file_path = file_path or self.config_file
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare configuration data
            config_data = {
                'engines': {
                    name: config.to_dict() 
                    for name, config in self._engine_configs.items()
                },
                'table_detection': self._table_detection_config.to_dict(),
                'processing': self._processing_config.to_dict()
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved configuration to file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {str(e)}")
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def validate_configuration(self) -> List[str]:
        """
        Validate current configuration and return list of issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Check if at least one engine is enabled
        enabled_engines = self.get_enabled_engines()
        if not enabled_engines:
            issues.append("No OCR engines are enabled")
        
        # Validate engine configurations
        for name, config in self._engine_configs.items():
            if config.confidence_threshold < 0 or config.confidence_threshold > 1:
                issues.append(f"Invalid confidence threshold for {name}: {config.confidence_threshold}")
            
            if config.dpi < 72 or config.dpi > 1200:
                issues.append(f"Invalid DPI for {name}: {config.dpi}")
            
            if config.timeout_seconds <= 0:
                issues.append(f"Invalid timeout for {name}: {config.timeout_seconds}")
        
        # Validate table detection configuration
        if self._table_detection_config.confidence_threshold < 0 or self._table_detection_config.confidence_threshold > 1:
            issues.append(f"Invalid table detection confidence threshold: {self._table_detection_config.confidence_threshold}")
        
        # Validate processing configuration
        if self._processing_config.resize_factor <= 0:
            issues.append(f"Invalid resize factor: {self._processing_config.resize_factor}")
        
        if self._processing_config.max_image_size <= 0:
            issues.append(f"Invalid max image size: {self._processing_config.max_image_size}")
        
        return issues
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            'enabled_engines': self.get_enabled_engines(),
            'total_engines': len(self._engine_configs),
            'table_detection_enabled': self._table_detection_config.enabled,
            'preprocessing_enabled': self._processing_config.preprocessing_enabled,
            'supported_formats': self._processing_config.supported_formats,
            'config_sources': dict(self._config_sources)
        }


# Global configuration manager instance
_global_config_manager = None


def get_global_config_manager() -> OCRConfigurationManager:
    """Get the global OCR configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = OCRConfigurationManager()
    return _global_config_manager


def create_config_manager(config_file: Optional[str] = None) -> OCRConfigurationManager:
    """Create a new OCR configuration manager instance."""
    return OCRConfigurationManager(config_file)