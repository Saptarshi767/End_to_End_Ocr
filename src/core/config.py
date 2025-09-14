"""
Configuration management for the OCR Table Analytics system.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .interfaces import ConfigurationManagerInterface
from .models import ValidationResult


@dataclass
class OCREngineConfig:
    """Configuration for OCR engines."""
    tesseract_path: str = "tesseract"
    tesseract_config: str = "--oem 3 --psm 6"
    easyocr_gpu: bool = False
    easyocr_languages: list = field(default_factory=lambda: ["en"])
    cloud_vision_credentials: Optional[str] = None
    confidence_threshold: float = 0.8
    preprocessing_enabled: bool = True


@dataclass
class LLMProviderConfig:
    """Configuration for LLM providers."""
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 2000
    openai_temperature: float = 0.1
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-3-sonnet-20240229"
    local_model_path: Optional[str] = None
    timeout_seconds: int = 30


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    host: str = "localhost"
    port: int = 5432
    database: str = "ocr_analytics"
    username: str = "postgres"
    password: Optional[str] = None
    connection_pool_size: int = 10
    connection_timeout: int = 30


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    max_file_size_mb: int = 100
    supported_formats: list = field(default_factory=lambda: ["pdf", "png", "jpg", "jpeg", "tiff"])
    temp_directory: str = "/tmp/ocr_processing"
    batch_size: int = 10
    max_concurrent_jobs: int = 5
    cleanup_temp_files: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
    default_chart_library: str = "plotly"
    max_data_points: int = 10000
    color_palette: list = field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    export_dpi: int = 300
    dashboard_refresh_interval: int = 30


@dataclass
class SystemConfig:
    """Main system configuration."""
    ocr: OCREngineConfig = field(default_factory=OCREngineConfig)
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    debug_mode: bool = False
    log_level: str = "INFO"


class ConfigurationManager(ConfigurationManagerInterface):
    """Configuration manager implementation."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.config = SystemConfig()
        self._load_config()
        self._load_environment_variables()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    self._update_config_from_dict(config_data)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        # OpenAI API Key
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.config.llm.openai_api_key = openai_key
        
        # Database configuration
        if os.getenv('DATABASE_URL'):
            # Parse DATABASE_URL if provided
            db_url = os.getenv('DATABASE_URL')
            # Simple parsing - in production, use proper URL parsing
            if 'postgresql://' in db_url:
                parts = db_url.replace('postgresql://', '').split('@')
                if len(parts) == 2:
                    user_pass, host_db = parts
                    if ':' in user_pass:
                        self.config.database.username, self.config.database.password = user_pass.split(':', 1)
                    if '/' in host_db:
                        host_port, database = host_db.split('/', 1)
                        if ':' in host_port:
                            self.config.database.host, port_str = host_port.split(':', 1)
                            self.config.database.port = int(port_str)
                        else:
                            self.config.database.host = host_port
                        self.config.database.database = database
        
        # Other environment variables
        if os.getenv('DEBUG'):
            self.config.debug_mode = os.getenv('DEBUG').lower() in ('true', '1', 'yes')
        
        if os.getenv('LOG_LEVEL'):
            self.config.log_level = os.getenv('LOG_LEVEL').upper()
        
        if os.getenv('MAX_FILE_SIZE_MB'):
            self.config.processing.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB'))
        
        if os.getenv('TEMP_DIRECTORY'):
            self.config.processing.temp_directory = os.getenv('TEMP_DIRECTORY')
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'ocr' in config_data:
            ocr_config = config_data['ocr']
            for key, value in ocr_config.items():
                if hasattr(self.config.ocr, key):
                    setattr(self.config.ocr, key, value)
        
        if 'llm' in config_data:
            llm_config = config_data['llm']
            for key, value in llm_config.items():
                if hasattr(self.config.llm, key):
                    setattr(self.config.llm, key, value)
        
        if 'database' in config_data:
            db_config = config_data['database']
            for key, value in db_config.items():
                if hasattr(self.config.database, key):
                    setattr(self.config.database, key, value)
        
        if 'processing' in config_data:
            proc_config = config_data['processing']
            for key, value in proc_config.items():
                if hasattr(self.config.processing, key):
                    setattr(self.config.processing, key, value)
        
        if 'visualization' in config_data:
            viz_config = config_data['visualization']
            for key, value in viz_config.items():
                if hasattr(self.config.visualization, key):
                    setattr(self.config.visualization, key, value)
        
        # Top-level config
        for key in ['debug_mode', 'log_level']:
            if key in config_data:
                setattr(self.config, key, config_data[key])
    
    def get_ocr_config(self, engine: str) -> Dict[str, Any]:
        """Get configuration for OCR engine."""
        base_config = {
            'confidence_threshold': self.config.ocr.confidence_threshold,
            'preprocessing_enabled': self.config.ocr.preprocessing_enabled
        }
        
        if engine == 'tesseract':
            base_config.update({
                'tesseract_path': self.config.ocr.tesseract_path,
                'tesseract_config': self.config.ocr.tesseract_config
            })
        elif engine == 'easyocr':
            base_config.update({
                'gpu': self.config.ocr.easyocr_gpu,
                'languages': self.config.ocr.easyocr_languages
            })
        elif engine == 'cloud_vision':
            base_config.update({
                'credentials': self.config.ocr.cloud_vision_credentials
            })
        
        return base_config
    
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for LLM provider."""
        base_config = {
            'timeout_seconds': self.config.llm.timeout_seconds
        }
        
        if provider == 'openai':
            base_config.update({
                'api_key': self.config.llm.openai_api_key,
                'model': self.config.llm.openai_model,
                'max_tokens': self.config.llm.openai_max_tokens,
                'temperature': self.config.llm.openai_temperature
            })
        elif provider == 'claude':
            base_config.update({
                'api_key': self.config.llm.claude_api_key,
                'model': self.config.llm.claude_model
            })
        elif provider == 'local':
            base_config.update({
                'model_path': self.config.llm.local_model_path
            })
        
        return base_config
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update configuration value."""
        if hasattr(self.config, section):
            section_obj = getattr(self.config, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {section}.{key}")
        else:
            raise ValueError(f"Invalid configuration section: {section}")
    
    def validate_config(self) -> ValidationResult:
        """Validate current configuration."""
        errors = []
        warnings = []
        
        # Validate LLM configuration
        if not self.config.llm.openai_api_key and not self.config.llm.claude_api_key:
            errors.append("No LLM API key configured. Set OPENAI_API_KEY or configure claude_api_key.")
        
        # Validate processing configuration
        if self.config.processing.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be greater than 0")
        
        if not self.config.processing.supported_formats:
            errors.append("No supported file formats configured")
        
        # Validate temp directory
        temp_dir = Path(self.config.processing.temp_directory)
        if not temp_dir.parent.exists():
            warnings.append(f"Temp directory parent does not exist: {temp_dir.parent}")
        
        # Validate database configuration
        if not self.config.database.host:
            errors.append("Database host not configured")
        
        if self.config.database.port <= 0 or self.config.database.port > 65535:
            errors.append("Invalid database port")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=1.0 if len(errors) == 0 else 0.0
        )
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        config_dict = {
            'ocr': {
                'tesseract_path': self.config.ocr.tesseract_path,
                'tesseract_config': self.config.ocr.tesseract_config,
                'easyocr_gpu': self.config.ocr.easyocr_gpu,
                'easyocr_languages': self.config.ocr.easyocr_languages,
                'confidence_threshold': self.config.ocr.confidence_threshold,
                'preprocessing_enabled': self.config.ocr.preprocessing_enabled
            },
            'llm': {
                'openai_model': self.config.llm.openai_model,
                'openai_max_tokens': self.config.llm.openai_max_tokens,
                'openai_temperature': self.config.llm.openai_temperature,
                'claude_model': self.config.llm.claude_model,
                'local_model_path': self.config.llm.local_model_path,
                'timeout_seconds': self.config.llm.timeout_seconds
            },
            'database': {
                'host': self.config.database.host,
                'port': self.config.database.port,
                'database': self.config.database.database,
                'username': self.config.database.username,
                'connection_pool_size': self.config.database.connection_pool_size,
                'connection_timeout': self.config.database.connection_timeout
            },
            'processing': {
                'max_file_size_mb': self.config.processing.max_file_size_mb,
                'supported_formats': self.config.processing.supported_formats,
                'temp_directory': self.config.processing.temp_directory,
                'batch_size': self.config.processing.batch_size,
                'max_concurrent_jobs': self.config.processing.max_concurrent_jobs,
                'cleanup_temp_files': self.config.processing.cleanup_temp_files
            },
            'visualization': {
                'default_chart_library': self.config.visualization.default_chart_library,
                'max_data_points': self.config.visualization.max_data_points,
                'color_palette': self.config.visualization.color_palette,
                'export_dpi': self.config.visualization.export_dpi,
                'dashboard_refresh_interval': self.config.visualization.dashboard_refresh_interval
            },
            'debug_mode': self.config.debug_mode,
            'log_level': self.config.log_level
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)


# Global configuration instance
config_manager = ConfigurationManager()