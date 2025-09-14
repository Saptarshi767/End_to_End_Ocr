"""
Configuration helper for Google Cloud Vision OCR engine.
"""

import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CloudVisionConfig:
    """Configuration helper for Cloud Vision engine."""
    
    @staticmethod
    def from_environment() -> Dict[str, Any]:
        """
        Create Cloud Vision configuration from environment variables.
        
        Returns:
            Dictionary with Cloud Vision configuration
        """
        config = {}
        
        # Credentials configuration
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            config['credentials_path'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
        if os.getenv('GOOGLE_CLOUD_VISION_CREDENTIALS'):
            config['credentials_path'] = os.getenv('GOOGLE_CLOUD_VISION_CREDENTIALS')
            
        if os.getenv('GOOGLE_CLOUD_PROJECT'):
            config['project_id'] = os.getenv('GOOGLE_CLOUD_PROJECT')
            
        # API settings
        if os.getenv('CLOUD_VISION_REQUESTS_PER_MINUTE'):
            try:
                config['requests_per_minute'] = int(os.getenv('CLOUD_VISION_REQUESTS_PER_MINUTE'))
            except ValueError:
                logger.warning("Invalid CLOUD_VISION_REQUESTS_PER_MINUTE value, using default")
                
        if os.getenv('CLOUD_VISION_DAILY_LIMIT'):
            try:
                config['daily_request_limit'] = int(os.getenv('CLOUD_VISION_DAILY_LIMIT'))
            except ValueError:
                logger.warning("Invalid CLOUD_VISION_DAILY_LIMIT value, using default")
                
        # Detection settings
        if os.getenv('CLOUD_VISION_DETECT_HANDWRITING'):
            config['detect_handwriting'] = os.getenv('CLOUD_VISION_DETECT_HANDWRITING').lower() == 'true'
            
        if os.getenv('CLOUD_VISION_DETECT_TABLES'):
            config['detect_tables'] = os.getenv('CLOUD_VISION_DETECT_TABLES').lower() == 'true'
            
        # Retry and circuit breaker settings
        if os.getenv('CLOUD_VISION_MAX_RETRIES'):
            try:
                config['max_retries'] = int(os.getenv('CLOUD_VISION_MAX_RETRIES'))
            except ValueError:
                logger.warning("Invalid CLOUD_VISION_MAX_RETRIES value, using default")
                
        if os.getenv('CLOUD_VISION_CIRCUIT_BREAKER_THRESHOLD'):
            try:
                config['circuit_breaker_threshold'] = int(os.getenv('CLOUD_VISION_CIRCUIT_BREAKER_THRESHOLD'))
            except ValueError:
                logger.warning("Invalid CLOUD_VISION_CIRCUIT_BREAKER_THRESHOLD value, using default")
                
        # Language hints
        if os.getenv('CLOUD_VISION_LANGUAGE_HINTS'):
            language_hints = os.getenv('CLOUD_VISION_LANGUAGE_HINTS').split(',')
            config['language_hints'] = [lang.strip() for lang in language_hints]
            
        # Confidence threshold
        if os.getenv('CLOUD_VISION_CONFIDENCE_THRESHOLD'):
            try:
                config['confidence_threshold'] = float(os.getenv('CLOUD_VISION_CONFIDENCE_THRESHOLD'))
            except ValueError:
                logger.warning("Invalid CLOUD_VISION_CONFIDENCE_THRESHOLD value, using default")
        
        logger.info(f"Loaded Cloud Vision configuration from environment: {len(config)} settings")
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate Cloud Vision configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
        """
        # Check for credentials
        has_credentials = any([
            config.get('credentials_path'),
            config.get('credentials_json'),
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            os.getenv('GOOGLE_CLOUD_VISION_CREDENTIALS')
        ])
        
        if not has_credentials:
            logger.warning("No Google Cloud credentials found in configuration")
            return False
            
        # Validate numeric values
        numeric_fields = {
            'requests_per_minute': (1, 10000),
            'daily_request_limit': (1, 1000000),
            'max_retries': (0, 10),
            'circuit_breaker_threshold': (1, 100),
            'confidence_threshold': (0.0, 1.0)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    logger.error(f"Invalid {field} value: {value}. Must be between {min_val} and {max_val}")
                    return False
                    
        return True
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default Cloud Vision configuration.
        
        Returns:
            Dictionary with default configuration values
        """
        return {
            'confidence_threshold': 0.9,
            'detect_handwriting': True,
            'detect_tables': True,
            'language_hints': ['en'],
            'max_results': 50,
            'batch_size': 16,
            'requests_per_minute': 600,
            'daily_request_limit': 1000,
            'max_retries': 3,
            'retry_delay': 1.0,
            'circuit_breaker_threshold': 5,
            'circuit_breaker_timeout': 300
        }
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        Later configs override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        for config in configs:
            if config:
                merged.update(config)
        return merged
    
    @staticmethod
    def create_engine_config() -> Dict[str, Any]:
        """
        Create complete Cloud Vision engine configuration.
        Combines default config, environment config, and validates the result.
        
        Returns:
            Complete configuration dictionary
        """
        default_config = CloudVisionConfig.get_default_config()
        env_config = CloudVisionConfig.from_environment()
        
        # Merge configurations (environment overrides defaults)
        config = CloudVisionConfig.merge_configs(default_config, env_config)
        
        # Validate the final configuration
        if not CloudVisionConfig.validate_config(config):
            logger.warning("Cloud Vision configuration validation failed, using defaults")
            config = default_config
            
        return config


def configure_cloud_vision_from_env() -> Dict[str, Any]:
    """
    Convenience function to configure Cloud Vision from environment.
    
    Returns:
        Cloud Vision configuration dictionary
    """
    return CloudVisionConfig.create_engine_config()