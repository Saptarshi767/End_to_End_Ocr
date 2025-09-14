"""
Tests for Cloud Vision configuration helper.
"""

import pytest
import os
from unittest.mock import patch

from src.ocr.cloud_vision_config import CloudVisionConfig, configure_cloud_vision_from_env


class TestCloudVisionConfig:
    """Test the Cloud Vision configuration helper."""
    
    def test_default_config(self):
        """Test getting default configuration."""
        config = CloudVisionConfig.get_default_config()
        
        assert config['confidence_threshold'] == 0.9
        assert config['detect_handwriting'] == True
        assert config['detect_tables'] == True
        assert config['language_hints'] == ['en']
        assert config['max_results'] == 50
        assert config['batch_size'] == 16
        assert config['requests_per_minute'] == 600
        assert config['daily_request_limit'] == 1000
        assert config['max_retries'] == 3
        assert config['retry_delay'] == 1.0
        assert config['circuit_breaker_threshold'] == 5
        assert config['circuit_breaker_timeout'] == 300
    
    def test_from_environment_empty(self):
        """Test configuration from empty environment."""
        with patch.dict(os.environ, {}, clear=True):
            config = CloudVisionConfig.from_environment()
            assert config == {}
    
    def test_from_environment_with_credentials(self):
        """Test configuration from environment with credentials."""
        env_vars = {
            'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/creds.json',
            'GOOGLE_CLOUD_PROJECT': 'test-project',
            'CLOUD_VISION_REQUESTS_PER_MINUTE': '300',
            'CLOUD_VISION_DAILY_LIMIT': '500',
            'CLOUD_VISION_DETECT_HANDWRITING': 'false',
            'CLOUD_VISION_DETECT_TABLES': 'true',
            'CLOUD_VISION_MAX_RETRIES': '5',
            'CLOUD_VISION_CIRCUIT_BREAKER_THRESHOLD': '10',
            'CLOUD_VISION_LANGUAGE_HINTS': 'en,es,fr',
            'CLOUD_VISION_CONFIDENCE_THRESHOLD': '0.85'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = CloudVisionConfig.from_environment()
            
            assert config['credentials_path'] == '/path/to/creds.json'
            assert config['project_id'] == 'test-project'
            assert config['requests_per_minute'] == 300
            assert config['daily_request_limit'] == 500
            assert config['detect_handwriting'] == False
            assert config['detect_tables'] == True
            assert config['max_retries'] == 5
            assert config['circuit_breaker_threshold'] == 10
            assert config['language_hints'] == ['en', 'es', 'fr']
            assert config['confidence_threshold'] == 0.85
    
    def test_from_environment_with_vision_credentials(self):
        """Test configuration with Cloud Vision specific credentials."""
        env_vars = {
            'GOOGLE_CLOUD_VISION_CREDENTIALS': '/path/to/vision-creds.json'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = CloudVisionConfig.from_environment()
            
            assert config['credentials_path'] == '/path/to/vision-creds.json'
    
    def test_from_environment_invalid_numeric_values(self):
        """Test configuration with invalid numeric values."""
        env_vars = {
            'CLOUD_VISION_REQUESTS_PER_MINUTE': 'invalid',
            'CLOUD_VISION_DAILY_LIMIT': 'not_a_number',
            'CLOUD_VISION_MAX_RETRIES': 'bad_value',
            'CLOUD_VISION_CIRCUIT_BREAKER_THRESHOLD': 'wrong',
            'CLOUD_VISION_CONFIDENCE_THRESHOLD': 'invalid_float'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # Should not raise exception, just log warnings
            config = CloudVisionConfig.from_environment()
            
            # Invalid values should not be included
            assert 'requests_per_minute' not in config
            assert 'daily_request_limit' not in config
            assert 'max_retries' not in config
            assert 'circuit_breaker_threshold' not in config
            assert 'confidence_threshold' not in config
    
    def test_validate_config_with_credentials(self):
        """Test configuration validation with credentials."""
        config = {
            'credentials_path': '/path/to/creds.json',
            'requests_per_minute': 600,
            'daily_request_limit': 1000,
            'max_retries': 3,
            'circuit_breaker_threshold': 5,
            'confidence_threshold': 0.9
        }
        
        assert CloudVisionConfig.validate_config(config) == True
    
    def test_validate_config_without_credentials(self):
        """Test configuration validation without credentials."""
        config = {
            'requests_per_minute': 600,
            'daily_request_limit': 1000
        }
        
        with patch.dict(os.environ, {}, clear=True):
            assert CloudVisionConfig.validate_config(config) == False
    
    def test_validate_config_with_env_credentials(self):
        """Test configuration validation with environment credentials."""
        config = {
            'requests_per_minute': 600
        }
        
        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/creds.json'}):
            assert CloudVisionConfig.validate_config(config) == True
    
    def test_validate_config_invalid_numeric_values(self):
        """Test configuration validation with invalid numeric values."""
        configs = [
            {'credentials_path': '/path', 'requests_per_minute': 0},  # Too low
            {'credentials_path': '/path', 'requests_per_minute': 20000},  # Too high
            {'credentials_path': '/path', 'daily_request_limit': 0},  # Too low
            {'credentials_path': '/path', 'max_retries': -1},  # Too low
            {'credentials_path': '/path', 'max_retries': 20},  # Too high
            {'credentials_path': '/path', 'circuit_breaker_threshold': 0},  # Too low
            {'credentials_path': '/path', 'circuit_breaker_threshold': 200},  # Too high
            {'credentials_path': '/path', 'confidence_threshold': -0.1},  # Too low
            {'credentials_path': '/path', 'confidence_threshold': 1.1},  # Too high
        ]
        
        for config in configs:
            assert CloudVisionConfig.validate_config(config) == False
    
    def test_merge_configs(self):
        """Test merging multiple configurations."""
        config1 = {
            'confidence_threshold': 0.8,
            'detect_handwriting': True,
            'max_retries': 3
        }
        
        config2 = {
            'confidence_threshold': 0.9,  # Should override
            'detect_tables': True,  # Should be added
            'max_retries': 5  # Should override
        }
        
        config3 = {
            'project_id': 'test-project'  # Should be added
        }
        
        merged = CloudVisionConfig.merge_configs(config1, config2, config3)
        
        assert merged['confidence_threshold'] == 0.9  # From config2
        assert merged['detect_handwriting'] == True  # From config1
        assert merged['detect_tables'] == True  # From config2
        assert merged['max_retries'] == 5  # From config2
        assert merged['project_id'] == 'test-project'  # From config3
    
    def test_merge_configs_with_none(self):
        """Test merging configurations with None values."""
        config1 = {'confidence_threshold': 0.8}
        config2 = None
        config3 = {'max_retries': 3}
        
        merged = CloudVisionConfig.merge_configs(config1, config2, config3)
        
        assert merged['confidence_threshold'] == 0.8
        assert merged['max_retries'] == 3
    
    def test_create_engine_config(self):
        """Test creating complete engine configuration."""
        env_vars = {
            'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/creds.json',
            'CLOUD_VISION_REQUESTS_PER_MINUTE': '300',
            'CLOUD_VISION_DETECT_HANDWRITING': 'false'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = CloudVisionConfig.create_engine_config()
            
            # Should have defaults
            assert config['confidence_threshold'] == 0.9
            assert config['detect_tables'] == True
            assert config['max_retries'] == 3
            
            # Should have environment overrides
            assert config['credentials_path'] == '/path/to/creds.json'
            assert config['requests_per_minute'] == 300
            assert config['detect_handwriting'] == False
    
    def test_create_engine_config_validation_failure(self):
        """Test engine configuration creation with validation failure."""
        env_vars = {
            'CLOUD_VISION_REQUESTS_PER_MINUTE': '0'  # Invalid value
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = CloudVisionConfig.create_engine_config()
            
            # Should fall back to defaults due to validation failure
            assert config == CloudVisionConfig.get_default_config()
    
    def test_configure_cloud_vision_from_env(self):
        """Test convenience function."""
        env_vars = {
            'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/creds.json',
            'CLOUD_VISION_MAX_RETRIES': '5'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = configure_cloud_vision_from_env()
            
            assert isinstance(config, dict)
            assert config['credentials_path'] == '/path/to/creds.json'
            assert config['max_retries'] == 5
            assert config['confidence_threshold'] == 0.9  # Default value