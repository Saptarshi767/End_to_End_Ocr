"""
Main application entry point for the OCR Table Analytics system.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from core.utils import setup_logging
from core.exceptions import ConfigurationError


def initialize_application():
    """Initialize the application with configuration and logging."""
    # Load environment variables from .env file
    from dotenv import load_dotenv
    
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    print(f"Loading .env from: {env_path}")
    print(f".env exists: {env_path.exists()}")
    load_dotenv(dotenv_path=env_path)
    
    # Debug: Check if API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"OPENAI_API_KEY loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API key starts with: {api_key[:10]}...")
    
    # Import config manager after loading environment variables
    from core.config import ConfigurationManager
    config_manager = ConfigurationManager()
    
    # Set up logging
    logger = setup_logging(
        log_level=config_manager.config.log_level,
        log_file="ocr_analytics.log" if not config_manager.config.debug_mode else None
    )
    
    # Validate configuration
    validation_result = config_manager.validate_config()
    if not validation_result.is_valid:
        logger.error("Configuration validation failed:")
        for error in validation_result.errors:
            logger.error(f"  - {error}")
        raise ConfigurationError("Invalid configuration")
    
    if validation_result.warnings:
        logger.warning("Configuration warnings:")
        for warning in validation_result.warnings:
            logger.warning(f"  - {warning}")
    
    logger.info("OCR Table Analytics system initialized successfully")
    
    # Return both logger and config_manager for use in main
    return logger, config_manager


def main():
    """Main application function."""
    try:
        logger, config_manager = initialize_application()
        
        # Print system information in debug mode
        if config_manager.config.debug_mode:
            from core.utils import get_system_info
            system_info = get_system_info()
            logger.debug("System Information:")
            for key, value in system_info.items():
                logger.debug(f"  {key}: {value}")
        
        logger.info("Application ready to process documents")
        
        # TODO: Start web server or CLI interface based on configuration
        print("OCR Table Analytics System")
        print("=" * 40)
        print("System initialized successfully!")
        print(f"Debug mode: {config_manager.config.debug_mode}")
        print(f"Log level: {config_manager.config.log_level}")
        print(f"OpenAI API configured: {'Yes' if config_manager.config.llm.openai_api_key else 'No'}")
        print("\nReady to process documents...")
        
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()