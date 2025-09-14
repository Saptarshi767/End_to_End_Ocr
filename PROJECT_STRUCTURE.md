# Project Structure Overview

This document provides an overview of the OCR Table Analytics system project structure.

## Directory Structure

```
ocr-table-analytics/
├── .env                        # Environment variables (configured with OpenAI API key)
├── .env.example               # Environment variables template
├── README.md                  # Project documentation
├── PROJECT_STRUCTURE.md       # This file
├── requirements.txt           # Python dependencies
├── config.json               # Optional configuration file
├── ocr_analytics.log         # Application log file (created at runtime)
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── main.py              # Application entry point
│   │
│   ├── core/                # Core system components
│   │   ├── __init__.py
│   │   ├── models.py        # Data models and type definitions
│   │   ├── interfaces.py    # Abstract base classes and interfaces
│   │   ├── config.py        # Configuration management
│   │   ├── exceptions.py    # Custom exception classes
│   │   └── utils.py         # Utility functions
│   │
│   ├── ocr/                 # OCR processing components
│   │   └── __init__.py
│   │
│   ├── data_processing/     # Data cleaning and standardization
│   │   └── __init__.py
│   │
│   ├── visualization/       # Dashboard and chart generation
│   │   └── __init__.py
│   │
│   └── ai/                  # Conversational AI and LLM integration
│       └── __init__.py
│
└── tests/                   # Test suite
    ├── __init__.py
    └── test_core.py         # Core functionality tests
```

## Key Components Implemented

### 1. Core Models (`src/core/models.py`)
- **Data Classes**: Complete set of dataclasses for all system entities
- **Enumerations**: Type-safe enums for processing status, OCR engines, data types, chart types
- **Type Definitions**: Comprehensive type hints for all data structures

### 2. Interfaces (`src/core/interfaces.py`)
- **Abstract Base Classes**: Interfaces for all major system components
- **Contract Definitions**: Clear contracts for OCR engines, data processing, visualization, and AI components
- **Extensibility**: Easy to add new implementations following defined interfaces

### 3. Configuration Management (`src/core/config.py`)
- **Environment Variables**: Automatic loading from .env file
- **Multi-source Configuration**: Support for file-based and environment-based configuration
- **Validation**: Built-in configuration validation with error reporting
- **API Key Management**: Secure handling of OpenAI and other service API keys

### 4. Error Handling (`src/core/exceptions.py`)
- **Custom Exceptions**: Specific exception types for different error categories
- **Context Information**: Rich error context for debugging
- **Error Codes**: Structured error identification

### 5. Utilities (`src/core/utils.py`)
- **Logging Setup**: Configurable logging with file and console output
- **File Operations**: File validation, hashing, and metadata extraction
- **Data Processing Helpers**: Column name sanitization, API key validation
- **System Information**: Runtime system information gathering

## Configuration

### Environment Variables
The system is configured with the following environment variables:

- `OPENAI_API_KEY`: OpenAI API key (configured)
- `DATABASE_URL`: PostgreSQL connection string
- `MAX_FILE_SIZE_MB`: Maximum file size for uploads
- `TEMP_DIRECTORY`: Temporary directory for processing
- `DEBUG`: Debug mode flag
- `LOG_LEVEL`: Logging level

### Supported Features
- ✅ Project structure and core interfaces
- ✅ Configuration management with OpenAI API key
- ✅ Core data models and type definitions
- ✅ Error handling framework
- ✅ Logging and utilities
- ✅ Basic testing framework

### Next Steps
The foundation is now ready for implementing the specific components:
1. Document processing and OCR engines
2. Table extraction and data cleaning
3. Visualization and dashboard generation
4. Conversational AI integration
5. Web interface and API endpoints

## Testing

Run tests with:
```bash
python -m pytest tests/ -v
```

## Running the Application

```bash
python src/main.py
```

The application will:
1. Load environment variables from .env
2. Initialize configuration
3. Validate settings
4. Set up logging
5. Display system status

## Requirements Addressed

This implementation addresses the following requirements from the specification:

- **Requirement 1.1**: Foundation for document upload and OCR processing
- **Requirement 5.1**: Framework for multiple OCR engine support
- **Requirement 8.1**: Comprehensive error handling and logging system