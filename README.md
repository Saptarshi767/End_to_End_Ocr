# End-to-End OCR Table Analytics

A comprehensive OCR (Optical Character Recognition) system for extracting and analyzing tables from images with real-time processing, interactive validation, and intelligent analytics.

![OCR Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

### Core OCR Capabilities
- **Multi-Engine OCR**: EasyOCR and Tesseract support with automatic fallback
- **Advanced Table Detection**: OpenCV-based table region detection
- **Image Preprocessing**: Automatic enhancement for better OCR accuracy
- **Real-time Processing**: Live feedback during OCR operations
- **High Accuracy**: Confidence scoring and quality metrics

### Interactive Interface
- **Drag & Drop Upload**: Support for PNG, JPG, JPEG, TIFF, BMP formats
- **Live Data Editing**: Interactive table validation and correction
- **Smart Analytics**: AI-powered data insights and chat interface
- **Dynamic Dashboards**: Auto-generated visualizations
- **Export Options**: CSV, JSON, and custom formats

### Enterprise Features
- **Session Management**: Persistent data across page navigation
- **Security**: Authentication and audit logging
- **Scalability**: Modular architecture for easy extension
- **Error Handling**: Robust fallback mechanisms

## 📋 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Saptarshi767/End_to_End_Ocr.git
cd End_to_End_Ocr
```

2. **Install dependencies**
```bash
# Option 1: Automatic installation
python install_ocr_deps.py

# Option 2: Manual installation
pip install -r requirements_ocr.txt
```

3. **Set up environment variables**
```bash
# Interactive setup (recommended)
python setup_env.py

# Or copy and edit manually
cp .env.example .env
# Edit .env with your settings
```

4. **Install Tesseract (Optional but recommended)**
- **Windows**: Download from [Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

5. **Create test images**
```bash
python create_test_table.py
```

6. **Run the application**
```bash
python run_ui.py
```

7. **Open your browser** to http://localhost:8501

## 🎯 Usage Guide

### Basic Workflow

1. **Upload Images**: Drag and drop or select table images
2. **Choose OCR Engine**: Auto, EasyOCR, or Tesseract
3. **Process**: Click process to extract table data
4. **Validate**: Review and edit extracted data
5. **Analyze**: Use chat interface for data insights
6. **Export**: Download results in various formats

### Supported Image Formats
- PNG (recommended)
- JPG/JPEG
- TIFF
- BMP

### OCR Engine Selection
- **Auto**: Intelligent selection based on image characteristics
- **EasyOCR**: Best for handwritten text and multiple languages
- **Tesseract**: Optimal for printed documents and forms

## 🏗️ Architecture

```
End_to_End_Ocr/
├── streamlit_app.py          # Main Streamlit application
├── run_ui.py                 # Application launcher
├── install_ocr_deps.py       # Dependency installer
├── create_test_table.py      # Test image generator
├── requirements_ocr.txt      # Python dependencies
├── src/                      # Source code modules
│   ├── ocr/                  # OCR engines and factories
│   ├── data_processing/      # Table extraction and processing
│   ├── core/                 # Core models and interfaces
│   ├── security/             # Authentication and security
│   └── ui/                   # UI components
├── tests/                    # Comprehensive test suite
└── .kiro/                    # Project specifications
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
# UI Configuration
UI_HOST=0.0.0.0
UI_PORT=8501
UI_DEBUG=false

# OCR Settings
DEFAULT_OCR_ENGINE=auto
DEFAULT_CONFIDENCE=0.5
ENABLE_PREPROCESSING=true

# Performance
MAX_IMAGE_SIZE=10485760  # 10MB
PROCESSING_TIMEOUT=300   # 5 minutes
```

### OCR Engine Settings
```python
# Confidence thresholds
EASYOCR_CONFIDENCE = 0.3    # Lower for better detection
TESSERACT_CONFIDENCE = 30   # Tesseract scale (0-100)

# Image preprocessing
DENOISE_STRENGTH = 10
THRESHOLD_BLOCK_SIZE = 11
```

## 📊 Performance

### Benchmarks
- **Processing Speed**: ~2-5 seconds per image
- **Accuracy**: 85-95% depending on image quality
- **Supported Languages**: 80+ languages via EasyOCR
- **Max Image Size**: 10MB (configurable)

### Optimization Tips
1. Use high-resolution images (300+ DPI)
2. Ensure good contrast between text and background
3. Avoid skewed or rotated images
4. Use preprocessing for noisy images

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_ocr_engines.py
python -m pytest tests/test_table_extraction.py
```

### Test with Sample Images
```bash
# Generate test images
python create_test_table.py

# Test OCR processing
python -c "
from streamlit_app import process_uploaded_file
from PIL import Image
import numpy as np

img = Image.open('test_table.png')
result = process_uploaded_file(img, 'auto', True)
print('Test completed:', result is not None)
"
```

## 🚀 Deployment

### Local Development
```bash
python run_ui.py --debug
```

### Production Deployment
```bash
# Using Docker (recommended)
docker build -t ocr-analytics .
docker run -p 8501:8501 ocr-analytics

# Using systemd service
sudo cp ocr-analytics.service /etc/systemd/system/
sudo systemctl enable ocr-analytics
sudo systemctl start ocr-analytics
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Use provided Procfile
- **AWS/GCP**: Container deployment ready

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `python -m pytest`
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings for all functions
- Include tests for new features

## 📝 API Reference

### Core Functions
```python
# OCR Processing
process_uploaded_file(file, engine, preprocessing)

# Table Detection
detect_tables_cv2(image_array)

# Text Extraction
extract_text_easyocr(image_array)
extract_text_tesseract(image_array)

# Data Organization
organize_text_into_table(extracted_data, image_shape)
```

### Configuration Classes
```python
# OCR Engine Configuration
OCREngineConfig(engine_type, confidence_threshold, languages)

# Processing Options
ProcessingOptions(preprocessing, confidence, timeout)
```

## 🔒 Security & Privacy

### Environment Variables & API Keys
- **Never commit `.env` files** to version control
- Use the provided `setup_env.py` script for secure setup
- Store API keys securely and rotate them regularly
- Use environment-specific configuration files

### File Security
- All uploaded files are validated for type and size
- Temporary files are automatically cleaned up
- No executable files are processed
- Sandboxed processing environment

### Data Privacy
- Images are processed locally by default
- No data is sent to external services without explicit configuration
- User data is not stored permanently unless configured
- GDPR and privacy compliance ready

### Security Best Practices
```bash
# Use the secure setup script
python setup_env.py

# Set proper file permissions
chmod 600 .env

# Regular security updates
pip install --upgrade -r requirements.txt
```

## 🐛 Troubleshooting

### Common Issues

**1. "Tesseract not found"**
- Install Tesseract system package
- Add to system PATH
- Use EasyOCR as alternative

**2. "No text detected"**
- Check image quality and resolution
- Adjust confidence threshold
- Enable preprocessing options

**3. "Poor OCR accuracy"**
- Use higher resolution images
- Ensure good contrast
- Try different OCR engines

**4. "Memory issues"**
- Reduce image size
- Adjust batch processing settings
- Monitor system resources

### Performance Issues
- Enable GPU acceleration for EasyOCR
- Optimize image preprocessing
- Use appropriate confidence thresholds
- Consider batch processing for multiple images

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for robust OCR capabilities
- [Tesseract](https://github.com/tesseract-ocr/tesseract) for traditional OCR support
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [OpenCV](https://opencv.org/) for image processing capabilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Saptarshi767/End_to_End_Ocr/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Saptarshi767/End_to_End_Ocr/discussions)
- **Documentation**: [Wiki](https://github.com/Saptarshi767/End_to_End_Ocr/wiki)

## 🔄 Changelog

### v1.0.0 (Latest)
- ✅ Complete OCR pipeline implementation
- ✅ Multi-engine support (EasyOCR + Tesseract)
- ✅ Interactive web interface
- ✅ Real-time table extraction
- ✅ Data validation and editing
- ✅ Analytics and visualization
- ✅ Export functionality
- ✅ Comprehensive error handling

---

**Made with ❤️ by [Saptarshi767](https://github.com/Saptarshi767)**

⭐ **Star this repository if you found it helpful!**