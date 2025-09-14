# OCR Table Analytics - Full Implementation

A robust OCR (Optical Character Recognition) system for extracting and analyzing tables from images.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Option 1: Install all dependencies at once
pip install -r requirements_ocr.txt

# Option 2: Use the installation script
python install_ocr_deps.py

# Option 3: Manual installation
pip install streamlit pandas numpy opencv-python pillow easyocr pytesseract plotly
```

### 2. Install Tesseract (Required for OCR)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### 3. Create Test Images (Optional)

```bash
python create_test_table.py
```

### 4. Run the Application

```bash
python run_ui.py
```

Visit: http://localhost:8501

## 📋 Features

### ✅ Implemented Features

- **Real OCR Processing**: Uses EasyOCR and Tesseract engines
- **Table Detection**: Automatically detects table regions in images
- **Image Preprocessing**: Enhances images for better OCR accuracy
- **Interactive Validation**: Edit and correct extracted data
- **Multiple OCR Engines**: Auto-selection or manual choice
- **Data Analysis**: Built-in chat interface for data insights
- **Interactive Dashboard**: Auto-generated visualizations
- **Export Options**: CSV download with various formats
- **Real-time Processing**: Live feedback during OCR processing

### 🔧 Technical Features

- **OpenCV Integration**: Advanced image preprocessing
- **Confidence Scoring**: Quality metrics for extracted data
- **Error Handling**: Graceful fallbacks and user feedback
- **Session Management**: Persistent data across page navigation
- **Responsive UI**: Works on desktop and mobile devices

## 📖 Usage Guide

### 1. Upload Documents

1. Go to the **Upload Documents** page
2. Drag and drop or select image files (PNG, JPG, JPEG, TIFF, BMP)
3. Choose OCR engine (Auto, EasyOCR, or Tesseract)
4. Enable image preprocessing for better results
5. Click **Process** to extract tables

### 2. Validate & Edit Data

1. Navigate to **Validate Tables** page
2. Review extracted data and confidence scores
3. Edit cells directly in the data editor
4. Use **Clean Data** for automatic cleanup
5. Save changes and export as needed

### 3. Analyze Data

1. Go to **Chat Analysis** page
2. Ask questions about your data
3. Use quick analysis buttons for common insights
4. Get real-time responses based on actual data

### 4. View Dashboard

1. Visit **Dashboard** page for auto-generated visualizations
2. Explore data distribution and relationships
3. Use filters and search functionality
4. Export data and statistics

## 🛠️ Configuration

### OCR Engine Settings

- **Auto**: Tries EasyOCR first, falls back to Tesseract
- **EasyOCR**: Better for handwritten text and multiple languages
- **Tesseract**: Better for printed text and documents

### Image Preprocessing Options

- **Denoising**: Removes image noise
- **Adaptive Thresholding**: Improves text contrast
- **Morphological Operations**: Cleans up text regions

### Confidence Thresholds

- **High (0.8+)**: Very reliable text detection
- **Medium (0.5-0.8)**: Good quality with some uncertainty
- **Low (0.3-0.5)**: May need manual verification

## 🔍 Troubleshooting

### Common Issues

**1. "OCR libraries not available"**
```bash
pip install easyocr pytesseract
```

**2. "Tesseract not found"**
- Install Tesseract system package
- Add to system PATH
- Restart terminal/IDE

**3. "No table detected"**
- Ensure image has clear table structure
- Try different confidence thresholds
- Use image preprocessing options
- Check image quality and resolution

**4. "Poor OCR accuracy"**
- Use higher resolution images
- Ensure good contrast
- Try different OCR engines
- Enable preprocessing options

### Performance Tips

1. **Image Quality**: Use high-resolution, clear images
2. **File Size**: Optimize large images before processing
3. **Table Structure**: Ensure clear borders and alignment
4. **Lighting**: Avoid shadows and glare in photos

## 📊 Supported Formats

### Input Formats
- PNG, JPG, JPEG (recommended)
- TIFF, BMP
- High resolution preferred (300+ DPI)

### Output Formats
- CSV (Comma-separated values)
- Interactive tables (Streamlit data editor)
- JSON (via API endpoints)

## 🔧 Advanced Configuration

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

### Custom OCR Settings

Edit `streamlit_app.py` to customize:

```python
# OCR confidence thresholds
EASYOCR_CONFIDENCE = 0.5
TESSERACT_CONFIDENCE = 30

# Image preprocessing parameters
DENOISE_STRENGTH = 10
THRESHOLD_BLOCK_SIZE = 11
MORPHOLOGY_KERNEL_SIZE = (1, 1)
```

## 🚀 Development

### Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── run_ui.py                 # Application launcher
├── requirements_ocr.txt      # Python dependencies
├── install_ocr_deps.py       # Dependency installer
├── create_test_table.py      # Test image generator
├── src/                      # Source code modules
│   ├── ocr/                  # OCR engines
│   ├── data_processing/      # Table extraction
│   └── core/                 # Core models and interfaces
└── tests/                    # Test files
```

### Adding New OCR Engines

1. Create engine class in `src/ocr/`
2. Implement `BaseOCREngine` interface
3. Register in `OCREngineFactory`
4. Update UI dropdown options

### Extending Functionality

- Add new preprocessing filters
- Implement additional export formats
- Create custom visualization types
- Add batch processing capabilities

## 📝 License

This project is open source. See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📞 Support

For issues and questions:
1. Check troubleshooting section
2. Review error messages carefully
3. Test with provided sample images
4. Check system requirements

---

**Happy OCR Processing! 🎉**