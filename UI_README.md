# OCR Table Analytics - User Interface

This document describes the user interface components for the OCR Table Analytics system, including document upload, table validation, and conversational chat interfaces.

## Overview

The UI is built using Streamlit and provides three main interfaces:

1. **Document Upload Interface** - Drag-and-drop file upload with progress tracking
2. **Table Validation Interface** - Side-by-side document and data editing
3. **Conversational Chat Interface** - Natural language data analysis

## Architecture

```
src/ui/
├── app.py                      # Main Streamlit application
├── components/
│   ├── document_upload.py      # Document upload interface
│   ├── table_validation.py     # Table validation and correction
│   └── chat_interface.py       # Conversational chat interface
└── __init__.py

tests/
├── test_ui_document_upload.py  # Upload interface tests
├── test_ui_table_validation.py # Validation interface tests
└── test_ui_chat_interface.py   # Chat interface tests
```

## Features

### 1. Document Upload Interface

#### Features
- **Drag-and-drop file upload** with support for PDF, PNG, JPG, JPEG, TIFF, BMP
- **Progress indicators** showing upload and processing status
- **Document preview** with metadata display
- **Batch upload capabilities** with queue management
- **Processing options** for OCR engine selection and parameters

#### Components
- File validation and size checking (max 50MB)
- Image preprocessing options
- OCR engine selection (Tesseract, EasyOCR, Cloud Vision, Auto)
- Confidence threshold configuration
- Upload queue with status tracking
- Upload history with document management

#### Usage
```python
from src.ui.components.document_upload import render_document_upload

# Render the upload interface
render_document_upload()
```

### 2. Table Validation Interface

#### Features
- **Side-by-side view** of original document and extracted data
- **Inline editing capabilities** for table corrections
- **Confidence indicators** and validation warnings
- **Data type detection** and correction suggestions
- **Missing value handling** with multiple strategies
- **Export functionality** for corrected data

#### Components
- Document preview with table region highlighting
- Editable data grid with real-time validation
- Confidence scoring and issue detection
- Change tracking and comparison
- Auto-correction suggestions
- Export to CSV, Excel, JSON formats

#### Usage
```python
from src.ui.components.table_validation import render_table_validation

# Render validation interface for a document
render_table_validation(document_id="doc_123")
```

### 3. Conversational Chat Interface

#### Features
- **Chat UI** with message history and context management
- **Quick question suggestions** based on data schema
- **Visualization embedding** in chat responses
- **Natural language processing** for data queries
- **Follow-up question suggestions**
- **Chat export functionality**

#### Components
- Message history with user and assistant messages
- Contextual question suggestions
- Embedded charts (bar, line, pie, scatter)
- Data summary displays
- Chat controls and settings
- Export chat history

#### Usage
```python
from src.ui.components.chat_interface import render_chat_interface

# Render chat interface for a document
render_chat_interface(document_id="doc_123")
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

#### Option 1: Minimal Installation (Recommended)
```bash
# Install only core dependencies needed for basic functionality
pip install -r requirements-ui-minimal.txt
```

#### Option 2: Full Installation
```bash
# Install all UI dependencies including optional components
pip install -r requirements-ui.txt
```

#### Option 3: Manual Installation
```bash
# Install core dependencies manually
pip install streamlit pandas plotly requests Pillow python-dotenv pydantic
```

### Environment Setup
Create a `.env` file with configuration:
```env
API_BASE_URL=http://localhost:8000
UI_HOST=0.0.0.0
UI_PORT=8501
UI_DEBUG=false
```

## Running the Application

### Quick Start
```bash
# Run with default settings
python run_ui.py

# Run on custom port
python run_ui.py --port 8502

# Run in debug mode
python run_ui.py --debug

# Run on localhost only
python run_ui.py --host 127.0.0.1
```

### Manual Start
```bash
# Start Streamlit directly
streamlit run src/ui/app.py --server.port 8501
```

### Access the Application
Open your browser and navigate to:
- Default: http://localhost:8501
- Custom: http://localhost:[PORT]

### Dependency Troubleshooting
If you encounter issues with package installation:

```bash
# Use minimal requirements (recommended)
pip install -r requirements-ui-minimal.txt

# Or install core packages manually
pip install streamlit pandas plotly requests Pillow python-dotenv pydantic

# Check dependencies before running
python run_ui.py --skip-checks
```

## Configuration

### Streamlit Configuration
The application creates a `.streamlit/config.toml` file with optimized settings:

```toml
[global]
developmentMode = false
showWarningOnDirectExecution = false

[server]
headless = true
enableCORS = false
maxUploadSize = 200

[theme]
primaryColor = "#0066CC"
backgroundColor = "#FFFFFF"
```

### API Configuration
Configure the backend API connection:
```python
# In your environment or code
API_BASE_URL = "http://localhost:8000"
```

## User Workflows

### 1. Document Processing Workflow
1. **Upload Documents**
   - Drag and drop files or click to browse
   - Configure OCR settings (engine, preprocessing, confidence)
   - Monitor upload progress and processing status

2. **Validate Tables**
   - Review extracted table data
   - Compare with original document
   - Edit cells inline and correct headers
   - Handle validation warnings and issues

3. **Analyze Data**
   - Ask questions in natural language
   - View generated visualizations
   - Explore suggested follow-up questions
   - Export results and chat history

### 2. Batch Processing Workflow
1. **Upload Multiple Documents**
   - Select multiple files for batch upload
   - Configure processing options for all files
   - Monitor queue progress

2. **Review and Validate**
   - Process each document's tables
   - Apply corrections and validations
   - Export corrected data

3. **Analyze Across Documents**
   - Compare data across multiple documents
   - Generate consolidated reports
   - Share insights with team

## API Integration

The UI components communicate with the backend API through standardized endpoints:

### Document Endpoints
- `POST /documents/upload` - Upload and process documents
- `GET /documents/{id}/status` - Check processing status
- `GET /documents/{id}/tables` - Get extracted tables

### Dashboard Endpoints
- `POST /dashboards/generate` - Generate dashboard
- `GET /dashboards/{id}` - Get dashboard data

### Chat Endpoints
- `POST /chat/ask` - Send natural language questions
- `GET /chat/history` - Get conversation history

### Export Endpoints
- `POST /export/table/{id}` - Export table data
- `POST /export/dashboard/{id}` - Export dashboard

## Testing

### Running Tests
```bash
# Run all UI tests
pytest tests/test_ui_*.py -v

# Run specific component tests
pytest tests/test_ui_document_upload.py -v
pytest tests/test_ui_table_validation.py -v
pytest tests/test_ui_chat_interface.py -v

# Run with coverage
pytest tests/test_ui_*.py --cov=src/ui --cov-report=html
```

### Test Structure
- **Unit Tests**: Test individual component methods
- **Integration Tests**: Test complete workflows
- **Mock Tests**: Test API interactions with mocked responses

### Test Coverage
The tests cover:
- File upload and validation
- Table editing and correction
- Chat message handling
- Visualization rendering
- Error handling and edge cases

## Customization

### Theming
Modify the Streamlit theme in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"        # Custom primary color
backgroundColor = "#F8F9FA"     # Custom background
secondaryBackgroundColor = "#E9ECEF"
textColor = "#212529"
```

### Custom Components
Add new UI components:
```python
# src/ui/components/custom_component.py
import streamlit as st

class CustomComponent:
    def render(self):
        st.title("Custom Component")
        # Your custom UI logic here
```

### Styling
Add custom CSS:
```python
# In your component
st.markdown("""
<style>
.custom-class {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)
```

## Performance Optimization

### Caching
Use Streamlit caching for expensive operations:
```python
@st.cache_data
def load_large_dataset():
    # Expensive data loading
    return data

@st.cache_resource
def initialize_model():
    # Model initialization
    return model
```

### Session State Management
Optimize session state usage:
```python
# Initialize once
if 'data' not in st.session_state:
    st.session_state.data = load_data()

# Use efficiently
data = st.session_state.data
```

### Memory Management
- Limit file upload sizes
- Use data pagination for large tables
- Clear unused session state variables

## Security Considerations

### File Upload Security
- Validate file types and sizes
- Scan uploaded files for malware
- Store files in secure locations
- Implement access controls

### Authentication
- Implement user authentication
- Use secure session management
- Validate API tokens
- Implement rate limiting

### Data Privacy
- Encrypt sensitive data
- Implement data retention policies
- Provide data deletion capabilities
- Comply with privacy regulations

## Troubleshooting

### Common Issues

#### 1. Application Won't Start
```bash
# Check dependencies
python -c "import streamlit, pandas, plotly"

# Check port availability
netstat -an | grep 8501

# Run with debug
python run_ui.py --debug
```

#### 2. File Upload Fails
- Check file size limits (max 50MB)
- Verify supported file formats
- Check API connectivity
- Review server logs

#### 3. API Connection Issues
- Verify API_BASE_URL configuration
- Check API server status
- Validate authentication tokens
- Review network connectivity

#### 4. Performance Issues
- Enable caching for data operations
- Optimize large file handling
- Use pagination for large datasets
- Monitor memory usage

### Debug Mode
Enable debug logging:
```bash
python run_ui.py --debug
```

### Logs
Check application logs:
- Streamlit logs: `~/.streamlit/logs/`
- Application logs: `ocr_analytics.log`
- Browser console for client-side issues

## Contributing

### Development Setup
1. Clone the repository
2. Install development dependencies
3. Set up pre-commit hooks
4. Run tests before committing

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions
- Write comprehensive tests

### Pull Request Process
1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review existing issues and discussions