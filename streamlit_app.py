"""
OCR Table Analytics - Full Implementation

Complete OCR table detection and analysis system.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List
import requests
from datetime import datetime
import uuid
import json
import io
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import sys

# Try to import OCR libraries directly (no complex modules needed)
try:
    import easyocr
    import pytesseract
    DIRECT_OCR_AVAILABLE = True
except ImportError:
    DIRECT_OCR_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="OCR Table Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'upload'
if 'selected_document' not in st.session_state:
    st.session_state.selected_document = None
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'upload_queue' not in st.session_state:
    st.session_state.upload_queue = []
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = {}
if 'extracted_tables' not in st.session_state:
    st.session_state.extracted_tables = {}

# OCR Processing Functions
@st.cache_data
def preprocess_image(image_array):
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def detect_tables_cv2(image_array):
    """Detect table regions using OpenCV"""
    # Preprocess image
    processed = preprocess_image(image_array)
    
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Extract horizontal and vertical lines
    horizontal_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine lines to find table structure
    table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    
    # Find contours (potential table regions)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    table_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter by size (tables should be reasonably large)
        if w > 100 and h > 50:
            table_regions.append((x, y, w, h))
    
    return table_regions, table_mask

def extract_text_easyocr(image_array, table_regions=None):
    """Extract text using EasyOCR"""
    if not DIRECT_OCR_AVAILABLE:
        return None
    
    try:
        reader = easyocr.Reader(['en'], verbose=False)
        results = reader.readtext(image_array)
        
        extracted_data = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Lower threshold for better detection
                # Convert EasyOCR bbox format to consistent format
                # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                if len(bbox) == 4 and len(bbox[0]) == 2:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    extracted_data.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': (x_min, y_min, x_max - x_min, y_max - y_min)  # (x, y, width, height)
                    })
                else:
                    # Fallback for unexpected bbox format
                    extracted_data.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        return extracted_data
    except Exception as e:
        st.error(f"EasyOCR error: {e}")
        return None

def extract_text_tesseract(image_array):
    """Extract text using Tesseract"""
    if not DIRECT_OCR_AVAILABLE:
        return None
    
    try:
        # Check if tesseract is available
        import subprocess
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            st.warning("⚠️ Tesseract not found in PATH. Using EasyOCR instead.")
            return None
    except (subprocess.SubprocessError, FileNotFoundError):
        st.warning("⚠️ Tesseract not installed. Using EasyOCR instead.")
        return None
    
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Get detailed data from Tesseract
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        extracted_data = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Filter low confidence
                text = data['text'][i].strip()
                if text:
                    extracted_data.append({
                        'text': text,
                        'confidence': int(data['conf'][i]) / 100.0,
                        'bbox': (data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i])
                    })
        
        return extracted_data
    except Exception as e:
        st.warning(f"Tesseract error: {e}. Falling back to EasyOCR.")
        return None

def organize_text_into_table(extracted_data, image_shape):
    """Organize extracted text into table structure"""
    if not extracted_data:
        return pd.DataFrame()
    
    try:
        # Sort by y-coordinate (top to bottom) then x-coordinate (left to right)
        sorted_data = sorted(extracted_data, key=lambda x: (get_y_coord(x['bbox']), get_x_coord(x['bbox'])))
        
        # Group text by rows (similar y-coordinates)
        rows = []
        current_row = []
        current_y = None
        y_threshold = 20  # Pixels tolerance for same row
        
        for item in sorted_data:
            y_coord = get_y_coord(item['bbox'])
            
            if current_y is None or abs(y_coord - current_y) <= y_threshold:
                current_row.append(item)
                current_y = y_coord if current_y is None else current_y
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda x: get_x_coord(x['bbox'])))
                current_row = [item]
                current_y = y_coord
        
        if current_row:
            rows.append(sorted(current_row, key=lambda x: get_x_coord(x['bbox'])))
        
        # Convert to DataFrame
        if not rows:
            return pd.DataFrame()
        
        # Find maximum number of columns
        max_cols = max(len(row) for row in rows)
        
        # Create table data
        table_data = []
        for row in rows:
            row_data = [item['text'] for item in row]
            # Pad with empty strings if needed
            while len(row_data) < max_cols:
                row_data.append('')
            table_data.append(row_data)
        
        # Create DataFrame with generic column names
        if table_data:
            columns = [f'Column_{i+1}' for i in range(max_cols)]
            df = pd.DataFrame(table_data, columns=columns)
            return df
        
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error organizing text into table: {e}")
        return pd.DataFrame()

def get_x_coord(bbox):
    """Extract x coordinate from bbox (handles different formats)"""
    try:
        if isinstance(bbox, (list, tuple)):
            if len(bbox) >= 4:  # (x, y, width, height) format
                return bbox[0]
            elif len(bbox) >= 2:
                if isinstance(bbox[0], (int, float)):
                    return bbox[0]
                elif isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
                    return bbox[0][0]  # [[x1,y1], [x2,y1], ...] format
        return 0
    except:
        return 0

def get_y_coord(bbox):
    """Extract y coordinate from bbox (handles different formats)"""
    try:
        if isinstance(bbox, (list, tuple)):
            if len(bbox) >= 4:  # (x, y, width, height) format
                return bbox[1]
            elif len(bbox) >= 2:
                if isinstance(bbox[1], (int, float)):
                    return bbox[1]
                elif isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
                    return bbox[0][1]  # [[x1,y1], [x2,y1], ...] format
        return 0
    except:
        return 0

def process_uploaded_file(uploaded_file, ocr_engine="auto", preprocessing=True):
    """Process uploaded file and extract tables"""
    try:
        # Read image
        if uploaded_file.type.startswith('image/'):
            image = Image.open(uploaded_file)
            image_array = np.array(image)
        else:
            st.error("Only image files are supported in this demo")
            return None, None
        
        # Detect table regions
        table_regions, table_mask = detect_tables_cv2(image_array)
        
        # Extract text based on selected engine
        extracted_data = None
        
        if ocr_engine == "easyocr" and DIRECT_OCR_AVAILABLE:
            extracted_data = extract_text_easyocr(image_array, table_regions)
        elif ocr_engine == "tesseract" and DIRECT_OCR_AVAILABLE:
            extracted_data = extract_text_tesseract(image_array)
            # If tesseract fails, fallback to EasyOCR
            if not extracted_data:
                st.info("🔄 Tesseract failed, trying EasyOCR...")
                extracted_data = extract_text_easyocr(image_array, table_regions)
        else:  # auto mode - try EasyOCR first, then Tesseract
            extracted_data = extract_text_easyocr(image_array, table_regions)
            if not extracted_data or len(extracted_data) < 3:  # If poor results, try Tesseract
                tesseract_data = extract_text_tesseract(image_array)
                if tesseract_data and len(tesseract_data) > len(extracted_data or []):
                    extracted_data = tesseract_data
        
        if not extracted_data:
            st.warning("No text detected in the image")
            st.info("💡 Tips for better results:")
            st.markdown("""
            - Ensure the image has clear, readable text
            - Try a higher resolution image
            - Make sure there's good contrast between text and background
            - Check that the table structure is clearly visible
            """)
            return None, None
        
        # Organize into table
        df = organize_text_into_table(extracted_data, image_array.shape)
        
        return df, {
            'table_regions': table_regions,
            'extracted_data': extracted_data,
            'confidence_avg': np.mean([item['confidence'] for item in extracted_data])
        }
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

def render_header():
    """Render application header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.image("https://via.placeholder.com/100x50/0066CC/FFFFFF?text=OCR", width=100)
    
    with col2:
        st.title("📊 OCR Table Analytics")
        st.markdown("*Transform documents into interactive data insights*")
    
    with col3:
        if st.session_state.user_authenticated:
            if st.button("🚪 Logout"):
                st.session_state.user_authenticated = False
                st.rerun()

def render_login():
    """Render login interface"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login", type="primary"):
                if username and password:
                    st.session_state.user_authenticated = True
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Please enter username and password")
        
        st.markdown("---")
        if st.button("🎮 Demo Mode", help="Try the app without authentication"):
            st.session_state.user_authenticated = True
            st.rerun()

def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.title("🧭 Navigation")
    
    # Page navigation
    pages = {
        'upload': '📤 Upload Documents',
        'validation': '🔍 Validate Tables',
        'chat': '💬 Chat Analysis',
        'dashboard': '📊 Dashboard'
    }
    
    for page_key, page_name in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}"):
            st.session_state.current_page = page_key
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Real document selector
    st.sidebar.subheader("📄 Processed Documents")
    
    if st.session_state.processed_documents:
        documents = list(st.session_state.processed_documents.keys())
        selected_doc = st.sidebar.selectbox("Select Document", documents)
        st.session_state.selected_document = selected_doc
        
        # Show document info
        if selected_doc:
            doc_data = st.session_state.processed_documents[selected_doc]
            df = doc_data['dataframe']
            st.sidebar.text(f"Rows: {len(df)}")
            st.sidebar.text(f"Columns: {len(df.columns)}")
            
            confidence = doc_data.get('metadata', {}).get('confidence_avg', 0)
            st.sidebar.text(f"Confidence: {confidence:.1%}")
    else:
        st.sidebar.info("No documents processed yet")
        st.sidebar.text("Upload and process documents first")

def render_upload_page():
    """Render document upload page"""
    st.title("📄 Document Upload & OCR Processing")
    st.markdown("Upload images containing tables for OCR processing and analysis")
    
    # Check OCR availability
    if not DIRECT_OCR_AVAILABLE:
        st.error("⚠️ OCR libraries not installed")
        st.markdown("""
        **To install OCR libraries:**
        ```bash
        pip install easyocr pytesseract
        ```
        """)
        st.info("💡 You can still explore the interface, but OCR processing won't work until libraries are installed.")
        return
    
    # Check Tesseract installation
    tesseract_available = False
    try:
        import subprocess
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        tesseract_available = (result.returncode == 0)
    except (subprocess.SubprocessError, FileNotFoundError):
        tesseract_available = False
    
    if not tesseract_available:
        st.info("ℹ️ Tesseract not found - using EasyOCR only (this is fine for most use cases)")
        with st.expander("📋 Optional: Install Tesseract for additional OCR options"):
            st.markdown("""
            - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
            - **macOS**: `brew install tesseract`
            - **Linux**: `sudo apt-get install tesseract-ocr`
            """)
    else:
        st.success("✅ Both EasyOCR and Tesseract are available")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose image files containing tables",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            help="Drag and drop image files here or click to browse"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            # Processing options
            with col2:
                st.subheader("Processing Options")
                
                # Check what engines are available
                available_engines = ["easyocr"]  # EasyOCR is always available if DIRECT_OCR_AVAILABLE
                try:
                    import subprocess
                    result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        available_engines.extend(["tesseract", "auto"])
                except:
                    pass
                
                if "auto" in available_engines:
                    engine_options = ["auto", "easyocr", "tesseract"]
                    default_engine = "auto"
                else:
                    engine_options = ["easyocr"]
                    default_engine = "easyocr"
                
                ocr_engine = st.selectbox("OCR Engine", engine_options, index=0)
                preprocessing = st.checkbox("Image Preprocessing", value=True)
                confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
                
                if len(available_engines) == 1:
                    st.info("ℹ️ Using EasyOCR (Tesseract not installed)")
                else:
                    st.success("✅ Multiple OCR engines available")
            
            # Process each file
            for i, file in enumerate(uploaded_files):
                with st.expander(f"📄 {file.name}", expanded=True):
                    col_a, col_b = st.columns([1, 2])
                    
                    with col_a:
                        st.text(f"Size: {file.size / 1024 / 1024:.2f} MB")
                        st.text(f"Type: {file.type}")
                        
                        if file.type.startswith('image/'):
                            st.image(file, width=200, caption="Original Image")
                    
                    with col_b:
                        if st.button(f"🚀 Process {file.name}", key=f"process_{i}"):
                            if DIRECT_OCR_AVAILABLE:
                                with st.spinner(f"Processing {file.name}..."):
                                    # Process the file
                                    df, metadata = process_uploaded_file(file, ocr_engine, preprocessing)
                                    
                                    if df is not None and not df.empty:
                                        # Store results
                                        st.session_state.processed_documents[file.name] = {
                                            'dataframe': df,
                                            'metadata': metadata,
                                            'processed_at': datetime.now()
                                        }
                                        st.session_state.extracted_tables[file.name] = df
                                        
                                        st.success("✅ Table extracted successfully!")
                                        
                                        # Show processing details
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            if metadata and 'confidence_avg' in metadata:
                                                confidence_score = metadata['confidence_avg']
                                                st.metric("Average Confidence", f"{confidence_score:.2%}")
                                        with col2:
                                            st.metric("Rows Extracted", len(df))
                                        with col3:
                                            st.metric("Columns Detected", len(df.columns))
                                        
                                        # Show extracted table
                                        st.subheader("📊 Extracted Table")
                                        st.dataframe(df, width='stretch')
                                        
                                        # Show debug info
                                        if metadata and 'extracted_data' in metadata:
                                            with st.expander("🔍 Debug Information"):
                                                st.write(f"**Text elements found:** {len(metadata['extracted_data'])}")
                                                for i, item in enumerate(metadata['extracted_data'][:5]):  # Show first 5
                                                    st.text(f"{i+1}. '{item['text']}' (confidence: {item['confidence']:.2%})")
                                                if len(metadata['extracted_data']) > 5:
                                                    st.text(f"... and {len(metadata['extracted_data']) - 5} more")
                                        
                                        # Download option
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            "📥 Download CSV",
                                            csv,
                                            f"{file.name}_extracted.csv",
                                            "text/csv",
                                            key=f"download_{i}"
                                        )
                                        
                                        # Set as selected document for other pages
                                        st.session_state.selected_document = file.name
                                        
                                    else:
                                        st.error("❌ No table data could be extracted from this image")
                                        st.info("💡 Try adjusting the confidence threshold or preprocessing options")
                            else:
                                st.error("❌ OCR libraries not available")
                                st.markdown("""
                                **To enable OCR processing:**
                                ```bash
                                pip install easyocr pytesseract
                                ```
                                Then restart the application.
                                """)
    
    with col2:
        if not uploaded_files:
            st.subheader("Processing Options")
            st.selectbox("OCR Engine", ["auto", "easyocr", "tesseract"], disabled=True)
            st.checkbox("Image Preprocessing", value=True, disabled=True)
            st.slider("Confidence Threshold", 0.1, 1.0, 0.5, disabled=True)
            
            st.info("💡 Tips for better results:")
            st.markdown("""
            - Use high-resolution images
            - Ensure good contrast
            - Tables should be clearly visible
            - Avoid skewed or rotated images
            """)
            
            # Demo mode when OCR is not available
            if not DIRECT_OCR_AVAILABLE:
                st.markdown("---")
                st.subheader("🎮 Demo Mode")
                st.info("Since OCR libraries aren't installed, you can try the demo with sample data")
                
                if st.button("📊 Load Sample Table Data"):
                    # Create sample data
                    sample_df = pd.DataFrame({
                        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
                        'Price': ['$999', '$25', '$75', '$299', '$150'],
                        'Stock': ['15', '50', '30', '8', '25'],
                        'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Audio']
                    })
                    
                    # Store as processed document
                    st.session_state.processed_documents['sample_data.png'] = {
                        'dataframe': sample_df,
                        'metadata': {
                            'confidence_avg': 0.95,
                            'table_regions': [(50, 50, 400, 200)],
                            'extracted_data': [
                                {'text': 'Product', 'confidence': 0.98},
                                {'text': 'Price', 'confidence': 0.96},
                                {'text': 'Stock', 'confidence': 0.94}
                            ]
                        },
                        'processed_at': datetime.now()
                    }
                    st.session_state.extracted_tables['sample_data.png'] = sample_df
                    st.session_state.selected_document = 'sample_data.png'
                    
                    st.success("✅ Sample data loaded! Check other pages to explore features.")
                    st.dataframe(sample_df, width='stretch')
    
    # Show processed documents summary
    if st.session_state.processed_documents:
        st.markdown("---")
        st.subheader("📋 Processed Documents")
        
        for doc_name, doc_data in st.session_state.processed_documents.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(f"📄 {doc_name}")
            with col2:
                st.text(f"Rows: {len(doc_data['dataframe'])}")
            with col3:
                if st.button(f"View", key=f"view_{doc_name}"):
                    st.session_state.selected_document = doc_name
                    st.session_state.current_page = 'validation'
                    st.rerun()

def render_validation_page():
    """Render table validation page"""
    if not st.session_state.selected_document:
        st.info("👆 Please upload and process a document first, or select one from the sidebar")
        return
    
    doc_name = st.session_state.selected_document
    
    if doc_name not in st.session_state.processed_documents:
        st.error(f"Document '{doc_name}' not found in processed documents")
        return
    
    doc_data = st.session_state.processed_documents[doc_name]
    df = doc_data['dataframe']
    metadata = doc_data.get('metadata', {})
    
    st.title("🔍 Table Validation & Correction")
    st.markdown(f"Reviewing tables from: **{doc_name}**")
    
    # Real confidence indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence = metadata.get('confidence_avg', 0)
        st.metric("Confidence Score", f"{confidence:.1%}")
    with col2:
        tables_found = len(metadata.get('table_regions', []))
        st.metric("Table Regions", str(tables_found))
    with col3:
        data_quality = "Good" if confidence > 0.7 else "Fair" if confidence > 0.5 else "Poor"
        st.metric("Data Quality", data_quality)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("� EProcessing Info")
        st.info(f"Processed: {doc_data['processed_at'].strftime('%Y-%m-%d %H:%M')}")
        
        if metadata.get('table_regions'):
            st.write(f"**Table regions detected:** {len(metadata['table_regions'])}")
            for i, region in enumerate(metadata['table_regions']):
                x, y, w, h = region
                st.text(f"Region {i+1}: {w}×{h} at ({x},{y})")
        
        if metadata.get('extracted_data'):
            st.write(f"**Text elements:** {len(metadata['extracted_data'])}")
            avg_conf = np.mean([item['confidence'] for item in metadata['extracted_data']])
            st.write(f"**Average confidence:** {avg_conf:.1%}")
    
    with col2:
        st.subheader("� Eaxtracted Data")
        
        if df.empty:
            st.warning("No table data extracted")
            return
        
        # Allow editing of the extracted data
        edited_df = st.data_editor(
            df,
            width='stretch',
            num_rows="dynamic",
            key=f"editor_{doc_name}"
        )
        
        # Check for changes
        if not df.equals(edited_df):
            st.info("📝 Changes detected! Don't forget to save.")
            # Update the stored dataframe
            st.session_state.processed_documents[doc_name]['dataframe'] = edited_df
            st.session_state.extracted_tables[doc_name] = edited_df
    
    # Validation controls
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Re-process Document"):
            # Clear the processed document to allow re-processing
            if doc_name in st.session_state.processed_documents:
                del st.session_state.processed_documents[doc_name]
            if doc_name in st.session_state.extracted_tables:
                del st.session_state.extracted_tables[doc_name]
            st.session_state.current_page = 'upload'
            st.info("Document cleared. Please re-upload and process.")
            st.rerun()
    
    with col2:
        if st.button("🤖 Clean Data"):
            # Simple data cleaning
            cleaned_df = edited_df.copy()
            
            # Remove empty rows and columns
            cleaned_df = cleaned_df.dropna(how='all').dropna(axis=1, how='all')
            
            # Strip whitespace
            for col in cleaned_df.select_dtypes(include=['object']).columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            
            # Update the data
            st.session_state.processed_documents[doc_name]['dataframe'] = cleaned_df
            st.session_state.extracted_tables[doc_name] = cleaned_df
            st.success("✅ Data cleaned!")
            st.rerun()
    
    with col3:
        if st.button("💾 Save Changes", type="primary"):
            # Save changes (in a real app, this would save to database)
            st.session_state.processed_documents[doc_name]['last_saved'] = datetime.now()
            st.success("✅ Changes saved!")
    
    with col4:
        # Export options
        csv = edited_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"{doc_name}_validated.csv",
            "text/csv"
        )
    
    # Data preview and statistics
    if not edited_df.empty:
        st.markdown("---")
        st.subheader("📈 Data Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape:**", f"{edited_df.shape[0]} rows × {edited_df.shape[1]} columns")
            st.write("**Memory usage:**", f"{edited_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        with col2:
            # Show data types
            st.write("**Column types:**")
            for col, dtype in edited_df.dtypes.items():
                st.text(f"{col}: {dtype}")
        
        # Show sample of the data
        st.subheader("📋 Data Preview")
        st.dataframe(edited_df.head(10), width='stretch')

def render_chat_page():
    """Render chat analysis page"""
    if not st.session_state.selected_document:
        st.info("👆 Please select a document from the sidebar to start chatting")
        return
    
    doc_name = st.session_state.selected_document
    if doc_name not in st.session_state.extracted_tables:
        st.error("No data available for this document")
        return
    
    df = st.session_state.extracted_tables[doc_name]
    
    st.title("💬 Data Analysis Chat")
    st.markdown(f"Ask questions about: **{doc_name}**")
    
    # Show data summary
    with st.expander("📊 Data Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
            st.metric("Text Columns", len(df.columns) - len(numeric_cols))
        
        st.dataframe(df.head(), width='stretch')
    
    # Quick analysis suggestions based on actual data
    st.markdown("**💡 Quick Analysis**")
    
    suggestions = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        suggestions.extend([
            f"Show statistics for {numeric_cols[0]}",
            "Create a summary of all numeric columns",
            "Show the distribution of values"
        ])
    
    if len(df.columns) > 1:
        suggestions.append("Compare columns in the data")
    
    suggestions.extend([
        "Show the first and last rows",
        "Find any missing or empty values",
        "Export insights as text"
    ])
    
    # Display suggestion buttons
    cols = st.columns(min(3, len(suggestions)))
    for i, suggestion in enumerate(suggestions[:3]):
        with cols[i % 3]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                # Add to chat history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': suggestion,
                    'timestamp': datetime.now()
                })
                
                # Generate real response based on data
                response = generate_data_response(df, suggestion)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
                st.rerun()
    
    # Chat history
    st.markdown("**💬 Conversation**")
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your data...")
    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Generate response based on actual data
        response = generate_data_response(df, user_input)
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        })
        st.rerun()

def generate_data_response(df, question):
    """Generate responses based on actual data"""
    question_lower = question.lower()
    
    try:
        if "statistic" in question_lower or "summary" in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats = df[numeric_cols].describe()
                return f"Here are the statistics for numeric columns:\n\n{stats.to_string()}"
            else:
                return "No numeric columns found for statistical analysis."
        
        elif "first" in question_lower and "last" in question_lower:
            return f"**First row:**\n{df.iloc[0].to_string()}\n\n**Last row:**\n{df.iloc[-1].to_string()}"
        
        elif "missing" in question_lower or "empty" in question_lower:
            missing_info = df.isnull().sum()
            if missing_info.sum() > 0:
                return f"Missing values per column:\n{missing_info[missing_info > 0].to_string()}"
            else:
                return "No missing values found in the data!"
        
        elif "distribution" in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                values = df[col].value_counts().head(10)
                return f"Distribution of {col}:\n{values.to_string()}"
            else:
                return "No numeric columns available for distribution analysis."
        
        elif "compare" in question_lower:
            if len(df.columns) >= 2:
                return f"Your data has {len(df.columns)} columns: {', '.join(df.columns)}. The data types are: {df.dtypes.to_string()}"
            else:
                return "Need at least 2 columns to compare data."
        
        else:
            # General response
            return f"""Based on your question about "{question}", here's what I can tell you about your data:

- **Shape:** {df.shape[0]} rows × {df.shape[1]} columns
- **Columns:** {', '.join(df.columns)}
- **Data types:** {len(df.select_dtypes(include=[np.number]).columns)} numeric, {len(df.select_dtypes(include=['object']).columns)} text columns

Ask me specific questions like:
- "Show statistics for [column name]"
- "What are the missing values?"
- "Show the distribution of values"
"""
    
    except Exception as e:
        return f"I encountered an error analyzing your data: {str(e)}. Please try a different question."

def render_dashboard_page():
    """Render dashboard page"""
    if not st.session_state.selected_document:
        st.info("👆 Please select a document from the sidebar to view dashboard")
        return
    
    doc_name = st.session_state.selected_document
    if doc_name not in st.session_state.extracted_tables:
        st.error("No data available for this document")
        return
    
    df = st.session_state.extracted_tables[doc_name]
    
    st.title("📊 Interactive Dashboard")
    st.markdown(f"Auto-generated dashboard for: **{doc_name}**")
    
    if df.empty:
        st.warning("No data to display in dashboard")
        return
    
    # Real KPIs based on actual data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            avg_val = df[numeric_cols[0]].mean()
            st.metric("Average Value", f"{avg_val:.2f}")
        else:
            st.metric("Numeric Columns", len(numeric_cols))
    
    with col3:
        if len(numeric_cols) > 0:
            max_val = df[numeric_cols[0]].max()
            st.metric("Max Value", f"{max_val:.2f}")
        else:
            st.metric("Text Columns", len(df.columns) - len(numeric_cols))
    
    with col4:
        # Calculate data quality based on missing values
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_pct = 100 - missing_pct
        st.metric("Data Quality", f"{quality_pct:.1f}%")
    
    # Charts based on actual data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Distribution")
        
        # Try to create meaningful visualizations
        if len(numeric_cols) > 0:
            # Histogram for first numeric column
            col_name = numeric_cols[0]
            fig = px.histogram(df, x=col_name, title=f"Distribution of {col_name}")
            st.plotly_chart(fig, width='stretch')
        else:
            # Value counts for first text column
            if len(df.columns) > 0:
                col_name = df.columns[0]
                value_counts = df[col_name].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"Top Values in {col_name}")
                fig.update_xaxes(title=col_name)
                fig.update_yaxes(title="Count")
                st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📈 Data Analysis")
        
        if len(numeric_cols) >= 2:
            # Scatter plot for two numeric columns
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                           title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
            st.plotly_chart(fig, width='stretch')
        elif len(numeric_cols) == 1:
            # Box plot for single numeric column
            fig = px.box(df, y=numeric_cols[0], title=f"Box Plot of {numeric_cols[0]}")
            st.plotly_chart(fig, width='stretch')
        else:
            # Show column information
            st.info("No numeric columns available for advanced visualization")
            st.write("**Column Information:**")
            for col in df.columns:
                unique_vals = df[col].nunique()
                st.text(f"{col}: {unique_vals} unique values")
    
    # Data table section
    st.markdown("---")
    st.subheader("📋 Raw Data")
    
    # Add filters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("**Filters:**")
        show_all = st.checkbox("Show all rows", value=False)
        
        if not show_all:
            max_rows = st.slider("Max rows to display", 5, min(50, len(df)), 10)
        else:
            max_rows = len(df)
    
    with col2:
        # Search functionality
        search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
        
        if search_term:
            # Filter dataframe based on search
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            filtered_df = df[mask]
            st.write(f"Found {len(filtered_df)} rows matching '{search_term}'")
        else:
            filtered_df = df
    
    # Display the data
    display_df = filtered_df.head(max_rows) if not show_all else filtered_df
    st.dataframe(display_df, width='stretch')
    
    # Export options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Full CSV", csv, f"{doc_name}_dashboard.csv", "text/csv")
    
    with col2:
        if search_term and not filtered_df.empty:
            filtered_csv = filtered_df.to_csv(index=False)
            st.download_button("📥 Download Filtered CSV", filtered_csv, f"{doc_name}_filtered.csv", "text/csv")
    
    with col3:
        # Summary statistics
        if len(numeric_cols) > 0:
            summary = df[numeric_cols].describe().to_csv()
            st.download_button("📊 Download Statistics", summary, f"{doc_name}_stats.csv", "text/csv")

def main():
    """Main application"""
    render_header()
    
    if not st.session_state.user_authenticated:
        render_login()
        return
    
    render_sidebar()
    
    # Render main content
    current_page = st.session_state.current_page
    
    if current_page == 'upload':
        render_upload_page()
    elif current_page == 'validation':
        render_validation_page()
    elif current_page == 'chat':
        render_chat_page()
    elif current_page == 'dashboard':
        render_dashboard_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "OCR Table Analytics v1.0.0 | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()