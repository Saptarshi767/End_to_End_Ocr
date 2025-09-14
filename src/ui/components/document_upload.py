"""
Document Upload Interface Component

Implements drag-and-drop file upload with progress indicators,
document preview, metadata display, and batch upload capabilities.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import asyncio
import requests
import json
from pathlib import Path
import mimetypes

try:
    from src.core.logging_system import get_logger
    from src.api.models import ProcessingOptionsRequest
except ImportError:
    # Mock implementations for standalone usage
    class MockLogger:
        def info(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
    
    def get_logger():
        return MockLogger()
    
    # Mock ProcessingOptionsRequest
    from pydantic import BaseModel
    from typing import Optional
    
    class ProcessingOptionsRequest(BaseModel):
        ocr_engine: Optional[str] = "auto"
        preprocessing: Optional[bool] = True
        table_detection: Optional[bool] = True
        confidence_threshold: Optional[float] = 0.8


class DocumentUploadInterface:
    """Document upload interface with drag-and-drop and batch processing"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.logger = get_logger()
        self.supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
    def render(self):
        """Render the document upload interface"""
        
        st.title("📄 Document Upload")
        st.markdown("Upload documents containing tables for OCR processing and analysis")
        
        # Upload options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_upload_area()
            
        with col2:
            self._render_processing_options()
        
        # Batch upload queue
        if 'upload_queue' in st.session_state and st.session_state.upload_queue:
            self._render_upload_queue()
        
        # Upload history
        self._render_upload_history()
    
    def _render_upload_area(self):
        """Render the main upload area with drag-and-drop"""
        
        st.subheader("Upload Documents")
        
        # File uploader with drag-and-drop
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            help="Drag and drop files here or click to browse"
        )
        
        if uploaded_files:
            self._handle_file_upload(uploaded_files)
    
    def _render_processing_options(self):
        """Render processing options panel"""
        
        st.subheader("Processing Options")
        
        with st.expander("OCR Settings", expanded=False):
            ocr_engine = st.selectbox(
                "OCR Engine",
                options=["auto", "tesseract", "easyocr", "cloud_vision"],
                index=0,
                help="Choose OCR engine or let system auto-select"
            )
            
            preprocessing = st.checkbox(
                "Image Preprocessing",
                value=True,
                help="Apply image enhancement for better OCR accuracy"
            )
            
            table_detection = st.checkbox(
                "Table Detection",
                value=True,
                help="Automatically detect table regions"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Minimum confidence for OCR results"
            )
        
        # Store options in session state
        st.session_state.processing_options = ProcessingOptionsRequest(
            ocr_engine=ocr_engine,
            preprocessing=preprocessing,
            table_detection=table_detection,
            confidence_threshold=confidence_threshold
        )
    
    def _handle_file_upload(self, uploaded_files: List):
        """Handle uploaded files and add to processing queue"""
        
        if 'upload_queue' not in st.session_state:
            st.session_state.upload_queue = []
        
        for uploaded_file in uploaded_files:
            # Validate file
            if not self._validate_file(uploaded_file):
                continue
            
            # Create upload item
            upload_item = {
                'id': str(uuid.uuid4()),
                'file': uploaded_file,
                'filename': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type,
                'status': 'queued',
                'progress': 0,
                'uploaded_at': datetime.now(),
                'metadata': self._extract_file_metadata(uploaded_file)
            }
            
            st.session_state.upload_queue.append(upload_item)
        
        # Show preview for first file
        if uploaded_files:
            self._show_document_preview(uploaded_files[0])
        
        # Process queue button
        if st.button("🚀 Process All Files", type="primary"):
            self._process_upload_queue()
    
    def _validate_file(self, uploaded_file) -> bool:
        """Validate uploaded file"""
        
        # Check file extension
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext not in self.supported_formats:
            st.error(f"Unsupported file format: {file_ext}")
            return False
        
        # Check file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if uploaded_file.size > max_size:
            st.error(f"File too large: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f}MB)")
            return False
        
        return True
    
    def _extract_file_metadata(self, uploaded_file) -> Dict[str, Any]:
        """Extract metadata from uploaded file"""
        
        return {
            'filename': uploaded_file.name,
            'size_bytes': uploaded_file.size,
            'size_mb': round(uploaded_file.size / 1024 / 1024, 2),
            'mime_type': uploaded_file.type,
            'extension': Path(uploaded_file.name).suffix.lower()
        }
    
    def _show_document_preview(self, uploaded_file):
        """Show document preview and metadata"""
        
        st.subheader("📋 Document Preview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show image preview for image files
            if uploaded_file.type.startswith('image/'):
                st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
            else:
                st.info(f"Preview not available for {uploaded_file.type}")
        
        with col2:
            # Show metadata
            st.markdown("**File Information**")
            metadata = self._extract_file_metadata(uploaded_file)
            
            st.text(f"Name: {metadata['filename']}")
            st.text(f"Size: {metadata['size_mb']} MB")
            st.text(f"Type: {metadata['mime_type']}")
            st.text(f"Extension: {metadata['extension']}")
    
    def _render_upload_queue(self):
        """Render the upload queue with progress indicators"""
        
        st.subheader("📋 Upload Queue")
        
        queue = st.session_state.upload_queue
        
        for i, item in enumerate(queue):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.text(f"📄 {item['filename']}")
                
                with col2:
                    st.text(f"{item['metadata']['size_mb']} MB")
                
                with col3:
                    status_color = {
                        'queued': '🟡',
                        'processing': '🔄',
                        'completed': '✅',
                        'failed': '❌'
                    }
                    st.text(f"{status_color.get(item['status'], '⚪')} {item['status'].title()}")
                
                with col4:
                    if item['status'] == 'processing':
                        st.progress(item['progress'] / 100)
                    elif item['status'] == 'completed':
                        if st.button(f"View Results", key=f"view_{item['id']}"):
                            st.session_state.selected_document = item['document_id']
                            st.rerun()
                
                st.divider()
    
    def _process_upload_queue(self):
        """Process all files in the upload queue"""
        
        if 'upload_queue' not in st.session_state:
            return
        
        queue = st.session_state.upload_queue
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, item in enumerate(queue):
            if item['status'] != 'queued':
                continue
            
            # Update status
            item['status'] = 'processing'
            item['progress'] = 0
            
            status_text.text(f"Processing {item['filename']}...")
            
            try:
                # Upload file
                document_id = self._upload_file_to_api(item)
                item['document_id'] = document_id
                
                # Monitor processing
                self._monitor_processing(item, progress_bar, status_text)
                
                item['status'] = 'completed'
                item['progress'] = 100
                
            except Exception as e:
                self.logger.error(f"Upload failed for {item['filename']}: {str(e)}")
                item['status'] = 'failed'
                item['error'] = str(e)
                st.error(f"Failed to process {item['filename']}: {str(e)}")
            
            # Update overall progress
            progress_bar.progress((i + 1) / len(queue))
        
        status_text.text("✅ All files processed!")
        st.success("Upload queue completed!")
    
    def _upload_file_to_api(self, item: Dict[str, Any]) -> str:
        """Upload file to API and return document ID"""
        
        files = {'file': (item['filename'], item['file'].getvalue(), item['type'])}
        
        # Get processing options
        options = st.session_state.get('processing_options')
        data = {}
        if options:
            data['processing_options'] = options.json()
        
        response = requests.post(
            f"{self.api_base_url}/documents/upload",
            files=files,
            data=data,
            headers={'Authorization': f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.text}")
        
        result = response.json()
        return result['document_id']
    
    def _monitor_processing(self, item: Dict[str, Any], progress_bar, status_text):
        """Monitor document processing progress"""
        
        document_id = item['document_id']
        
        while True:
            response = requests.get(
                f"{self.api_base_url}/documents/{document_id}/status",
                headers={'Authorization': f"Bearer {st.session_state.get('auth_token', '')}"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Status check failed: {response.text}")
            
            status_data = response.json()
            item['progress'] = status_data['progress']
            
            status_text.text(f"Processing {item['filename']}... {item['progress']}%")
            
            if status_data['status'] in ['completed', 'failed']:
                break
            
            # Wait before next check
            import time
            time.sleep(2)
    
    def _render_upload_history(self):
        """Render upload history"""
        
        st.subheader("📚 Recent Uploads")
        
        # Get upload history from session state or API
        if 'upload_history' not in st.session_state:
            st.session_state.upload_history = []
        
        history = st.session_state.upload_history
        
        if not history:
            st.info("No recent uploads")
            return
        
        # Create DataFrame for display
        df_data = []
        for item in history[-10:]:  # Show last 10 uploads
            df_data.append({
                'Filename': item['filename'],
                'Size (MB)': item['metadata']['size_mb'],
                'Status': item['status'],
                'Uploaded': item['uploaded_at'].strftime('%Y-%m-%d %H:%M'),
                'Document ID': item.get('document_id', 'N/A')
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, width='stretch')


def render_document_upload():
    """Render document upload interface"""
    upload_interface = DocumentUploadInterface()
    upload_interface.render()


if __name__ == "__main__":
    render_document_upload()