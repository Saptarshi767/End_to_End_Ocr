"""
Tests for Document Upload UI Component

Tests drag-and-drop file upload, progress indicators,
document preview, metadata display, and batch upload capabilities.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import io
from PIL import Image

from src.ui.components.document_upload import DocumentUploadInterface


class TestDocumentUploadInterface:
    """Test cases for DocumentUploadInterface"""
    
    @pytest.fixture
    def upload_interface(self):
        """Create DocumentUploadInterface instance"""
        return DocumentUploadInterface(api_base_url="http://test-api:8000")
    
    @pytest.fixture
    def mock_uploaded_file(self):
        """Create mock uploaded file"""
        mock_file = Mock()
        mock_file.name = "test_document.pdf"
        mock_file.size = 1024 * 1024  # 1MB
        mock_file.type = "application/pdf"
        mock_file.getvalue.return_value = b"mock file content"
        return mock_file
    
    @pytest.fixture
    def mock_image_file(self):
        """Create mock image file"""
        mock_file = Mock()
        mock_file.name = "test_image.png"
        mock_file.size = 512 * 1024  # 512KB
        mock_file.type = "image/png"
        
        # Create mock image content
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        mock_file.getvalue.return_value = img_bytes.getvalue()
        
        return mock_file
    
    def test_initialization(self, upload_interface):
        """Test interface initialization"""
        
        assert upload_interface.api_base_url == "http://test-api:8000"
        assert upload_interface.supported_formats == ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    
    def test_validate_file_valid(self, upload_interface, mock_uploaded_file):
        """Test file validation with valid file"""
        
        result = upload_interface._validate_file(mock_uploaded_file)
        assert result is True
    
    def test_validate_file_invalid_format(self, upload_interface):
        """Test file validation with invalid format"""
        
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.size = 1024
        
        with patch('streamlit.error') as mock_error:
            result = upload_interface._validate_file(mock_file)
            assert result is False
            mock_error.assert_called_once()
    
    def test_validate_file_too_large(self, upload_interface):
        """Test file validation with oversized file"""
        
        mock_file = Mock()
        mock_file.name = "large_file.pdf"
        mock_file.size = 60 * 1024 * 1024  # 60MB (over 50MB limit)
        
        with patch('streamlit.error') as mock_error:
            result = upload_interface._validate_file(mock_file)
            assert result is False
            mock_error.assert_called_once()
    
    def test_extract_file_metadata(self, upload_interface, mock_uploaded_file):
        """Test file metadata extraction"""
        
        metadata = upload_interface._extract_file_metadata(mock_uploaded_file)
        
        assert metadata['filename'] == "test_document.pdf"
        assert metadata['size_bytes'] == 1024 * 1024
        assert metadata['size_mb'] == 1.0
        assert metadata['mime_type'] == "application/pdf"
        assert metadata['extension'] == ".pdf"
    
    @patch('streamlit.session_state', {})
    def test_handle_file_upload_single_file(self, upload_interface, mock_uploaded_file):
        """Test handling single file upload"""
        
        with patch('streamlit.button', return_value=False):
            with patch.object(upload_interface, '_show_document_preview') as mock_preview:
                upload_interface._handle_file_upload([mock_uploaded_file])
                
                # Check that file was added to queue
                assert 'upload_queue' in st.session_state
                assert len(st.session_state.upload_queue) == 1
                
                queue_item = st.session_state.upload_queue[0]
                assert queue_item['filename'] == "test_document.pdf"
                assert queue_item['status'] == 'queued'
                assert queue_item['progress'] == 0
                
                # Check preview was shown
                mock_preview.assert_called_once_with(mock_uploaded_file)
    
    @patch('streamlit.session_state', {})
    def test_handle_file_upload_multiple_files(self, upload_interface, mock_uploaded_file, mock_image_file):
        """Test handling multiple file uploads"""
        
        files = [mock_uploaded_file, mock_image_file]
        
        with patch('streamlit.button', return_value=False):
            with patch.object(upload_interface, '_show_document_preview'):
                upload_interface._handle_file_upload(files)
                
                # Check that both files were added to queue
                assert len(st.session_state.upload_queue) == 2
                
                filenames = [item['filename'] for item in st.session_state.upload_queue]
                assert "test_document.pdf" in filenames
                assert "test_image.png" in filenames
    
    @patch('streamlit.session_state', {})
    def test_handle_file_upload_with_processing(self, upload_interface, mock_uploaded_file):
        """Test file upload with processing trigger"""
        
        with patch('streamlit.button', return_value=True):
            with patch.object(upload_interface, '_process_upload_queue') as mock_process:
                with patch.object(upload_interface, '_show_document_preview'):
                    upload_interface._handle_file_upload([mock_uploaded_file])
                    
                    # Check that processing was triggered
                    mock_process.assert_called_once()
    
    @patch('requests.post')
    def test_upload_file_to_api_success(self, mock_post, upload_interface):
        """Test successful file upload to API"""
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'document_id': 'test_doc_123'}
        mock_post.return_value = mock_response
        
        # Create upload item
        upload_item = {
            'filename': 'test.pdf',
            'file': Mock(),
            'type': 'application/pdf'
        }
        upload_item['file'].getvalue.return_value = b"test content"
        
        with patch('streamlit.session_state', {'auth_token': 'test_token'}):
            document_id = upload_interface._upload_file_to_api(upload_item)
            
            assert document_id == 'test_doc_123'
            mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_upload_file_to_api_failure(self, mock_post, upload_interface):
        """Test failed file upload to API"""
        
        # Mock failed API response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Upload failed"
        mock_post.return_value = mock_response
        
        upload_item = {
            'filename': 'test.pdf',
            'file': Mock(),
            'type': 'application/pdf'
        }
        upload_item['file'].getvalue.return_value = b"test content"
        
        with patch('streamlit.session_state', {'auth_token': 'test_token'}):
            with pytest.raises(Exception, match="Upload failed"):
                upload_interface._upload_file_to_api(upload_item)
    
    @patch('requests.get')
    @patch('time.sleep')
    def test_monitor_processing_success(self, mock_sleep, mock_get, upload_interface):
        """Test successful processing monitoring"""
        
        # Mock API responses for status checks
        responses = [
            {'status': 'processing', 'progress': 25},
            {'status': 'processing', 'progress': 50},
            {'status': 'processing', 'progress': 75},
            {'status': 'completed', 'progress': 100}
        ]
        
        mock_responses = []
        for resp_data in responses:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = resp_data
            mock_responses.append(mock_resp)
        
        mock_get.side_effect = mock_responses
        
        upload_item = {
            'document_id': 'test_doc_123',
            'filename': 'test.pdf',
            'progress': 0
        }
        
        mock_progress_bar = Mock()
        mock_status_text = Mock()
        
        with patch('streamlit.session_state', {'auth_token': 'test_token'}):
            upload_interface._monitor_processing(upload_item, mock_progress_bar, mock_status_text)
            
            # Check that progress was updated
            assert upload_item['progress'] == 100
            
            # Check that API was called multiple times
            assert mock_get.call_count == 4
    
    @patch('streamlit.session_state', {'upload_queue': []})
    def test_process_upload_queue_empty(self, upload_interface):
        """Test processing empty upload queue"""
        
        # Should handle empty queue gracefully
        upload_interface._process_upload_queue()
        
        # No errors should occur
        assert len(st.session_state.upload_queue) == 0
    
    @patch('streamlit.session_state', {})
    def test_process_upload_queue_with_files(self, upload_interface):
        """Test processing upload queue with files"""
        
        # Setup queue with test items
        st.session_state.upload_queue = [
            {
                'id': 'item1',
                'filename': 'test1.pdf',
                'status': 'queued',
                'progress': 0,
                'file': Mock(),
                'type': 'application/pdf'
            },
            {
                'id': 'item2',
                'filename': 'test2.png',
                'status': 'queued',
                'progress': 0,
                'file': Mock(),
                'type': 'image/png'
            }
        ]
        
        with patch('streamlit.progress') as mock_progress:
            with patch('streamlit.empty') as mock_empty:
                with patch.object(upload_interface, '_upload_file_to_api', return_value='doc_123'):
                    with patch.object(upload_interface, '_monitor_processing'):
                        
                        upload_interface._process_upload_queue()
                        
                        # Check that all items were processed
                        for item in st.session_state.upload_queue:
                            assert item['status'] == 'completed'
                            assert item['progress'] == 100
                            assert 'document_id' in item
    
    def test_show_document_preview_image(self, upload_interface, mock_image_file):
        """Test document preview for image files"""
        
        with patch('streamlit.columns') as mock_columns:
            with patch('streamlit.image') as mock_image:
                with patch('streamlit.text') as mock_text:
                    
                    mock_col1, mock_col2 = Mock(), Mock()
                    mock_columns.return_value = [mock_col1, mock_col2]
                    
                    upload_interface._show_document_preview(mock_image_file)
                    
                    # Check that image was displayed
                    mock_image.assert_called_once()
    
    def test_show_document_preview_pdf(self, upload_interface, mock_uploaded_file):
        """Test document preview for PDF files"""
        
        with patch('streamlit.columns') as mock_columns:
            with patch('streamlit.info') as mock_info:
                with patch('streamlit.text') as mock_text:
                    
                    mock_col1, mock_col2 = Mock(), Mock()
                    mock_columns.return_value = [mock_col1, mock_col2]
                    
                    upload_interface._show_document_preview(mock_uploaded_file)
                    
                    # Check that info message was shown for non-image files
                    mock_info.assert_called_once()
    
    @patch('streamlit.session_state', {'upload_history': []})
    def test_render_upload_history_empty(self, upload_interface):
        """Test rendering empty upload history"""
        
        with patch('streamlit.info') as mock_info:
            upload_interface._render_upload_history()
            
            mock_info.assert_called_once_with("No recent uploads")
    
    @patch('streamlit.session_state', {})
    def test_render_upload_history_with_data(self, upload_interface):
        """Test rendering upload history with data"""
        
        # Setup history data
        st.session_state.upload_history = [
            {
                'filename': 'test1.pdf',
                'status': 'completed',
                'uploaded_at': datetime.now(),
                'document_id': 'doc_123',
                'metadata': {'size_mb': 1.5}
            },
            {
                'filename': 'test2.png',
                'status': 'processing',
                'uploaded_at': datetime.now(),
                'document_id': 'doc_456',
                'metadata': {'size_mb': 0.8}
            }
        ]
        
        with patch('streamlit.dataframe') as mock_dataframe:
            upload_interface._render_upload_history()
            
            # Check that dataframe was displayed
            mock_dataframe.assert_called_once()
            
            # Check dataframe content
            call_args = mock_dataframe.call_args[0][0]
            assert len(call_args) == 2  # Two history items
            assert 'test1.pdf' in call_args['Filename'].values
            assert 'test2.png' in call_args['Filename'].values


@pytest.mark.integration
class TestDocumentUploadIntegration:
    """Integration tests for document upload workflow"""
    
    @pytest.fixture
    def upload_interface(self):
        """Create DocumentUploadInterface instance"""
        return DocumentUploadInterface()
    
    @patch('streamlit.session_state', {})
    @patch('requests.post')
    @patch('requests.get')
    def test_complete_upload_workflow(self, mock_get, mock_post, upload_interface):
        """Test complete upload workflow from file selection to completion"""
        
        # Mock file upload API response
        upload_response = Mock()
        upload_response.status_code = 200
        upload_response.json.return_value = {'document_id': 'test_doc_123'}
        mock_post.return_value = upload_response
        
        # Mock status monitoring API responses
        status_responses = [
            {'status': 'processing', 'progress': 50},
            {'status': 'completed', 'progress': 100}
        ]
        
        mock_status_responses = []
        for resp_data in status_responses:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = resp_data
            mock_status_responses.append(mock_resp)
        
        mock_get.side_effect = mock_status_responses
        
        # Create mock uploaded file
        mock_file = Mock()
        mock_file.name = "integration_test.pdf"
        mock_file.size = 1024 * 1024
        mock_file.type = "application/pdf"
        mock_file.getvalue.return_value = b"test content"
        
        # Simulate complete workflow
        with patch('streamlit.button', return_value=True):
            with patch('streamlit.progress') as mock_progress:
                with patch('streamlit.empty') as mock_status:
                    with patch('time.sleep'):
                        
                        # Handle file upload
                        upload_interface._handle_file_upload([mock_file])
                        
                        # Verify file was queued
                        assert len(st.session_state.upload_queue) == 1
                        
                        # Verify final status
                        queue_item = st.session_state.upload_queue[0]
                        assert queue_item['status'] == 'completed'
                        assert queue_item['progress'] == 100
                        assert queue_item['document_id'] == 'test_doc_123'


if __name__ == "__main__":
    pytest.main([__file__])