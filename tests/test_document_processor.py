"""
Unit tests for document processor module.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from src.data_processing.document_processor import (
    DocumentProcessor, DocumentMetadata, ValidationConfig, ValidationResult
)
from src.core.models import ProcessingOptions
from src.core.exceptions import DocumentProcessingError, ValidationError


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_image(self, filename: str, size: tuple = (800, 600), format: str = 'PNG') -> str:
        """Create a test image file."""
        file_path = os.path.join(self.temp_dir, filename)
        image = Image.new('RGB', size, color='white')
        image.save(file_path, format=format)
        return file_path

    def create_test_pdf(self, filename: str) -> str:
        """Create a minimal test PDF file."""
        file_path = os.path.join(self.temp_dir, filename)
        # Create a minimal PDF content
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
190
%%EOF"""
        with open(file_path, 'wb') as f:
            f.write(pdf_content)
        return file_path

    def test_process_document_valid_image(self):
        """Test processing a valid image document."""
        image_path = self.create_test_image('test.png')
        
        result = self.processor.process_document(image_path)
        
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata.filename == 'test.png'
        assert result.metadata.mime_type == 'image/png'
        assert result.metadata.file_extension == '.png'
        assert result.metadata.is_valid is True
        assert result.metadata.image_dimensions == (800, 600)

    def test_process_document_valid_pdf(self):
        """Test processing a valid PDF document."""
        pdf_path = self.create_test_pdf('test.pdf')
        
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.__len__ = Mock(return_value=1)
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            result = self.processor.process_document(pdf_path)
            
            assert result.success is True
            assert result.metadata.filename == 'test.pdf'
            assert result.metadata.mime_type == 'application/pdf'
            assert result.metadata.page_count == 1

    def test_process_document_nonexistent_file(self):
        """Test processing a non-existent file."""
        with pytest.raises(DocumentProcessingError, match="File not found"):
            self.processor.process_document('/nonexistent/file.pdf')

    def test_process_document_invalid_extension(self):
        """Test processing a file with invalid extension."""
        # Create a file with invalid extension
        invalid_path = os.path.join(self.temp_dir, 'test.exe')
        with open(invalid_path, 'w') as f:
            f.write('test content')
        
        with pytest.raises(ValidationError, match="File extension '.exe' is not allowed"):
            self.processor.process_document(invalid_path)

    def test_process_document_oversized_file(self):
        """Test processing an oversized file."""
        # Create a large image
        large_image_path = self.create_test_image('large.png', size=(5000, 5000))
        
        # Use a small file size limit for testing
        config = ValidationConfig(max_file_size_mb=1)
        processor = DocumentProcessor(config)
        
        with pytest.raises(ValidationError, match="File size .* exceeds maximum allowed size"):
            processor.process_document(large_image_path)

    def test_validate_format_valid_extensions(self):
        """Test format validation for valid extensions."""
        image_path = self.create_test_image('test.png')
        assert self.processor.validate_format(image_path) is True
        
        pdf_path = self.create_test_pdf('test.pdf')
        assert self.processor.validate_format(pdf_path) is True

    def test_validate_format_invalid_extension(self):
        """Test format validation for invalid extensions."""
        invalid_path = os.path.join(self.temp_dir, 'test.txt')
        with open(invalid_path, 'w') as f:
            f.write('test')
        
        assert self.processor.validate_format(invalid_path) is False

    def test_extract_metadata_image(self):
        """Test metadata extraction for image files."""
        image_path = self.create_test_image('test.jpg', format='JPEG')
        
        metadata = self.processor._extract_metadata(image_path)
        
        assert metadata.filename == 'test.jpg'
        assert metadata.file_extension == '.jpg'
        assert metadata.image_dimensions == (800, 600)
        assert metadata.file_size > 0
        assert len(metadata.file_hash) == 64  # SHA-256 hash length

    def test_extract_metadata_pdf(self):
        """Test metadata extraction for PDF files."""
        pdf_path = self.create_test_pdf('test.pdf')
        
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.__len__ = Mock(return_value=3)
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            metadata = self.processor._extract_metadata(pdf_path)
            
            assert metadata.filename == 'test.pdf'
            assert metadata.file_extension == '.pdf'
            assert metadata.page_count == 3

    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        test_path = os.path.join(self.temp_dir, 'test.txt')
        with open(test_path, 'w') as f:
            f.write('test content')
        
        hash1 = self.processor._calculate_file_hash(test_path)
        hash2 = self.processor._calculate_file_hash(test_path)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hash length

    def test_detect_mime_type(self):
        """Test MIME type detection."""
        image_path = self.create_test_image('test.png')
        mime_type = self.processor._detect_mime_type(image_path)
        assert mime_type == 'image/png'
        
        pdf_path = self.create_test_pdf('test.pdf')
        mime_type = self.processor._detect_mime_type(pdf_path)
        assert mime_type == 'application/pdf'

    def test_validate_file_integrity_valid_image(self):
        """Test file integrity validation for valid image."""
        image_path = self.create_test_image('test.png')
        metadata = DocumentMetadata(
            filename='test.png',
            file_path=image_path,
            file_size=1000,
            mime_type='image/png',
            file_hash='test_hash',
            upload_timestamp=None
        )
        
        assert self.processor._validate_file_integrity(image_path, metadata) is True

    def test_validate_file_integrity_corrupted_file(self):
        """Test file integrity validation for corrupted file."""
        # Create a corrupted image file
        corrupted_path = os.path.join(self.temp_dir, 'corrupted.png')
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a valid image')
        
        metadata = DocumentMetadata(
            filename='corrupted.png',
            file_path=corrupted_path,
            file_size=100,
            mime_type='image/png',
            file_hash='test_hash',
            upload_timestamp=None
        )
        
        assert self.processor._validate_file_integrity(corrupted_path, metadata) is False

    def test_perform_security_scan_safe_file(self):
        """Test security scan for safe file."""
        image_path = self.create_test_image('safe_image.png')
        metadata = DocumentMetadata(
            filename='safe_image.png',
            file_path=image_path,
            file_size=1000,
            mime_type='image/png',
            file_hash='test_hash',
            upload_timestamp=None
        )
        
        issues = self.processor._perform_security_scan(image_path, metadata)
        assert len(issues) == 0

    def test_perform_security_scan_suspicious_filename(self):
        """Test security scan for suspicious filename."""
        image_path = self.create_test_image('../suspicious.png')
        metadata = DocumentMetadata(
            filename='../suspicious.png',
            file_path=image_path,
            file_size=1000,
            mime_type='image/png',
            file_hash='test_hash',
            upload_timestamp=None
        )
        
        issues = self.processor._perform_security_scan(image_path, metadata)
        assert any('suspicious characters' in issue for issue in issues)

    def test_perform_security_scan_executable_extension(self):
        """Test security scan for executable extension."""
        exe_path = os.path.join(self.temp_dir, 'malware.exe')
        with open(exe_path, 'wb') as f:
            f.write(b'fake executable')
        
        metadata = DocumentMetadata(
            filename='malware.exe',
            file_path=exe_path,
            file_size=100,
            mime_type='application/octet-stream',
            file_hash='test_hash',
            upload_timestamp=None,
            file_extension='.exe'
        )
        
        issues = self.processor._perform_security_scan(exe_path, metadata)
        assert any('executable extension' in issue for issue in issues)

    def test_validation_config_defaults(self):
        """Test validation configuration defaults."""
        config = ValidationConfig()
        
        assert config.max_file_size_mb == 50
        assert '.pdf' in config.allowed_extensions
        assert '.png' in config.allowed_extensions
        assert 'application/pdf' in config.allowed_mime_types
        assert 'image/png' in config.allowed_mime_types
        assert config.min_image_dimensions == (100, 100)
        assert config.max_image_dimensions == (10000, 10000)

    def test_validation_config_custom(self):
        """Test custom validation configuration."""
        config = ValidationConfig(
            max_file_size_mb=10,
            allowed_extensions=['.pdf'],
            allowed_mime_types=['application/pdf']
        )
        
        assert config.max_file_size_mb == 10
        assert config.allowed_extensions == ['.pdf']
        assert config.allowed_mime_types == ['application/pdf']

    def test_document_metadata_post_init(self):
        """Test DocumentMetadata post-initialization."""
        metadata = DocumentMetadata(
            filename='test.pdf',
            file_path='/path/to/test.pdf',
            file_size=1000,
            mime_type='application/pdf',
            file_hash='test_hash',
            upload_timestamp=None
        )
        
        assert metadata.validation_errors == []
        assert metadata.is_valid is True

    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            is_valid=False,
            errors=['Error 1', 'Error 2'],
            warnings=['Warning 1']
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1