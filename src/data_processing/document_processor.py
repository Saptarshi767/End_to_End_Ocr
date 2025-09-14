"""
Document processing module for handling file uploads and validation.
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None
import PIL.Image
from PIL import Image, ImageStat
import fitz  # PyMuPDF for PDF handling

from src.core.exceptions import DocumentProcessingError, ValidationError
from src.core.models import ProcessingOptions, ProcessingResult


@dataclass
class DocumentMetadata:
    """Document metadata container."""
    filename: str
    file_path: str
    file_size: int
    mime_type: str
    file_hash: str
    upload_timestamp: datetime
    page_count: Optional[int] = None
    image_dimensions: Optional[Tuple[int, int]] = None
    file_extension: str = ""
    is_valid: bool = True
    validation_errors: List[str] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


@dataclass
class ValidationConfig:
    """Configuration for document validation."""
    max_file_size_mb: int = 50
    allowed_extensions: List[str] = None
    allowed_mime_types: List[str] = None
    min_image_dimensions: Tuple[int, int] = (100, 100)
    max_image_dimensions: Tuple[int, int] = (10000, 10000)
    scan_for_malware: bool = True

    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        if self.allowed_mime_types is None:
            self.allowed_mime_types = [
                'application/pdf',
                'image/png',
                'image/jpeg',
                'image/tiff',
                'image/bmp'
            ]


class DocumentProcessor:
    """
    Handles document upload, validation, and metadata extraction.
    """

    def __init__(self, validation_config: Optional[ValidationConfig] = None):
        """
        Initialize document processor.
        
        Args:
            validation_config: Configuration for document validation
        """
        self.validation_config = validation_config or ValidationConfig()
        self._setup_magic()

    def _setup_magic(self):
        """Setup python-magic for MIME type detection."""
        if MAGIC_AVAILABLE:
            try:
                self.magic_mime = magic.Magic(mime=True)
            except Exception as e:
                # Fallback to mimetypes module if python-magic setup fails
                self.magic_mime = None
        else:
            self.magic_mime = None

    def process_document(self, file_path: str, options: Optional[ProcessingOptions] = None) -> ProcessingResult:
        """
        Process uploaded document and extract metadata.
        
        Args:
            file_path: Path to the document file
            options: Processing options
            
        Returns:
            ProcessingResult containing metadata and validation status
            
        Raises:
            DocumentProcessingError: If document processing fails
            ValidationError: If document validation fails
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise DocumentProcessingError(f"File not found: {file_path}")

            # Extract metadata
            metadata = self._extract_metadata(file_path)
            
            # Validate document
            validation_result = self._validate_document(file_path, metadata)
            metadata.is_valid = validation_result.is_valid
            metadata.validation_errors = validation_result.errors

            if not validation_result.is_valid:
                raise ValidationError(f"Document validation failed: {', '.join(validation_result.errors)}")

            # Create processing result
            result = ProcessingResult(
                success=True,
                metadata=metadata,
                message="Document processed successfully"
            )

            return result

        except (DocumentProcessingError, ValidationError):
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Unexpected error processing document: {str(e)}")

    def _extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract comprehensive metadata from document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentMetadata object
        """
        path_obj = Path(file_path)
        
        # Basic file information
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        filename = path_obj.name
        file_extension = path_obj.suffix.lower()
        
        # Generate file hash for integrity checking
        file_hash = self._calculate_file_hash(file_path)
        
        # Detect MIME type
        mime_type = self._detect_mime_type(file_path)
        
        # Initialize metadata
        metadata = DocumentMetadata(
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
            file_hash=file_hash,
            upload_timestamp=datetime.now(),
            file_extension=file_extension
        )

        # Extract format-specific metadata
        try:
            if mime_type == 'application/pdf':
                metadata.page_count = self._get_pdf_page_count(file_path)
            elif mime_type.startswith('image/'):
                metadata.image_dimensions = self._get_image_dimensions(file_path)
        except Exception as e:
            # Don't fail metadata extraction for format-specific issues
            metadata.validation_errors.append(f"Could not extract format-specific metadata: {str(e)}")

        return metadata

    def _validate_document(self, file_path: str, metadata: DocumentMetadata) -> 'ValidationResult':
        """
        Validate document against security and format requirements.
        
        Args:
            file_path: Path to the document file
            metadata: Document metadata
            
        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []

        # File size validation
        max_size_bytes = self.validation_config.max_file_size_mb * 1024 * 1024
        if metadata.file_size > max_size_bytes:
            errors.append(f"File size ({metadata.file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({self.validation_config.max_file_size_mb}MB)")

        # File extension validation
        if metadata.file_extension not in self.validation_config.allowed_extensions:
            errors.append(f"File extension '{metadata.file_extension}' is not allowed. Allowed extensions: {', '.join(self.validation_config.allowed_extensions)}")

        # MIME type validation
        if metadata.mime_type not in self.validation_config.allowed_mime_types:
            errors.append(f"MIME type '{metadata.mime_type}' is not allowed. Allowed types: {', '.join(self.validation_config.allowed_mime_types)}")

        # Image dimension validation for image files
        if metadata.image_dimensions:
            width, height = metadata.image_dimensions
            min_width, min_height = self.validation_config.min_image_dimensions
            max_width, max_height = self.validation_config.max_image_dimensions
            
            if width < min_width or height < min_height:
                errors.append(f"Image dimensions ({width}x{height}) are below minimum required ({min_width}x{min_height})")
            
            if width > max_width or height > max_height:
                warnings.append(f"Image dimensions ({width}x{height}) are very large and may impact processing performance")

        # File integrity validation
        if not self._validate_file_integrity(file_path, metadata):
            errors.append("File appears to be corrupted or incomplete")

        # Security validation
        if self.validation_config.scan_for_malware:
            security_issues = self._perform_security_scan(file_path, metadata)
            errors.extend(security_issues)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for integrity checking."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type of file."""
        if self.magic_mime:
            try:
                return self.magic_mime.from_file(file_path)
            except Exception:
                pass
        
        # Fallback to mimetypes module
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'

    def _get_pdf_page_count(self, file_path: str) -> int:
        """Get number of pages in PDF document."""
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            raise DocumentProcessingError(f"Could not read PDF file: {str(e)}")

    def _get_image_dimensions(self, file_path: str) -> Tuple[int, int]:
        """Get dimensions of image file."""
        try:
            with Image.open(file_path) as img:
                return img.size
        except Exception as e:
            raise DocumentProcessingError(f"Could not read image file: {str(e)}")

    def _validate_file_integrity(self, file_path: str, metadata: DocumentMetadata) -> bool:
        """
        Validate file integrity by attempting to open/read the file.
        
        Args:
            file_path: Path to the file
            metadata: Document metadata
            
        Returns:
            True if file appears to be valid, False otherwise
        """
        try:
            if metadata.mime_type == 'application/pdf':
                # Try to open PDF
                doc = fitz.open(file_path)
                doc.close()
                return True
            elif metadata.mime_type.startswith('image/'):
                # Try to open image
                with Image.open(file_path) as img:
                    img.verify()
                return True
            else:
                # For other file types, just check if we can read some bytes
                with open(file_path, 'rb') as f:
                    f.read(1024)
                return True
        except Exception:
            return False

    def _perform_security_scan(self, file_path: str, metadata: DocumentMetadata) -> List[str]:
        """
        Perform basic security scanning on the file.
        
        Args:
            file_path: Path to the file
            metadata: Document metadata
            
        Returns:
            List of security issues found
        """
        issues = []
        
        # Check for suspicious file names
        suspicious_patterns = ['..', '/', '\\', '<', '>', '|', ':', '*', '?', '"']
        if any(pattern in metadata.filename for pattern in suspicious_patterns):
            issues.append("Filename contains suspicious characters")
        
        # Check for executable extensions disguised as documents
        if metadata.file_extension in ['.exe', '.bat', '.cmd', '.scr', '.com']:
            issues.append("File has executable extension")
        
        # Check for extremely large files that might be zip bombs
        if metadata.file_size > 100 * 1024 * 1024:  # 100MB
            issues.append("File is extremely large and may pose security risks")
        
        # Basic content validation for PDFs
        if metadata.mime_type == 'application/pdf':
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'%PDF-'):
                        issues.append("PDF file has invalid header")
            except Exception:
                issues.append("Could not validate PDF file structure")
        
        return issues

    def validate_format(self, file_path: str) -> bool:
        """
        Validate if file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if format is supported, False otherwise
        """
        try:
            path_obj = Path(file_path)
            file_extension = path_obj.suffix.lower()
            
            if file_extension not in self.validation_config.allowed_extensions:
                return False
            
            mime_type = self._detect_mime_type(file_path)
            if mime_type not in self.validation_config.allowed_mime_types:
                return False
            
            return True
        except Exception:
            return False


@dataclass
class ValidationResult:
    """Result of document validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]