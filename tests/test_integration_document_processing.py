"""
Integration tests for document processing foundation.
"""

import os
import tempfile
import numpy as np
from PIL import Image

from src.data_processing import DocumentProcessor, ImagePreprocessor


class TestDocumentProcessingIntegration:
    """Integration tests for document processing components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.document_processor = DocumentProcessor()
        self.image_preprocessor = ImagePreprocessor()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_image(self, filename: str, size: tuple = (800, 600)) -> str:
        """Create a test image file."""
        file_path = os.path.join(self.temp_dir, filename)
        image = Image.new('RGB', size, color='white')
        # Add some content to make it more realistic
        pixels = image.load()
        for i in range(100, 200):
            for j in range(100, 200):
                pixels[i, j] = (0, 0, 0)  # Black square
        image.save(file_path, 'PNG')
        return file_path

    def test_document_upload_and_preprocessing_workflow(self):
        """Test complete workflow from document upload to preprocessing."""
        # Create test image
        image_path = self.create_test_image('test_document.png')
        
        # Step 1: Process document (upload and validation)
        result = self.document_processor.process_document(image_path)
        
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata.is_valid is True
        assert result.metadata.image_dimensions == (800, 600)
        
        # Step 2: Load image for preprocessing
        image = np.array(Image.open(image_path))
        
        # Step 3: Assess image quality
        quality_metrics = self.image_preprocessor.assess_image_quality(image)
        
        assert quality_metrics is not None
        assert 0 <= quality_metrics.brightness <= 1
        assert 0 <= quality_metrics.contrast <= 1
        assert 0 <= quality_metrics.overall_score <= 1
        
        # Step 4: Preprocess image
        preprocessed_image = self.image_preprocessor.preprocess_image(image, quality_metrics)
        
        assert preprocessed_image is not None
        assert preprocessed_image.shape == image.shape
        assert preprocessed_image.dtype == np.uint8

    def test_format_validation_and_conversion(self):
        """Test format validation and image conversion."""
        # Create test image
        image_path = self.create_test_image('test_format.png')
        
        # Validate format
        is_valid = self.document_processor.validate_format(image_path)
        assert is_valid is True
        
        # Load and convert image format
        image = np.array(Image.open(image_path))
        
        # Test format conversions
        rgb_image = self.image_preprocessor.convert_format(image, 'RGB')
        gray_image = self.image_preprocessor.convert_format(image, 'GRAY')
        
        assert rgb_image.shape == image.shape
        assert len(gray_image.shape) == 2
        assert gray_image.shape[:2] == image.shape[:2]

    def test_error_handling_integration(self):
        """Test error handling across components."""
        # Test with non-existent file
        try:
            self.document_processor.process_document('/nonexistent/file.png')
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "File not found" in str(e)
        
        # Test with invalid image data
        try:
            invalid_image = np.array([])
            self.image_preprocessor.preprocess_image(invalid_image)
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "preprocessing failed" in str(e).lower()

    def test_quality_assessment_and_recommendations(self):
        """Test quality assessment and preprocessing recommendations."""
        # Create a low-quality image (very dark)
        image_path = os.path.join(self.temp_dir, 'low_quality.png')
        dark_image = Image.new('RGB', (400, 300), color=(30, 30, 30))  # Very dark
        dark_image.save(image_path, 'PNG')
        
        # Process document
        result = self.document_processor.process_document(image_path)
        assert result.success is True
        
        # Load and assess image
        image = np.array(Image.open(image_path))
        quality_metrics = self.image_preprocessor.assess_image_quality(image)
        
        # Should detect low brightness
        assert quality_metrics.brightness < 0.5
        assert len(quality_metrics.recommendations) > 0
        assert any('dark' in rec.lower() or 'brightness' in rec.lower() 
                  for rec in quality_metrics.recommendations)
        
        # Preprocessing should improve the image
        preprocessed_image = self.image_preprocessor.preprocess_image(image, quality_metrics)
        
        # Check that preprocessing was applied
        assert not np.array_equal(image, preprocessed_image)