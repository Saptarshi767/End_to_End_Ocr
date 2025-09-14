#!/usr/bin/env python3
"""
Demo script for document processing foundation functionality.
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, 'src')

from data_processing import DocumentProcessor, ImagePreprocessor


def create_sample_document():
    """Create a sample document for testing."""
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, 'sample_document.png')
    
    # Create a sample document image with some text-like content
    image = Image.new('RGB', (800, 600), color='white')
    pixels = image.load()
    
    # Add some black rectangles to simulate text
    for i in range(100, 700, 150):
        for j in range(50, 550, 100):
            for x in range(i, min(i + 100, 800)):
                for y in range(j, min(j + 20, 600)):
                    pixels[x, y] = (0, 0, 0)
    
    image.save(image_path, 'PNG')
    return image_path, temp_dir


def main():
    """Main demo function."""
    print("OCR Table Analytics - Document Processing Foundation Demo")
    print("=" * 60)
    
    # Create sample document
    print("\n1. Creating sample document...")
    image_path, temp_dir = create_sample_document()
    print(f"   Sample document created: {os.path.basename(image_path)}")
    
    # Initialize processors
    print("\n2. Initializing processors...")
    document_processor = DocumentProcessor()
    image_preprocessor = ImagePreprocessor()
    print("   Document processor and image preprocessor initialized")
    
    # Process document
    print("\n3. Processing document (validation and metadata extraction)...")
    try:
        result = document_processor.process_document(image_path)
        
        if result.success:
            print("   ✓ Document processing successful!")
            print(f"   - Filename: {result.metadata.filename}")
            print(f"   - File size: {result.metadata.file_size:,} bytes")
            print(f"   - MIME type: {result.metadata.mime_type}")
            print(f"   - Dimensions: {result.metadata.image_dimensions}")
            print(f"   - File hash: {result.metadata.file_hash[:16]}...")
            print(f"   - Valid: {result.metadata.is_valid}")
        else:
            print("   ✗ Document processing failed!")
            return
            
    except Exception as e:
        print(f"   ✗ Error processing document: {e}")
        return
    
    # Load image for preprocessing
    print("\n4. Loading image for preprocessing...")
    image = np.array(Image.open(image_path))
    print(f"   Image loaded: {image.shape} shape, {image.dtype} dtype")
    
    # Assess image quality
    print("\n5. Assessing image quality...")
    quality_metrics = image_preprocessor.assess_image_quality(image)
    print(f"   - Brightness: {quality_metrics.brightness:.3f}")
    print(f"   - Contrast: {quality_metrics.contrast:.3f}")
    print(f"   - Sharpness: {quality_metrics.sharpness:.3f}")
    print(f"   - Noise level: {quality_metrics.noise_level:.3f}")
    print(f"   - Skew angle: {quality_metrics.skew_angle:.2f}°")
    print(f"   - Overall score: {quality_metrics.overall_score:.3f}")
    
    print("\n   Recommendations:")
    for i, rec in enumerate(quality_metrics.recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Preprocess image
    print("\n6. Preprocessing image...")
    try:
        preprocessed_image = image_preprocessor.preprocess_image(image, quality_metrics)
        print(f"   ✓ Image preprocessing successful!")
        print(f"   - Original shape: {image.shape}")
        print(f"   - Preprocessed shape: {preprocessed_image.shape}")
        print(f"   - Changes applied: {not np.array_equal(image, preprocessed_image)}")
        
    except Exception as e:
        print(f"   ✗ Error preprocessing image: {e}")
        return
    
    # Test format conversion
    print("\n7. Testing format conversion...")
    try:
        rgb_image = image_preprocessor.convert_format(image, 'RGB')
        gray_image = image_preprocessor.convert_format(image, 'GRAY')
        
        print(f"   ✓ Format conversion successful!")
        print(f"   - Original: {image.shape}")
        print(f"   - RGB: {rgb_image.shape}")
        print(f"   - Grayscale: {gray_image.shape}")
        
    except Exception as e:
        print(f"   ✗ Error in format conversion: {e}")
        return
    
    # Test validation
    print("\n8. Testing format validation...")
    is_valid = document_processor.validate_format(image_path)
    print(f"   Format validation result: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Cleanup
    print("\n9. Cleaning up...")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("   Temporary files cleaned up")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully! 🎉")
    print("\nThe document processing foundation is ready for:")
    print("- File upload and validation (PDF, PNG, JPEG, TIFF)")
    print("- Document metadata extraction and storage")
    print("- File size and security validation")
    print("- Image enhancement (noise reduction, contrast adjustment)")
    print("- Image format conversion utilities")
    print("- Image quality assessment metrics")


if __name__ == "__main__":
    main()