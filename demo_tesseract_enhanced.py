"""
Demo script showing the enhanced Tesseract OCR engine functionality.
This script demonstrates the improved confidence scoring, bounding box extraction, and error handling.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import logging

from src.ocr.tesseract_engine import TesseractEngine, TESSERACT_AVAILABLE
from src.core.exceptions import OCREngineError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_image(text: str, width: int = 400, height: int = 100) -> np.ndarray:
    """Create a sample image with text for testing."""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), text, fill='black', font=font)
    return np.array(img)


def create_table_image() -> np.ndarray:
    """Create a sample table image for testing."""
    rows, cols = 3, 3
    cell_width, cell_height = 80, 30
    width = cols * cell_width
    height = rows * cell_height
    
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            # Draw cell border
            draw.rectangle([x1, y1, x2, y2], outline='black', width=1)
            
            # Add cell content
            if row == 0:
                text = f"Header{col+1}"
            else:
                text = f"R{row}C{col+1}"
            
            # Center text in cell
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = x1 + (cell_width - text_width) // 2
            text_y = y1 + (cell_height - text_height) // 2
            
            draw.text((text_x, text_y), text, fill='black', font=font)
    
    return np.array(img)


def demo_basic_ocr():
    """Demonstrate basic OCR functionality."""
    print("\n=== Basic OCR Demo ===")
    
    if not TESSERACT_AVAILABLE:
        print("Tesseract not available - skipping OCR demo")
        return
    
    try:
        # Initialize engine
        engine = TesseractEngine(confidence_threshold=0.6)
        engine.initialize()
        
        # Create test image
        test_text = "Hello World! This is a test of enhanced OCR."
        image = create_sample_image(test_text)
        
        print(f"Original text: {test_text}")
        
        # Extract text
        result = engine.extract_text(image)
        
        print(f"Extracted text: {result.text}")
        print(f"Overall confidence: {result.confidence:.3f}")
        print(f"Processing time: {result.processing_time_ms}ms")
        print(f"Number of words detected: {len(result.word_level_data)}")
        print(f"Number of line boxes: {len(result.bounding_boxes)}")
        
        # Show word-level details
        print("\nWord-level details:")
        for i, word in enumerate(result.word_level_data[:5]):  # Show first 5 words
            bbox = word.bounding_box
            print(f"  Word {i+1}: '{word.text}' (confidence: {word.confidence:.3f}, "
                  f"bbox: {bbox.x},{bbox.y},{bbox.width},{bbox.height})")
        
    except OCREngineError as e:
        print(f"OCR Error: {e.message} (Code: {e.error_code})")
    except Exception as e:
        print(f"Unexpected error: {e}")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n=== Error Handling Demo ===")
    
    if not TESSERACT_AVAILABLE:
        print("Tesseract not available - skipping error handling demo")
        return
    
    try:
        engine = TesseractEngine()
        engine.initialize()
        
        # Test with invalid image
        print("Testing with None image...")
        try:
            engine.extract_text(None)
        except OCREngineError as e:
            print(f"  Caught expected error: {e.error_code} - {e.message}")
        
        # Test with too small image
        print("Testing with too small image...")
        try:
            tiny_image = np.ones((5, 5, 3), dtype=np.uint8)
            engine.extract_text(tiny_image)
        except OCREngineError as e:
            print(f"  Caught expected error: {e.error_code} - {e.message}")
        
        # Test with invalid shape
        print("Testing with invalid image shape...")
        try:
            invalid_image = np.ones((10,))  # 1D array
            engine.extract_text(invalid_image)
        except OCREngineError as e:
            print(f"  Caught expected error: {e.error_code} - {e.message}")
        
        print("Error handling working correctly!")
        
    except Exception as e:
        print(f"Unexpected error in error handling demo: {e}")


def demo_table_detection():
    """Demonstrate table detection functionality."""
    print("\n=== Table Detection Demo ===")
    
    if not TESSERACT_AVAILABLE:
        print("Tesseract not available - skipping table detection demo")
        return
    
    try:
        engine = TesseractEngine()
        engine.initialize()
        
        # Create table image
        table_image = create_table_image()
        
        print("Detecting tables in sample image...")
        table_regions = engine.detect_tables(table_image)
        
        print(f"Number of table regions detected: {len(table_regions)}")
        
        for i, region in enumerate(table_regions):
            bbox = region.bounding_box
            print(f"  Table {i+1}: confidence={region.confidence:.3f}, "
                  f"bbox=({bbox.x},{bbox.y},{bbox.width},{bbox.height}), "
                  f"page={region.page_number}")
        
    except OCREngineError as e:
        print(f"Table detection error: {e.message} (Code: {e.error_code})")
    except Exception as e:
        print(f"Unexpected error in table detection: {e}")


def demo_configuration():
    """Demonstrate configuration options."""
    print("\n=== Configuration Demo ===")
    
    engine = TesseractEngine()
    
    print("Default configuration:")
    info = engine.get_engine_info()
    print(f"  Name: {info['name']}")
    print(f"  Confidence threshold: {info['confidence_threshold']}")
    print(f"  Supported languages: {info['supported_languages'][:3]}...")  # Show first 3
    print(f"  Initialized: {info['is_initialized']}")
    
    # Test configuration changes
    print("\nTesting configuration changes...")
    engine.configure({
        'psm': 8,  # Single word mode
        'confidence_threshold': 0.9,
        'preserve_interword_spaces': '1'
    })
    
    updated_info = engine.get_engine_info()
    print(f"  Updated confidence threshold: {updated_info['confidence_threshold']}")
    print(f"  PSM setting: {engine.psm}")
    print(f"  Config contains preserve_interword_spaces: {'preserve_interword_spaces' in updated_info['config']}")


def demo_confidence_scoring():
    """Demonstrate enhanced confidence scoring."""
    print("\n=== Confidence Scoring Demo ===")
    
    if not TESSERACT_AVAILABLE:
        print("Tesseract not available - skipping confidence scoring demo")
        return
    
    try:
        engine = TesseractEngine()
        engine.initialize()
        
        # Test with clear text
        clear_text = "CLEAR TEXT EXAMPLE"
        clear_image = create_sample_image(clear_text, width=300, height=60)
        clear_result = engine.extract_text(clear_image)
        
        # Test with noisy text
        noisy_image = clear_image.copy()
        noise = np.random.normal(0, 25, noisy_image.shape).astype(np.int16)
        noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        noisy_result = engine.extract_text(noisy_image)
        
        print(f"Clear text confidence: {clear_result.confidence:.3f}")
        print(f"Noisy text confidence: {noisy_result.confidence:.3f}")
        print(f"Confidence difference: {clear_result.confidence - noisy_result.confidence:.3f}")
        
        # Show word-level confidence variation
        if clear_result.word_level_data:
            confidences = [word.confidence for word in clear_result.word_level_data]
            print(f"Word confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        
    except Exception as e:
        print(f"Error in confidence scoring demo: {e}")


def main():
    """Run all demos."""
    print("Enhanced Tesseract OCR Engine Demo")
    print("=" * 40)
    
    if not TESSERACT_AVAILABLE:
        print("WARNING: Tesseract dependencies not available.")
        print("Install with: pip install pytesseract pillow")
        print("Also ensure Tesseract OCR is installed on your system.")
        print("\nRunning configuration demo only...")
        demo_configuration()
        return
    
    # Run all demos
    demo_basic_ocr()
    demo_error_handling()
    demo_table_detection()
    demo_configuration()
    demo_confidence_scoring()
    
    print("\n" + "=" * 40)
    print("Demo completed!")


if __name__ == "__main__":
    main()