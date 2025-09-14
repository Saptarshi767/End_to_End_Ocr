"""
Demo script for table structure extraction functionality.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import logging

from src.core.models import OCRResult, BoundingBox, WordData, TableRegion
from src.data_processing.table_structure_extractor import TableStructureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_table_image() -> np.ndarray:
    """Create a sample table image for demonstration."""
    # Create a white image
    width, height = 600, 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a default font
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw table structure
    # Headers
    headers = ["Employee Name", "Department", "Salary", "Start Date"]
    header_y = 50
    col_widths = [150, 120, 100, 120]
    col_x_positions = [50, 200, 320, 420]
    
    # Draw headers
    for i, (header, x_pos) in enumerate(zip(headers, col_x_positions)):
        draw.text((x_pos, header_y), header, fill='black', font=font)
    
    # Draw header underline
    draw.line([(40, header_y + 25), (width - 40, header_y + 25)], fill='black', width=2)
    
    # Sample data
    data_rows = [
        ["John Smith", "Engineering", "$75,000", "2020-01-15"],
        ["Jane Doe", "Marketing", "$65,000", "2019-03-22"],
        ["Bob Johnson", "Sales", "$55,000", "2021-06-10"],
        ["Alice Brown", "HR", "$60,000", "2018-11-05"],
        ["Charlie Wilson", "Engineering", "$80,000", "2020-08-30"]
    ]
    
    # Draw data rows
    row_height = 30
    for row_idx, row_data in enumerate(data_rows):
        y_pos = header_y + 50 + (row_idx * row_height)
        for col_idx, (cell_data, x_pos) in enumerate(zip(row_data, col_x_positions)):
            draw.text((x_pos, y_pos), cell_data, fill='black', font=font)
    
    # Convert PIL image to numpy array
    return np.array(image)


def create_mock_ocr_result(image: np.ndarray) -> OCRResult:
    """Create mock OCR result for the sample table image."""
    # Define the expected text and word positions based on our sample table
    word_data = []
    
    # Headers
    headers = ["Employee", "Name", "Department", "Salary", "Start", "Date"]
    header_positions = [(50, 50), (120, 50), (200, 50), (320, 50), (420, 50), (470, 50)]
    header_widths = [60, 40, 80, 50, 40, 40]
    
    for i, (word, (x, y), width) in enumerate(zip(headers, header_positions, header_widths)):
        word_data.append(WordData(
            text=word,
            confidence=0.9,
            bounding_box=BoundingBox(x, y, width, 20, 0.9)
        ))
    
    # Data rows
    data_rows = [
        ["John", "Smith", "Engineering", "$75,000", "2020-01-15"],
        ["Jane", "Doe", "Marketing", "$65,000", "2019-03-22"],
        ["Bob", "Johnson", "Sales", "$55,000", "2021-06-10"],
        ["Alice", "Brown", "HR", "$60,000", "2018-11-05"],
        ["Charlie", "Wilson", "Engineering", "$80,000", "2020-08-30"]
    ]
    
    base_y = 100
    row_height = 30
    col_positions = [50, 120, 200, 320, 420]
    col_widths = [60, 60, 80, 60, 80]
    
    for row_idx, row in enumerate(data_rows):
        y_pos = base_y + (row_idx * row_height)
        for col_idx, (cell, x_pos, width) in enumerate(zip(row, col_positions, col_widths)):
            word_data.append(WordData(
                text=cell,
                confidence=0.85,
                bounding_box=BoundingBox(x_pos, y_pos, width, 20, 0.85)
            ))
    
    # Create full text
    all_text = " ".join(word.text for word in word_data)
    
    return OCRResult(
        text=all_text,
        confidence=0.87,
        bounding_boxes=[],
        word_level_data=word_data
    )


def demo_basic_table_extraction():
    """Demonstrate basic table structure extraction."""
    print("\n=== Basic Table Structure Extraction Demo ===")
    
    # Create sample table image
    print("Creating sample table image...")
    image = create_sample_table_image()
    
    # Create mock OCR result
    print("Creating mock OCR result...")
    ocr_result = create_mock_ocr_result(image)
    
    # Define table region (covers the entire table)
    table_region = TableRegion(
        bounding_box=BoundingBox(40, 40, 520, 200, 0.8),
        confidence=0.8,
        page_number=1
    )
    
    # Initialize table structure extractor
    extractor = TableStructureExtractor(confidence_threshold=0.6)
    
    # Extract table structure
    print("Extracting table structure...")
    tables = extractor.extract_table_structure(ocr_result, [table_region])
    
    # Display results
    if tables:
        table = tables[0]
        print(f"\n✓ Successfully extracted table with confidence: {table.confidence:.2f}")
        print(f"  - Headers: {table.headers}")
        print(f"  - Rows: {len(table.rows)}")
        print(f"  - Columns: {len(table.headers)}")
        
        print("\nTable Data:")
        print("-" * 80)
        
        # Print headers
        header_row = " | ".join(f"{header:15}" for header in table.headers)
        print(header_row)
        print("-" * len(header_row))
        
        # Print data rows
        for i, row in enumerate(table.rows):
            row_str = " | ".join(f"{cell:15}" for cell in row)
            print(f"{row_str}")
        
        print("-" * 80)
        
    else:
        print("❌ No tables extracted")


def demo_table_validation():
    """Demonstrate table validation functionality."""
    print("\n=== Table Validation Demo ===")
    
    # Create sample table image and extract structure
    image = create_sample_table_image()
    ocr_result = create_mock_ocr_result(image)
    table_region = TableRegion(
        bounding_box=BoundingBox(40, 40, 520, 200, 0.8),
        confidence=0.8
    )
    
    extractor = TableStructureExtractor()
    tables = extractor.extract_table_structure(ocr_result, [table_region])
    
    if tables:
        table = tables[0]
        
        # Validate the extracted table
        print("Validating extracted table...")
        validation_result = extractor.validate_table_structure(table)
        
        print(f"\nValidation Results:")
        print(f"  - Valid: {validation_result.is_valid}")
        print(f"  - Confidence: {validation_result.confidence:.2f}")
        
        if validation_result.errors:
            print(f"  - Errors: {len(validation_result.errors)}")
            for error in validation_result.errors:
                print(f"    • {error}")
        
        if validation_result.warnings:
            print(f"  - Warnings: {len(validation_result.warnings)}")
            for warning in validation_result.warnings:
                print(f"    • {warning}")
        
        if validation_result.is_valid and not validation_result.warnings:
            print("  ✓ Table structure is valid with no issues!")
    else:
        print("❌ No tables to validate")


def demo_table_fragment_merging():
    """Demonstrate table fragment merging functionality."""
    print("\n=== Table Fragment Merging Demo ===")
    
    # Create two table fragments with same headers
    from src.core.models import Table
    
    fragment1 = Table(
        headers=["Name", "Department", "Salary"],
        rows=[
            ["John Smith", "Engineering", "$75,000"],
            ["Jane Doe", "Marketing", "$65,000"]
        ],
        confidence=0.8,
        metadata={"fragment": 1}
    )
    
    fragment2 = Table(
        headers=["Name", "Department", "Salary"],
        rows=[
            ["Bob Johnson", "Sales", "$55,000"],
            ["Alice Brown", "HR", "$60,000"]
        ],
        confidence=0.75,
        metadata={"fragment": 2}
    )
    
    # Create a different table (not a fragment)
    different_table = Table(
        headers=["Product", "Price", "Stock"],
        rows=[
            ["Widget A", "$10.99", "100"],
            ["Widget B", "$15.99", "50"]
        ],
        confidence=0.9
    )
    
    print("Original tables:")
    print(f"  - Fragment 1: {len(fragment1.rows)} rows")
    print(f"  - Fragment 2: {len(fragment2.rows)} rows")
    print(f"  - Different table: {len(different_table.rows)} rows")
    
    # Merge fragments
    extractor = TableStructureExtractor()
    tables = [fragment1, fragment2, different_table]
    merged_tables = extractor.merge_table_fragments(tables)
    
    print(f"\nAfter merging:")
    print(f"  - Total tables: {len(merged_tables)}")
    
    for i, table in enumerate(merged_tables):
        print(f"  - Table {i+1}: {len(table.rows)} rows, headers: {table.headers}")
        if 'merged_from' in table.metadata:
            print(f"    (Merged from {table.metadata['merged_from']} fragments)")


def demo_complex_table_scenarios():
    """Demonstrate handling of complex table scenarios."""
    print("\n=== Complex Table Scenarios Demo ===")
    
    # Test with irregular table (missing cells, merged headers, etc.)
    print("\n1. Testing irregular table structure...")
    
    # Create word data for irregular table
    irregular_words = [
        # Multi-part header
        WordData("Employee", 0.9, BoundingBox(50, 50, 80, 20, 0.9)),
        WordData("Information", 0.8, BoundingBox(140, 50, 80, 20, 0.8)),
        WordData("Salary", 0.9, BoundingBox(300, 50, 60, 20, 0.9)),
        
        # Sub-headers
        WordData("First", 0.8, BoundingBox(50, 75, 40, 15, 0.8)),
        WordData("Last", 0.8, BoundingBox(100, 75, 40, 15, 0.8)),
        WordData("Dept", 0.8, BoundingBox(150, 75, 40, 15, 0.8)),
        WordData("Amount", 0.8, BoundingBox(300, 75, 60, 15, 0.8)),
        
        # Data with missing cell
        WordData("John", 0.9, BoundingBox(50, 100, 40, 20, 0.9)),
        WordData("Smith", 0.9, BoundingBox(100, 100, 40, 20, 0.9)),
        # Missing department for John
        WordData("$75000", 0.8, BoundingBox(300, 100, 60, 20, 0.8)),
        
        # Complete row
        WordData("Jane", 0.9, BoundingBox(50, 125, 40, 20, 0.9)),
        WordData("Doe", 0.9, BoundingBox(100, 125, 40, 20, 0.9)),
        WordData("Marketing", 0.8, BoundingBox(150, 125, 70, 20, 0.8)),
        WordData("$65000", 0.8, BoundingBox(300, 125, 60, 20, 0.8)),
    ]
    
    irregular_ocr = OCRResult(
        text=" ".join(word.text for word in irregular_words),
        confidence=0.8,
        bounding_boxes=[],
        word_level_data=irregular_words
    )
    
    irregular_region = TableRegion(
        bounding_box=BoundingBox(40, 40, 340, 120, 0.8),
        confidence=0.8
    )
    
    extractor = TableStructureExtractor()
    irregular_tables = extractor.extract_table_structure(irregular_ocr, [irregular_region])
    
    if irregular_tables:
        table = irregular_tables[0]
        print(f"  ✓ Extracted irregular table: {len(table.rows)} rows, {len(table.headers)} columns")
        print(f"  - Headers: {table.headers}")
        print(f"  - Confidence: {table.confidence:.2f}")
        
        # Validate the irregular table
        validation = extractor.validate_table_structure(table)
        print(f"  - Validation warnings: {len(validation.warnings)}")
    else:
        print("  ❌ Could not extract irregular table")
    
    print("\n2. Testing single-row table (should be rejected)...")
    
    # Single row should not be processed as a table
    single_row_words = [
        WordData("Name", 0.9, BoundingBox(50, 50, 50, 20, 0.9)),
        WordData("Age", 0.8, BoundingBox(120, 50, 30, 20, 0.8)),
        WordData("City", 0.9, BoundingBox(170, 50, 40, 20, 0.9)),
    ]
    
    single_row_ocr = OCRResult(
        text=" ".join(word.text for word in single_row_words),
        confidence=0.85,
        bounding_boxes=[],
        word_level_data=single_row_words
    )
    
    single_row_region = TableRegion(
        bounding_box=BoundingBox(40, 40, 180, 40, 0.8),
        confidence=0.8
    )
    
    single_row_tables = extractor.extract_table_structure(single_row_ocr, [single_row_region])
    
    if single_row_tables:
        print(f"  ❌ Unexpectedly extracted {len(single_row_tables)} tables from single row")
    else:
        print("  ✓ Correctly rejected single-row as table")


def main():
    """Run all table structure extraction demos."""
    print("🔍 Table Structure Extraction Demo")
    print("=" * 50)
    
    try:
        demo_basic_table_extraction()
        demo_table_validation()
        demo_table_fragment_merging()
        demo_complex_table_scenarios()
        
        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        logger.exception("Demo failed")


if __name__ == "__main__":
    main()