"""
Tests for table structure extraction functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.core.models import OCRResult, BoundingBox, WordData, TableRegion, Table, ValidationResult
from src.data_processing.table_structure_extractor import TableStructureExtractor, Cell, TableStructure
from src.core.exceptions import OCREngineError


class TestTableStructureExtractor:
    """Test cases for TableStructureExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TableStructureExtractor(confidence_threshold=0.6)
    
    def create_sample_word_data(self) -> list:
        """Create sample word data for testing."""
        return [
            # Header row
            WordData("Name", 0.9, BoundingBox(10, 10, 50, 20, 0.9)),
            WordData("Age", 0.8, BoundingBox(80, 10, 30, 20, 0.8)),
            WordData("City", 0.9, BoundingBox(130, 10, 40, 20, 0.9)),
            
            # Data row 1
            WordData("John", 0.9, BoundingBox(10, 40, 40, 20, 0.9)),
            WordData("25", 0.8, BoundingBox(80, 40, 20, 20, 0.8)),
            WordData("NYC", 0.7, BoundingBox(130, 40, 30, 20, 0.7)),
            
            # Data row 2
            WordData("Jane", 0.8, BoundingBox(10, 70, 40, 20, 0.8)),
            WordData("30", 0.9, BoundingBox(80, 70, 20, 20, 0.9)),
            WordData("LA", 0.8, BoundingBox(130, 70, 25, 20, 0.8)),
        ]
    
    def create_sample_table_region(self) -> TableRegion:
        """Create sample table region for testing."""
        return TableRegion(
            bounding_box=BoundingBox(5, 5, 200, 100, 0.8),
            confidence=0.8,
            page_number=1
        )
    
    def create_sample_ocr_result(self) -> OCRResult:
        """Create sample OCR result for testing."""
        word_data = self.create_sample_word_data()
        text = " ".join(word.text for word in word_data)
        
        return OCRResult(
            text=text,
            confidence=0.85,
            bounding_boxes=[],
            word_level_data=word_data
        )
    
    def test_extract_table_structure_basic(self):
        """Test basic table structure extraction."""
        ocr_result = self.create_sample_ocr_result()
        table_regions = [self.create_sample_table_region()]
        
        tables = self.extractor.extract_table_structure(ocr_result, table_regions)
        
        assert len(tables) == 1
        table = tables[0]
        
        assert len(table.headers) == 3
        assert table.headers == ["Name", "Age", "City"]
        assert len(table.rows) == 2
        assert table.rows[0] == ["John", "25", "NYC"]
        assert table.rows[1] == ["Jane", "30", "LA"]
        assert table.confidence > 0.0
    
    def test_extract_words_in_region(self):
        """Test extraction of words within a table region."""
        word_data = self.create_sample_word_data()
        region = self.create_sample_table_region()
        
        words_in_region = self.extractor._extract_words_in_region(word_data, region)
        
        # All sample words should be within the region
        assert len(words_in_region) == len(word_data)
        
        # Test with region that excludes some words
        small_region = TableRegion(
            bounding_box=BoundingBox(5, 5, 100, 50, 0.8),
            confidence=0.8
        )
        
        words_in_small_region = self.extractor._extract_words_in_region(word_data, small_region)
        assert len(words_in_small_region) < len(word_data)
    
    def test_group_words_into_rows(self):
        """Test grouping words into rows."""
        word_data = self.create_sample_word_data()
        
        rows = self.extractor._group_words_into_rows(word_data)
        
        assert len(rows) == 3  # Header + 2 data rows
        
        # Check first row (header)
        assert len(rows[0]) == 3
        assert [w.text for w in rows[0]] == ["Name", "Age", "City"]
        
        # Check second row
        assert len(rows[1]) == 3
        assert [w.text for w in rows[1]] == ["John", "25", "NYC"]
        
        # Check third row
        assert len(rows[2]) == 3
        assert [w.text for w in rows[2]] == ["Jane", "30", "LA"]
    
    def test_detect_column_boundaries(self):
        """Test column boundary detection."""
        word_data = self.create_sample_word_data()
        rows = self.extractor._group_words_into_rows(word_data)
        
        boundaries = self.extractor._detect_column_boundaries(rows)
        
        # Should have 4 boundaries for 3 columns (left, between columns, right)
        assert len(boundaries) >= 3
        
        # Boundaries should be sorted
        assert boundaries == sorted(boundaries)
    
    def test_cluster_positions(self):
        """Test position clustering for column detection."""
        positions = [10, 12, 11, 80, 82, 130, 132, 131]
        
        clusters = self.extractor._cluster_positions(positions, threshold=5)
        
        # Should create 3 clusters
        assert len(clusters) == 3
        
        # Clusters should be approximately at the expected positions
        assert abs(clusters[0] - 11) <= 2  # Around 10-12
        assert abs(clusters[1] - 81) <= 2  # Around 80-82
        assert abs(clusters[2] - 131) <= 2  # Around 130-132
    
    def test_create_cell_grid(self):
        """Test cell grid creation."""
        word_data = self.create_sample_word_data()
        rows = self.extractor._group_words_into_rows(word_data)
        boundaries = [5, 70, 120, 180]  # 3 columns
        
        cells = self.extractor._create_cell_grid(rows, boundaries)
        
        # Should have 9 cells (3 rows × 3 columns)
        assert len(cells) == 9
        
        # Check some specific cells
        header_cells = [cell for cell in cells if cell.row == 0]
        assert len(header_cells) == 3
        assert any(cell.content == "Name" for cell in header_cells)
        assert any(cell.content == "Age" for cell in header_cells)
        assert any(cell.content == "City" for cell in header_cells)
    
    def test_identify_headers(self):
        """Test header identification."""
        # Create cells with header-like content
        cells = [
            Cell("Name", 0.9, 0, 0, BoundingBox(10, 10, 50, 20, 0.9)),
            Cell("Age", 0.8, 0, 1, BoundingBox(80, 10, 30, 20, 0.8)),
            Cell("Total Amount", 0.9, 0, 2, BoundingBox(130, 10, 80, 20, 0.9)),
            Cell("John", 0.9, 1, 0, BoundingBox(10, 40, 40, 20, 0.9)),
            Cell("25", 0.8, 1, 1, BoundingBox(80, 40, 20, 20, 0.8)),
            Cell("$100", 0.7, 1, 2, BoundingBox(130, 40, 40, 20, 0.7)),
        ]
        
        self.extractor._identify_headers(cells, [])
        
        # First row should be identified as headers
        first_row_cells = [cell for cell in cells if cell.row == 0]
        assert all(cell.is_header for cell in first_row_cells)
        
        # Second row should not be headers
        second_row_cells = [cell for cell in cells if cell.row == 1]
        assert not any(cell.is_header for cell in second_row_cells)
    
    def test_is_likely_header(self):
        """Test header content detection."""
        # Test header-like content
        assert self.extractor._is_likely_header("Name")
        assert self.extractor._is_likely_header("Total Amount")
        assert self.extractor._is_likely_header("Description:")
        assert self.extractor._is_likely_header("Customer ID")
        
        # Test non-header content
        assert not self.extractor._is_likely_header("John Smith")
        assert not self.extractor._is_likely_header("12345")
        assert not self.extractor._is_likely_header("")
        assert not self.extractor._is_likely_header("   ")
    
    def test_has_header_formatting(self):
        """Test header formatting detection."""
        # Test uppercase formatting
        assert self.extractor._has_header_formatting("NAME")
        assert self.extractor._has_header_formatting("TOTAL AMOUNT")
        
        # Test title case
        assert self.extractor._has_header_formatting("Customer Name")
        assert self.extractor._has_header_formatting("Order Date")
        
        # Test non-header formatting
        assert not self.extractor._has_header_formatting("john smith")
        assert not self.extractor._has_header_formatting("12345")
        assert not self.extractor._has_header_formatting("")
    
    def test_calculate_structure_confidence(self):
        """Test structure confidence calculation."""
        # Create high-confidence cells
        cells = [
            Cell("Name", 0.9, 0, 0, BoundingBox(10, 10, 50, 20, 0.9), is_header=True),
            Cell("Age", 0.8, 0, 1, BoundingBox(80, 10, 30, 20, 0.8), is_header=True),
            Cell("John", 0.9, 1, 0, BoundingBox(10, 40, 40, 20, 0.9)),
            Cell("25", 0.8, 1, 1, BoundingBox(80, 40, 20, 20, 0.8)),
        ]
        
        rows = [[], []]  # 2 rows
        boundaries = [0, 60, 120]  # 2 columns
        
        confidence = self.extractor._calculate_structure_confidence(cells, rows, boundaries)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high for good structure
    
    def test_extract_header_names(self):
        """Test header name extraction."""
        cells = [
            Cell("First", 0.9, 0, 0, BoundingBox(10, 10, 50, 20, 0.9), is_header=True),
            Cell("Name", 0.8, 1, 0, BoundingBox(10, 30, 50, 20, 0.8), is_header=True),
            Cell("Age", 0.9, 0, 1, BoundingBox(80, 10, 30, 20, 0.9), is_header=True),
            Cell("John", 0.9, 2, 0, BoundingBox(10, 60, 40, 20, 0.9)),
            Cell("25", 0.8, 2, 1, BoundingBox(80, 60, 20, 20, 0.8)),
        ]
        
        headers = self.extractor._extract_header_names(cells)
        
        assert len(headers) == 2
        assert headers[0] == "First Name"  # Combined multi-row header
        assert headers[1] == "Age"
    
    def test_merge_table_fragments(self):
        """Test merging of table fragments."""
        # Create two tables with same headers (fragments)
        table1 = Table(
            headers=["Name", "Age", "City"],
            rows=[["John", "25", "NYC"]],
            confidence=0.8
        )
        
        table2 = Table(
            headers=["Name", "Age", "City"],
            rows=[["Jane", "30", "LA"]],
            confidence=0.7
        )
        
        # Create a different table (not a fragment)
        table3 = Table(
            headers=["Product", "Price"],
            rows=[["Widget", "$10"]],
            confidence=0.9
        )
        
        tables = [table1, table2, table3]
        merged = self.extractor.merge_table_fragments(tables)
        
        # Should merge table1 and table2, keep table3 separate
        assert len(merged) == 2
        
        # Find the merged table
        merged_table = next((t for t in merged if len(t.rows) == 2), None)
        assert merged_table is not None
        assert merged_table.headers == ["Name", "Age", "City"]
        assert len(merged_table.rows) == 2
    
    def test_are_table_fragments(self):
        """Test fragment detection."""
        table1 = Table(headers=["Name", "Age", "City"], rows=[])
        table2 = Table(headers=["Name", "Age", "City"], rows=[])
        table3 = Table(headers=["Product", "Price"], rows=[])
        
        # Same headers should be fragments
        assert self.extractor._are_table_fragments(table1, table2)
        
        # Different headers should not be fragments
        assert not self.extractor._are_table_fragments(table1, table3)
        
        # Different number of columns should not be fragments
        table4 = Table(headers=["Name", "Age"], rows=[])
        assert not self.extractor._are_table_fragments(table1, table4)
    
    def test_validate_table_structure(self):
        """Test table structure validation."""
        # Test valid table
        valid_table = Table(
            headers=["Name", "Age", "City"],
            rows=[
                ["John", "25", "NYC"],
                ["Jane", "30", "LA"]
            ],
            confidence=0.8
        )
        
        result = self.extractor.validate_table_structure(valid_table)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.confidence > 0.0
    
    def test_validate_table_structure_errors(self):
        """Test table validation with errors."""
        # Test table with no headers
        invalid_table = Table(
            headers=[],
            rows=[["John", "25"]],
            confidence=0.8
        )
        
        result = self.extractor.validate_table_structure(invalid_table)
        
        assert not result.is_valid
        assert "no headers" in result.errors[0].lower()
    
    def test_validate_table_structure_warnings(self):
        """Test table validation with warnings."""
        # Test table with inconsistent columns
        warning_table = Table(
            headers=["Name", "Age", "City"],
            rows=[
                ["John", "25", "NYC"],
                ["Jane", "30"]  # Missing city
            ],
            confidence=0.4  # Low confidence
        )
        
        result = self.extractor.validate_table_structure(warning_table)
        
        assert result.is_valid  # Still valid despite warnings
        assert len(result.warnings) > 0
        assert any("columns" in warning.lower() for warning in result.warnings)
        assert any("confidence" in warning.lower() for warning in result.warnings)
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        # Test with empty OCR result
        empty_ocr = OCRResult("", 0.0, [], [])
        table_regions = [self.create_sample_table_region()]
        
        tables = self.extractor.extract_table_structure(empty_ocr, table_regions)
        assert len(tables) == 0
        
        # Test with empty table regions
        ocr_result = self.create_sample_ocr_result()
        empty_regions = []
        
        tables = self.extractor.extract_table_structure(ocr_result, empty_regions)
        assert len(tables) == 0
    
    def test_complex_table_layout(self):
        """Test extraction from complex table layout."""
        # Create word data for a more complex table with merged cells and irregular spacing
        complex_words = [
            # Header row with merged cell
            WordData("Employee", 0.9, BoundingBox(10, 10, 80, 20, 0.9)),
            WordData("Information", 0.8, BoundingBox(100, 10, 80, 20, 0.8)),
            WordData("Salary", 0.9, BoundingBox(200, 10, 50, 20, 0.9)),
            
            # Subheader row
            WordData("First", 0.8, BoundingBox(10, 35, 35, 15, 0.8)),
            WordData("Last", 0.8, BoundingBox(55, 35, 35, 15, 0.8)),
            WordData("Department", 0.8, BoundingBox(100, 35, 80, 15, 0.8)),
            WordData("Amount", 0.8, BoundingBox(200, 35, 50, 15, 0.8)),
            
            # Data rows
            WordData("John", 0.9, BoundingBox(10, 60, 35, 20, 0.9)),
            WordData("Doe", 0.9, BoundingBox(55, 60, 35, 20, 0.9)),
            WordData("Engineering", 0.8, BoundingBox(100, 60, 80, 20, 0.8)),
            WordData("$75000", 0.8, BoundingBox(200, 60, 50, 20, 0.8)),
        ]
        
        ocr_result = OCRResult(
            text=" ".join(word.text for word in complex_words),
            confidence=0.85,
            bounding_boxes=[],
            word_level_data=complex_words
        )
        
        table_regions = [TableRegion(
            bounding_box=BoundingBox(5, 5, 260, 100, 0.8),
            confidence=0.8
        )]
        
        tables = self.extractor.extract_table_structure(ocr_result, table_regions)
        
        # Should still extract a table despite complexity
        assert len(tables) >= 1
        table = tables[0]
        assert len(table.headers) > 0
        assert len(table.rows) > 0
    
    def test_error_handling(self):
        """Test error handling in table extraction."""
        # Test with invalid input that should raise an exception
        with patch.object(self.extractor, '_detect_table_structure', side_effect=Exception("Test error")):
            ocr_result = self.create_sample_ocr_result()
            table_regions = [self.create_sample_table_region()]
            
            with pytest.raises(OCREngineError):
                self.extractor.extract_table_structure(ocr_result, table_regions)
    
    def test_find_column_index(self):
        """Test column index finding."""
        boundaries = [0, 50, 100, 150]
        
        # Test positions within columns
        assert self.extractor._find_column_index(25, boundaries) == 0
        assert self.extractor._find_column_index(75, boundaries) == 1
        assert self.extractor._find_column_index(125, boundaries) == 2
        
        # Test edge cases
        assert self.extractor._find_column_index(0, boundaries) == 0
        assert self.extractor._find_column_index(49, boundaries) == 0
        assert self.extractor._find_column_index(200, boundaries) == 2  # Beyond last boundary
    
    def test_create_cell_bounding_box(self):
        """Test cell bounding box creation."""
        words = [
            WordData("Hello", 0.9, BoundingBox(10, 20, 30, 15, 0.9)),
            WordData("World", 0.8, BoundingBox(45, 20, 35, 15, 0.8))
        ]
        
        bbox = self.extractor._create_cell_bounding_box(words)
        
        assert bbox.x == 10  # Leftmost x
        assert bbox.y == 20  # Topmost y
        assert bbox.width == 70  # 45 + 35 - 10
        assert bbox.height == 15  # Same height
        assert abs(bbox.confidence - 0.85) < 0.001  # Average confidence (with tolerance)
    
    def test_single_row_table(self):
        """Test handling of single-row table (should not be processed as table)."""
        single_row_words = [
            WordData("Name", 0.9, BoundingBox(10, 10, 50, 20, 0.9)),
            WordData("Age", 0.8, BoundingBox(80, 10, 30, 20, 0.8)),
            WordData("City", 0.9, BoundingBox(130, 10, 40, 20, 0.9)),
        ]
        
        ocr_result = OCRResult(
            text=" ".join(word.text for word in single_row_words),
            confidence=0.85,
            bounding_boxes=[],
            word_level_data=single_row_words
        )
        
        table_regions = [self.create_sample_table_region()]
        
        tables = self.extractor.extract_table_structure(ocr_result, table_regions)
        
        # Should not extract table from single row
        assert len(tables) == 0
    
    def test_large_table_performance(self):
        """Test performance with larger table."""
        # Create a larger table (10x10)
        large_word_data = []
        
        for row in range(10):
            for col in range(10):
                x = col * 60 + 10
                y = row * 25 + 10
                text = f"R{row}C{col}" if row > 0 else f"Header{col}"
                confidence = 0.8 + (row + col) * 0.01  # Vary confidence slightly
                
                large_word_data.append(
                    WordData(text, confidence, BoundingBox(x, y, 50, 20, confidence))
                )
        
        ocr_result = OCRResult(
            text=" ".join(word.text for word in large_word_data),
            confidence=0.85,
            bounding_boxes=[],
            word_level_data=large_word_data
        )
        
        large_region = TableRegion(
            bounding_box=BoundingBox(5, 5, 600, 260, 0.8),
            confidence=0.8
        )
        
        tables = self.extractor.extract_table_structure(ocr_result, [large_region])
        
        # Should successfully extract the large table
        assert len(tables) == 1
        table = tables[0]
        assert len(table.headers) == 10
        assert len(table.rows) == 9  # 10 rows minus header
        assert all(len(row) == 10 for row in table.rows)


if __name__ == "__main__":
    pytest.main([__file__])