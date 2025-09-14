"""
Table structure extraction service for converting OCR results into structured table data.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import cv2
from dataclasses import dataclass
import re

from ..core.models import OCRResult, Table, TableRegion, BoundingBox, WordData, ValidationResult
from ..core.interfaces import TableExtractionInterface
from ..core.exceptions import OCREngineError

logger = logging.getLogger(__name__)


@dataclass
class Cell:
    """Represents a single table cell."""
    content: str
    confidence: float
    row: int
    column: int
    bounding_box: BoundingBox
    is_header: bool = False


@dataclass
class TableStructure:
    """Represents the complete structure of a table."""
    cells: List[Cell]
    num_rows: int
    num_columns: int
    headers: List[str]
    confidence: float


class TableStructureExtractor(TableExtractionInterface):
    """
    Service for extracting structured table data from OCR results.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.header_keywords = {
            'name', 'title', 'description', 'type', 'category', 'id', 'number',
            'date', 'time', 'amount', 'price', 'cost', 'total', 'sum', 'count',
            'quantity', 'status', 'state', 'location', 'address', 'phone', 'email'
        }
        
    def extract_table_structure(self, ocr_result: OCRResult, table_regions: List[TableRegion]) -> List[Table]:
        """
        Extract structured table data from OCR results and table regions.
        
        Args:
            ocr_result: OCR result containing text and word-level data
            table_regions: List of detected table regions
            
        Returns:
            List of extracted tables with structure
        """
        try:
            tables = []
            
            for i, region in enumerate(table_regions):
                logger.info(f"Processing table region {i+1}/{len(table_regions)}")
                
                # Extract words within this table region
                table_words = self._extract_words_in_region(ocr_result.word_level_data, region)
                
                if not table_words:
                    logger.warning(f"No words found in table region {i+1}")
                    continue
                
                # Detect table structure
                structure = self._detect_table_structure(table_words, region)
                
                if structure and structure.num_rows > 0 and structure.num_columns > 0:
                    # Convert to Table object
                    table = self._create_table_from_structure(structure, region, i)
                    tables.append(table)
                    logger.info(f"Extracted table {i+1}: {structure.num_rows}x{structure.num_columns}")
                else:
                    logger.warning(f"Could not detect valid structure for table region {i+1}")
            
            return tables
            
        except Exception as e:
            logger.error(f"Table structure extraction failed: {str(e)}")
            raise OCREngineError(
                f"Table structure extraction failed: {str(e)}",
                error_code="TABLE_STRUCTURE_EXTRACTION_FAILED",
                context={"original_error": str(e)}
            )
    
    def _extract_words_in_region(self, word_data: List[WordData], region: TableRegion) -> List[WordData]:
        """Extract words that fall within the specified table region."""
        region_bbox = region.bounding_box
        words_in_region = []
        
        for word in word_data:
            word_bbox = word.bounding_box
            
            # Check if word center is within the region
            word_center_x = word_bbox.x + word_bbox.width // 2
            word_center_y = word_bbox.y + word_bbox.height // 2
            
            if (region_bbox.x <= word_center_x <= region_bbox.x + region_bbox.width and
                region_bbox.y <= word_center_y <= region_bbox.y + region_bbox.height):
                words_in_region.append(word)
        
        return words_in_region
    
    def _detect_table_structure(self, words: List[WordData], region: TableRegion) -> Optional[TableStructure]:
        """
        Detect the structure of a table from word-level data.
        
        Args:
            words: List of words within the table region
            region: Table region information
            
        Returns:
            TableStructure object or None if structure cannot be detected
        """
        try:
            if not words:
                return None
            
            # Step 1: Detect rows by grouping words with similar y-coordinates
            rows = self._group_words_into_rows(words)
            
            if len(rows) < 2:  # Need at least 2 rows for a table
                logger.debug("Insufficient rows detected for table structure")
                return None
            
            # Step 2: Detect columns by analyzing word positions across rows
            column_boundaries = self._detect_column_boundaries(rows)
            
            if len(column_boundaries) < 2:  # Need at least 1 column (2 boundaries)
                logger.debug("Insufficient columns detected for table structure")
                return None
            
            # Step 3: Create cell grid
            cells = self._create_cell_grid(rows, column_boundaries)
            
            # Step 4: Identify headers
            self._identify_headers(cells, rows)
            
            # Step 5: Calculate overall confidence
            overall_confidence = self._calculate_structure_confidence(cells, rows, column_boundaries)
            
            # Step 6: Extract header names
            headers = self._extract_header_names(cells)
            
            return TableStructure(
                cells=cells,
                num_rows=len(rows),
                num_columns=len(column_boundaries) - 1,
                headers=headers,
                confidence=overall_confidence
            )
            
        except Exception as e:
            logger.error(f"Error detecting table structure: {str(e)}")
            return None
    
    def _group_words_into_rows(self, words: List[WordData]) -> List[List[WordData]]:
        """Group words into rows based on y-coordinate proximity."""
        if not words:
            return []
        
        # Sort words by y-coordinate
        sorted_words = sorted(words, key=lambda w: w.bounding_box.y)
        
        rows = []
        current_row = [sorted_words[0]]
        row_height_threshold = 15  # Pixels - adjust based on typical text height
        
        for word in sorted_words[1:]:
            # Calculate current row's average y-coordinate
            current_row_y = np.mean([w.bounding_box.y for w in current_row])
            
            # Check if word belongs to current row
            if abs(word.bounding_box.y - current_row_y) <= row_height_threshold:
                current_row.append(word)
            else:
                # Start new row
                if current_row:
                    # Sort current row by x-coordinate
                    current_row.sort(key=lambda w: w.bounding_box.x)
                    rows.append(current_row)
                current_row = [word]
        
        # Add last row
        if current_row:
            current_row.sort(key=lambda w: w.bounding_box.x)
            rows.append(current_row)
        
        return rows
    
    def _detect_column_boundaries(self, rows: List[List[WordData]]) -> List[int]:
        """
        Detect column boundaries by analyzing word positions across all rows.
        
        Args:
            rows: List of rows, each containing words
            
        Returns:
            List of x-coordinates representing column boundaries
        """
        if not rows:
            return []
        
        # Collect word start positions (left edges) - these are more reliable for column detection
        word_starts = []
        for row in rows:
            for word in row:
                word_starts.append(word.bounding_box.x)
        
        if not word_starts:
            return []
        
        # Sort positions
        word_starts.sort()
        
        # Find consistent column boundaries using clustering
        boundaries = self._cluster_positions(word_starts, threshold=30)  # Increased threshold
        
        # Ensure we have proper table boundaries
        if boundaries:
            # Calculate table extent
            all_word_ends = []
            for row in rows:
                for word in row:
                    all_word_ends.append(word.bounding_box.x + word.bounding_box.width)
            
            min_x = min(word_starts)
            max_x = max(all_word_ends) if all_word_ends else max(word_starts)
            
            # Add left boundary if not present
            if not boundaries or boundaries[0] > min_x + 10:
                boundaries.insert(0, min_x)
            
            # Add right boundary if not present
            if not boundaries or boundaries[-1] < max_x - 10:
                boundaries.append(max_x)
        
        return boundaries
    
    def _cluster_positions(self, positions: List[int], threshold: int = 20) -> List[int]:
        """
        Cluster similar positions to find column boundaries.
        
        Args:
            positions: List of x-coordinates
            threshold: Maximum distance between positions in the same cluster
            
        Returns:
            List of cluster centers representing column boundaries
        """
        if not positions:
            return []
        
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] <= threshold:
                current_cluster.append(pos)
            else:
                # End current cluster and start new one
                if current_cluster:
                    cluster_center = int(np.mean(current_cluster))
                    clusters.append(cluster_center)
                current_cluster = [pos]
        
        # Add last cluster
        if current_cluster:
            cluster_center = int(np.mean(current_cluster))
            clusters.append(cluster_center)
        
        # Remove clusters that are too close together
        filtered_clusters = []
        for cluster in clusters:
            if not filtered_clusters or cluster - filtered_clusters[-1] > threshold:
                filtered_clusters.append(cluster)
        
        return filtered_clusters
    
    def _create_cell_grid(self, rows: List[List[WordData]], column_boundaries: List[int]) -> List[Cell]:
        """
        Create a grid of cells by assigning words to rows and columns.
        
        Args:
            rows: List of word rows
            column_boundaries: List of column boundary x-coordinates
            
        Returns:
            List of Cell objects
        """
        cells = []
        
        for row_idx, row_words in enumerate(rows):
            # Create cells for this row
            row_cells = [[] for _ in range(len(column_boundaries) - 1)]
            
            # Assign words to columns
            for word in row_words:
                word_center_x = word.bounding_box.x + word.bounding_box.width // 2
                
                # Find which column this word belongs to
                col_idx = self._find_column_index(word_center_x, column_boundaries)
                
                if 0 <= col_idx < len(row_cells):
                    row_cells[col_idx].append(word)
            
            # Create Cell objects for each column in this row
            for col_idx, cell_words in enumerate(row_cells):
                if cell_words:
                    # Combine words in this cell
                    cell_content = ' '.join(word.text for word in cell_words)
                    cell_confidence = np.mean([word.confidence for word in cell_words])
                    
                    # Create bounding box for the cell
                    cell_bbox = self._create_cell_bounding_box(cell_words)
                    
                    cell = Cell(
                        content=cell_content.strip(),
                        confidence=cell_confidence,
                        row=row_idx,
                        column=col_idx,
                        bounding_box=cell_bbox
                    )
                    cells.append(cell)
                else:
                    # Empty cell
                    cell = Cell(
                        content="",
                        confidence=0.0,
                        row=row_idx,
                        column=col_idx,
                        bounding_box=BoundingBox(
                            x=column_boundaries[col_idx],
                            y=0,  # Will be updated if needed
                            width=column_boundaries[col_idx + 1] - column_boundaries[col_idx],
                            height=0,
                            confidence=0.0
                        )
                    )
                    cells.append(cell)
        
        return cells
    
    def _find_column_index(self, x_position: int, column_boundaries: List[int]) -> int:
        """Find which column a given x-position belongs to."""
        for i in range(len(column_boundaries) - 1):
            if column_boundaries[i] <= x_position < column_boundaries[i + 1]:
                return i
        
        # If not found, assign to the last column
        return len(column_boundaries) - 2 if len(column_boundaries) > 1 else 0
    
    def _create_cell_bounding_box(self, words: List[WordData]) -> BoundingBox:
        """Create a bounding box that encompasses all words in a cell."""
        if not words:
            return BoundingBox(0, 0, 0, 0, 0.0)
        
        min_x = min(word.bounding_box.x for word in words)
        max_x = max(word.bounding_box.x + word.bounding_box.width for word in words)
        min_y = min(word.bounding_box.y for word in words)
        max_y = max(word.bounding_box.y + word.bounding_box.height for word in words)
        
        avg_confidence = np.mean([word.confidence for word in words])
        
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            confidence=avg_confidence
        )
    
    def _identify_headers(self, cells: List[Cell], rows: List[List[WordData]]) -> None:
        """
        Identify header cells based on position and content analysis.
        
        Args:
            cells: List of cells to analyze
            rows: Original row data for additional context
        """
        if not cells:
            return
        
        # Strategy 1: First row is likely headers
        first_row_cells = [cell for cell in cells if cell.row == 0]
        
        # Strategy 2: Look for header-like content
        header_found = False
        for cell in first_row_cells:
            if self._is_likely_header(cell.content):
                cell.is_header = True
                header_found = True
        
        # Strategy 3: If no headers found in first row, look for formatting clues
        if not header_found:
            for cell in first_row_cells:
                if self._has_header_formatting(cell.content):
                    cell.is_header = True
                    header_found = True
        
        # Strategy 4: If still no headers, mark first row as headers by default
        if not header_found:
            for cell in first_row_cells:
                if cell.content.strip():  # Only non-empty cells
                    cell.is_header = True
    
    def _is_likely_header(self, content: str) -> bool:
        """Check if content is likely a table header."""
        if not content or len(content.strip()) == 0:
            return False
        
        content_lower = content.lower().strip()
        
        # Check for common header keywords
        for keyword in self.header_keywords:
            if keyword in content_lower:
                return True
        
        # Check for header-like patterns
        # Headers often end with colons
        if content.strip().endswith(':'):
            return True
        
        # Headers are often short and descriptive, but not personal names
        if len(content.strip()) <= 20 and not content.strip().isdigit():
            # Avoid common personal names that might appear in data
            personal_name_patterns = [
                r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # "John Smith" pattern
                r'^\d+$',  # Pure numbers
                r'^\$\d+',  # Currency values
            ]
            
            for pattern in personal_name_patterns:
                if re.match(pattern, content.strip()):
                    return False
            
            # Headers often contain only letters and spaces
            if re.match(r'^[a-zA-Z\s]+$', content.strip()):
                return True
        
        return False
    
    def _has_header_formatting(self, content: str) -> bool:
        """Check if content has formatting that suggests it's a header."""
        if not content:
            return False
        
        # Check if all uppercase (common header formatting)
        if content.isupper() and len(content.strip()) > 1:
            return True
        
        # Check if title case
        if content.istitle():
            return True
        
        return False
    
    def _calculate_structure_confidence(self, cells: List[Cell], rows: List[List[WordData]], 
                                      column_boundaries: List[int]) -> float:
        """Calculate confidence score for the detected table structure."""
        if not cells or not rows:
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Average cell confidence
        cell_confidences = [cell.confidence for cell in cells if cell.confidence > 0]
        if cell_confidences:
            confidence_factors.append(np.mean(cell_confidences))
        
        # Factor 2: Structure regularity (consistent number of columns per row)
        row_column_counts = {}
        for cell in cells:
            if cell.row not in row_column_counts:
                row_column_counts[cell.row] = 0
            if cell.content.strip():  # Only count non-empty cells
                row_column_counts[cell.row] += 1
        
        if row_column_counts:
            column_counts = list(row_column_counts.values())
            if column_counts:
                # Higher confidence for more consistent column counts
                consistency = 1.0 - (np.std(column_counts) / np.mean(column_counts)) if np.mean(column_counts) > 0 else 0.0
                confidence_factors.append(max(0.0, consistency))
        
        # Factor 3: Presence of headers
        has_headers = any(cell.is_header for cell in cells)
        if has_headers:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Factor 4: Table size (larger tables are more likely to be real tables)
        num_rows = len(rows)
        num_cols = len(column_boundaries) - 1 if column_boundaries else 0
        
        if num_rows >= 3 and num_cols >= 2:
            confidence_factors.append(0.9)
        elif num_rows >= 2 and num_cols >= 2:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Calculate overall confidence
        if confidence_factors:
            return min(1.0, np.mean(confidence_factors))
        else:
            return 0.0
    
    def _extract_header_names(self, cells: List[Cell]) -> List[str]:
        """Extract header names from identified header cells."""
        headers = []
        
        # Get header cells sorted by column
        header_cells = [cell for cell in cells if cell.is_header]
        header_cells.sort(key=lambda c: c.column)
        
        # Group by column to handle multi-row headers
        column_headers = {}
        for cell in header_cells:
            if cell.column not in column_headers:
                column_headers[cell.column] = []
            column_headers[cell.column].append(cell.content.strip())
        
        # Create header names
        max_column = max(column_headers.keys()) if column_headers else -1
        
        for col in range(max_column + 1):
            if col in column_headers:
                # Combine multiple header parts if present
                header_parts = [part for part in column_headers[col] if part]
                if header_parts:
                    headers.append(' '.join(header_parts))
                else:
                    headers.append(f"Column_{col + 1}")
            else:
                headers.append(f"Column_{col + 1}")
        
        return headers
    
    def _create_table_from_structure(self, structure: TableStructure, region: TableRegion, 
                                   table_index: int) -> Table:
        """Convert TableStructure to Table object."""
        # Create rows from cells
        rows = []
        
        # Group cells by row
        row_cells = {}
        for cell in structure.cells:
            if cell.row not in row_cells:
                row_cells[cell.row] = {}
            row_cells[cell.row][cell.column] = cell
        
        # Create row data (skip header row)
        header_row = 0  # Assume first row is header
        for row_idx in sorted(row_cells.keys()):
            if row_idx == header_row:
                continue  # Skip header row
            
            row_data = []
            max_col = max(row_cells[row_idx].keys()) if row_cells[row_idx] else -1
            
            for col_idx in range(max_col + 1):
                if col_idx in row_cells[row_idx]:
                    row_data.append(row_cells[row_idx][col_idx].content)
                else:
                    row_data.append("")  # Empty cell
            
            rows.append(row_data)
        
        return Table(
            headers=structure.headers,
            rows=rows,
            confidence=structure.confidence,
            region=region,
            metadata={
                'num_rows': structure.num_rows,
                'num_columns': structure.num_columns,
                'table_index': table_index,
                'extraction_method': 'structure_analysis'
            }
        )
    
    def merge_table_fragments(self, tables: List[Table]) -> List[Table]:
        """
        Merge table fragments that span across multiple pages or regions.
        
        Args:
            tables: List of tables that might be fragments
            
        Returns:
            List of merged tables
        """
        if len(tables) <= 1:
            return tables
        
        try:
            merged_tables = []
            used_indices = set()
            
            for i, table in enumerate(tables):
                if i in used_indices:
                    continue
                
                # Find tables that might be fragments of this table
                fragments = [table]
                fragment_indices = {i}
                
                for j, other_table in enumerate(tables[i+1:], start=i+1):
                    if j in used_indices:
                        continue
                    
                    if self._are_table_fragments(table, other_table):
                        fragments.append(other_table)
                        fragment_indices.add(j)
                
                # Merge fragments if found
                if len(fragments) > 1:
                    merged_table = self._merge_table_fragments(fragments)
                    merged_tables.append(merged_table)
                    used_indices.update(fragment_indices)
                else:
                    merged_tables.append(table)
                    used_indices.add(i)
            
            logger.info(f"Merged {len(tables)} tables into {len(merged_tables)} tables")
            return merged_tables
            
        except Exception as e:
            logger.error(f"Table fragment merging failed: {str(e)}")
            return tables  # Return original tables if merging fails
    
    def _are_table_fragments(self, table1: Table, table2: Table) -> bool:
        """Check if two tables are fragments of the same logical table."""
        # Check if headers match
        if len(table1.headers) != len(table2.headers):
            return False
        
        # Check header similarity
        header_matches = 0
        for h1, h2 in zip(table1.headers, table2.headers):
            if h1.lower().strip() == h2.lower().strip():
                header_matches += 1
        
        header_similarity = header_matches / len(table1.headers) if table1.headers else 0
        
        # Tables are fragments if headers are very similar
        return header_similarity >= 0.8
    
    def _merge_table_fragments(self, fragments: List[Table]) -> Table:
        """Merge multiple table fragments into a single table."""
        if not fragments:
            return None
        
        if len(fragments) == 1:
            return fragments[0]
        
        # Use the first fragment as the base
        base_table = fragments[0]
        
        # Merge rows from all fragments
        all_rows = list(base_table.rows)
        
        for fragment in fragments[1:]:
            all_rows.extend(fragment.rows)
        
        # Calculate merged confidence
        confidences = [table.confidence for table in fragments]
        merged_confidence = np.mean(confidences)
        
        # Create merged metadata
        merged_metadata = dict(base_table.metadata)
        merged_metadata['merged_from'] = len(fragments)
        merged_metadata['fragment_confidences'] = confidences
        
        return Table(
            headers=base_table.headers,
            rows=all_rows,
            confidence=merged_confidence,
            region=base_table.region,  # Use first region
            metadata=merged_metadata
        )
    
    def validate_table_structure(self, table: Table) -> ValidationResult:
        """
        Validate extracted table structure and data quality.
        
        Args:
            table: Table to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        try:
            errors = []
            warnings = []
            
            # Check basic structure
            if not table.headers:
                errors.append("Table has no headers")
            
            if not table.rows:
                errors.append("Table has no data rows")
            
            if table.headers and table.rows:
                # Check column consistency
                expected_columns = len(table.headers)
                
                for i, row in enumerate(table.rows):
                    if len(row) != expected_columns:
                        warnings.append(f"Row {i+1} has {len(row)} columns, expected {expected_columns}")
            
            # Check confidence levels
            if table.confidence < 0.3:
                warnings.append(f"Low table confidence: {table.confidence:.2f}")
            elif table.confidence < 0.6:
                warnings.append(f"Moderate table confidence: {table.confidence:.2f}")
            
            # Check for empty content
            if table.rows:
                empty_rows = sum(1 for row in table.rows if all(not cell.strip() for cell in row))
                if empty_rows > 0:
                    warnings.append(f"Found {empty_rows} empty rows")
                
                # Check for data quality issues
                total_cells = sum(len(row) for row in table.rows)
                empty_cells = sum(1 for row in table.rows for cell in row if not cell.strip())
                
                if total_cells > 0:
                    empty_ratio = empty_cells / total_cells
                    if empty_ratio > 0.5:
                        warnings.append(f"High empty cell ratio: {empty_ratio:.1%}")
            
            # Check header quality
            if table.headers:
                generic_headers = sum(1 for header in table.headers if header.startswith('Column_'))
                if generic_headers > 0:
                    warnings.append(f"Found {generic_headers} generic column headers")
            
            # Determine overall validation result
            is_valid = len(errors) == 0
            confidence = max(0.0, table.confidence - len(warnings) * 0.1)
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Table validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                confidence=0.0
            )