"""
Table validation and correction system for ensuring data quality and accuracy.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from ..core.models import Table, ValidationResult, DataType, ColumnInfo
from ..core.interfaces import ValidationInterface
from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class CellValidation:
    """Validation result for a single table cell."""
    row: int
    column: int
    original_value: str
    suggested_value: Optional[str] = None
    confidence: float = 0.0
    issues: List[str] = field(default_factory=list)
    data_type: Optional[DataType] = None


@dataclass
class TableValidationReport:
    """Comprehensive validation report for a table."""
    table_id: str
    overall_confidence: float
    is_valid: bool
    structure_issues: List[str] = field(default_factory=list)
    data_quality_issues: List[str] = field(default_factory=list)
    cell_validations: List[CellValidation] = field(default_factory=list)
    suggested_corrections: Dict[str, Any] = field(default_factory=dict)
    validation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CorrectionAction:
    """Represents a correction action for table data."""
    action_type: str  # 'replace', 'delete', 'insert', 'merge'
    row: int
    column: int
    original_value: str
    new_value: str
    confidence: float
    reason: str


class TableValidator(ValidationInterface):
    """
    Comprehensive table validation and correction system.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.data_type_patterns = {
            DataType.NUMBER: [
                r'^\d+$',  # Integer
                r'^\d+\.\d+$',  # Decimal
                r'^-?\d+\.?\d*$',  # Signed number
                r'^\d{1,3}(,\d{3})*(\.\d+)?$'  # Number with commas
            ],
            DataType.CURRENCY: [
                r'^\$\d+\.?\d*$',  # Dollar amounts
                r'^\d+\.?\d*\s*(USD|EUR|GBP)$',  # Currency codes
                r'^[€£¥]\d+\.?\d*$'  # Currency symbols
            ],
            DataType.PERCENTAGE: [
                r'^\d+\.?\d*%$',  # Percentage
                r'^0\.\d+$'  # Decimal percentage
            ],
            DataType.DATE: [
                r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
                r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                r'^\d{1,2}-\d{1,2}-\d{4}$',  # DD-MM-YYYY
                r'^[A-Za-z]{3}\s+\d{1,2},?\s+\d{4}$'  # Month DD, YYYY
            ],
            DataType.BOOLEAN: [
                r'^(true|false)$',  # Boolean
                r'^(yes|no)$',  # Yes/No
                r'^(y|n)$',  # Y/N
                r'^(1|0)$'  # Binary
            ]
        }
        
        self.common_corrections = {
            # Common OCR mistakes
            'O': '0',  # Letter O to zero
            'l': '1',  # Lowercase L to one
            'I': '1',  # Uppercase I to one
            'S': '5',  # S to 5 in numbers
            'G': '6',  # G to 6 in numbers
            'B': '8',  # B to 8 in numbers
        }
        
    def validate_table_structure(self, table: Table) -> TableValidationReport:
        """
        Perform comprehensive validation of table structure and data quality.
        
        Args:
            table: Table to validate
            
        Returns:
            TableValidationReport with detailed validation results
        """
        try:
            logger.info(f"Validating table structure for table {table.id}")
            
            report = TableValidationReport(
                table_id=table.id,
                overall_confidence=table.confidence,
                is_valid=True
            )
            
            # Validate basic structure
            self._validate_basic_structure(table, report)
            
            # Validate headers
            self._validate_headers(table, report)
            
            # Validate data consistency
            self._validate_data_consistency(table, report)
            
            # Validate individual cells
            self._validate_cells(table, report)
            
            # Calculate overall validation score
            self._calculate_validation_score(report)
            
            # Generate correction suggestions
            self._generate_correction_suggestions(table, report)
            
            logger.info(f"Table validation completed. Overall confidence: {report.overall_confidence:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"Table validation failed: {str(e)}")
            raise ValidationError(
                f"Table validation failed: {str(e)}",
                error_code="TABLE_VALIDATION_FAILED",
                context={"table_id": table.id, "original_error": str(e)}
            )
    
    def _validate_basic_structure(self, table: Table, report: TableValidationReport) -> None:
        """Validate basic table structure requirements."""
        # Check for headers
        if not table.headers:
            report.structure_issues.append("Table has no headers")
            report.is_valid = False
        
        # Check for data rows
        if not table.rows:
            report.structure_issues.append("Table has no data rows")
            report.is_valid = False
        
        # Check minimum table size
        if len(table.rows) < 1:
            report.structure_issues.append("Table must have at least one data row")
        
        if table.headers and len(table.headers) < 2:
            report.structure_issues.append("Table should have at least two columns")
        
        # Check column consistency
        if table.headers and table.rows:
            expected_columns = len(table.headers)
            inconsistent_rows = []
            
            for i, row in enumerate(table.rows):
                if len(row) != expected_columns:
                    inconsistent_rows.append(i + 1)
            
            if inconsistent_rows:
                report.structure_issues.append(
                    f"Rows {inconsistent_rows} have inconsistent column counts"
                )
    
    def _validate_headers(self, table: Table, report: TableValidationReport) -> None:
        """Validate table headers for quality and consistency."""
        if not table.headers:
            return
        
        # Check for duplicate headers
        header_counts = {}
        for header in table.headers:
            header_lower = header.lower().strip()
            header_counts[header_lower] = header_counts.get(header_lower, 0) + 1
        
        duplicates = [header for header, count in header_counts.items() if count > 1]
        if duplicates:
            report.structure_issues.append(f"Duplicate headers found: {duplicates}")
        
        # Check for empty headers
        empty_headers = [i for i, header in enumerate(table.headers) if not header.strip()]
        if empty_headers:
            report.structure_issues.append(f"Empty headers at positions: {empty_headers}")
        
        # Check for generic headers
        generic_headers = [header for header in table.headers if header.startswith('Column_')]
        if generic_headers:
            report.data_quality_issues.append(
                f"Generic headers detected: {len(generic_headers)} columns"
            )
        
        # Check header length and format
        for i, header in enumerate(table.headers):
            if len(header.strip()) > 50:
                report.data_quality_issues.append(f"Header {i+1} is unusually long")
            
            if header.strip() != header:
                report.data_quality_issues.append(f"Header {i+1} has leading/trailing whitespace")
    
    def _validate_data_consistency(self, table: Table, report: TableValidationReport) -> None:
        """Validate data consistency across columns and rows."""
        if not table.rows or not table.headers:
            return
        
        # Analyze each column for data type consistency
        for col_idx, header in enumerate(table.headers):
            column_values = []
            
            for row in table.rows:
                if col_idx < len(row):
                    value = row[col_idx].strip()
                    if value:  # Skip empty values
                        column_values.append(value)
            
            if column_values:
                # Detect data type for this column
                detected_type = self._detect_column_data_type(column_values)
                
                # Check for type consistency
                inconsistent_values = []
                for i, row in enumerate(table.rows):
                    if col_idx < len(row):
                        value = row[col_idx].strip()
                        if value and not self._matches_data_type(value, detected_type):
                            inconsistent_values.append((i + 1, value))
                
                if inconsistent_values and len(inconsistent_values) > len(column_values) * 0.1:
                    report.data_quality_issues.append(
                        f"Column '{header}' has inconsistent data types. "
                        f"Expected {detected_type.value}, found {len(inconsistent_values)} inconsistent values"
                    )
        
        # Check for completely empty rows
        empty_rows = []
        for i, row in enumerate(table.rows):
            if all(not cell.strip() for cell in row):
                empty_rows.append(i + 1)
        
        if empty_rows:
            report.data_quality_issues.append(f"Empty rows found: {empty_rows}")
        
        # Check for duplicate rows
        row_signatures = {}
        duplicate_rows = []
        
        for i, row in enumerate(table.rows):
            row_signature = '|'.join(cell.strip().lower() for cell in row)
            if row_signature in row_signatures:
                duplicate_rows.append((i + 1, row_signatures[row_signature] + 1))
            else:
                row_signatures[row_signature] = i
        
        if duplicate_rows:
            report.data_quality_issues.append(f"Duplicate rows found: {duplicate_rows}")
    
    def _validate_cells(self, table: Table, report: TableValidationReport) -> None:
        """Validate individual cells for data quality and suggest corrections."""
        if not table.rows or not table.headers:
            return
        
        # Detect column data types first
        column_types = {}
        for col_idx, header in enumerate(table.headers):
            column_values = [
                row[col_idx].strip() for row in table.rows 
                if col_idx < len(row) and row[col_idx].strip()
            ]
            if column_values:
                column_types[col_idx] = self._detect_column_data_type(column_values)
        
        # Validate each cell
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell_value in enumerate(row):
                if col_idx >= len(table.headers):
                    continue
                
                cell_validation = CellValidation(
                    row=row_idx,
                    column=col_idx,
                    original_value=cell_value,
                    confidence=1.0
                )
                
                # Check for common OCR errors
                suggested_value = self._suggest_cell_correction(
                    cell_value, column_types.get(col_idx)
                )
                
                if suggested_value != cell_value:
                    cell_validation.suggested_value = suggested_value
                    cell_validation.confidence = 0.8
                    cell_validation.issues.append("Potential OCR error detected")
                
                # Check data type consistency
                if col_idx in column_types:
                    expected_type = column_types[col_idx]
                    cell_validation.data_type = expected_type
                    
                    if cell_value.strip() and not self._matches_data_type(cell_value, expected_type):
                        cell_validation.issues.append(f"Value doesn't match expected type: {expected_type.value}")
                        cell_validation.confidence *= 0.7
                
                # Check for suspicious patterns
                if self._has_suspicious_patterns(cell_value):
                    cell_validation.issues.append("Suspicious character patterns detected")
                    cell_validation.confidence *= 0.8
                
                # Only add cell validation if there are issues or suggestions
                if cell_validation.issues or cell_validation.suggested_value:
                    report.cell_validations.append(cell_validation)
    
    def _detect_column_data_type(self, values: List[str]) -> DataType:
        """Detect the most likely data type for a column based on its values."""
        if not values:
            return DataType.TEXT
        
        type_scores = {data_type: 0 for data_type in DataType}
        
        for value in values:
            value = value.strip()
            if not value:
                continue
            
            for data_type, patterns in self.data_type_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, value, re.IGNORECASE):
                        type_scores[data_type] += 1
                        break
        
        # Find the type with the highest score
        best_type = max(type_scores.items(), key=lambda x: x[1])
        
        # If no specific type matches well, default to TEXT
        if best_type[1] < len(values) * 0.5:
            return DataType.TEXT
        
        return best_type[0]
    
    def _matches_data_type(self, value: str, data_type: DataType) -> bool:
        """Check if a value matches the expected data type."""
        if not value.strip():
            return True  # Empty values are generally acceptable
        
        if data_type not in self.data_type_patterns:
            return True  # If no patterns defined, accept any value
        
        patterns = self.data_type_patterns[data_type]
        return any(re.match(pattern, value.strip(), re.IGNORECASE) for pattern in patterns)
    
    def _suggest_cell_correction(self, value: str, expected_type: Optional[DataType] = None) -> str:
        """Suggest corrections for common OCR errors in cell values."""
        if not value.strip():
            return value
        
        corrected = value
        
        # Apply common OCR corrections for numeric data
        if expected_type in [DataType.NUMBER, DataType.CURRENCY, DataType.PERCENTAGE]:
            for mistake, correction in self.common_corrections.items():
                # Only replace if it would make the value more numeric
                test_value = corrected.replace(mistake, correction)
                if self._is_more_numeric(test_value, corrected):
                    corrected = test_value
        
        # Clean up common formatting issues
        corrected = self._clean_formatting(corrected, expected_type)
        
        return corrected
    
    def _is_more_numeric(self, value1: str, value2: str) -> bool:
        """Check if value1 is more numeric than value2."""
        numeric_chars1 = sum(1 for c in value1 if c.isdigit() or c in '.,')
        numeric_chars2 = sum(1 for c in value2 if c.isdigit() or c in '.,')
        
        return numeric_chars1 > numeric_chars2
    
    def _clean_formatting(self, value: str, expected_type: Optional[DataType] = None) -> str:
        """Clean common formatting issues in cell values."""
        cleaned = value.strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Fix common punctuation issues
        if expected_type == DataType.NUMBER:
            # Remove spaces in numbers
            if re.match(r'^\d+\s+\d+$', cleaned):
                cleaned = cleaned.replace(' ', '')
            
            # Fix decimal separators
            if cleaned.count('.') > 1 and cleaned.count(',') == 0:
                # Multiple dots, likely thousands separator
                parts = cleaned.split('.')
                if len(parts[-1]) <= 2:  # Last part looks like cents
                    cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
        
        elif expected_type == DataType.CURRENCY:
            # Standardize currency format
            cleaned = re.sub(r'[^\d.,\$€£¥]', '', cleaned)
        
        return cleaned
    
    def _has_suspicious_patterns(self, value: str) -> bool:
        """Check for suspicious character patterns that might indicate OCR errors."""
        if not value.strip():
            return False
        
        # Check for mixed character types in unexpected ways
        suspicious_patterns = [
            r'[0-9][a-zA-Z][0-9]',  # Number-letter-number
            r'[a-zA-Z][0-9][a-zA-Z]',  # Letter-number-letter
            r'[Il1]{2,}',  # Multiple I, l, or 1 characters
            r'[O0]{3,}',  # Multiple O or 0 characters
            r'[^\w\s.,;:!?()-]',  # Unusual special characters
        ]
        
        return any(re.search(pattern, value) for pattern in suspicious_patterns)
    
    def _calculate_validation_score(self, report: TableValidationReport) -> None:
        """Calculate overall validation confidence score."""
        base_confidence = report.overall_confidence
        
        # Penalize for structural issues
        structure_penalty = len(report.structure_issues) * 0.2
        
        # Penalize for data quality issues
        quality_penalty = len(report.data_quality_issues) * 0.1
        
        # Penalize for cell-level issues
        cell_penalty = len(report.cell_validations) * 0.05
        
        # Calculate final confidence
        final_confidence = max(0.0, base_confidence - structure_penalty - quality_penalty - cell_penalty)
        
        # Update report
        report.overall_confidence = final_confidence
        
        # Mark as invalid if confidence is too low or there are critical issues
        if final_confidence < self.confidence_threshold or report.structure_issues:
            report.is_valid = False
    
    def _generate_correction_suggestions(self, table: Table, report: TableValidationReport) -> None:
        """Generate high-level correction suggestions for the table."""
        suggestions = {}
        
        # Header corrections
        if any("duplicate headers" in issue.lower() for issue in report.structure_issues):
            suggestions["headers"] = "Consider renaming duplicate headers to make them unique"
        
        if any("generic headers" in issue.lower() for issue in report.data_quality_issues):
            suggestions["headers"] = "Consider providing more descriptive column names"
        
        # Data corrections
        if report.cell_validations:
            cell_corrections = len([cv for cv in report.cell_validations if cv.suggested_value])
            if cell_corrections > 0:
                suggestions["cells"] = f"Apply {cell_corrections} suggested cell corrections"
        
        # Structure corrections
        if any("empty rows" in issue.lower() for issue in report.data_quality_issues):
            suggestions["structure"] = "Consider removing empty rows"
        
        if any("duplicate rows" in issue.lower() for issue in report.data_quality_issues):
            suggestions["structure"] = "Consider removing or consolidating duplicate rows"
        
        report.suggested_corrections = suggestions
    
    def apply_corrections(self, table: Table, corrections: List[CorrectionAction]) -> Table:
        """
        Apply a list of correction actions to a table.
        
        Args:
            table: Original table
            corrections: List of correction actions to apply
            
        Returns:
            New table with corrections applied
        """
        try:
            logger.info(f"Applying {len(corrections)} corrections to table {table.id}")
            
            # Create a copy of the table
            corrected_table = Table(
                headers=table.headers.copy(),
                rows=[row.copy() for row in table.rows],
                confidence=table.confidence,
                region=table.region,
                metadata=table.metadata.copy()
            )
            
            # Sort corrections by row and column to apply them consistently
            sorted_corrections = sorted(corrections, key=lambda c: (c.row, c.column))
            
            applied_corrections = 0
            
            for correction in sorted_corrections:
                try:
                    if correction.action_type == "replace":
                        if (0 <= correction.row < len(corrected_table.rows) and
                            0 <= correction.column < len(corrected_table.rows[correction.row])):
                            
                            corrected_table.rows[correction.row][correction.column] = correction.new_value
                            applied_corrections += 1
                    
                    elif correction.action_type == "delete":
                        if correction.row < len(corrected_table.rows):
                            if correction.column == -1:  # Delete entire row
                                del corrected_table.rows[correction.row]
                            else:  # Delete cell (set to empty)
                                if correction.column < len(corrected_table.rows[correction.row]):
                                    corrected_table.rows[correction.row][correction.column] = ""
                            applied_corrections += 1
                    
                    # Note: Insert and merge operations would require more complex logic
                    # and are not implemented in this basic version
                    
                except Exception as e:
                    logger.warning(f"Failed to apply correction: {correction}. Error: {str(e)}")
            
            # Update metadata
            corrected_table.metadata["corrections_applied"] = applied_corrections
            corrected_table.metadata["correction_timestamp"] = datetime.now().isoformat()
            
            # Recalculate confidence based on corrections
            if applied_corrections > 0:
                correction_boost = min(0.2, applied_corrections * 0.05)
                corrected_table.confidence = min(1.0, corrected_table.confidence + correction_boost)
            
            logger.info(f"Applied {applied_corrections} corrections successfully")
            return corrected_table
            
        except Exception as e:
            logger.error(f"Failed to apply corrections: {str(e)}")
            raise ValidationError(
                f"Failed to apply corrections: {str(e)}",
                error_code="CORRECTION_APPLICATION_FAILED",
                context={"table_id": table.id, "corrections_count": len(corrections)}
            )
    
    def validate_table(self, table: Table) -> ValidationResult:
        """
        Validate table and return basic ValidationResult for interface compatibility.
        
        Args:
            table: Table to validate
            
        Returns:
            ValidationResult with basic validation information
        """
        try:
            report = self.validate_table_structure(table)
            
            return ValidationResult(
                is_valid=report.is_valid,
                errors=report.structure_issues,
                warnings=report.data_quality_issues,
                confidence=report.overall_confidence
            )
            
        except Exception as e:
            logger.error(f"Table validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                confidence=0.0
            )