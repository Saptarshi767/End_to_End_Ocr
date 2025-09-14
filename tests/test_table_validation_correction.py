"""
Tests for table validation and correction system.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.core.models import Table, TableRegion, BoundingBox, DataType, ValidationResult
from src.data_processing.table_validator import (
    TableValidator, TableValidationReport, CellValidation, CorrectionAction
)
from src.data_processing.correction_interface import (
    TableCorrectionInterface, CorrectionSession, CorrectionSuggestion
)


class TestTableValidator:
    """Test cases for TableValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a TableValidator instance for testing."""
        return TableValidator(confidence_threshold=0.7)
    
    @pytest.fixture
    def sample_table(self):
        """Create a sample table for testing."""
        return Table(
            headers=["Name", "Age", "Salary", "Date"],
            rows=[
                ["John Doe", "30", "$50,000", "2023-01-15"],
                ["Jane Smith", "25", "$45,000", "2023-02-20"],
                ["Bob Johnson", "35", "$60,000", "2023-03-10"]
            ],
            confidence=0.85,
            region=TableRegion(
                bounding_box=BoundingBox(0, 0, 500, 200, 0.9),
                confidence=0.9
            )
        )
    
    @pytest.fixture
    def problematic_table(self):
        """Create a table with various issues for testing."""
        return Table(
            headers=["Name", "Age", "Age", ""],  # Duplicate and empty header
            rows=[
                ["John Doe", "3O", "$5O,OOO", "2O23-O1-15"],  # OCR errors (O instead of 0)
                ["", "", "", ""],  # Empty row
                ["Jane Smith", "25", "$45,000", "2023-02-20"],
                ["Jane Smith", "25", "$45,000", "2023-02-20"],  # Duplicate row
                ["Bob", "thirty-five", "60000", "March 10, 2023"],  # Inconsistent formats
            ],
            confidence=0.65,
            region=TableRegion(
                bounding_box=BoundingBox(0, 0, 500, 300, 0.7),
                confidence=0.7
            )
        )
    
    def test_validate_basic_structure_valid_table(self, validator, sample_table):
        """Test validation of a well-structured table."""
        report = validator.validate_table_structure(sample_table)
        
        assert report.table_id == sample_table.id
        assert report.is_valid
        assert len(report.structure_issues) == 0
        assert report.overall_confidence >= 0.7
    
    def test_validate_basic_structure_no_headers(self, validator):
        """Test validation of table without headers."""
        table = Table(
            headers=[],
            rows=[["data1", "data2"], ["data3", "data4"]],
            confidence=0.8
        )
        
        report = validator.validate_table_structure(table)
        
        assert not report.is_valid
        assert "no headers" in str(report.structure_issues).lower()
    
    def test_validate_basic_structure_no_rows(self, validator):
        """Test validation of table without data rows."""
        table = Table(
            headers=["Col1", "Col2"],
            rows=[],
            confidence=0.8
        )
        
        report = validator.validate_table_structure(table)
        
        assert not report.is_valid
        assert "no data rows" in str(report.structure_issues).lower()
    
    def test_validate_headers_duplicates(self, validator, problematic_table):
        """Test detection of duplicate headers."""
        report = validator.validate_table_structure(problematic_table)
        
        duplicate_issues = [issue for issue in report.structure_issues 
                          if "duplicate" in issue.lower()]
        assert len(duplicate_issues) > 0
    
    def test_validate_headers_empty(self, validator, problematic_table):
        """Test detection of empty headers."""
        report = validator.validate_table_structure(problematic_table)
        
        empty_issues = [issue for issue in report.structure_issues 
                       if "empty" in issue.lower()]
        assert len(empty_issues) > 0
    
    def test_detect_column_data_type_number(self, validator):
        """Test data type detection for numeric columns."""
        values = ["123", "456.78", "1,234", "-567"]
        detected_type = validator._detect_column_data_type(values)
        assert detected_type == DataType.NUMBER
    
    def test_detect_column_data_type_currency(self, validator):
        """Test data type detection for currency columns."""
        values = ["$123.45", "$1,234", "$567.89"]
        detected_type = validator._detect_column_data_type(values)
        assert detected_type == DataType.CURRENCY
    
    def test_detect_column_data_type_date(self, validator):
        """Test data type detection for date columns."""
        values = ["2023-01-15", "2023-02-20", "2023-03-10"]
        detected_type = validator._detect_column_data_type(values)
        assert detected_type == DataType.DATE
    
    def test_detect_column_data_type_text(self, validator):
        """Test data type detection for text columns."""
        values = ["John Doe", "Jane Smith", "Bob Johnson"]
        detected_type = validator._detect_column_data_type(values)
        assert detected_type == DataType.TEXT
    
    def test_suggest_cell_correction_ocr_errors(self, validator):
        """Test OCR error correction suggestions."""
        # Test O to 0 correction in numbers
        corrected = validator._suggest_cell_correction("5O,OOO", DataType.CURRENCY)
        assert "50,000" in corrected or corrected.count('0') > "5O,OOO".count('0')
        
        # Test I to 1 correction
        corrected = validator._suggest_cell_correction("I23", DataType.NUMBER)
        assert corrected != "I23"  # Should be different
    
    def test_validate_data_consistency_empty_rows(self, validator, problematic_table):
        """Test detection of empty rows."""
        report = validator.validate_table_structure(problematic_table)
        
        empty_row_issues = [issue for issue in report.data_quality_issues 
                           if "empty rows" in issue.lower()]
        assert len(empty_row_issues) > 0
    
    def test_validate_data_consistency_duplicate_rows(self, validator, problematic_table):
        """Test detection of duplicate rows."""
        report = validator.validate_table_structure(problematic_table)
        
        duplicate_row_issues = [issue for issue in report.data_quality_issues 
                               if "duplicate rows" in issue.lower()]
        assert len(duplicate_row_issues) > 0
    
    def test_validate_cells_ocr_errors(self, validator, problematic_table):
        """Test cell-level validation and OCR error detection."""
        report = validator.validate_table_structure(problematic_table)
        
        # Should detect OCR errors in cells with O instead of 0
        ocr_errors = [cv for cv in report.cell_validations 
                     if cv.suggested_value and cv.suggested_value != cv.original_value]
        assert len(ocr_errors) > 0
    
    def test_calculate_validation_score(self, validator, sample_table, problematic_table):
        """Test validation score calculation."""
        good_report = validator.validate_table_structure(sample_table)
        bad_report = validator.validate_table_structure(problematic_table)
        
        assert good_report.overall_confidence > bad_report.overall_confidence
        assert good_report.is_valid
        assert not bad_report.is_valid
    
    def test_apply_corrections(self, validator, problematic_table):
        """Test applying corrections to a table."""
        corrections = [
            CorrectionAction(
                action_type="replace",
                row=0,
                column=1,
                original_value="3O",
                new_value="30",
                confidence=0.9,
                reason="OCR correction: O to 0"
            ),
            CorrectionAction(
                action_type="replace",
                row=0,
                column=2,
                original_value="$5O,OOO",
                new_value="$50,000",
                confidence=0.9,
                reason="OCR correction: O to 0"
            )
        ]
        
        corrected_table = validator.apply_corrections(problematic_table, corrections)
        
        assert corrected_table.rows[0][1] == "30"
        assert corrected_table.rows[0][2] == "$50,000"
        assert corrected_table.confidence >= problematic_table.confidence
        assert "corrections_applied" in corrected_table.metadata
    
    def test_validate_table_interface_compatibility(self, validator, sample_table):
        """Test ValidationInterface compatibility."""
        result = validator.validate_table(sample_table)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.confidence, float)


class TestTableCorrectionInterface:
    """Test cases for TableCorrectionInterface class."""
    
    @pytest.fixture
    def validator(self):
        """Create a mock validator for testing."""
        validator = Mock(spec=TableValidator)
        validator.validate_table_structure.return_value = TableValidationReport(
            table_id="test-table",
            overall_confidence=0.8,
            is_valid=True,
            cell_validations=[
                CellValidation(
                    row=0,
                    column=1,
                    original_value="3O",
                    suggested_value="30",
                    confidence=0.9,
                    issues=["OCR error detected"]
                )
            ]
        )
        return validator
    
    @pytest.fixture
    def interface(self, validator):
        """Create a TableCorrectionInterface instance for testing."""
        return TableCorrectionInterface(validator)
    
    @pytest.fixture
    def sample_table(self):
        """Create a sample table for testing."""
        return Table(
            id="test-table",
            headers=["Name", "Age", "Salary"],
            rows=[
                ["John Doe", "3O", "$50,000"],
                ["Jane Smith", "25", "$45,000"]
            ],
            confidence=0.75
        )
    
    def test_start_correction_session(self, interface, sample_table):
        """Test starting a correction session."""
        session_id = interface.start_correction_session(sample_table, user_id="test-user")
        
        assert session_id is not None
        assert session_id in interface.active_sessions
        
        session = interface.active_sessions[session_id]
        assert session.table_id == sample_table.id
        assert session.user_id == "test-user"
        assert session.original_table.id == sample_table.id
        assert session.current_table.id == sample_table.id
    
    def test_get_correction_suggestions(self, interface, sample_table):
        """Test getting correction suggestions."""
        session_id = interface.start_correction_session(sample_table)
        suggestions = interface.get_correction_suggestions(session_id)
        
        assert len(suggestions) > 0
        suggestion = suggestions[0]
        assert suggestion.row == 0
        assert suggestion.column == 1
        assert suggestion.original_value == "3O"
        assert suggestion.suggested_value == "30"
    
    def test_apply_cell_correction(self, interface, sample_table):
        """Test applying a cell correction."""
        session_id = interface.start_correction_session(sample_table)
        
        success = interface.apply_cell_correction(
            session_id=session_id,
            row=0,
            column=1,
            new_value="30",
            reason="Manual correction"
        )
        
        assert success
        
        session = interface.active_sessions[session_id]
        assert session.current_table.rows[0][1] == "30"
        assert len(session.applied_corrections) == 1
        
        correction = session.applied_corrections[0]
        assert correction.action_type == "replace"
        assert correction.row == 0
        assert correction.column == 1
        assert correction.new_value == "30"
    
    def test_apply_cell_correction_invalid_indices(self, interface, sample_table):
        """Test applying correction with invalid indices."""
        session_id = interface.start_correction_session(sample_table)
        
        # Invalid row
        success = interface.apply_cell_correction(
            session_id=session_id,
            row=10,
            column=1,
            new_value="30"
        )
        assert not success
        
        # Invalid column
        success = interface.apply_cell_correction(
            session_id=session_id,
            row=0,
            column=10,
            new_value="30"
        )
        assert not success
    
    def test_apply_bulk_corrections(self, interface, sample_table):
        """Test applying multiple corrections at once."""
        session_id = interface.start_correction_session(sample_table)
        
        corrections = [
            {"row": 0, "column": 1, "new_value": "30", "reason": "OCR fix"},
            {"row": 1, "column": 1, "new_value": "25", "reason": "Validation"}
        ]
        
        results = interface.apply_bulk_corrections(session_id, corrections)
        
        assert results["applied"] == 2
        assert results["failed"] == 0
        
        session = interface.active_sessions[session_id]
        assert len(session.applied_corrections) == 2
    
    def test_accept_suggestion(self, interface, sample_table):
        """Test accepting a correction suggestion."""
        session_id = interface.start_correction_session(sample_table)
        
        success = interface.accept_suggestion(session_id, row=0, column=1)
        
        assert success
        
        session = interface.active_sessions[session_id]
        assert session.current_table.rows[0][1] == "30"  # Should be corrected value
    
    def test_reject_suggestion(self, interface, sample_table):
        """Test rejecting a correction suggestion."""
        session_id = interface.start_correction_session(sample_table)
        
        success = interface.reject_suggestion(session_id, row=0, column=1)
        
        assert success
        
        session = interface.active_sessions[session_id]
        # Original value should remain unchanged
        assert session.current_table.rows[0][1] == "3O"
    
    def test_get_session_status(self, interface, sample_table):
        """Test getting session status."""
        session_id = interface.start_correction_session(sample_table)
        status = interface.get_session_status(session_id)
        
        assert status["session_id"] == session_id
        assert status["table_id"] == sample_table.id
        assert "created_at" in status
        assert "corrections_applied" in status
        assert "pending_suggestions" in status
        assert "validation_score" in status
    
    def test_finalize_corrections(self, interface, sample_table):
        """Test finalizing corrections."""
        session_id = interface.start_correction_session(sample_table)
        
        # Apply a correction
        interface.apply_cell_correction(session_id, 0, 1, "30", "Test correction")
        
        final_table = interface.finalize_corrections(session_id)
        
        assert final_table.id == sample_table.id
        assert final_table.rows[0][1] == "30"
        assert "correction_session_id" in final_table.metadata
        assert "corrections_applied" in final_table.metadata
        assert final_table.confidence >= sample_table.confidence
    
    def test_export_session_data(self, interface, sample_table):
        """Test exporting session data."""
        session_id = interface.start_correction_session(sample_table)
        
        # Apply a correction
        interface.apply_cell_correction(session_id, 0, 1, "30", "Test correction")
        
        export_data = interface.export_session_data(session_id)
        
        assert "session_info" in export_data
        assert "original_table" in export_data
        assert "corrected_table" in export_data
        assert "validation_report" in export_data
        assert "corrections" in export_data
        assert "suggestions" in export_data
        
        # Check that correction is recorded
        assert len(export_data["corrections"]) == 1
        assert export_data["corrections"][0]["new_value"] == "30"
    
    def test_close_session(self, interface, sample_table):
        """Test closing a correction session."""
        session_id = interface.start_correction_session(sample_table)
        
        assert session_id in interface.active_sessions
        
        success = interface.close_session(session_id)
        
        assert success
        assert session_id not in interface.active_sessions
    
    def test_correction_callbacks(self, interface, sample_table):
        """Test correction callback functionality."""
        callback_calls = []
        
        def test_callback(session_id, correction):
            callback_calls.append((session_id, correction))
        
        interface.add_correction_callback(test_callback)
        
        session_id = interface.start_correction_session(sample_table)
        interface.apply_cell_correction(session_id, 0, 1, "30", "Test")
        
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == session_id
        assert callback_calls[0][1].new_value == "30"
    
    def test_cleanup_old_sessions(self, interface, sample_table):
        """Test cleanup of old sessions."""
        # Create a session
        session_id = interface.start_correction_session(sample_table)
        
        # Mock old timestamp
        session = interface.active_sessions[session_id]
        old_time = datetime.now().replace(year=2020)  # Very old
        session.last_modified = old_time
        
        # Cleanup sessions older than 1 hour
        cleaned_count = interface.cleanup_old_sessions(max_age_hours=1)
        
        assert cleaned_count == 1
        assert session_id not in interface.active_sessions
    
    def test_session_not_found_errors(self, interface):
        """Test error handling for non-existent sessions."""
        fake_session_id = "non-existent-session"
        
        with pytest.raises(ValueError, match="Session .* not found"):
            interface.get_correction_suggestions(fake_session_id)
        
        with pytest.raises(ValueError, match="Session .* not found"):
            interface.get_session_status(fake_session_id)
        
        with pytest.raises(ValueError, match="Session .* not found"):
            interface.finalize_corrections(fake_session_id)


class TestIntegrationValidationCorrection:
    """Integration tests for validation and correction workflow."""
    
    @pytest.fixture
    def complete_system(self):
        """Create a complete validation and correction system."""
        validator = TableValidator(confidence_threshold=0.7)
        interface = TableCorrectionInterface(validator)
        return validator, interface
    
    @pytest.fixture
    def realistic_table(self):
        """Create a realistic table with various issues."""
        return Table(
            headers=["Employee Name", "Age", "Salary", "Start Date", "Active"],
            rows=[
                ["John Doe", "3O", "$5O,OOO", "2O23-O1-15", "true"],  # OCR errors
                ["Jane Smith", "25", "$45,000", "2023-02-20", "yes"],
                ["", "", "", "", ""],  # Empty row
                ["Bob Johnson", "thirty-five", "60000", "March 10, 2023", "1"],  # Format issues
                ["Alice Brown", "28", "$52,000", "2023-04-05", "true"],
                ["Alice Brown", "28", "$52,000", "2023-04-05", "true"],  # Duplicate
            ],
            confidence=0.65
        )
    
    def test_complete_validation_correction_workflow(self, complete_system, realistic_table):
        """Test the complete workflow from validation to correction."""
        validator, interface = complete_system
        
        # Step 1: Start correction session
        session_id = interface.start_correction_session(realistic_table)
        assert session_id is not None
        
        # Step 2: Get validation report
        session = interface.active_sessions[session_id]
        report = session.validation_report
        
        # Should detect various issues
        assert not report.is_valid  # Table has issues
        assert len(report.structure_issues) > 0 or len(report.data_quality_issues) > 0
        
        # Step 3: Get correction suggestions
        suggestions = interface.get_correction_suggestions(session_id)
        assert len(suggestions) > 0
        
        # Step 4: Apply some corrections
        corrections_applied = 0
        for suggestion in suggestions[:3]:  # Apply first 3 suggestions
            success = interface.accept_suggestion(
                session_id, suggestion.row, suggestion.column
            )
            if success:
                corrections_applied += 1
        
        assert corrections_applied > 0
        
        # Step 5: Check session status
        status = interface.get_session_status(session_id)
        assert status["corrections_applied"] == corrections_applied
        
        # Step 6: Finalize corrections
        final_table = interface.finalize_corrections(session_id)
        
        # Final table should have improvements
        assert final_table.confidence >= realistic_table.confidence
        assert "corrections_applied" in final_table.metadata
        
        # Step 7: Export session data
        export_data = interface.export_session_data(session_id)
        assert len(export_data["corrections"]) == corrections_applied
        
        # Step 8: Close session
        success = interface.close_session(session_id)
        assert success
    
    def test_validation_accuracy_metrics(self, complete_system):
        """Test validation accuracy with known good and bad tables."""
        validator, interface = complete_system
        
        # Good table
        good_table = Table(
            headers=["Name", "Age", "Salary"],
            rows=[
                ["John Doe", "30", "$50,000"],
                ["Jane Smith", "25", "$45,000"],
                ["Bob Johnson", "35", "$60,000"]
            ],
            confidence=0.9
        )
        
        # Bad table
        bad_table = Table(
            headers=["", "Age", "Age"],  # Empty and duplicate headers
            rows=[
                ["John", "3O", "$5O,OOO"],  # OCR errors
                ["", "", ""],  # Empty row
                ["Jane", "twenty-five", "45000"]  # Inconsistent format
            ],
            confidence=0.4
        )
        
        good_report = validator.validate_table_structure(good_table)
        bad_report = validator.validate_table_structure(bad_table)
        
        # Good table should validate better
        assert good_report.overall_confidence > bad_report.overall_confidence
        assert good_report.is_valid
        assert not bad_report.is_valid
        
        # Bad table should have more issues
        total_good_issues = len(good_report.structure_issues) + len(good_report.data_quality_issues)
        total_bad_issues = len(bad_report.structure_issues) + len(bad_report.data_quality_issues)
        
        assert total_bad_issues > total_good_issues
    
    def test_correction_effectiveness(self, complete_system, realistic_table):
        """Test that corrections actually improve table quality."""
        validator, interface = complete_system
        
        # Initial validation
        initial_report = validator.validate_table_structure(realistic_table)
        initial_confidence = initial_report.overall_confidence
        
        # Start correction session and apply all suggestions
        session_id = interface.start_correction_session(realistic_table)
        suggestions = interface.get_correction_suggestions(session_id)
        
        for suggestion in suggestions:
            interface.accept_suggestion(session_id, suggestion.row, suggestion.column)
        
        # Get corrected table
        corrected_table = interface.finalize_corrections(session_id)
        
        # Validate corrected table
        corrected_report = validator.validate_table_structure(corrected_table)
        
        # Corrected table should be better
        assert corrected_report.overall_confidence >= initial_confidence
        
        # Should have fewer cell-level issues
        initial_cell_issues = len(initial_report.cell_validations)
        corrected_cell_issues = len(corrected_report.cell_validations)
        
        # After applying corrections, there should be fewer issues
        assert corrected_cell_issues <= initial_cell_issues
        
        interface.close_session(session_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])