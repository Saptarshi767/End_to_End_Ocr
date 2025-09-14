"""
Demo script for table validation and correction system.
"""

import logging
from src.core.models import Table, TableRegion, BoundingBox
from src.data_processing.table_validator import TableValidator
from src.data_processing.correction_interface import TableCorrectionInterface

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_table_with_issues():
    """Create a sample table with various data quality issues."""
    return Table(
        headers=["Employee Name", "Age", "Salary", "Start Date", "Active"],
        rows=[
            ["John Doe", "3O", "$5O,OOO", "2O23-O1-15", "true"],  # OCR errors (O instead of 0)
            ["Jane Smith", "25", "$45,000", "2023-02-20", "yes"],
            ["", "", "", "", ""],  # Empty row
            ["Bob Johnson", "thirty-five", "60000", "March 10, 2023", "1"],  # Format inconsistencies
            ["Alice Brown", "28", "$52,000", "2023-04-05", "true"],
            ["Alice Brown", "28", "$52,000", "2023-04-05", "true"],  # Duplicate row
            ["Charlie Wilson", "3l", "$48,OOO", "2023-05-15", "false"],  # More OCR errors (l instead of 1)
        ],
        confidence=0.65,
        region=TableRegion(
            bounding_box=BoundingBox(0, 0, 600, 400, 0.7),
            confidence=0.7
        )
    )


def demo_table_validation():
    """Demonstrate table validation functionality."""
    print("=" * 60)
    print("TABLE VALIDATION DEMO")
    print("=" * 60)
    
    # Create validator
    validator = TableValidator(confidence_threshold=0.7)
    
    # Create sample table with issues
    table = create_sample_table_with_issues()
    
    print(f"\nOriginal Table:")
    print(f"Headers: {table.headers}")
    print(f"Rows ({len(table.rows)}):")
    for i, row in enumerate(table.rows):
        print(f"  {i+1}: {row}")
    print(f"Original Confidence: {table.confidence:.2f}")
    
    # Validate the table
    print(f"\nValidating table structure...")
    report = validator.validate_table_structure(table)
    
    print(f"\nValidation Results:")
    print(f"  Overall Confidence: {report.overall_confidence:.2f}")
    print(f"  Is Valid: {report.is_valid}")
    print(f"  Structure Issues: {len(report.structure_issues)}")
    for issue in report.structure_issues:
        print(f"    - {issue}")
    
    print(f"  Data Quality Issues: {len(report.data_quality_issues)}")
    for issue in report.data_quality_issues:
        print(f"    - {issue}")
    
    print(f"  Cell-level Issues: {len(report.cell_validations)}")
    for cv in report.cell_validations[:5]:  # Show first 5
        if cv.suggested_value:
            print(f"    - Row {cv.row+1}, Col {cv.column+1}: '{cv.original_value}' -> '{cv.suggested_value}' "
                  f"(confidence: {cv.confidence:.2f})")
    
    if len(report.cell_validations) > 5:
        print(f"    ... and {len(report.cell_validations) - 5} more")
    
    return table, report


def demo_correction_interface():
    """Demonstrate the correction interface functionality."""
    print("\n" + "=" * 60)
    print("CORRECTION INTERFACE DEMO")
    print("=" * 60)
    
    # Create validator and interface
    validator = TableValidator(confidence_threshold=0.7)
    interface = TableCorrectionInterface(validator)
    
    # Create sample table
    table = create_sample_table_with_issues()
    
    # Start correction session
    print(f"\nStarting correction session...")
    session_id = interface.start_correction_session(table, user_id="demo-user")
    print(f"Session ID: {session_id}")
    
    # Get correction suggestions
    suggestions = interface.get_correction_suggestions(session_id)
    print(f"\nFound {len(suggestions)} correction suggestions:")
    
    for i, suggestion in enumerate(suggestions[:10]):  # Show first 10
        print(f"  {i+1}. Row {suggestion.row+1}, Col {suggestion.column+1}: "
              f"'{suggestion.original_value}' -> '{suggestion.suggested_value}' "
              f"(confidence: {suggestion.confidence:.2f})")
        print(f"      Reason: {suggestion.reason}")
    
    if len(suggestions) > 10:
        print(f"  ... and {len(suggestions) - 10} more suggestions")
    
    # Apply some corrections automatically
    print(f"\nApplying first 5 suggestions automatically...")
    applied_count = 0
    for suggestion in suggestions[:5]:
        success = interface.accept_suggestion(session_id, suggestion.row, suggestion.column)
        if success:
            applied_count += 1
    
    print(f"Applied {applied_count} corrections successfully")
    
    # Apply a manual correction
    print(f"\nApplying manual correction...")
    manual_success = interface.apply_cell_correction(
        session_id=session_id,
        row=3,  # Bob Johnson's row
        column=1,  # Age column
        new_value="35",
        reason="Manual correction: converted text to number"
    )
    
    if manual_success:
        print("Manual correction applied successfully")
    
    # Get session status
    status = interface.get_session_status(session_id)
    print(f"\nSession Status:")
    print(f"  Corrections Applied: {status['corrections_applied']}")
    print(f"  Pending Suggestions: {status['pending_suggestions']}")
    print(f"  Validation Score: {status['validation_score']:.2f}")
    print(f"  Is Valid: {status['is_valid']}")
    
    # Finalize corrections
    print(f"\nFinalizing corrections...")
    corrected_table = interface.finalize_corrections(session_id)
    
    print(f"\nCorrected Table:")
    print(f"Headers: {corrected_table.headers}")
    print(f"Rows ({len(corrected_table.rows)}):")
    for i, row in enumerate(corrected_table.rows):
        print(f"  {i+1}: {row}")
    print(f"Final Confidence: {corrected_table.confidence:.2f}")
    print(f"Corrections Applied: {corrected_table.metadata.get('corrections_applied', 0)}")
    
    # Export session data
    print(f"\nExporting session data...")
    export_data = interface.export_session_data(session_id)
    print(f"Export includes:")
    print(f"  - Session info")
    print(f"  - Original table ({len(export_data['original_table']['rows'])} rows)")
    print(f"  - Corrected table ({len(export_data['corrected_table']['rows'])} rows)")
    print(f"  - {len(export_data['corrections'])} corrections")
    print(f"  - {len(export_data['suggestions'])} suggestions")
    
    # Close session
    interface.close_session(session_id)
    print(f"\nSession closed successfully")
    
    return corrected_table


def demo_validation_accuracy():
    """Demonstrate validation accuracy with different table qualities."""
    print("\n" + "=" * 60)
    print("VALIDATION ACCURACY DEMO")
    print("=" * 60)
    
    validator = TableValidator(confidence_threshold=0.7)
    
    # Good quality table
    good_table = Table(
        headers=["Name", "Age", "Salary", "Department"],
        rows=[
            ["John Doe", "30", "$50000", "Engineering"],
            ["Jane Smith", "25", "$45000", "Marketing"],
            ["Bob Johnson", "35", "$60000", "Sales"]
        ],
        confidence=0.95
    )
    
    # Poor quality table
    poor_table = Table(
        headers=["", "Age", "Age", "Dept"],  # Empty and duplicate headers
        rows=[
            ["John", "3O", "$5O,OOO", "Eng"],  # OCR errors
            ["", "", "", ""],  # Empty row
            ["Jane", "twenty-five", "45000", "Mkt"],  # Inconsistent formats
            ["Jane", "twenty-five", "45000", "Mkt"],  # Duplicate
        ],
        confidence=0.4
    )
    
    print(f"\nValidating GOOD quality table...")
    good_report = validator.validate_table_structure(good_table)
    print(f"  Confidence: {good_report.overall_confidence:.2f}")
    print(f"  Valid: {good_report.is_valid}")
    print(f"  Issues: {len(good_report.structure_issues) + len(good_report.data_quality_issues)}")
    
    print(f"\nValidating POOR quality table...")
    poor_report = validator.validate_table_structure(poor_table)
    print(f"  Confidence: {poor_report.overall_confidence:.2f}")
    print(f"  Valid: {poor_report.is_valid}")
    print(f"  Issues: {len(poor_report.structure_issues) + len(poor_report.data_quality_issues)}")
    
    print(f"\nValidation Accuracy:")
    print(f"  Good table confidence > Poor table confidence: {good_report.overall_confidence > poor_report.overall_confidence}")
    print(f"  Good table valid, Poor table invalid: {good_report.is_valid and not poor_report.is_valid}")
    print(f"  Confidence difference: {good_report.overall_confidence - poor_report.overall_confidence:.2f}")


def main():
    """Run all demos."""
    print("Table Validation and Correction System Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Basic validation
        table, report = demo_table_validation()
        
        # Demo 2: Correction interface
        corrected_table = demo_correction_interface()
        
        # Demo 3: Validation accuracy
        demo_validation_accuracy()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Table structure validation")
        print("✓ Data quality assessment")
        print("✓ OCR error detection and correction")
        print("✓ Manual correction interface")
        print("✓ Confidence scoring")
        print("✓ Session management")
        print("✓ Bulk correction application")
        print("✓ Data export capabilities")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()