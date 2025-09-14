"""
Demo script showcasing the enhanced missing value handling system.
"""

import pandas as pd
import numpy as np
from src.data_processing.data_cleaner import DataCleaningService
from src.data_processing.missing_value_handler import (
    MissingValueStrategy, MissingValuePolicy, handle_missing_values_simple
)
from src.core.models import DataType


def create_sample_data():
    """Create sample data with various missing value scenarios."""
    return pd.DataFrame({
        'sales_amount': [100.50, 200.75, np.nan, 400.25, 500.00, '', 700.80, np.nan, 850.25],
        'product_name': ['Widget A', 'Widget B', 'NULL', 'Widget D', '', 'Widget F', 'Widget G', 'N/A', 'Widget I'],
        'order_date': ['2023-01-01', '2023-01-02', np.nan, '2023-01-04', '2023-01-05', '', '2023-01-07', 'unknown', '2023-01-09'],
        'is_premium': [True, False, np.nan, True, False, '', True, 'NULL', False],
        'discount_pct': ['10%', '15%', np.nan, '20%', '25%', '-', '30%', 'N/A', '5%'],
        'customer_rating': [4.5, 3.8, np.nan, 4.2, 4.9, 'missing', 3.5, np.nan, 4.7]
    })


def demo_basic_missing_value_handling():
    """Demonstrate basic missing value handling."""
    print("=== Basic Missing Value Handling Demo ===\n")
    
    # Create sample data
    data = create_sample_data()
    print("Original Data:")
    print(data)
    print(f"\nMissing values per column:")
    print(data.isna().sum())
    print(f"Total missing values: {data.isna().sum().sum()}")
    
    # Simple automatic handling
    print("\n--- Automatic Missing Value Handling ---")
    result_auto = handle_missing_values_simple(data, strategy='auto')
    print("After automatic handling:")
    print(result_auto)
    print(f"\nMissing values per column:")
    print(result_auto.isna().sum())
    print(f"Total missing values: {result_auto.isna().sum().sum()}")
    
    # Simple removal strategy
    print("\n--- Remove Strategy ---")
    result_remove = handle_missing_values_simple(data, strategy='remove')
    print("After removal strategy:")
    print(result_remove)
    print(f"Rows remaining: {len(result_remove)} out of {len(data)}")


def demo_advanced_missing_value_handling():
    """Demonstrate advanced missing value handling with custom policies."""
    print("\n\n=== Advanced Missing Value Handling Demo ===\n")
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize data cleaning service
    service = DataCleaningService()
    
    # Get missing value report
    print("--- Missing Value Analysis Report ---")
    report = service.get_missing_value_report(data)
    
    print(f"Total cells: {report['total_cells']}")
    print(f"Missing cells: {report['missing_cells']}")
    print(f"Missing percentage: {report['missing_percentage']:.1%}")
    
    print("\nColumn Quality Assessments:")
    for col, assessment in report['column_assessments'].items():
        print(f"  {col}:")
        print(f"    Completeness: {assessment.completeness_score:.2f}")
        print(f"    Consistency: {assessment.consistency_score:.2f}")
        print(f"    Validity: {assessment.validity_score:.2f}")
        print(f"    Uniqueness: {assessment.uniqueness_score:.2f}")
        if assessment.quality_issues:
            print(f"    Issues: {', '.join(assessment.quality_issues)}")
    
    print("\nRecommended Strategies:")
    for col, strategy in report['recommended_strategies'].items():
        print(f"  {col}: {strategy.value}")
    
    # Create custom policies
    print("\n--- Custom Policy Application ---")
    custom_policies = {
        'sales_amount': service.create_missing_value_policy(
            'sales_amount', DataType.NUMBER, 'median'
        ),
        'product_name': service.create_missing_value_policy(
            'product_name', DataType.TEXT, 'mode'
        ),
        'order_date': service.create_missing_value_policy(
            'order_date', DataType.DATE, 'remove'
        ),
        'is_premium': service.create_missing_value_policy(
            'is_premium', DataType.BOOLEAN, 'false'
        ),
        'discount_pct': service.create_missing_value_policy(
            'discount_pct', DataType.PERCENTAGE, 'zero'
        ),
        'customer_rating': service.create_missing_value_policy(
            'customer_rating', DataType.NUMBER, 'custom_value', custom_value=4.0
        )
    }
    
    # Apply custom policies
    result_df, processing_report = service.handle_missing_values_with_policies(data, custom_policies)
    
    print("Processing Report:")
    print(f"  Original missing count: {processing_report.original_missing_count}")
    print(f"  Final missing count: {processing_report.final_missing_count}")
    print(f"  Quality improvement: {processing_report.quality_improvement:.1%}")
    print(f"  Rows removed: {processing_report.rows_removed}")
    print(f"  Values imputed: {processing_report.values_imputed}")
    print(f"  Processing time: {processing_report.processing_time_ms}ms")
    
    print("\nStrategies Applied:")
    for col, strategy in processing_report.strategies_applied.items():
        print(f"  {col}: {strategy}")
    
    if processing_report.warnings:
        print("\nWarnings:")
        for warning in processing_report.warnings:
            print(f"  - {warning}")
    
    print("\nFinal Result:")
    print(result_df)
    print(f"\nMissing values per column:")
    print(result_df.isna().sum())


def demo_strategy_comparison():
    """Demonstrate different strategies on the same data."""
    print("\n\n=== Strategy Comparison Demo ===\n")
    
    # Create data with numeric missing values
    numeric_data = pd.DataFrame({
        'values': [10, 20, np.nan, 40, 50, np.nan, 70, 80, np.nan, 100]
    })
    
    print("Original numeric data:")
    print(numeric_data)
    print(f"Missing values: {numeric_data['values'].isna().sum()}")
    
    strategies = ['mean', 'median', 'zero', 'interpolate_linear', 'remove']
    
    print("\nStrategy Comparison:")
    for strategy in strategies:
        try:
            result = handle_missing_values_simple(numeric_data, strategy=strategy)
            print(f"\n{strategy.upper()} strategy:")
            print(f"  Result: {result['values'].tolist()}")
            print(f"  Missing values: {result['values'].isna().sum()}")
            print(f"  Rows: {len(result)}")
        except Exception as e:
            print(f"\n{strategy.upper()} strategy: Failed - {e}")


def demo_data_quality_assessment():
    """Demonstrate data quality assessment features."""
    print("\n\n=== Data Quality Assessment Demo ===\n")
    
    # Create data with quality issues
    quality_data = pd.DataFrame({
        'high_quality': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'some_missing': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        'many_missing': [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8, 9, 10],
        'inconsistent': ['1', '2.0', 'three', '4', 'five', '6.0', 'seven', '8', '9.0', 'ten'],
        'duplicates': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'C', 'A', 'B']
    })
    
    service = DataCleaningService()
    
    print("Sample data with quality issues:")
    print(quality_data)
    
    # Get quality assessment
    data_types = {
        'high_quality': DataType.NUMBER,
        'some_missing': DataType.NUMBER,
        'many_missing': DataType.NUMBER,
        'inconsistent': DataType.NUMBER,
        'duplicates': DataType.TEXT
    }
    
    assessments = service.missing_value_handler.get_quality_assessment(quality_data, data_types)
    
    print("\nQuality Assessment Results:")
    for col, assessment in assessments.items():
        print(f"\n{col.upper()}:")
        print(f"  Completeness: {assessment.completeness_score:.2f}")
        print(f"  Consistency: {assessment.consistency_score:.2f}")
        print(f"  Validity: {assessment.validity_score:.2f}")
        print(f"  Uniqueness: {assessment.uniqueness_score:.2f}")
        print(f"  Missing %: {assessment.missing_percentage:.1%}")
        
        if assessment.quality_issues:
            print(f"  Issues: {', '.join(assessment.quality_issues)}")
        
        if assessment.recommendations:
            print(f"  Recommendations: {', '.join(assessment.recommendations)}")


def main():
    """Run all demos."""
    print("Missing Value Handling System Demo")
    print("=" * 50)
    
    try:
        demo_basic_missing_value_handling()
        demo_advanced_missing_value_handling()
        demo_strategy_comparison()
        demo_data_quality_assessment()
        
        print("\n\n=== Demo Complete ===")
        print("The missing value handling system provides:")
        print("- Configurable strategies for different data types")
        print("- Data quality assessment and recommendations")
        print("- User-configurable policies")
        print("- Comprehensive error handling")
        print("- Performance monitoring and reporting")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()