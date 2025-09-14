#!/usr/bin/env python3
"""
Enhanced Schema Detection and Management Demo

This demo showcases the comprehensive schema detection and management system
that supports automatic schema inference, validation, compatibility checking,
versioning, and AI integration features.
"""

import pandas as pd
import json
from datetime import datetime
from src.data_processing.schema_manager import SchemaManager


def main():
    print("=== Enhanced Schema Detection and Management Demo ===\n")
    
    # Initialize schema manager
    schema_manager = SchemaManager()
    
    # Create sample employee data
    print("1. Creating sample employee dataset...")
    employee_data = pd.DataFrame({
        'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
        'age': [25, 30, 35, 28, 32],
        'salary': ['$50000', '$60000', '$55000', '$70000', '$65000'],
        'is_active': [True, False, True, True, False],
        'join_date': ['2020-01-15', '2019-06-20', '2021-03-10', '2020-11-05', '2021-07-22'],
        'completion_rate': ['85%', '92%', '78%', '95%', '88%'],
        'department': ['Engineering', 'Data Science', 'Product', 'Design', 'Analytics']
    })
    
    print(f"Dataset shape: {employee_data.shape}")
    print(f"Columns: {list(employee_data.columns)}\n")
    
    # Detect schema
    print("2. Detecting schema from data...")
    schema = schema_manager.detect_schema(employee_data, "employees")
    
    print(f"Detected {len(schema.columns)} columns:")
    for col in schema.columns:
        print(f"  - {col.name}: {col.data_type.value} (nullable: {col.nullable}, unique: {col.unique_values})")
    print()
    
    # Validate schema
    print("3. Validating schema...")
    validation_result = schema_manager.validate_schema(schema)
    print(f"Schema is valid: {validation_result.is_valid}")
    if validation_result.warnings:
        print(f"Warnings: {validation_result.warnings}")
    print()
    
    # Create schema version
    print("4. Creating schema version...")
    version_id = schema_manager.create_schema_version(schema, "Initial employee schema")
    print(f"Created schema version: {version_id}\n")
    
    # Export schema for AI consumption
    print("5. Exporting schema for AI integration...")
    ai_schema = schema_manager.export_schema_for_ai(schema)
    
    print("AI-friendly schema summary:")
    print(f"  - Table info: {ai_schema['table_info']}")
    print(f"  - Data patterns: {ai_schema['data_patterns']}")
    print(f"  - Query suggestions ({len(ai_schema['query_suggestions'])}):")
    for i, suggestion in enumerate(ai_schema['query_suggestions'][:5], 1):
        print(f"    {i}. {suggestion}")
    print()
    
    # Demonstrate schema evolution
    print("6. Demonstrating schema evolution...")
    
    # Create modified dataset
    modified_data = employee_data.copy()
    modified_data['email'] = [
        'alice@company.com', 'bob@company.com', 'charlie@company.com',
        'diana@company.com', 'eve@company.com'
    ]
    modified_data['performance_score'] = [4.2, 3.8, 4.5, 4.9, 4.1]
    
    print("Modified dataset with new columns: email, performance_score")
    
    # Detect evolution
    evolution_report = schema_manager.detect_schema_evolution(
        employee_data, modified_data, "employees"
    )
    
    print(f"Schema compatibility: {evolution_report['compatibility']['level']}")
    print(f"Can migrate: {evolution_report['compatibility']['can_migrate']}")
    print(f"Changes detected: {len(evolution_report['changes'])}")
    
    for change in evolution_report['changes']:
        print(f"  - {change['type']}: {change['column']} - {change['description']}")
    
    print(f"\nMigration recommendations: {len(evolution_report['migration_recommendations'])}")
    for rec in evolution_report['migration_recommendations']:
        print(f"  - {rec['change_type']} ({rec['priority']} priority): {rec['column']}")
        print(f"    Actions: {rec['actions'][0] if rec['actions'] else 'None'}")
    print()
    
    # Demonstrate advanced data type detection
    print("7. Advanced data type detection examples...")
    
    complex_data = pd.DataFrame({
        'scientific_numbers': ['1.5e10', '2.3e-5', '4.7e8', '1.2e-3', '9.8e6'],
        'mixed_currency': ['$100.00', '$200', '€150', '$75.50', '$999'],
        'phone_numbers': ['555-123-4567', '555-987-6543', '555-456-7890', '555-321-0987', '555-654-3210'],
        'zip_codes': ['12345', '67890', '54321', '09876', '13579'],
        'boolean_variants': ['TRUE', 'FALSE', 'True', 'False', 'true'],
        'percentage_mixed': ['85%', '92.5%', '78', '95%', '88.2%']
    })
    
    complex_schema = schema_manager.detect_schema(complex_data, "complex_types")
    
    print("Complex data type detection results:")
    for col in complex_schema.columns:
        print(f"  - {col.name}: {col.data_type.value}")
        if hasattr(col, 'metadata') and col.metadata:
            patterns = col.metadata.get('common_patterns', [])
            if patterns:
                print(f"    Patterns: {patterns}")
    print()
    
    # Show column metadata example
    print("8. Detailed column metadata example...")
    age_column = next(col for col in schema.columns if col.name == 'age')
    print(f"Age column metadata:")
    if hasattr(age_column, 'metadata') and age_column.metadata:
        for key, value in age_column.metadata.items():
            if key not in ['memory_usage']:  # Skip technical details
                print(f"  - {key}: {value}")
    print()
    
    print("=== Demo completed successfully! ===")
    print("\nKey features demonstrated:")
    print("✓ Automatic schema inference from structured data")
    print("✓ Data type detection (numbers, dates, text, currency, percentage, boolean)")
    print("✓ Schema validation and compatibility checking")
    print("✓ Schema versioning for data evolution tracking")
    print("✓ AI-friendly schema export for conversational interfaces")
    print("✓ Schema evolution detection and migration recommendations")
    print("✓ Advanced pattern recognition and metadata extraction")


if __name__ == "__main__":
    main()