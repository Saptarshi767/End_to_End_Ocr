#!/usr/bin/env python3
"""
Enhanced Duplicate Detection and Consolidation Demo

This demo showcases the comprehensive duplicate detection and consolidation system
that handles:
1. Exact duplicate row removal
2. Fuzzy duplicate detection with similarity matching
3. Header consolidation for multi-page tables
4. Meaningful column name generation for missing headers

Requirements addressed: 2.3, 2.5
"""

import pandas as pd
import numpy as np
from src.data_processing.data_cleaner import (
    DataCleaningService, DuplicateDetector, DuplicateDetectionConfig
)
from src.core.models import DataType


def create_sample_data_with_duplicates():
    """Create sample data with various types of duplicates and issues."""
    
    # Sample 1: Multi-page table with duplicate headers
    print("=== Sample 1: Multi-page Table with Duplicate Headers ===")
    multi_page_data = pd.DataFrame({
        'Product Name': [
            'Product Name',  # Duplicate header from page 2
            'Apple iPhone 14',
            'Samsung Galaxy S23',
            'Product Name',  # Another duplicate header from page 3
            'Google Pixel 7',
            'OnePlus 11'
        ],
        'Price': [
            'Price',  # Duplicate header
            '$999.99',
            '$899.99', 
            'Price',  # Another duplicate header
            '$699.99',
            '$799.99'
        ],
        'Category': [
            'Category',  # Duplicate header
            'Smartphone',
            'Smartphone',
            'Category',  # Another duplicate header
            'Smartphone',
            'Smartphone'
        ]
    })
    
    return multi_page_data


def create_fuzzy_duplicate_data():
    """Create data with fuzzy duplicates for similarity matching."""
    
    print("\n=== Sample 2: Data with Fuzzy Duplicates ===")
    fuzzy_data = pd.DataFrame({
        'Company': [
            'Microsoft Corporation',
            'Microsoft Corp',  # Similar to above
            'Apple Inc.',
            'Apple Incorporated',  # Similar to above
            'Google LLC',
            'Alphabet Inc.',  # Related but different
            'Amazon.com Inc',
            'Amazon Inc',  # Similar to above
            'Meta Platforms Inc',
            'Facebook Inc'  # Old name, similar
        ],
        'Revenue': [
            198000, 198000, 365000, 365000, 282000, 282000, 
            469000, 469000, 117000, 117000
        ],
        'Employees': [
            221000, 221000, 164000, 164000, 156000, 156000,
            1540000, 1540000, 77000, 77000
        ]
    })
    
    return fuzzy_data


def create_generic_column_data():
    """Create data with generic column names that need meaningful names."""
    
    print("\n=== Sample 3: Data with Generic Column Names ===")
    generic_data = pd.DataFrame({
        'Unnamed: 0': [
            'John Doe',
            'Jane Smith', 
            'Bob Johnson',
            'Alice Brown'
        ],
        '0': [
            'john.doe@email.com',
            'jane.smith@email.com',
            'bob.johnson@email.com', 
            'alice.brown@email.com'
        ],
        '1': [
            '$75,000',
            '$82,000',
            '$68,000',
            '$91,000'
        ],
        'Column3': [
            '2023-01-15',
            '2023-02-20',
            '2023-03-10',
            '2023-04-05'
        ],
        'A': [
            'Manager',
            'Developer',
            'Analyst', 
            'Director'
        ]
    })
    
    return generic_data


def demonstrate_duplicate_detection():
    """Demonstrate comprehensive duplicate detection capabilities."""
    
    print("🔍 DUPLICATE DETECTION AND CONSOLIDATION DEMO")
    print("=" * 60)
    
    # Initialize the data cleaning service
    service = DataCleaningService()
    
    # Configure for more aggressive fuzzy matching
    service.configure_duplicate_detection(
        similarity_threshold=0.8,
        enable_fuzzy_matching=True,
        header_similarity_threshold=0.9
    )
    
    # Demo 1: Multi-page table header consolidation
    print("\n📄 DEMO 1: Multi-page Table Header Consolidation")
    print("-" * 50)
    
    multi_page_data = create_sample_data_with_duplicates()
    print("Original data with duplicate headers:")
    print(multi_page_data)
    print(f"Original shape: {multi_page_data.shape}")
    
    # Apply duplicate detection and removal
    cleaned_data, detection_result = service.remove_duplicates(multi_page_data), service.detect_duplicates_detailed(multi_page_data)
    
    print(f"\nCleaned data:")
    print(cleaned_data)
    print(f"Final shape: {cleaned_data.shape}")
    print(f"\nDetection Results:")
    print(f"- Original rows: {detection_result.original_rows}")
    print(f"- Final rows: {detection_result.final_rows}")
    print(f"- Exact duplicates removed: {detection_result.exact_duplicates_removed}")
    print(f"- Headers consolidated: {detection_result.headers_consolidated}")
    print(f"- Fuzzy duplicates removed: {detection_result.fuzzy_duplicates_removed}")
    
    for log_entry in detection_result.consolidation_log:
        print(f"- {log_entry}")
    
    # Demo 2: Fuzzy duplicate detection
    print("\n🔍 DEMO 2: Fuzzy Duplicate Detection")
    print("-" * 50)
    
    fuzzy_data = create_fuzzy_duplicate_data()
    print("Original data with fuzzy duplicates:")
    print(fuzzy_data)
    print(f"Original shape: {fuzzy_data.shape}")
    
    # Find similar entries before removal
    similar_entries = service.find_similar_entries(fuzzy_data, 'Company')
    print(f"\nSimilar entries found in 'Company' column:")
    if 'Company' in similar_entries:
        for i, group in enumerate(similar_entries['Company']):
            print(f"  Group {i+1}: {[entry[1] for entry in group]}")
    
    # Apply fuzzy duplicate removal
    cleaned_fuzzy = service.remove_duplicates(fuzzy_data)
    fuzzy_result = service.detect_duplicates_detailed(fuzzy_data)
    
    print(f"\nCleaned data after fuzzy duplicate removal:")
    print(cleaned_fuzzy)
    print(f"Final shape: {cleaned_fuzzy.shape}")
    print(f"\nFuzzy Detection Results:")
    print(f"- Original rows: {fuzzy_result.original_rows}")
    print(f"- Final rows: {fuzzy_result.final_rows}")
    print(f"- Exact duplicates removed: {fuzzy_result.exact_duplicates_removed}")
    print(f"- Fuzzy duplicates removed: {fuzzy_result.fuzzy_duplicates_removed}")
    
    # Demo 3: Meaningful column name generation
    print("\n📝 DEMO 3: Meaningful Column Name Generation")
    print("-" * 50)
    
    generic_data = create_generic_column_data()
    print("Original data with generic column names:")
    print(generic_data)
    print(f"Original columns: {list(generic_data.columns)}")
    
    # Generate meaningful column names
    meaningful_data = service.generate_meaningful_column_names(generic_data)
    
    print(f"\nData with meaningful column names:")
    print(meaningful_data)
    print(f"New columns: {list(meaningful_data.columns)}")
    
    # Show column mapping
    print(f"\nColumn name mapping:")
    for old_col, new_col in zip(generic_data.columns, meaningful_data.columns):
        if old_col != new_col:
            print(f"  '{old_col}' → '{new_col}'")
        else:
            print(f"  '{old_col}' (unchanged)")
    
    # Demo 4: Complete integration test
    print("\n🔄 DEMO 4: Complete Integration Test")
    print("-" * 50)
    
    # Create complex data with all issues
    complex_data = pd.DataFrame({
        'Unnamed: 0': [
            'Customer Name',  # Header row
            'John Doe',
            'Jon Doe',  # Fuzzy duplicate
            'Jane Smith',
            'John Doe'  # Exact duplicate
        ],
        '1': [
            'Email',  # Header row
            'john@email.com',
            'jon@email.com',
            'jane@email.com', 
            'john@email.com'  # Exact duplicate
        ],
        'A': [
            'Salary',  # Header row
            '$50,000',
            '$50,000',
            '$60,000',
            '$50,000'  # Exact duplicate
        ]
    })
    
    print("Complex data with multiple issues:")
    print(complex_data)
    print(f"Original shape: {complex_data.shape}")
    print(f"Original columns: {list(complex_data.columns)}")
    
    # Apply complete processing
    detector = DuplicateDetector(service.duplicate_config)
    final_data, final_result = detector.detect_and_remove_duplicates(complex_data)
    
    print(f"\nFinal processed data:")
    print(final_data)
    print(f"Final shape: {final_data.shape}")
    print(f"Final columns: {list(final_data.columns)}")
    
    print(f"\nComplete Processing Results:")
    print(f"- Original rows: {final_result.original_rows}")
    print(f"- Final rows: {final_result.final_rows}")
    print(f"- Exact duplicates removed: {final_result.exact_duplicates_removed}")
    print(f"- Headers consolidated: {final_result.headers_consolidated}")
    print(f"- Fuzzy duplicates removed: {final_result.fuzzy_duplicates_removed}")
    
    print(f"\nProcessing log:")
    for log_entry in final_result.consolidation_log:
        print(f"- {log_entry}")
    
    # Demo 5: Configuration options
    print("\n⚙️ DEMO 5: Configuration Options")
    print("-" * 50)
    
    # Test with different similarity thresholds
    test_data = pd.DataFrame({
        'Name': ['Microsoft Corp', 'Microsoft Corporation', 'Apple Inc', 'Apple Incorporated'],
        'Value': [100, 100, 200, 200]
    })
    
    print("Test data for configuration demo:")
    print(test_data)
    
    # High similarity threshold (strict)
    service.configure_duplicate_detection(similarity_threshold=0.95)
    strict_result = service.remove_duplicates(test_data)
    print(f"\nWith strict similarity (0.95): {len(strict_result)} rows remaining")
    print(strict_result)
    
    # Low similarity threshold (lenient)
    service.configure_duplicate_detection(similarity_threshold=0.7)
    lenient_result = service.remove_duplicates(test_data)
    print(f"\nWith lenient similarity (0.7): {len(lenient_result)} rows remaining")
    print(lenient_result)
    
    # Disable fuzzy matching
    service.configure_duplicate_detection(
        similarity_threshold=0.85,
        enable_fuzzy_matching=False
    )
    no_fuzzy_result = service.remove_duplicates(test_data)
    print(f"\nWith fuzzy matching disabled: {len(no_fuzzy_result)} rows remaining")
    print(no_fuzzy_result)
    
    print("\n✅ DUPLICATE DETECTION DEMO COMPLETED!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Exact duplicate row removal")
    print("✓ Fuzzy duplicate detection with similarity matching")
    print("✓ Header consolidation for multi-page tables")
    print("✓ Meaningful column name generation")
    print("✓ Configurable similarity thresholds")
    print("✓ Comprehensive processing logs")
    print("✓ Integration with data cleaning pipeline")


if __name__ == "__main__":
    demonstrate_duplicate_detection()