"""
Demo script for duplicate detection and consolidation functionality.

This script demonstrates the comprehensive duplicate detection capabilities
including exact duplicates, fuzzy matching, and header consolidation.
"""

import pandas as pd
import numpy as np
from src.data_processing.data_cleaner import (
    DataCleaningService, DuplicateDetectionConfig, DuplicateDetector
)


def create_sample_data_with_duplicates():
    """Create sample data with various types of duplicates."""
    
    # Sample 1: Multi-page table with repeated headers
    print("=== Sample 1: Multi-page Table with Repeated Headers ===")
    multi_page_data = pd.DataFrame({
        'Product': ['Product', 'Apple iPhone', 'Samsung Galaxy', 'Product', 'Google Pixel', 'OnePlus Nord'],
        'Price': ['Price', '$999', '$899', 'Price', '$799', '$399'],
        'Stock': ['Stock', '50', '30', 'Stock', '25', '40'],
        'Category': ['Category', 'Phone', 'Phone', 'Category', 'Phone', 'Phone']
    })
    
    print("Original data:")
    print(multi_page_data)
    print(f"Shape: {multi_page_data.shape}")
    
    return multi_page_data


def create_fuzzy_duplicate_data():
    """Create data with fuzzy duplicates."""
    
    print("\n=== Sample 2: Data with Fuzzy Duplicates ===")
    fuzzy_data = pd.DataFrame({
        'Company': [
            'Microsoft Corporation',
            'Microsoft Corp.',
            'Apple Inc.',
            'Apple Incorporated',
            'Google LLC',
            'Alphabet Inc.',
            'Amazon.com Inc.',
            'Amazon Inc',
            'Meta Platforms Inc.',
            'Facebook Inc.'
        ],
        'Revenue': [198000, 198000, 365000, 365000, 257000, 257000, 469000, 469000, 117000, 117000],
        'Employees': [221000, 221000, 154000, 154000, 156000, 156000, 1540000, 1540000, 77000, 77000]
    })
    
    print("Original data:")
    print(fuzzy_data)
    print(f"Shape: {fuzzy_data.shape}")
    
    return fuzzy_data


def create_exact_duplicate_data():
    """Create data with exact duplicates."""
    
    print("\n=== Sample 3: Data with Exact Duplicates ===")
    exact_data = pd.DataFrame({
        'Name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'Jane Smith', 'Alice Brown'],
        'Age': [25, 30, 25, 35, 30, 28],
        'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles', 'Boston'],
        'Salary': [50000, 60000, 50000, 70000, 60000, 55000]
    })
    
    print("Original data:")
    print(exact_data)
    print(f"Shape: {exact_data.shape}")
    
    return exact_data


def demo_basic_duplicate_removal():
    """Demonstrate basic duplicate removal functionality."""
    
    print("\n" + "="*60)
    print("DEMO: Basic Duplicate Removal")
    print("="*60)
    
    # Create service with default configuration
    service = DataCleaningService()
    
    # Test with multi-page table
    data = create_sample_data_with_duplicates()
    
    print("\nRemoving duplicates...")
    cleaned_data = service.remove_duplicates(data)
    
    print("\nCleaned data:")
    print(cleaned_data)
    print(f"Shape: {cleaned_data.shape}")
    print(f"Rows removed: {len(data) - len(cleaned_data)}")


def demo_fuzzy_matching():
    """Demonstrate fuzzy matching capabilities."""
    
    print("\n" + "="*60)
    print("DEMO: Fuzzy Matching")
    print("="*60)
    
    # Create service with fuzzy matching enabled
    service = DataCleaningService()
    service.configure_duplicate_detection(
        similarity_threshold=0.8,  # Lower threshold for more aggressive matching
        enable_fuzzy_matching=True
    )
    
    # Test with fuzzy duplicates
    data = create_fuzzy_duplicate_data()
    
    print("\nDetecting similar entries...")
    similar_entries = service.find_similar_entries(data, 'Company')
    
    if 'Company' in similar_entries:
        print("\nSimilar company names found:")
        for i, group in enumerate(similar_entries['Company']):
            print(f"Group {i+1}:")
            for idx, value in group:
                print(f"  - Row {idx}: {value}")
    
    print("\nRemoving fuzzy duplicates...")
    cleaned_data = service.remove_duplicates(data)
    
    print("\nCleaned data:")
    print(cleaned_data)
    print(f"Shape: {cleaned_data.shape}")
    print(f"Rows removed: {len(data) - len(cleaned_data)}")


def demo_detailed_detection():
    """Demonstrate detailed duplicate detection analysis."""
    
    print("\n" + "="*60)
    print("DEMO: Detailed Duplicate Detection Analysis")
    print("="*60)
    
    service = DataCleaningService()
    data = create_exact_duplicate_data()
    
    print("\nPerforming detailed duplicate analysis...")
    detection_result = service.detect_duplicates_detailed(data)
    
    print(f"\nDetection Results:")
    print(f"Original rows: {detection_result.original_rows}")
    print(f"Final rows: {detection_result.final_rows}")
    print(f"Exact duplicates removed: {detection_result.exact_duplicates_removed}")
    print(f"Fuzzy duplicates removed: {detection_result.fuzzy_duplicates_removed}")
    print(f"Headers consolidated: {detection_result.headers_consolidated}")
    
    if detection_result.duplicate_groups:
        print(f"\nDuplicate groups found: {len(detection_result.duplicate_groups)}")
        for i, group in enumerate(detection_result.duplicate_groups):
            print(f"Group {i+1}: Rows {group}")
    
    if detection_result.consolidation_log:
        print("\nConsolidation log:")
        for log_entry in detection_result.consolidation_log:
            print(f"  - {log_entry}")


def demo_header_consolidation():
    """Demonstrate header consolidation for multi-page tables."""
    
    print("\n" + "="*60)
    print("DEMO: Header Consolidation")
    print("="*60)
    
    service = DataCleaningService()
    
    # Create data simulating a multi-page PDF table
    multi_page_table = pd.DataFrame({
        'Product Name': ['Product Name', 'Laptop', 'Mouse', 'Product Name', 'Keyboard', 'Monitor'],
        'Price ($)': ['Price ($)', '999.99', '29.99', 'Price ($)', '79.99', '299.99'],
        'In Stock': ['In Stock', 'Yes', 'No', 'In Stock', 'Yes', 'Yes'],
        'Supplier': ['Supplier', 'Dell', 'Logitech', 'Supplier', 'Corsair', 'Samsung']
    })
    
    print("Original multi-page table data:")
    print(multi_page_table)
    print(f"Shape: {multi_page_table.shape}")
    
    print("\nConsolidating headers only...")
    consolidated_data = service.consolidate_headers_only(multi_page_table)
    
    print("\nData after header consolidation:")
    print(consolidated_data)
    print(f"Shape: {consolidated_data.shape}")
    print(f"Header rows removed: {len(multi_page_table) - len(consolidated_data)}")


def demo_configuration_options():
    """Demonstrate different configuration options."""
    
    print("\n" + "="*60)
    print("DEMO: Configuration Options")
    print("="*60)
    
    # Create test data
    test_data = pd.DataFrame({
        'Name': ['John Doe', 'JOHN DOE', 'Jon Doe', 'Jane Smith'],
        'Email': ['john@email.com', 'JOHN@EMAIL.COM', 'jon@email.com', 'jane@email.com']
    })
    
    print("Test data:")
    print(test_data)
    
    # Test 1: Case sensitive matching
    print("\n--- Test 1: Case Sensitive Matching ---")
    service1 = DataCleaningService()
    service1.configure_duplicate_detection(
        ignore_case=False,
        similarity_threshold=0.9
    )
    
    result1 = service1.remove_duplicates(test_data)
    print(f"Result (case sensitive): {len(result1)} rows")
    print(result1)
    
    # Test 2: Case insensitive matching
    print("\n--- Test 2: Case Insensitive Matching ---")
    service2 = DataCleaningService()
    service2.configure_duplicate_detection(
        ignore_case=True,
        similarity_threshold=0.9
    )
    
    result2 = service2.remove_duplicates(test_data)
    print(f"Result (case insensitive): {len(result2)} rows")
    print(result2)
    
    # Test 3: High similarity threshold
    print("\n--- Test 3: High Similarity Threshold (0.95) ---")
    service3 = DataCleaningService()
    service3.configure_duplicate_detection(
        similarity_threshold=0.95,
        enable_fuzzy_matching=True
    )
    
    result3 = service3.remove_duplicates(test_data)
    print(f"Result (high threshold): {len(result3)} rows")
    print(result3)
    
    # Test 4: Low similarity threshold
    print("\n--- Test 4: Low Similarity Threshold (0.7) ---")
    service4 = DataCleaningService()
    service4.configure_duplicate_detection(
        similarity_threshold=0.7,
        enable_fuzzy_matching=True
    )
    
    result4 = service4.remove_duplicates(test_data)
    print(f"Result (low threshold): {len(result4)} rows")
    print(result4)


def demo_performance_with_large_dataset():
    """Demonstrate performance with a larger dataset."""
    
    print("\n" + "="*60)
    print("DEMO: Performance with Large Dataset")
    print("="*60)
    
    # Create a larger dataset with duplicates
    np.random.seed(42)
    
    names = ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    
    # Create base data
    large_data = []
    for i in range(1000):
        name = np.random.choice(names)
        city = np.random.choice(cities)
        age = np.random.randint(20, 70)
        salary = np.random.randint(30000, 120000)
        
        # Add some variations to create fuzzy duplicates
        if np.random.random() < 0.1:  # 10% chance of variation
            if 'John' in name:
                name = name.replace('John', 'Jon')
            elif 'Jane' in name:
                name = name.replace('Jane', 'Jain')
        
        large_data.append([name, age, city, salary])
    
    # Add exact duplicates
    for i in range(100):
        large_data.append(large_data[i % 100])
    
    df = pd.DataFrame(large_data, columns=['Name', 'Age', 'City', 'Salary'])
    
    print(f"Large dataset created: {len(df)} rows")
    print(f"Sample data:")
    print(df.head(10))
    
    # Time the duplicate detection
    import time
    
    service = DataCleaningService()
    service.configure_duplicate_detection(
        similarity_threshold=0.85,
        enable_fuzzy_matching=True
    )
    
    start_time = time.time()
    cleaned_df = service.remove_duplicates(df)
    end_time = time.time()
    
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Original rows: {len(df)}")
    print(f"Final rows: {len(cleaned_df)}")
    print(f"Duplicates removed: {len(df) - len(cleaned_df)}")
    print(f"Duplicate percentage: {((len(df) - len(cleaned_df)) / len(df) * 100):.1f}%")


def main():
    """Run all demo functions."""
    
    print("DUPLICATE DETECTION AND CONSOLIDATION DEMO")
    print("=" * 80)
    print("This demo showcases the comprehensive duplicate detection capabilities")
    print("including exact duplicates, fuzzy matching, and header consolidation.")
    print("=" * 80)
    
    try:
        # Run all demos
        demo_basic_duplicate_removal()
        demo_fuzzy_matching()
        demo_detailed_detection()
        demo_header_consolidation()
        demo_configuration_options()
        demo_performance_with_large_dataset()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Features Demonstrated:")
        print("✓ Exact duplicate removal")
        print("✓ Fuzzy duplicate detection with configurable similarity threshold")
        print("✓ Header consolidation for multi-page tables")
        print("✓ Configurable matching options (case sensitivity, whitespace)")
        print("✓ Detailed analysis and reporting")
        print("✓ Performance with large datasets")
        print("✓ Similar entry detection across columns")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()