#!/usr/bin/env python3
"""
Demo script for data type detection and conversion functionality.
"""

import pandas as pd
import numpy as np
from src.data_processing.data_cleaner import (
    DataTypeDetector, DataCleaningService, 
    detect_data_types, standardize_dataframe
)
from src.core.models import DataType

def main():
    print("=== OCR Table Analytics - Data Type Detection Demo ===\n")
    
    # Create sample data that might come from OCR extraction
    sample_data = pd.DataFrame({
        'product_id': ['001', '002', '003', '004', '005'],
        'product_name': ['Widget A', 'Gadget B', 'Tool C', 'Device D', 'Component E'],
        'price': ['$29.99', '$45.50', '$12.75', '$89.00', '$156.25'],
        'quantity': ['100', '250', '75', '300', '150'],
        'in_stock': ['true', 'false', 'true', 'true', 'false'],
        'discount': ['10%', '15%', '5%', '20%', '12%'],
        'last_updated': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12'],
        'rating': ['4.5', '3.8', '4.9', '4.2', '3.5']
    })
    
    print("Original OCR-extracted data:")
    print(sample_data)
    print(f"\nData types: {sample_data.dtypes.to_dict()}")
    print()
    
    # Demonstrate data type detection
    print("=== Data Type Detection ===")
    detector = DataTypeDetector()
    
    for column in sample_data.columns:
        result = detector.detect_column_type(sample_data[column])
        print(f"{column:15} -> {result.detected_type.value:10} (confidence: {result.confidence:.2f})")
    
    print()
    
    # Demonstrate automatic data type standardization
    print("=== Automatic Data Standardization ===")
    service = DataCleaningService()
    standardized_data = service.standardize_data_types(sample_data)
    
    print("Standardized data:")
    print(standardized_data)
    print(f"\nNew data types: {standardized_data.dtypes.to_dict()}")
    print()
    
    # Demonstrate schema detection
    print("=== Schema Detection ===")
    schema = service.detect_schema(standardized_data)
    
    print(f"Detected schema for {schema.row_count} rows:")
    for col_info in schema.columns:
        print(f"  {col_info.name:15} -> {col_info.data_type.value:10} "
              f"(unique: {col_info.unique_values}, nullable: {col_info.nullable})")
    print()
    
    # Demonstrate handling missing values
    print("=== Missing Value Handling ===")
    data_with_missing = sample_data.copy()
    data_with_missing.loc[1, 'price'] = 'N/A'
    data_with_missing.loc[3, 'quantity'] = ''
    data_with_missing.loc[2, 'in_stock'] = 'unknown'
    
    print("Data with missing values:")
    print(data_with_missing)
    print()
    
    cleaned_data = service.handle_missing_values(data_with_missing)
    print("After handling missing values:")
    print(cleaned_data)
    print()
    
    # Demonstrate convenience function
    print("=== Convenience Function Demo ===")
    final_data = standardize_dataframe(sample_data)
    print("Fully processed data using convenience function:")
    print(final_data)
    print(f"\nFinal data types: {final_data.dtypes.to_dict()}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()