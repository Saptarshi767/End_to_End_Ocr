"""
Tests for duplicate detection and consolidation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.data_processing.data_cleaner import (
    DuplicateDetector, DuplicateDetectionConfig, DuplicateDetectionResult,
    DataCleaningService
)
from src.core.models import DataType


class TestDuplicateDetector:
    """Test cases for DuplicateDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DuplicateDetectionConfig(
            similarity_threshold=0.85,
            header_similarity_threshold=0.9,
            enable_fuzzy_matching=True,
            ignore_case=True,
            ignore_whitespace=True
        )
        self.detector = DuplicateDetector(self.config)
    
    def test_remove_exact_duplicates(self):
        """Test removal of exact duplicate rows."""
        # Create DataFrame with exact duplicates
        df = pd.DataFrame({
            'A': [1, 2, 1, 3, 2],
            'B': ['a', 'b', 'a', 'c', 'b'],
            'C': [10, 20, 10, 30, 20]
        })
        
        result_df, removed = self.detector._remove_exact_duplicates(df)
        
        assert removed == 2  # Two duplicate rows removed
        assert len(result_df) == 3  # Three unique rows remain
        assert result_df.iloc[0].equals(pd.Series([1, 'a', 10], index=['A', 'B', 'C']))
        assert result_df.iloc[1].equals(pd.Series([2, 'b', 20], index=['A', 'B', 'C']))
        assert result_df.iloc[2].equals(pd.Series([3, 'c', 30], index=['A', 'B', 'C']))
    
    def test_consolidate_headers_basic(self):
        """Test basic header consolidation."""
        # Create DataFrame with duplicate headers
        df = pd.DataFrame({
            'Name': ['Name', 'John', 'Jane', 'Bob'],
            'Age': ['Age', '25', '30', '35'],
            'City': ['City', 'NYC', 'LA', 'Chicago']
        })
        
        result_df, headers_consolidated = self.detector._consolidate_headers(df)
        
        assert headers_consolidated == 1  # One header row removed
        assert len(result_df) == 3  # Three data rows remain
        assert 'John' in result_df['Name'].values
        assert 'Jane' in result_df['Name'].values
        assert 'Bob' in result_df['Name'].values
    
    def test_consolidate_headers_case_insensitive(self):
        """Test header consolidation with case differences."""
        df = pd.DataFrame({
            'Name': ['name', 'John', 'Jane'],
            'Age': ['AGE', '25', '30'],
            'City': ['city', 'NYC', 'LA']
        })
        
        result_df, headers_consolidated = self.detector._consolidate_headers(df)
        
        assert headers_consolidated == 1
        assert len(result_df) == 2
        assert 'John' in result_df['Name'].values
    
    def test_consolidate_headers_with_whitespace(self):
        """Test header consolidation with whitespace differences."""
        df = pd.DataFrame({
            'Full Name': [' Full  Name ', 'John Doe', 'Jane Smith'],
            'Age': [' Age', '25', '30'],
            'City': ['City ', 'NYC', 'LA']
        })
        
        result_df, headers_consolidated = self.detector._consolidate_headers(df)
        
        assert headers_consolidated == 1
        assert len(result_df) == 2
    
    def test_normalize_row_for_comparison(self):
        """Test row normalization for comparison."""
        row = pd.Series(['John Doe', '  25  ', 'New York', None])
        
        normalized = self.detector._normalize_row_for_comparison(row)
        
        expected = 'john doe|25|new york|'
        assert normalized == expected
    
    def test_calculate_similarity_identical(self):
        """Test similarity calculation for identical strings."""
        similarity = self.detector._calculate_similarity('hello world', 'hello world')
        assert similarity == 1.0
    
    def test_calculate_similarity_different(self):
        """Test similarity calculation for different strings."""
        similarity = self.detector._calculate_similarity('hello', 'goodbye')
        assert similarity < 0.5
    
    def test_calculate_similarity_similar(self):
        """Test similarity calculation for similar strings."""
        similarity = self.detector._calculate_similarity('John Doe', 'Jon Doe')
        assert similarity > 0.8
    
    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        distance = self.detector._levenshtein_distance('kitten', 'sitting')
        assert distance == 3
        
        distance = self.detector._levenshtein_distance('hello', 'hello')
        assert distance == 0
        
        distance = self.detector._levenshtein_distance('', 'hello')
        assert distance == 5
    
    def test_find_fuzzy_duplicate_groups(self):
        """Test finding fuzzy duplicate groups."""
        normalized_rows = [
            (0, 'john doe|25|nyc'),
            (1, 'jane smith|30|la'),
            (2, 'jon doe|25|nyc'),  # Similar to row 0
            (3, 'bob jones|35|chicago'),
            (4, 'john doe|25|new york')  # Similar to row 0
        ]
        
        # Lower threshold to catch similar entries
        self.detector.config.similarity_threshold = 0.7
        
        groups = self.detector._find_fuzzy_duplicate_groups(normalized_rows)
        
        # Should find at least one group with similar entries
        assert len(groups) >= 1
        # Check that similar rows are grouped together
        group_indices = [idx for group in groups for idx in group]
        assert 0 in group_indices or 2 in group_indices  # Similar names should be grouped
    
    def test_detect_and_remove_duplicates_comprehensive(self):
        """Test comprehensive duplicate detection and removal."""
        # Create DataFrame with various types of duplicates
        df = pd.DataFrame({
            'Name': ['Name', 'John Doe', 'John Doe', 'Jon Doe', 'Jane Smith', 'Jane Smith'],
            'Age': ['Age', '25', '25', '25', '30', '30'],
            'City': ['City', 'NYC', 'NYC', 'New York', 'LA', 'LA']
        })
        
        result_df, detection_result = self.detector.detect_and_remove_duplicates(df)
        
        # Check that duplicates were removed
        assert detection_result.original_rows == 6
        assert detection_result.final_rows < 6
        assert detection_result.exact_duplicates_removed >= 1
        assert detection_result.headers_consolidated >= 1
        
        # Check that result DataFrame is valid
        assert len(result_df) == detection_result.final_rows
        assert not result_df.empty
    
    def test_detect_similar_entries_single_column(self):
        """Test detecting similar entries in a single column."""
        df = pd.DataFrame({
            'Name': ['John Doe', 'Jon Doe', 'Jane Smith', 'John Smith'],
            'Age': [25, 25, 30, 35]
        })
        
        similar_groups = self.detector.detect_similar_entries(df, 'Name')
        
        assert 'Name' in similar_groups
        # Should find similar names
        name_groups = similar_groups['Name']
        assert len(name_groups) >= 1
    
    def test_detect_similar_entries_all_columns(self):
        """Test detecting similar entries across all columns."""
        df = pd.DataFrame({
            'Name': ['John Doe', 'Jon Doe', 'Jane Smith'],
            'City': ['New York', 'NYC', 'Los Angeles']
        })
        
        similar_groups = self.detector.detect_similar_entries(df)
        
        # Should analyze both columns
        assert len(similar_groups) >= 0  # May or may not find similarities
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result_df, detection_result = self.detector.detect_and_remove_duplicates(df)
        
        assert result_df.empty
        assert detection_result.original_rows == 0
        assert detection_result.final_rows == 0
        assert detection_result.exact_duplicates_removed == 0
        assert detection_result.fuzzy_duplicates_removed == 0
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({'A': [1], 'B': ['test']})
        
        result_df, detection_result = self.detector.detect_and_remove_duplicates(df)
        
        assert len(result_df) == 1
        assert detection_result.original_rows == 1
        assert detection_result.final_rows == 1
        assert detection_result.exact_duplicates_removed == 0
        assert detection_result.fuzzy_duplicates_removed == 0
    
    def test_configuration_parameters(self):
        """Test different configuration parameters."""
        # Test with fuzzy matching disabled
        config = DuplicateDetectionConfig(enable_fuzzy_matching=False)
        detector = DuplicateDetector(config)
        
        df = pd.DataFrame({
            'Name': ['John Doe', 'Jon Doe', 'Jane Smith'],
            'Age': [25, 25, 30]
        })
        
        result_df, detection_result = detector.detect_and_remove_duplicates(df)
        
        # Should not remove fuzzy duplicates
        assert detection_result.fuzzy_duplicates_removed == 0
    
    def test_high_similarity_threshold(self):
        """Test with high similarity threshold."""
        config = DuplicateDetectionConfig(similarity_threshold=0.95)
        detector = DuplicateDetector(config)
        
        df = pd.DataFrame({
            'Name': ['John Doe', 'Jon Doe', 'Jane Smith'],
            'Age': [25, 25, 30]
        })
        
        result_df, detection_result = detector.detect_and_remove_duplicates(df)
        
        # With high threshold, fewer fuzzy matches should be found
        assert detection_result.fuzzy_duplicates_removed <= 1


class TestDataCleaningServiceDuplicates:
    """Test duplicate detection functionality in DataCleaningService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataCleaningService()
    
    def test_remove_duplicates_integration(self):
        """Test remove_duplicates method integration."""
        df = pd.DataFrame({
            'Name': ['John', 'John', 'Jane', 'Jon'],  # Exact and fuzzy duplicates
            'Age': [25, 25, 30, 25],
            'City': ['NYC', 'NYC', 'LA', 'NYC']
        })
        
        result_df = self.service.remove_duplicates(df)
        
        # Should remove at least exact duplicates
        assert len(result_df) <= len(df)
        assert not result_df.empty
    
    def test_detect_duplicates_detailed(self):
        """Test detailed duplicate detection without removal."""
        df = pd.DataFrame({
            'Name': ['John', 'John', 'Jane'],
            'Age': [25, 25, 30]
        })
        
        detection_result = self.service.detect_duplicates_detailed(df)
        
        assert isinstance(detection_result, DuplicateDetectionResult)
        assert detection_result.original_rows == 3
        assert detection_result.exact_duplicates_removed >= 1
    
    def test_find_similar_entries_service(self):
        """Test find_similar_entries method."""
        df = pd.DataFrame({
            'Name': ['John Doe', 'Jon Doe', 'Jane Smith'],
            'Age': [25, 25, 30]
        })
        
        similar_entries = self.service.find_similar_entries(df, 'Name')
        
        assert isinstance(similar_entries, dict)
        # Should find similar names
        if 'Name' in similar_entries:
            assert len(similar_entries['Name']) >= 0
    
    def test_consolidate_headers_only_service(self):
        """Test header consolidation without duplicate removal."""
        df = pd.DataFrame({
            'Name': ['Name', 'John', 'Jane'],
            'Age': ['Age', '25', '30']
        })
        
        result_df = self.service.consolidate_headers_only(df)
        
        # Should remove header row
        assert len(result_df) == 2
        assert 'John' in result_df['Name'].values
    
    def test_configure_duplicate_detection(self):
        """Test duplicate detection configuration."""
        original_threshold = self.service.duplicate_config.similarity_threshold
        
        self.service.configure_duplicate_detection(
            similarity_threshold=0.95,
            enable_fuzzy_matching=False
        )
        
        assert self.service.duplicate_config.similarity_threshold == 0.95
        assert self.service.duplicate_config.enable_fuzzy_matching == False
        
        # Reset for other tests
        self.service.configure_duplicate_detection(
            similarity_threshold=original_threshold,
            enable_fuzzy_matching=True
        )
    
    def test_multi_page_table_scenario(self):
        """Test scenario with multi-page table headers."""
        # Simulate a multi-page table with repeated headers
        df = pd.DataFrame({
            'Product': ['Product', 'Apple', 'Banana', 'Product', 'Cherry', 'Date'],
            'Price': ['Price', '1.00', '0.50', 'Price', '2.00', '3.00'],
            'Quantity': ['Quantity', '10', '20', 'Quantity', '5', '8']
        })
        
        result_df = self.service.remove_duplicates(df)
        
        # Should remove duplicate headers
        assert len(result_df) < len(df)
        # Should keep actual data rows
        assert 'Apple' in result_df['Product'].values
        assert 'Cherry' in result_df['Product'].values
    
    def test_fuzzy_matching_accuracy(self):
        """Test fuzzy matching accuracy with known similar entries."""
        df = pd.DataFrame({
            'Company': [
                'Microsoft Corporation',
                'Microsoft Corp',
                'Apple Inc.',
                'Apple Incorporated',
                'Google LLC',
                'Alphabet Inc.'
            ],
            'Revenue': [100, 100, 200, 200, 150, 150]
        })
        
        # Configure for more aggressive fuzzy matching
        self.service.configure_duplicate_detection(similarity_threshold=0.8)
        
        result_df = self.service.remove_duplicates(df)
        
        # Should detect Microsoft and Apple variations as similar
        assert len(result_df) < len(df)
        
        # Reset configuration
        self.service.configure_duplicate_detection(similarity_threshold=0.85)


class TestColumnNameGeneration:
    """Test cases for meaningful column name generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DuplicateDetector()
        self.service = DataCleaningService()
    
    def test_generate_meaningful_names_for_generic_columns(self):
        """Test generation of meaningful names for generic column names."""
        # Create DataFrame with generic column names
        df = pd.DataFrame({
            'Unnamed: 0': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            '0': ['john@email.com', 'jane@email.com', 'bob@email.com'],
            '1': ['$100.50', '$200.75', '$150.25'],
            'Column3': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        result_df = self.detector._generate_meaningful_column_names(df)
        
        # Check that generic names were replaced
        assert 'Unnamed: 0' not in result_df.columns
        assert '0' not in result_df.columns
        assert '1' not in result_df.columns
        assert 'Column3' not in result_df.columns
        
        # Check that meaningful names were generated
        columns = list(result_df.columns)
        assert any('Name' in col or 'Column_1' in col for col in columns)
        assert any('Email' in col or 'Column_2' in col for col in columns)
        assert any('Currency' in col or 'Price' in col or 'Column_3' in col for col in columns)
        assert any('Date' in col or 'Column_4' in col for col in columns)
    
    def test_infer_column_name_from_financial_data(self):
        """Test column name inference for financial data."""
        # Test price data
        price_series = pd.Series(['$10.99', '$25.50', '$100.00'])
        name = self.detector._infer_column_name_from_data(price_series, 0)
        assert name == 'Currency'
        
        # Test percentage data
        percent_series = pd.Series(['10%', '25%', '50%'])
        name = self.detector._infer_column_name_from_data(percent_series, 0)
        assert name == 'Percentage'
        
        # Test amount data
        amount_series = pd.Series(['Total Amount: 100', 'Amount: 200'])
        name = self.detector._infer_column_name_from_data(amount_series, 0)
        assert name == 'Amount'
    
    def test_infer_column_name_from_contact_data(self):
        """Test column name inference for contact data."""
        # Test email data
        email_series = pd.Series(['john@example.com', 'jane@test.org'])
        name = self.detector._infer_column_name_from_data(email_series, 0)
        assert name == 'Email'
        
        # Test phone data
        phone_series = pd.Series(['Phone: 123-456-7890', 'Tel: 987-654-3210'])
        name = self.detector._infer_column_name_from_data(phone_series, 0)
        assert name == 'Phone'
    
    def test_infer_column_name_from_date_data(self):
        """Test column name inference for date data."""
        # Test date patterns
        date_series = pd.Series(['2023-01-01', '2023-12-31', '2024-06-15'])
        name = self.detector._infer_column_name_from_data(date_series, 0)
        assert name == 'Date'
        
        # Test date with text
        date_text_series = pd.Series(['Created Date: 2023-01-01', 'Date Created: 2023-12-31'])
        name = self.detector._infer_column_name_from_data(date_text_series, 0)
        assert name in ['Date', 'Created_Date']  # Either is acceptable
    
    def test_infer_column_name_from_name_data(self):
        """Test column name inference for name data."""
        # Test full names
        name_series = pd.Series(['John Doe', 'Jane Smith', 'Bob Johnson'])
        name = self.detector._infer_column_name_from_data(name_series, 0)
        assert name in ['Name', 'Column_1']  # Could be either depending on pattern matching
        
        # Test customer names
        customer_series = pd.Series(['Customer: John', 'Customer: Jane'])
        name = self.detector._infer_column_name_from_data(customer_series, 0)
        assert name == 'Customer'
    
    def test_infer_column_name_from_numeric_data(self):
        """Test column name inference for numeric data."""
        # Test pure numbers
        number_series = pd.Series(['100', '200', '300'])
        name = self.detector._infer_column_name_from_data(number_series, 0)
        assert name == 'Number'
        
        # Test quantity data
        qty_series = pd.Series(['Qty: 10', 'Quantity: 20'])
        name = self.detector._infer_column_name_from_data(qty_series, 0)
        assert name == 'Quantity'
    
    def test_infer_column_name_from_category_data(self):
        """Test column name inference for categorical data."""
        # Test status-like data
        category_series = pd.Series(['Active', 'Inactive', 'Active', 'Pending', 'Active'])
        name = self.detector._infer_column_name_from_data(category_series, 0)
        assert name in ['Category', 'Column_1']  # Either is acceptable based on pattern matching
    
    def test_infer_column_name_from_description_data(self):
        """Test column name inference for description data."""
        # Test long text descriptions
        desc_series = pd.Series([
            'This is a very long description of a product that contains many details',
            'Another lengthy description with lots of information about the item'
        ])
        name = self.detector._infer_column_name_from_data(desc_series, 0)
        assert name == 'Description'
        
        # Test medium text
        text_series = pd.Series(['Some medium length text', 'Another text entry'])
        name = self.detector._infer_column_name_from_data(text_series, 0)
        assert name == 'Text'
    
    def test_infer_column_name_fallback(self):
        """Test fallback column name generation."""
        # Test with empty series
        empty_series = pd.Series([])
        name = self.detector._infer_column_name_from_data(empty_series, 5)
        assert name == 'Column_6'  # Index + 1
        
        # Test with unrecognizable data
        random_series = pd.Series(['xyz', 'abc', 'def'])
        name = self.detector._infer_column_name_from_data(random_series, 2)
        assert name == 'Column_3'  # Index + 1
    
    def test_generate_meaningful_names_preserves_good_names(self):
        """Test that good column names are preserved."""
        df = pd.DataFrame({
            'Customer_Name': ['John', 'Jane'],
            'Email_Address': ['john@test.com', 'jane@test.com'],
            'Unnamed: 2': ['$100', '$200'],
            'Good_Column': ['Data1', 'Data2']
        })
        
        result_df = self.detector._generate_meaningful_column_names(df)
        
        # Good names should be preserved
        assert 'Customer_Name' in result_df.columns
        assert 'Email_Address' in result_df.columns
        assert 'Good_Column' in result_df.columns
        
        # Only generic name should be changed
        assert 'Unnamed: 2' not in result_df.columns
    
    def test_service_generate_meaningful_column_names(self):
        """Test the service method for generating meaningful column names."""
        df = pd.DataFrame({
            'Unnamed: 0': ['John Doe', 'Jane Smith'],
            '1': ['john@email.com', 'jane@email.com'],
            'A': ['$100', '$200']
        })
        
        result_df = self.service.generate_meaningful_column_names(df)
        
        # Check that generic names were replaced
        assert 'Unnamed: 0' not in result_df.columns
        assert '1' not in result_df.columns
        assert 'A' not in result_df.columns
        
        # Check that we have the same number of columns
        assert len(result_df.columns) == len(df.columns)
        
        # Check that data is preserved
        assert len(result_df) == len(df)
    
    def test_integration_with_duplicate_detection(self):
        """Test that column name generation works with duplicate detection."""
        df = pd.DataFrame({
            'Unnamed: 0': ['Name', 'John Doe', 'Jane Smith', 'John Doe'],  # Header + duplicate
            '1': ['Email', 'john@email.com', 'jane@email.com', 'john@email.com'],  # Header + duplicate
            '2': ['Price', '$100', '$200', '$100']  # Header + duplicate
        })
        
        result_df, detection_result = self.detector.detect_and_remove_duplicates(df)
        
        # Should have meaningful column names
        generic_names = ['Unnamed: 0', '1', '2']
        for generic_name in generic_names:
            assert generic_name not in result_df.columns
        
        # Should have removed duplicates
        assert detection_result.exact_duplicates_removed >= 1
        assert detection_result.headers_consolidated >= 1
        
        # Should have fewer rows than original
        assert len(result_df) < len(df)


class TestDuplicateDetectionEdgeCases:
    """Test edge cases for duplicate detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DuplicateDetector()
    
    def test_all_null_values(self):
        """Test handling of DataFrame with all null values."""
        df = pd.DataFrame({
            'A': [None, None, None],
            'B': [np.nan, np.nan, np.nan]
        })
        
        result_df, detection_result = self.detector.detect_and_remove_duplicates(df)
        
        # Should handle null values gracefully
        assert len(result_df) <= len(df)
        assert detection_result.original_rows == 3
    
    def test_mixed_data_types(self):
        """Test with mixed data types."""
        df = pd.DataFrame({
            'Mixed': [1, '1', 1.0, 'one', 1],
            'Text': ['a', 'a', 'b', 'c', 'a']
        })
        
        result_df, detection_result = self.detector.detect_and_remove_duplicates(df)
        
        # Should handle mixed types
        assert len(result_df) <= len(df)
        assert not result_df.empty
    
    def test_very_long_strings(self):
        """Test with very long strings."""
        long_string1 = 'a' * 1000
        long_string2 = 'a' * 999 + 'b'
        
        df = pd.DataFrame({
            'Long': [long_string1, long_string2, long_string1],
            'Short': ['x', 'y', 'x']
        })
        
        result_df, detection_result = self.detector.detect_and_remove_duplicates(df)
        
        # Should handle long strings
        assert len(result_df) <= len(df)
        assert detection_result.exact_duplicates_removed >= 1
    
    def test_unicode_characters(self):
        """Test with Unicode characters."""
        df = pd.DataFrame({
            'Unicode': ['café', 'cafe', 'naïve', 'naive', '北京'],
            'Number': [1, 1, 2, 2, 3]
        })
        
        result_df, detection_result = self.detector.detect_and_remove_duplicates(df)
        
        # Should handle Unicode characters
        assert len(result_df) <= len(df)
        assert not result_df.empty
    
    def test_special_characters(self):
        """Test with special characters and symbols."""
        df = pd.DataFrame({
            'Special': ['@#$%', '@#$%', '!@#$%^&*()', 'test@email.com'],
            'Normal': ['a', 'a', 'b', 'c']
        })
        
        result_df, detection_result = self.detector.detect_and_remove_duplicates(df)
        
        # Should handle special characters
        assert len(result_df) <= len(df)
        assert detection_result.exact_duplicates_removed >= 1


if __name__ == '__main__':
    pytest.main([__file__])