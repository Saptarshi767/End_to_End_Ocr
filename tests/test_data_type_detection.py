"""
Unit tests for data type detection and conversion functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from decimal import Decimal

from src.data_processing.data_cleaner import (
    DataTypeDetector, DataCleaningService, TypeDetectionResult,
    DataTypeDetectionConfig, MissingValueConfig,
    detect_data_types, standardize_dataframe
)
from src.core.models import DataType


class TestDataTypeDetector:
    """Test cases for DataTypeDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DataTypeDetector()
        self.config = DataTypeDetectionConfig(confidence_threshold=0.6)
        self.custom_detector = DataTypeDetector(self.config)
    
    def test_detect_boolean_column(self):
        """Test detection of boolean columns."""
        # Test various boolean formats
        boolean_data = pd.Series(['true', 'false', 'true', 'false', 'true'])
        result = self.detector.detect_column_type(boolean_data)
        
        assert result.detected_type == DataType.BOOLEAN
        assert result.confidence >= 0.7
        assert result.conversion_function is not None
        
        # Test yes/no format
        yes_no_data = pd.Series(['yes', 'no', 'yes', 'no', 'yes'])
        result = self.detector.detect_column_type(yes_no_data)
        
        assert result.detected_type == DataType.BOOLEAN
        assert result.confidence >= 0.7
        
        # Test 1/0 format
        binary_data = pd.Series(['1', '0', '1', '0', '1'])
        result = self.detector.detect_column_type(binary_data)
        
        assert result.detected_type == DataType.BOOLEAN
        assert result.confidence >= 0.7
    
    def test_detect_currency_column(self):
        """Test detection of currency columns."""
        currency_data = pd.Series(['$100.50', '$25.00', '$1,500.75', '$0.99'])
        result = self.detector.detect_column_type(currency_data)
        
        assert result.detected_type == DataType.CURRENCY
        assert result.confidence >= 0.7
        assert result.conversion_function is not None
        
        # Test conversion
        converted_value = result.conversion_function('$100.50')
        assert converted_value == 100.50
        
        # Test different currency symbols
        euro_data = pd.Series(['€50.25', '€100.00', '€25.50'])
        result = self.detector.detect_column_type(euro_data)
        
        assert result.detected_type == DataType.CURRENCY
        assert result.confidence >= 0.7
    
    def test_detect_percentage_column(self):
        """Test detection of percentage columns."""
        percentage_data = pd.Series(['25%', '50%', '75%', '100%', '12.5%'])
        result = self.detector.detect_column_type(percentage_data)
        
        assert result.detected_type == DataType.PERCENTAGE
        assert result.confidence >= 0.7
        assert result.conversion_function is not None
        
        # Test conversion
        converted_value = result.conversion_function('25%')
        assert converted_value == 0.25
        
        converted_value = result.conversion_function('12.5%')
        assert converted_value == 0.125
    
    def test_detect_date_column(self):
        """Test detection of date columns."""
        date_data = pd.Series(['2023-01-15', '2023-02-20', '2023-03-25', '2023-04-30'])
        result = self.detector.detect_column_type(date_data)
        
        assert result.detected_type == DataType.DATE
        assert result.confidence >= 0.7
        assert result.conversion_function is not None
        
        # Test conversion
        converted_value = result.conversion_function('2023-01-15')
        assert isinstance(converted_value, date)
        assert converted_value == date(2023, 1, 15)
        
        # Test different date formats
        us_date_data = pd.Series(['01/15/2023', '02/20/2023', '03/25/2023'])
        result = self.detector.detect_column_type(us_date_data)
        
        assert result.detected_type == DataType.DATE
        assert result.confidence >= 0.7
    
    def test_detect_number_column(self):
        """Test detection of numeric columns."""
        # Test integers
        int_data = pd.Series(['100', '200', '300', '400', '500'])
        result = self.detector.detect_column_type(int_data)
        
        assert result.detected_type == DataType.NUMBER
        assert result.confidence >= 0.7
        assert result.conversion_function is not None
        
        # Test conversion
        converted_value = result.conversion_function('100')
        assert converted_value == 100
        assert isinstance(converted_value, int)
        
        # Test floats
        float_data = pd.Series(['100.5', '200.25', '300.75', '400.0'])
        result = self.detector.detect_column_type(float_data)
        
        assert result.detected_type == DataType.NUMBER
        assert result.confidence >= 0.7
        
        converted_value = result.conversion_function('100.5')
        assert converted_value == 100.5
        assert isinstance(converted_value, float)
        
        # Test numbers with commas
        comma_data = pd.Series(['1,000', '2,500', '10,000', '100,000'])
        result = self.detector.detect_column_type(comma_data)
        
        assert result.detected_type == DataType.NUMBER
        assert result.confidence >= 0.7
    
    def test_detect_text_column(self):
        """Test detection of text columns."""
        text_data = pd.Series(['apple', 'banana', 'cherry', 'date', 'elderberry'])
        result = self.detector.detect_column_type(text_data)
        
        assert result.detected_type == DataType.TEXT
        assert result.confidence == 1.0
        assert result.conversion_function is not None
    
    def test_mixed_data_column(self):
        """Test detection with mixed data types."""
        # Mixed data should default to text
        mixed_data = pd.Series(['100', 'apple', '2023-01-01', 'true', '$50.00'])
        result = self.detector.detect_column_type(mixed_data)
        
        # Should detect as text since no single type has high confidence
        assert result.detected_type == DataType.TEXT
    
    def test_empty_column(self):
        """Test detection with empty or all-null columns."""
        empty_data = pd.Series([])
        result = self.detector.detect_column_type(empty_data)
        
        assert result.detected_type == DataType.TEXT
        assert result.confidence == 0.0
        
        null_data = pd.Series([None, None, None, None])
        result = self.detector.detect_column_type(null_data)
        
        assert result.detected_type == DataType.TEXT
        assert result.confidence == 0.0
    
    def test_column_with_missing_values(self):
        """Test detection with missing values."""
        data_with_nulls = pd.Series(['100', '200', None, '400', 'N/A', '600'])
        result = self.detector.detect_column_type(data_with_nulls)
        
        # Should still detect as number based on valid values
        assert result.detected_type == DataType.NUMBER
        assert result.confidence >= 0.6
    
    def test_custom_configuration(self):
        """Test detection with custom configuration."""
        # Test with lower confidence threshold
        low_confidence_config = DataTypeDetectionConfig(confidence_threshold=0.3)
        detector = DataTypeDetector(low_confidence_config)
        
        # Mixed data that might not meet high confidence threshold
        mixed_numeric = pd.Series(['100', '200', 'abc', '400'])  # 75% numeric
        result = detector.detect_column_type(mixed_numeric)
        
        # With lower threshold, might detect as number
        assert result.confidence > 0.3
    
    def test_convert_column_functionality(self):
        """Test column conversion functionality."""
        # Test converting text numbers to actual numbers
        text_numbers = pd.Series(['100', '200', '300', '400'])
        converted = self.detector.convert_column(text_numbers, DataType.NUMBER)
        
        assert converted.dtype in [np.int64, np.float64]
        assert converted.iloc[0] == 100
        
        # Test converting to text
        numbers = pd.Series([100, 200, 300, 400])
        converted = self.detector.convert_column(numbers, DataType.TEXT)
        
        assert converted.dtype == object
        assert converted.iloc[0] == '100'


class TestDataCleaningService:
    """Test cases for DataCleaningService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataCleaningService()
    
    def test_standardize_data_types(self):
        """Test data type standardization."""
        # Create test DataFrame with mixed types
        test_data = pd.DataFrame({
            'numbers': ['100', '200', '300', '400'],
            'currencies': ['$50.00', '$75.25', '$100.50', '$25.75'],
            'percentages': ['25%', '50%', '75%', '100%'],
            'booleans': ['true', 'false', 'true', 'false'],
            'dates': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
            'text': ['apple', 'banana', 'cherry', 'date']
        })
        
        standardized = self.service.standardize_data_types(test_data)
        
        # Verify that data was converted appropriately
        assert len(standardized) == len(test_data)
        assert list(standardized.columns) == list(test_data.columns)
        
        # Check that original DataFrame wasn't modified
        assert test_data['numbers'].iloc[0] == '100'
    
    def test_handle_missing_values_auto_strategy(self):
        """Test automatic missing value handling."""
        test_data = pd.DataFrame({
            'numbers': [100, 200, None, 400, 500],
            'text': ['apple', 'banana', None, 'date', ''],
            'mixed': [100, 'text', None, 400, 'N/A']
        })
        
        cleaned = self.service.handle_missing_values(test_data, strategy='auto')
        
        # Verify DataFrame structure is maintained
        assert len(cleaned.columns) == len(test_data.columns)
        assert list(cleaned.columns) == list(test_data.columns)
    
    def test_handle_missing_values_specific_strategies(self):
        """Test specific missing value strategies."""
        from src.data_processing.missing_value_handler import handle_missing_values_simple
        from src.core.models import DataType
        
        numeric_data = pd.DataFrame({
            'values': [100, 200, None, 400, 500]
        })
        
        data_types = {'values': DataType.NUMBER}
        
        # Test mean strategy
        mean_filled = handle_missing_values_simple(numeric_data, 'mean', data_types)
        expected_mean = (100 + 200 + 400 + 500) / 4
        assert mean_filled['values'].iloc[2] == expected_mean
        
        # Test median strategy
        median_filled = handle_missing_values_simple(numeric_data, 'median', data_types)
        expected_median = 300.0  # median of [100, 200, 400, 500]
        assert median_filled['values'].iloc[2] == expected_median
        
        # Test zero strategy
        zero_filled = handle_missing_values_simple(numeric_data, 'zero', data_types)
        assert zero_filled['values'].iloc[2] == 0
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        test_data = pd.DataFrame({
            'col1': [1, 2, 3, 2, 4, 1],
            'col2': ['a', 'b', 'c', 'b', 'd', 'a']
        })
        
        deduplicated = self.service.remove_duplicates(test_data)
        
        # Should have fewer rows after removing duplicates
        assert len(deduplicated) < len(test_data)
        
        # Should have unique combinations
        assert len(deduplicated) == len(deduplicated.drop_duplicates())
    
    def test_detect_schema(self):
        """Test schema detection."""
        test_data = pd.DataFrame({
            'numbers': [100, 200, 300, 400],
            'text': ['apple', 'banana', 'cherry', 'date'],
            'mixed': [100, 'text', 300, 'more_text']
        })
        
        schema = self.service.detect_schema(test_data)
        
        assert len(schema.columns) == 3
        assert schema.row_count == 4
        assert len(schema.data_types) == 3
        
        # Check that column info is populated
        for column_info in schema.columns:
            assert column_info.name in test_data.columns
            assert isinstance(column_info.data_type, DataType)
            assert column_info.unique_values > 0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_detect_data_types_function(self):
        """Test the detect_data_types convenience function."""
        test_data = pd.DataFrame({
            'numbers': ['100', '200', '300'],
            'text': ['apple', 'banana', 'cherry']
        })
        
        results = detect_data_types(test_data)
        
        assert len(results) == 2
        assert 'numbers' in results
        assert 'text' in results
        
        assert isinstance(results['numbers'], TypeDetectionResult)
        assert isinstance(results['text'], TypeDetectionResult)
        
        assert results['numbers'].detected_type == DataType.NUMBER
        assert results['text'].detected_type == DataType.TEXT
    
    def test_standardize_dataframe_function(self):
        """Test the standardize_dataframe convenience function."""
        test_data = pd.DataFrame({
            'numbers': ['100', '200', None, '400'],
            'text': ['apple', 'banana', '', 'date'],
            'duplicates': [1, 2, 1, 3]
        })
        
        # Add a duplicate row
        test_data = pd.concat([test_data, test_data.iloc[[0]]], ignore_index=True)
        
        standardized = standardize_dataframe(test_data)
        
        # Should have processed missing values and removed duplicates
        assert len(standardized) <= len(test_data)
        assert len(standardized.columns) == len(test_data.columns)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DataTypeDetector()
        self.service = DataCleaningService()
    
    def test_single_value_column(self):
        """Test detection with single value columns."""
        single_value = pd.Series(['100'])
        result = self.detector.detect_column_type(single_value)
        
        assert result.detected_type == DataType.NUMBER
        assert result.confidence >= 0.7
    
    def test_all_whitespace_column(self):
        """Test detection with whitespace-only values."""
        whitespace_data = pd.Series(['   ', '\t', '\n', '  \t  '])
        result = self.detector.detect_column_type(whitespace_data)
        
        # Should be treated as empty and default to text
        assert result.detected_type == DataType.TEXT
    
    def test_very_large_numbers(self):
        """Test detection with very large numbers."""
        large_numbers = pd.Series(['999999999999', '888888888888', '777777777777'])
        result = self.detector.detect_column_type(large_numbers)
        
        assert result.detected_type == DataType.NUMBER
        assert result.confidence >= 0.7
    
    def test_scientific_notation(self):
        """Test detection with scientific notation."""
        scientific_data = pd.Series(['1.5e10', '2.3e-5', '4.7e8'])
        result = self.detector.detect_column_type(scientific_data)
        
        assert result.detected_type == DataType.NUMBER
        assert result.confidence >= 0.7
    
    def test_malformed_data(self):
        """Test handling of malformed data."""
        malformed_data = pd.DataFrame({
            'bad_numbers': ['12.34.56', '78..90', 'abc123'],
            'bad_dates': ['2023-13-45', '99/99/9999', 'not-a-date'],
            'bad_currency': ['$$100', '€€50', 'money']
        })
        
        # Should not raise exceptions
        standardized = self.service.standardize_data_types(malformed_data)
        assert len(standardized) == len(malformed_data)
        
        schema = self.service.detect_schema(malformed_data)
        assert len(schema.columns) == 3
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_data = pd.DataFrame({
            'unicode_text': ['café', 'naïve', '北京', 'москва'],
            'special_chars': ['@#$%', '!@#$%^&*()', '[]{}|\\', '<>?/']
        })
        
        # Should handle unicode gracefully
        standardized = self.service.standardize_data_types(unicode_data)
        assert len(standardized) == len(unicode_data)
        
        schema = self.service.detect_schema(unicode_data)
        assert len(schema.columns) == 2


if __name__ == '__main__':
    pytest.main([__file__])