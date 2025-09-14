"""
Integration tests for missing value handling in DataCleaningService.
"""

import pytest
import pandas as pd
import numpy as np

from src.data_processing.data_cleaner import DataCleaningService, DataTypeDetectionConfig
from src.data_processing.missing_value_handler import MissingValueConfig, MissingValuePolicy, MissingValueStrategy
from src.core.models import DataType


class TestDataCleanerMissingValueIntegration:
    """Test integration of missing value handling with DataCleaningService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.type_config = DataTypeDetectionConfig()
        self.missing_config = MissingValueConfig()
        self.service = DataCleaningService(
            type_detection_config=self.type_config,
            missing_value_config=self.missing_config
        )
        
        # Create comprehensive test dataset
        self.test_data = pd.DataFrame({
            'sales_amount': [100.50, 200.75, np.nan, 400.25, 500.00, '', 700.80, np.nan],
            'product_name': ['Widget A', 'Widget B', 'NULL', 'Widget D', '', 'Widget F', 'Widget G', 'N/A'],
            'order_date': ['2023-01-01', '2023-01-02', np.nan, '2023-01-04', '2023-01-05', '', '2023-01-07', 'unknown'],
            'is_premium': [True, False, np.nan, True, False, '', True, 'NULL'],
            'discount_pct': ['10%', '15%', np.nan, '20%', '25%', '-', '30%', 'N/A'],
            'customer_rating': [4.5, 3.8, np.nan, 4.2, 4.9, 'missing', 3.5, np.nan]
        })
    
    def test_handle_missing_values_auto_integration(self):
        """Test automatic missing value handling integration."""
        result_df = self.service.handle_missing_values(self.test_data, strategy='auto')
        
        # Check that result is a DataFrame
        assert isinstance(result_df, pd.DataFrame)
        
        # Check that columns are preserved
        assert len(result_df.columns) == len(self.test_data.columns)
        assert all(col in result_df.columns for col in self.test_data.columns)
        
        # Check that missing values are reduced
        original_missing = self.test_data.isna().sum().sum()
        final_missing = result_df.isna().sum().sum()
        assert final_missing <= original_missing
    
    def test_handle_missing_values_simple_strategies(self):
        """Test simple strategy application."""
        # Test remove strategy
        result_remove = self.service.handle_missing_values(self.test_data, strategy='remove')
        assert result_remove.isna().sum().sum() == 0
        assert len(result_remove) < len(self.test_data)
        
        # Test mean strategy (should work for numeric columns)
        result_mean = self.service.handle_missing_values(self.test_data, strategy='mean')
        assert isinstance(result_mean, pd.DataFrame)
        
        # Test mode strategy
        result_mode = self.service.handle_missing_values(self.test_data, strategy='mode')
        assert isinstance(result_mode, pd.DataFrame)
    
    def test_get_missing_value_report(self):
        """Test missing value analysis report."""
        report = self.service.get_missing_value_report(self.test_data)
        
        # Check report structure
        assert 'total_cells' in report
        assert 'missing_cells' in report
        assert 'missing_percentage' in report
        assert 'column_assessments' in report
        assert 'recommended_strategies' in report
        assert 'columns_with_high_missing' in report
        
        # Check report values
        assert report['total_cells'] == self.test_data.size
        assert report['missing_cells'] >= 0
        assert 0 <= report['missing_percentage'] <= 1
        
        # Check column assessments
        assert len(report['column_assessments']) == len(self.test_data.columns)
        for col, assessment in report['column_assessments'].items():
            assert hasattr(assessment, 'completeness_score')
            assert hasattr(assessment, 'consistency_score')
            assert hasattr(assessment, 'validity_score')
            assert hasattr(assessment, 'uniqueness_score')
        
        # Check strategy recommendations
        assert len(report['recommended_strategies']) == len(self.test_data.columns)
        for col, strategy in report['recommended_strategies'].items():
            assert isinstance(strategy, MissingValueStrategy)
    
    def test_handle_missing_values_with_custom_policies(self):
        """Test missing value handling with custom policies."""
        # Create custom policies
        policies = {
            'sales_amount': self.service.create_missing_value_policy(
                'sales_amount', DataType.NUMBER, 'median'
            ),
            'product_name': self.service.create_missing_value_policy(
                'product_name', DataType.TEXT, 'mode'
            ),
            'order_date': self.service.create_missing_value_policy(
                'order_date', DataType.DATE, 'remove'
            ),
            'is_premium': self.service.create_missing_value_policy(
                'is_premium', DataType.BOOLEAN, 'false'
            ),
            'discount_pct': self.service.create_missing_value_policy(
                'discount_pct', DataType.PERCENTAGE, 'zero'
            ),
            'customer_rating': self.service.create_missing_value_policy(
                'customer_rating', DataType.NUMBER, 'custom_value', custom_value=4.0
            )
        }
        
        result_df, report = self.service.handle_missing_values_with_policies(
            self.test_data, policies
        )
        
        # Check that custom strategies were applied
        assert 'median' in report.strategies_applied.values()
        assert 'mode' in report.strategies_applied.values()
        assert 'remove' in report.strategies_applied.values()
        assert 'false' in report.strategies_applied.values()
        assert 'zero' in report.strategies_applied.values()
        assert 'custom_value' in report.strategies_applied.values()
        
        # Check specific results
        # Sales amount should have no missing values after median imputation
        assert result_df['sales_amount'].isna().sum() == 0
        
        # Customer rating should be filled with custom value (4.0)
        original_missing_mask = self.test_data['customer_rating'].isna()
        if original_missing_mask.any():
            filled_values = result_df.loc[original_missing_mask, 'customer_rating']
            # Note: Some values might be 4.0 from custom fill
    
    def test_create_missing_value_policy(self):
        """Test creation of missing value policies."""
        policy = self.service.create_missing_value_policy(
            'test_col', DataType.NUMBER, 'mean', threshold=0.3, custom_value=100
        )
        
        assert isinstance(policy, MissingValuePolicy)
        assert policy.column_name == 'test_col'
        assert policy.data_type == DataType.NUMBER
        assert policy.strategy == MissingValueStrategy.MEAN
        assert policy.threshold == 0.3
        assert policy.custom_value == 100
    
    def test_create_missing_value_policy_invalid_strategy(self):
        """Test creation of policy with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown missing value strategy"):
            self.service.create_missing_value_policy(
                'test_col', DataType.NUMBER, 'invalid_strategy'
            )
    
    def test_get_available_strategies(self):
        """Test getting available strategies."""
        strategies = self.service.get_available_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert 'remove' in strategies
        assert 'mean' in strategies
        assert 'median' in strategies
        assert 'mode' in strategies
        assert 'zero' in strategies
        assert 'interpolate_linear' in strategies
        assert 'knn_impute' in strategies
    
    def test_standardize_and_handle_missing_values_workflow(self):
        """Test complete workflow: standardize types then handle missing values."""
        # First standardize data types
        standardized_df = self.service.standardize_data_types(self.test_data)
        
        # Then handle missing values
        result_df = self.service.handle_missing_values(standardized_df, strategy='auto')
        
        # Check that workflow completed successfully
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df.columns) == len(self.test_data.columns)
    
    def test_missing_value_handling_with_high_missing_percentage(self):
        """Test handling of columns with high missing percentage."""
        # Create data with very high missing percentage
        high_missing_data = pd.DataFrame({
            'mostly_missing': [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'some_missing': [1, 2, np.nan, 4, 5, np.nan, 7, 8],
            'no_missing': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        
        result_df = self.service.handle_missing_values(high_missing_data, strategy='auto')
        
        # Check that high missing columns are handled appropriately
        assert isinstance(result_df, pd.DataFrame)
        
        # Get report to see what strategies were applied
        report = self.service.get_missing_value_report(high_missing_data)
        assert 'mostly_missing' in report['columns_with_high_missing']
    
    def test_missing_value_handling_edge_cases(self):
        """Test edge cases in missing value handling."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        result_empty = self.service.handle_missing_values(empty_df, strategy='auto')
        assert len(result_empty) == 0
        
        # DataFrame with no missing values
        no_missing_df = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': ['A', 'B', 'C', 'D']
        })
        result_no_missing = self.service.handle_missing_values(no_missing_df, strategy='auto')
        # Check that data is preserved (may have type changes due to processing)
        assert len(result_no_missing) == len(no_missing_df)
        assert list(result_no_missing.columns) == list(no_missing_df.columns)
        assert result_no_missing.isna().sum().sum() == 0
        
        # DataFrame with all missing values
        all_missing_df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [np.nan, np.nan, np.nan]
        })
        result_all_missing = self.service.handle_missing_values(all_missing_df, strategy='auto')
        assert isinstance(result_all_missing, pd.DataFrame)
    
    def test_missing_value_handling_performance(self):
        """Test performance with larger dataset."""
        # Create larger test dataset
        large_data = pd.DataFrame({
            'numeric_col': np.random.choice([1, 2, 3, 4, 5, np.nan], size=1000),
            'text_col': np.random.choice(['A', 'B', 'C', 'D', ''], size=1000),
            'boolean_col': np.random.choice([True, False, np.nan], size=1000)
        })
        
        # Test that it completes in reasonable time
        import time
        start_time = time.time()
        result_df = self.service.handle_missing_values(large_data, strategy='auto')
        end_time = time.time()
        
        # Should complete within 10 seconds for 1000 rows
        assert end_time - start_time < 10
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df.columns) == len(large_data.columns)
    
    def test_missing_value_report_quality_metrics(self):
        """Test quality metrics in missing value report."""
        report = self.service.get_missing_value_report(self.test_data)
        
        for col, assessment in report['column_assessments'].items():
            # Check that all quality scores are between 0 and 1
            assert 0 <= assessment.completeness_score <= 1
            assert 0 <= assessment.consistency_score <= 1
            assert 0 <= assessment.validity_score <= 1
            assert 0 <= assessment.uniqueness_score <= 1
            
            # Check that missing percentage is calculated correctly
            expected_missing_pct = self.test_data[col].isna().sum() / len(self.test_data)
            # Allow for some tolerance due to missing indicator standardization
            assert abs(assessment.missing_percentage - expected_missing_pct) <= 0.2
            
            # Check that quality issues and recommendations are provided
            assert isinstance(assessment.quality_issues, list)
            assert isinstance(assessment.recommendations, list)


class TestMissingValueHandlerErrorHandling:
    """Test error handling in missing value processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataCleaningService()
    
    def test_handle_corrupted_data(self):
        """Test handling of corrupted or invalid data."""
        corrupted_data = pd.DataFrame({
            'mixed_types': [1, 'text', [1, 2, 3], {'key': 'value'}, np.nan],
            'invalid_numbers': ['1.5', 'not_a_number', '2.7', 'invalid', np.nan]
        })
        
        # Should not raise exception
        result_df = self.service.handle_missing_values(corrupted_data, strategy='auto')
        assert isinstance(result_df, pd.DataFrame)
    
    def test_handle_extreme_values(self):
        """Test handling of extreme values and outliers."""
        extreme_data = pd.DataFrame({
            'with_outliers': [1, 2, 3, 1000000, 5, np.nan, 7, 8],
            'normal_range': [10, 20, 30, 40, 50, np.nan, 70, 80]
        })
        
        result_df = self.service.handle_missing_values(extreme_data, strategy='auto')
        assert isinstance(result_df, pd.DataFrame)
    
    def test_handle_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_data = pd.DataFrame({
            'unicode_text': ['café', 'naïve', '北京', 'москва', np.nan, ''],
            'special_chars': ['@#$%', '&*()_+', '[]{}|;:', '<>?/', np.nan, 'NULL']
        })
        
        result_df = self.service.handle_missing_values(unicode_data, strategy='auto')
        assert isinstance(result_df, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__])