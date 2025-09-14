"""
Tests for the enhanced missing value handling system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.data_processing.missing_value_handler import (
    MissingValueHandler, MissingValueConfig, MissingValuePolicy,
    MissingValueStrategy, DataQualityAssessor, DataQualityAssessment,
    handle_missing_values_simple, assess_data_quality
)
from src.core.models import DataType


class TestMissingValueHandler:
    """Test cases for MissingValueHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MissingValueConfig()
        self.handler = MissingValueHandler(self.config)
        
        # Create test DataFrame with various missing value scenarios
        self.test_data = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0],
            'text_col': ['A', 'B', '', 'D', 'E', 'NULL', 'G'],
            'date_col': ['2023-01-01', '2023-01-02', np.nan, '2023-01-04', 
                        '2023-01-05', 'N/A', '2023-01-07'],
            'boolean_col': [True, False, np.nan, True, False, '', True],
            'currency_col': ['$100', '$200', np.nan, '$400', '$500', '-', '$700'],
            'percentage_col': ['10%', '20%', np.nan, '40%', '50%', 'unknown', '70%']
        })
        
        self.data_types = {
            'numeric_col': DataType.NUMBER,
            'text_col': DataType.TEXT,
            'date_col': DataType.DATE,
            'boolean_col': DataType.BOOLEAN,
            'currency_col': DataType.CURRENCY,
            'percentage_col': DataType.PERCENTAGE
        }
    
    def test_standardize_missing_indicators(self):
        """Test standardization of missing value indicators."""
        result_df = self.handler._standardize_missing_indicators(self.test_data)
        
        # Check that missing indicators are converted to NaN
        assert result_df['text_col'].isna().sum() == 2  # '' and 'NULL'
        assert result_df['date_col'].isna().sum() == 2  # np.nan and 'N/A'
        assert result_df['boolean_col'].isna().sum() == 2  # np.nan and ''
        assert result_df['currency_col'].isna().sum() == 2  # np.nan and '-'
        assert result_df['percentage_col'].isna().sum() == 2  # np.nan and 'unknown'
    
    def test_generate_default_policies(self):
        """Test generation of default policies based on data types."""
        policies = self.handler._generate_default_policies(self.test_data, self.data_types)
        
        assert len(policies) == len(self.test_data.columns)
        
        # Check that appropriate strategies are selected
        assert policies['numeric_col'].strategy in [
            MissingValueStrategy.MEAN, MissingValueStrategy.MEDIAN, 
            MissingValueStrategy.INTERPOLATE_LINEAR
        ]
        assert policies['text_col'].strategy in [
            MissingValueStrategy.MODE, MissingValueStrategy.EMPTY_STRING,
            MissingValueStrategy.REMOVE
        ]
        assert policies['date_col'].strategy in [
            MissingValueStrategy.FORWARD_FILL, MissingValueStrategy.REMOVE
        ]
        assert policies['boolean_col'].strategy in [
            MissingValueStrategy.MODE, MissingValueStrategy.FALSE
        ]
    
    def test_handle_missing_values_auto(self):
        """Test automatic missing value handling."""
        result_df, report = self.handler.handle_missing_values(
            self.test_data, data_types=self.data_types
        )
        
        # Check that missing values are reduced
        assert report.original_missing_count > report.final_missing_count
        assert report.quality_improvement >= 0
        assert len(report.strategies_applied) == len(self.test_data.columns)
        
        # Check that DataFrame structure is maintained
        assert len(result_df.columns) == len(self.test_data.columns)
        assert all(col in result_df.columns for col in self.test_data.columns)
    
    def test_handle_missing_values_with_custom_policies(self):
        """Test missing value handling with custom policies."""
        policies = {
            'numeric_col': MissingValuePolicy(
                column_name='numeric_col',
                data_type=DataType.NUMBER,
                strategy=MissingValueStrategy.MEAN
            ),
            'text_col': MissingValuePolicy(
                column_name='text_col',
                data_type=DataType.TEXT,
                strategy=MissingValueStrategy.MODE
            )
        }
        
        result_df, report = self.handler.handle_missing_values(
            self.test_data, policies=policies, data_types=self.data_types
        )
        
        # Check that specified strategies were applied
        assert report.strategies_applied['numeric_col'] == 'mean'
        assert report.strategies_applied['text_col'] == 'mode'
        
        # Check that numeric column has no missing values after mean imputation
        assert result_df['numeric_col'].isna().sum() == 0
    
    def test_mean_strategy(self):
        """Test mean imputation strategy."""
        policy = MissingValuePolicy(
            column_name='numeric_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.MEAN
        )
        
        result_df = self.handler._apply_mean_strategy(self.test_data, 'numeric_col', policy)
        
        # Check that missing values are filled with mean
        expected_mean = self.test_data['numeric_col'].mean()
        filled_values = result_df.loc[self.test_data['numeric_col'].isna(), 'numeric_col']
        assert all(val == expected_mean for val in filled_values)
    
    def test_median_strategy(self):
        """Test median imputation strategy."""
        policy = MissingValuePolicy(
            column_name='numeric_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.MEDIAN
        )
        
        result_df = self.handler._apply_median_strategy(self.test_data, 'numeric_col', policy)
        
        # Check that missing values are filled with median
        expected_median = self.test_data['numeric_col'].median()
        filled_values = result_df.loc[self.test_data['numeric_col'].isna(), 'numeric_col']
        assert all(val == expected_median for val in filled_values)
    
    def test_mode_strategy(self):
        """Test mode imputation strategy."""
        policy = MissingValuePolicy(
            column_name='text_col',
            data_type=DataType.TEXT,
            strategy=MissingValueStrategy.MODE
        )
        
        # First standardize missing indicators
        standardized_df = self.handler._standardize_missing_indicators(self.test_data)
        result_df = self.handler._apply_mode_strategy(standardized_df, 'text_col', policy)
        
        # Check that missing values are filled with mode
        mode_value = standardized_df['text_col'].mode().iloc[0]
        filled_values = result_df.loc[standardized_df['text_col'].isna(), 'text_col']
        assert all(val == mode_value for val in filled_values)
    
    def test_remove_strategy(self):
        """Test removal strategy."""
        policy = MissingValuePolicy(
            column_name='numeric_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.REMOVE
        )
        
        original_length = len(self.test_data)
        missing_count = self.test_data['numeric_col'].isna().sum()
        
        result_df = self.handler._apply_remove_strategy(self.test_data, 'numeric_col', policy)
        
        # Check that rows with missing values are removed
        assert len(result_df) == original_length - missing_count
        assert result_df['numeric_col'].isna().sum() == 0
    
    def test_zero_strategy(self):
        """Test zero fill strategy."""
        policy = MissingValuePolicy(
            column_name='numeric_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.ZERO
        )
        
        result_df = self.handler._apply_zero_strategy(self.test_data, 'numeric_col', policy)
        
        # Check that missing values are filled with zero
        filled_values = result_df.loc[self.test_data['numeric_col'].isna(), 'numeric_col']
        assert all(val == 0 for val in filled_values)
    
    def test_custom_value_strategy(self):
        """Test custom value strategy."""
        custom_value = 999
        policy = MissingValuePolicy(
            column_name='numeric_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.CUSTOM_VALUE,
            custom_value=custom_value
        )
        
        result_df = self.handler._apply_custom_value_strategy(self.test_data, 'numeric_col', policy)
        
        # Check that missing values are filled with custom value
        filled_values = result_df.loc[self.test_data['numeric_col'].isna(), 'numeric_col']
        assert all(val == custom_value for val in filled_values)
    
    def test_linear_interpolation_strategy(self):
        """Test linear interpolation strategy."""
        policy = MissingValuePolicy(
            column_name='numeric_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.INTERPOLATE_LINEAR
        )
        
        result_df = self.handler._apply_linear_interpolation(self.test_data, 'numeric_col', policy)
        
        # Check that missing values are interpolated
        assert result_df['numeric_col'].isna().sum() < self.test_data['numeric_col'].isna().sum()
    
    @patch('src.data_processing.missing_value_handler.KNNImputer')
    def test_knn_imputation_strategy(self, mock_knn):
        """Test KNN imputation strategy."""
        # Mock KNNImputer
        mock_imputer = Mock()
        mock_imputer.fit_transform.return_value = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])
        mock_knn.return_value = mock_imputer
        
        policy = MissingValuePolicy(
            column_name='numeric_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.KNN_IMPUTE,
            knn_neighbors=3
        )
        
        # Create a DataFrame with numeric columns for KNN
        numeric_df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0],
            'other_numeric': [10, 20, 30, 40, 50, 60, 70]
        })
        
        result_df = self.handler._apply_knn_imputation(numeric_df, 'numeric_col', policy)
        
        # Check that KNN imputation was attempted
        mock_knn.assert_called_once_with(n_neighbors=3)
    
    def test_forward_fill_strategy(self):
        """Test forward fill strategy."""
        policy = MissingValuePolicy(
            column_name='numeric_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.FORWARD_FILL
        )
        
        result_df = self.handler._apply_forward_fill_strategy(self.test_data, 'numeric_col', policy)
        
        # Check that forward fill was applied
        assert result_df['numeric_col'].isna().sum() <= self.test_data['numeric_col'].isna().sum()
    
    def test_backward_fill_strategy(self):
        """Test backward fill strategy."""
        policy = MissingValuePolicy(
            column_name='numeric_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.BACKWARD_FILL
        )
        
        result_df = self.handler._apply_backward_fill_strategy(self.test_data, 'numeric_col', policy)
        
        # Check that backward fill was applied
        assert result_df['numeric_col'].isna().sum() <= self.test_data['numeric_col'].isna().sum()
    
    def test_create_user_policy(self):
        """Test creation of user-defined policies."""
        policy = self.handler.create_user_policy(
            column_name='test_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.MEAN,
            threshold=0.3,
            custom_value=100
        )
        
        assert policy.column_name == 'test_col'
        assert policy.data_type == DataType.NUMBER
        assert policy.strategy == MissingValueStrategy.MEAN
        assert policy.threshold == 0.3
        assert policy.custom_value == 100
    
    def test_get_quality_assessment(self):
        """Test data quality assessment."""
        assessments = self.handler.get_quality_assessment(self.test_data, self.data_types)
        
        assert len(assessments) == len(self.test_data.columns)
        
        for col, assessment in assessments.items():
            assert isinstance(assessment, DataQualityAssessment)
            assert assessment.column_name == col
            assert 0 <= assessment.completeness_score <= 1
            assert 0 <= assessment.consistency_score <= 1
            assert 0 <= assessment.validity_score <= 1
            assert 0 <= assessment.uniqueness_score <= 1
    
    def test_recommend_strategies(self):
        """Test strategy recommendations."""
        recommendations = self.handler.recommend_strategies(self.test_data, self.data_types)
        
        assert len(recommendations) == len(self.test_data.columns)
        
        for col, strategy in recommendations.items():
            assert isinstance(strategy, MissingValueStrategy)


class TestDataQualityAssessor:
    """Test cases for DataQualityAssessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MissingValueConfig()
        self.assessor = DataQualityAssessor(self.config)
    
    def test_assess_column_quality_numeric(self):
        """Test quality assessment for numeric column."""
        series = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7])
        assessment = self.assessor.assess_column_quality(series, 'test_col', DataType.NUMBER)
        
        assert assessment.column_name == 'test_col'
        assert assessment.total_rows == 7
        assert assessment.missing_count == 2
        assert assessment.missing_percentage == 2/7
        assert 0 <= assessment.completeness_score <= 1
        assert 0 <= assessment.consistency_score <= 1
        assert 0 <= assessment.validity_score <= 1
        assert 0 <= assessment.uniqueness_score <= 1
    
    def test_assess_column_quality_text(self):
        """Test quality assessment for text column."""
        series = pd.Series(['A', 'B', '', 'D', 'E', 'NULL', 'G'])
        assessment = self.assessor.assess_column_quality(series, 'test_col', DataType.TEXT)
        
        assert assessment.column_name == 'test_col'
        assert assessment.total_rows == 7
        assert assessment.uniqueness_score > 0  # Should have unique values
    
    def test_assess_column_quality_empty_series(self):
        """Test quality assessment for empty series."""
        series = pd.Series([], dtype=object)
        assessment = self.assessor.assess_column_quality(series, 'test_col', DataType.TEXT)
        
        # Empty series has completeness score of 1.0 (no missing values in empty set)
        assert assessment.completeness_score == 1.0
        assert assessment.total_rows == 0
        assert assessment.missing_count == 0
    
    def test_assess_dataset_quality(self):
        """Test quality assessment for entire dataset."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': ['A', 'B', '', 'D']
        })
        data_types = {'col1': DataType.NUMBER, 'col2': DataType.TEXT}
        
        assessments = self.assessor.assess_dataset_quality(df, data_types)
        
        assert len(assessments) == 2
        assert 'col1' in assessments
        assert 'col2' in assessments
        assert all(isinstance(assessment, DataQualityAssessment) 
                  for assessment in assessments.values())


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'text_col': ['A', 'B', '', 'D', 'E']
        })
        
        self.data_types = {
            'numeric_col': DataType.NUMBER,
            'text_col': DataType.TEXT
        }
    
    def test_handle_missing_values_simple_auto(self):
        """Test simple missing value handling with auto strategy."""
        result_df = handle_missing_values_simple(self.test_data, 'auto', self.data_types)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df.columns) == len(self.test_data.columns)
    
    def test_handle_missing_values_simple_remove(self):
        """Test simple missing value handling with remove strategy."""
        result_df = handle_missing_values_simple(self.test_data, 'remove')
        
        # Should remove rows with any missing values
        assert len(result_df) < len(self.test_data)
        assert result_df.isna().sum().sum() == 0
    
    def test_handle_missing_values_simple_mean(self):
        """Test simple missing value handling with mean strategy."""
        result_df = handle_missing_values_simple(self.test_data, 'mean', self.data_types)
        
        assert isinstance(result_df, pd.DataFrame)
        # Numeric column should have no missing values after mean imputation
        assert result_df['numeric_col'].isna().sum() == 0
    
    def test_handle_missing_values_simple_unknown_strategy(self):
        """Test simple missing value handling with unknown strategy."""
        result_df = handle_missing_values_simple(self.test_data, 'unknown_strategy')
        
        # Should fallback to removal
        assert result_df.isna().sum().sum() == 0
    
    def test_assess_data_quality_function(self):
        """Test data quality assessment convenience function."""
        assessments = assess_data_quality(self.test_data, self.data_types)
        
        assert len(assessments) == len(self.test_data.columns)
        assert all(isinstance(assessment, DataQualityAssessment) 
                  for assessment in assessments.values())
    
    def test_assess_data_quality_no_types(self):
        """Test data quality assessment without data types."""
        assessments = assess_data_quality(self.test_data)
        
        assert len(assessments) == len(self.test_data.columns)
        # Should default to TEXT type for all columns
        assert all(isinstance(assessment, DataQualityAssessment) 
                  for assessment in assessments.values())


class TestMissingValueConfig:
    """Test cases for MissingValueConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MissingValueConfig()
        
        assert isinstance(config.missing_indicators, list)
        assert len(config.missing_indicators) > 0
        assert '' in config.missing_indicators
        assert 'NULL' in config.missing_indicators
        assert 'N/A' in config.missing_indicators
        
        assert isinstance(config.quality_thresholds, dict)
        assert 'completeness' in config.quality_thresholds
        assert 'consistency' in config.quality_thresholds
        assert 'validity' in config.quality_thresholds
        assert 'uniqueness' in config.quality_thresholds
        
        assert isinstance(config.auto_detect_outliers, bool)
        assert isinstance(config.outlier_z_threshold, float)
        assert isinstance(config.max_missing_percentage, float)
        assert isinstance(config.enable_advanced_imputation, bool)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_indicators = ['', 'missing', 'void']
        custom_thresholds = {'completeness': 0.9, 'validity': 0.8}
        
        config = MissingValueConfig(
            missing_indicators=custom_indicators,
            quality_thresholds=custom_thresholds,
            outlier_z_threshold=2.5,
            max_missing_percentage=0.6
        )
        
        assert config.missing_indicators == custom_indicators
        assert config.quality_thresholds == custom_thresholds
        assert config.outlier_z_threshold == 2.5
        assert config.max_missing_percentage == 0.6


class TestMissingValuePolicy:
    """Test cases for MissingValuePolicy class."""
    
    def test_policy_creation(self):
        """Test creation of missing value policy."""
        policy = MissingValuePolicy(
            column_name='test_col',
            data_type=DataType.NUMBER,
            strategy=MissingValueStrategy.MEAN,
            custom_value=100,
            threshold=0.3,
            fallback_strategy=MissingValueStrategy.REMOVE
        )
        
        assert policy.column_name == 'test_col'
        assert policy.data_type == DataType.NUMBER
        assert policy.strategy == MissingValueStrategy.MEAN
        assert policy.custom_value == 100
        assert policy.threshold == 0.3
        assert policy.fallback_strategy == MissingValueStrategy.REMOVE
    
    def test_policy_defaults(self):
        """Test default values in policy."""
        policy = MissingValuePolicy(
            column_name='test_col',
            data_type=DataType.TEXT,
            strategy=MissingValueStrategy.MODE
        )
        
        assert policy.threshold == 0.5  # Default threshold
        assert policy.custom_value is None
        assert policy.fallback_strategy is None
        assert policy.interpolation_method == 'linear'
        assert policy.knn_neighbors == 5
        assert policy.polynomial_order == 2


if __name__ == '__main__':
    pytest.main([__file__])