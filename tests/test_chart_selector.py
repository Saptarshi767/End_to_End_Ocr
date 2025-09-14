"""
Tests for chart type selection engine.

This module tests the automatic chart type selection logic, data pattern detection,
and chart configuration generation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.core.models import ChartType, ChartConfig, DataType
from src.visualization.chart_selector import (
    ChartTypeSelector, DataPattern, ColumnAnalysis, DataAnalysis
)


class TestChartTypeSelector:
    """Test cases for ChartTypeSelector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = ChartTypeSelector()
    
    def test_analyze_categorical_data(self):
        """Test analysis of categorical data."""
        # Create test data with categorical column
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'A'],
            'value': [10, 20, 15, 12, 18, 14]
        })
        
        analysis = self.selector.analyze_dataframe(data)
        
        # Check column analysis
        assert len(analysis.columns) == 2
        
        category_col = next(col for col in analysis.columns if col.name == 'category')
        assert category_col.is_categorical
        assert not category_col.is_numerical
        assert category_col.unique_count == 3
        
        value_col = next(col for col in analysis.columns if col.name == 'value')
        assert not value_col.is_categorical
        assert value_col.is_numerical
        
        # Check patterns
        assert DataPattern.CATEGORICAL in analysis.patterns
        assert DataPattern.NUMERICAL in analysis.patterns
        assert DataPattern.CATEGORICAL_NUMERICAL in analysis.patterns
    
    def test_analyze_numerical_data(self):
        """Test analysis of numerical data."""
        # Create test data with numerical columns
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        })
        
        analysis = self.selector.analyze_dataframe(data)
        
        # Check column analysis
        assert len(analysis.columns) == 2
        
        for col in analysis.columns:
            assert col.is_numerical
            assert not col.is_categorical
            assert col.data_type == DataType.NUMBER
        
        # Check patterns
        assert DataPattern.NUMERICAL in analysis.patterns
        assert DataPattern.NUMERICAL_NUMERICAL in analysis.patterns
        assert DataPattern.CORRELATION in analysis.patterns
    
    def test_analyze_time_series_data(self):
        """Test analysis of time series data."""
        # Create test data with datetime column
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, 50)
        })
        
        analysis = self.selector.analyze_dataframe(data)
        
        # Check column analysis
        date_col = next(col for col in analysis.columns if col.name == 'date')
        assert date_col.is_temporal
        assert date_col.data_type == DataType.DATE
        
        value_col = next(col for col in analysis.columns if col.name == 'value')
        assert value_col.is_numerical
        
        # Check patterns
        assert DataPattern.TIME_SERIES in analysis.patterns
    
    def test_analyze_distribution_data(self):
        """Test analysis of single numerical column for distribution."""
        # Create test data with single numerical column
        data = pd.DataFrame({
            'values': np.random.normal(50, 15, 200)
        })
        
        analysis = self.selector.analyze_dataframe(data)
        
        # Check patterns
        assert DataPattern.DISTRIBUTION in analysis.patterns
        assert DataPattern.NUMERICAL in analysis.patterns
    
    def test_detect_data_types(self):
        """Test data type detection."""
        # Create test data with various types
        data = pd.DataFrame({
            'integers': [1, 2, 3, 4, 5],
            'floats': [1.1, 2.2, 3.3, 4.4, 5.5],
            'strings': ['a', 'b', 'c', 'd', 'e'],
            'booleans': [True, False, True, False, True],
            'dates': pd.date_range('2023-01-01', periods=5),
            'string_numbers': ['1', '2', '3', '4', '5']
        })
        
        analysis = self.selector.analyze_dataframe(data)
        
        # Check data type detection
        type_map = {col.name: col.data_type for col in analysis.columns}
        
        assert type_map['integers'] == DataType.NUMBER
        assert type_map['floats'] == DataType.NUMBER
        assert type_map['strings'] == DataType.TEXT
        assert type_map['booleans'] == DataType.BOOLEAN
        assert type_map['dates'] == DataType.DATE
        # String numbers might be detected as NUMBER depending on implementation
    
    def test_chart_type_recommendations(self):
        """Test chart type recommendation logic."""
        # Test categorical data -> bar chart
        categorical_data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'count': [10, 20, 15, 12, 18]
        })
        
        chart_type = self.selector.select_chart_type(categorical_data)
        assert chart_type == ChartType.BAR
        
        # Test time series data -> line chart
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        time_series_data = pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, 20)
        })
        
        chart_type = self.selector.select_chart_type(time_series_data)
        assert chart_type == ChartType.LINE
        
        # Test two numerical columns -> scatter plot
        numerical_data = pd.DataFrame({
            'x': np.random.normal(0, 1, 50),
            'y': np.random.normal(0, 1, 50)
        })
        
        chart_type = self.selector.select_chart_type(numerical_data)
        assert chart_type == ChartType.SCATTER
        
        # Test single numerical column -> histogram
        distribution_data = pd.DataFrame({
            'values': np.random.normal(50, 15, 100)
        })
        
        chart_type = self.selector.select_chart_type(distribution_data)
        assert chart_type == ChartType.HISTOGRAM
    
    def test_generate_chart_config_bar(self):
        """Test bar chart configuration generation."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'value': [10, 20, 15, 12, 18]
        })
        
        config = self.selector.generate_chart_config(data, ChartType.BAR)
        
        assert config.chart_type == ChartType.BAR
        assert config.x_column == 'category'
        assert config.y_column == 'value'
        assert config.aggregation == 'sum'
        assert 'category' in config.title.lower()
    
    def test_generate_chart_config_line(self):
        """Test line chart configuration generation."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, 10)
        })
        
        config = self.selector.generate_chart_config(data, ChartType.LINE)
        
        assert config.chart_type == ChartType.LINE
        assert config.x_column == 'date'
        assert config.y_column == 'value'
        assert config.aggregation == 'avg'
    
    def test_generate_chart_config_pie(self):
        """Test pie chart configuration generation."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A']
        })
        
        config = self.selector.generate_chart_config(data, ChartType.PIE)
        
        assert config.chart_type == ChartType.PIE
        assert config.x_column == 'category'
        assert config.y_column is None
        assert config.aggregation == 'count'
    
    def test_generate_chart_config_scatter(self):
        """Test scatter chart configuration generation."""
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 50),
            'y': np.random.normal(0, 1, 50),
            'category': np.random.choice(['A', 'B', 'C'], 50)
        })
        
        config = self.selector.generate_chart_config(data, ChartType.SCATTER)
        
        assert config.chart_type == ChartType.SCATTER
        assert config.x_column == 'x'
        assert config.y_column == 'y'
        assert config.color_column == 'category'
        assert config.aggregation == 'none'
    
    def test_generate_chart_config_histogram(self):
        """Test histogram configuration generation."""
        data = pd.DataFrame({
            'values': np.random.normal(50, 15, 100)
        })
        
        config = self.selector.generate_chart_config(data, ChartType.HISTOGRAM)
        
        assert config.chart_type == ChartType.HISTOGRAM
        assert config.x_column == 'values'
        assert config.aggregation == 'count'
        assert config.options.get('bins') == 20
    
    def test_get_chart_recommendations(self):
        """Test multiple chart recommendations."""
        # Create mixed data that could support multiple chart types
        data = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 20,
            'value': np.random.normal(100, 20, 60),
            'date': pd.date_range('2023-01-01', periods=60, freq='D')
        })
        
        recommendations = self.selector.get_chart_recommendations(data, max_recommendations=3)
        
        assert len(recommendations) <= 3
        assert all(isinstance(config, ChartConfig) for config in recommendations)
        
        # Should include different chart types
        chart_types = [config.chart_type for config in recommendations]
        assert len(set(chart_types)) >= 2  # At least 2 different types
    
    def test_cardinality_threshold(self):
        """Test categorical detection based on cardinality threshold."""
        # High cardinality data should not be considered categorical
        high_cardinality_data = pd.DataFrame({
            'ids': range(1000),  # 1000 unique values
            'values': np.random.normal(0, 1, 1000)
        })
        
        analysis = self.selector.analyze_dataframe(high_cardinality_data)
        
        ids_col = next(col for col in analysis.columns if col.name == 'ids')
        # Should not be considered categorical due to high cardinality
        assert not ids_col.is_categorical
        
        # Low cardinality data should be considered categorical
        low_cardinality_data = pd.DataFrame({
            'status': np.random.choice(['active', 'inactive', 'pending'], 100)
        })
        
        analysis = self.selector.analyze_dataframe(low_cardinality_data)
        
        status_col = next(col for col in analysis.columns if col.name == 'status')
        assert status_col.is_categorical
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_data = pd.DataFrame()
        
        analysis = self.selector.analyze_dataframe(empty_data)
        
        assert len(analysis.columns) == 0
        assert analysis.row_count == 0
        assert len(analysis.patterns) == 0
        assert len(analysis.recommended_charts) == 0
    
    def test_single_column_dataframe(self):
        """Test handling of single column dataframe."""
        single_col_data = pd.DataFrame({
            'values': [1, 2, 3, 4, 5]
        })
        
        analysis = self.selector.analyze_dataframe(single_col_data)
        
        assert len(analysis.columns) == 1
        assert analysis.columns[0].name == 'values'
        
        # Should recommend appropriate chart for single column
        chart_type = self.selector.select_chart_type(single_col_data)
        assert chart_type in [ChartType.HISTOGRAM, ChartType.BAR]
    
    def test_missing_values_handling(self):
        """Test handling of missing values in analysis."""
        data_with_nulls = pd.DataFrame({
            'category': ['A', 'B', None, 'A', 'B', None],
            'value': [10, None, 15, 12, 18, None]
        })
        
        analysis = self.selector.analyze_dataframe(data_with_nulls)
        
        # Check null counts are recorded
        category_col = next(col for col in analysis.columns if col.name == 'category')
        value_col = next(col for col in analysis.columns if col.name == 'value')
        
        assert category_col.null_count == 2
        assert value_col.null_count == 2
        
        # Should still generate valid recommendations
        recommendations = self.selector.get_chart_recommendations(data_with_nulls)
        assert len(recommendations) > 0
    
    def test_column_relationships(self):
        """Test detection of column relationships."""
        # Create correlated data
        x = np.random.normal(0, 1, 100)
        y = 2 * x + np.random.normal(0, 0.1, 100)  # Strong correlation
        z = np.random.normal(0, 1, 100)  # No correlation
        
        data = pd.DataFrame({'x': x, 'y': y, 'z': z})
        
        analysis = self.selector.analyze_dataframe(data)
        
        # Should detect relationship between x and y
        assert 'x' in analysis.column_relationships or 'y' in analysis.column_relationships
    
    def test_time_series_keyword_detection(self):
        """Test detection of temporal columns by keywords."""
        data = pd.DataFrame({
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'year': [2021, 2022, 2023],
            'month_name': ['Jan', 'Feb', 'Mar'],
            'value': [10, 20, 30]
        })
        
        analysis = self.selector.analyze_dataframe(data)
        
        # Check temporal detection
        temporal_cols = [col.name for col in analysis.columns if col.is_temporal]
        
        # Should detect timestamp and year as temporal
        assert 'timestamp' in temporal_cols or 'year' in temporal_cols
    
    def test_pie_chart_heuristics(self):
        """Test pie chart recommendation heuristics."""
        # Small number of categories - should boost pie chart
        small_categories = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 10
        })
        
        analysis = self.selector.analyze_dataframe(small_categories)
        chart_types = [chart_type for chart_type, _ in analysis.recommended_charts]
        
        # Pie chart should be recommended for small categorical data
        assert ChartType.PIE in chart_types
        
        # Large number of categories - should penalize pie chart
        large_categories = pd.DataFrame({
            'category': [f'Cat_{i}' for i in range(20)] * 5
        })
        
        analysis = self.selector.analyze_dataframe(large_categories)
        
        # Pie chart should be less likely or not recommended
        pie_score = 0
        for chart_type, score in analysis.recommended_charts:
            if chart_type == ChartType.PIE:
                pie_score = score
                break
        
        # Score should be lower for large categories
        assert pie_score < 0.8  # Assuming original score would be higher


if __name__ == '__main__':
    pytest.main([__file__])