"""
Tests for chart generation engines.

This module tests the chart generation functionality for different visualization
libraries including Plotly and Chart.js engines.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.models import ChartType, ChartConfig, Chart
from src.visualization.chart_engines import (
    PlotlyChartEngine, ChartJSEngine, ChartEngineFactory
)


class TestPlotlyChartEngine:
    """Test cases for PlotlyChartEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PlotlyChartEngine()
        
        # Sample data for testing
        self.categorical_data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'value': [10, 20, 15, 12, 18]
        })
        
        self.numerical_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        self.time_series_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'value': [100, 110, 105, 120, 115]
        })
    
    def test_get_supported_chart_types(self):
        """Test getting supported chart types."""
        supported_types = self.engine.get_supported_chart_types()
        
        expected_types = ['bar', 'line', 'pie', 'scatter', 'histogram', 'heatmap', 'table']
        assert all(chart_type in supported_types for chart_type in expected_types)
    
    def test_create_bar_chart(self):
        """Test creating Plotly bar chart."""
        config = ChartConfig(
            chart_type=ChartType.BAR,
            title="Test Bar Chart",
            x_column="category",
            y_column="value",
            aggregation="sum"
        )
        
        chart = self.engine.create_chart(config, self.categorical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.BAR
        assert "data" in chart.data
        assert "layout" in chart.data
        
        # Check data structure
        chart_data = chart.data["data"][0]
        assert chart_data["type"] == "bar"
        assert len(chart_data["x"]) > 0
        assert len(chart_data["y"]) > 0
    
    def test_create_line_chart(self):
        """Test creating Plotly line chart."""
        config = ChartConfig(
            chart_type=ChartType.LINE,
            title="Test Line Chart",
            x_column="date",
            y_column="value",
            aggregation="avg"
        )
        
        chart = self.engine.create_chart(config, self.time_series_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.LINE
        
        # Check data structure
        chart_data = chart.data["data"][0]
        assert chart_data["type"] == "scatter"
        assert chart_data["mode"] == "lines+markers"
        assert len(chart_data["x"]) == len(self.time_series_data)
    
    def test_create_pie_chart(self):
        """Test creating Plotly pie chart."""
        config = ChartConfig(
            chart_type=ChartType.PIE,
            title="Test Pie Chart",
            x_column="category",
            aggregation="count"
        )
        
        chart = self.engine.create_chart(config, self.categorical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.PIE
        
        # Check data structure
        chart_data = chart.data["data"][0]
        assert chart_data["type"] == "pie"
        assert "labels" in chart_data
        assert "values" in chart_data
        assert len(chart_data["labels"]) == len(chart_data["values"])
    
    def test_create_scatter_chart(self):
        """Test creating Plotly scatter chart."""
        config = ChartConfig(
            chart_type=ChartType.SCATTER,
            title="Test Scatter Chart",
            x_column="x",
            y_column="y",
            aggregation="none"
        )
        
        chart = self.engine.create_chart(config, self.numerical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.SCATTER
        
        # Check data structure
        chart_data = chart.data["data"][0]
        assert chart_data["type"] == "scatter"
        assert chart_data["mode"] == "markers"
        assert len(chart_data["x"]) == len(self.numerical_data)
    
    def test_create_histogram_chart(self):
        """Test creating Plotly histogram chart."""
        config = ChartConfig(
            chart_type=ChartType.HISTOGRAM,
            title="Test Histogram",
            x_column="value",
            aggregation="count",
            options={"bins": 10}
        )
        
        chart = self.engine.create_chart(config, self.categorical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.HISTOGRAM
        
        # Check data structure
        chart_data = chart.data["data"][0]
        assert chart_data["type"] == "histogram"
        assert "x" in chart_data
        assert chart_data["nbinsx"] == 10
    
    def test_create_table_chart(self):
        """Test creating Plotly table chart."""
        config = ChartConfig(
            chart_type=ChartType.TABLE,
            title="Test Table",
            x_column="category",
            aggregation="none"
        )
        
        chart = self.engine.create_chart(config, self.categorical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.TABLE
        
        # Check data structure
        chart_data = chart.data["data"][0]
        assert chart_data["type"] == "table"
        assert "header" in chart_data
        assert "cells" in chart_data
    
    def test_auto_select_chart_type(self):
        """Test automatic chart type selection."""
        # Single categorical column
        chart_type = self.engine.auto_select_chart_type(
            self.categorical_data, ["category"]
        )
        assert chart_type == "bar"
        
        # Single numerical column
        chart_type = self.engine.auto_select_chart_type(
            self.numerical_data, ["x"]
        )
        assert chart_type == "histogram"
        
        # Two columns
        chart_type = self.engine.auto_select_chart_type(
            self.numerical_data, ["x", "y"]
        )
        assert chart_type == "scatter"


class TestChartJSEngine:
    """Test cases for ChartJSEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ChartJSEngine()
        
        # Sample data for testing
        self.categorical_data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'value': [10, 20, 15, 12, 18]
        })
        
        self.numerical_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
    
    def test_get_supported_chart_types(self):
        """Test getting supported chart types."""
        supported_types = self.engine.get_supported_chart_types()
        
        expected_types = ['bar', 'line', 'pie', 'scatter', 'histogram', 'heatmap', 'table']
        assert all(chart_type in supported_types for chart_type in expected_types)
    
    def test_create_bar_chart(self):
        """Test creating Chart.js bar chart."""
        config = ChartConfig(
            chart_type=ChartType.BAR,
            title="Test Bar Chart",
            x_column="category",
            y_column="value",
            aggregation="sum"
        )
        
        chart = self.engine.create_chart(config, self.categorical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.BAR
        assert chart.data["type"] == "bar"
        assert "data" in chart.data
        assert "options" in chart.data
        
        # Check data structure
        datasets = chart.data["data"]["datasets"]
        assert len(datasets) > 0
        assert "data" in datasets[0]
        assert len(datasets[0]["data"]) > 0
    
    def test_create_line_chart(self):
        """Test creating Chart.js line chart."""
        config = ChartConfig(
            chart_type=ChartType.LINE,
            title="Test Line Chart",
            x_column="x",
            y_column="y",
            aggregation="avg"
        )
        
        chart = self.engine.create_chart(config, self.numerical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.LINE
        assert chart.data["type"] == "line"
        
        # Check data structure
        datasets = chart.data["data"]["datasets"]
        assert len(datasets) > 0
        assert "data" in datasets[0]
        assert len(datasets[0]["data"]) == len(self.numerical_data)
    
    def test_create_pie_chart(self):
        """Test creating Chart.js pie chart."""
        config = ChartConfig(
            chart_type=ChartType.PIE,
            title="Test Pie Chart",
            x_column="category",
            aggregation="count"
        )
        
        chart = self.engine.create_chart(config, self.categorical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.PIE
        assert chart.data["type"] == "pie"
        
        # Check data structure
        chart_data = chart.data["data"]
        assert "labels" in chart_data
        assert "datasets" in chart_data
        assert len(chart_data["datasets"]) > 0
    
    def test_create_scatter_chart(self):
        """Test creating Chart.js scatter chart."""
        config = ChartConfig(
            chart_type=ChartType.SCATTER,
            title="Test Scatter Chart",
            x_column="x",
            y_column="y",
            aggregation="none"
        )
        
        chart = self.engine.create_chart(config, self.numerical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.SCATTER
        assert chart.data["type"] == "scatter"
        
        # Check data structure
        datasets = chart.data["data"]["datasets"]
        assert len(datasets) > 0
        scatter_data = datasets[0]["data"]
        assert len(scatter_data) == len(self.numerical_data)
        assert all("x" in point and "y" in point for point in scatter_data)
    
    def test_create_histogram_chart(self):
        """Test creating Chart.js histogram chart."""
        # Create data suitable for histogram
        histogram_data = pd.DataFrame({
            'values': np.random.normal(50, 15, 100)
        })
        
        config = ChartConfig(
            chart_type=ChartType.HISTOGRAM,
            title="Test Histogram",
            x_column="values",
            aggregation="count",
            options={"bins": 10}
        )
        
        chart = self.engine.create_chart(config, histogram_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.HISTOGRAM
        assert chart.data["type"] == "bar"  # Chart.js uses bar for histogram
        
        # Check data structure
        chart_data = chart.data["data"]
        assert "labels" in chart_data
        assert "datasets" in chart_data
    
    def test_create_table_chart(self):
        """Test creating Chart.js table chart."""
        config = ChartConfig(
            chart_type=ChartType.TABLE,
            title="Test Table",
            x_column="category",
            aggregation="none"
        )
        
        chart = self.engine.create_chart(config, self.categorical_data)
        
        assert isinstance(chart, Chart)
        assert chart.config.chart_type == ChartType.TABLE
        assert chart.data["type"] == "table"
        
        # Check data structure
        chart_data = chart.data["data"]
        assert "headers" in chart_data
        assert "rows" in chart_data
        assert len(chart_data["headers"]) == len(self.categorical_data.columns)


class TestChartEngineFactory:
    """Test cases for ChartEngineFactory class."""
    
    def test_create_plotly_engine(self):
        """Test creating Plotly engine."""
        engine = ChartEngineFactory.create_engine("plotly")
        assert isinstance(engine, PlotlyChartEngine)
        assert engine.library_name == "plotly"
    
    def test_create_chartjs_engine(self):
        """Test creating Chart.js engine."""
        engine = ChartEngineFactory.create_engine("chartjs")
        assert isinstance(engine, ChartJSEngine)
        assert engine.library_name == "chartjs"
    
    def test_create_default_engine(self):
        """Test creating default engine."""
        engine = ChartEngineFactory.create_engine()
        assert isinstance(engine, PlotlyChartEngine)  # Default should be Plotly
    
    def test_create_unknown_engine(self):
        """Test creating unknown engine type."""
        engine = ChartEngineFactory.create_engine("unknown")
        assert isinstance(engine, PlotlyChartEngine)  # Should fallback to default
    
    def test_get_available_engines(self):
        """Test getting available engines."""
        engines = ChartEngineFactory.get_available_engines()
        assert "plotly" in engines
        assert "chartjs" in engines
        assert len(engines) >= 2


class TestChartEngineErrorHandling:
    """Test error handling in chart engines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotly_engine = PlotlyChartEngine()
        self.chartjs_engine = ChartJSEngine()
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframe."""
        empty_data = pd.DataFrame()
        
        config = ChartConfig(
            chart_type=ChartType.BAR,
            title="Empty Data Test",
            x_column="nonexistent",
            aggregation="count"
        )
        
        # Should not crash and return fallback chart
        plotly_chart = self.plotly_engine.create_chart(config, empty_data)
        chartjs_chart = self.chartjs_engine.create_chart(config, empty_data)
        
        assert isinstance(plotly_chart, Chart)
        assert isinstance(chartjs_chart, Chart)
    
    def test_invalid_column_handling(self):
        """Test handling of invalid column names."""
        data = pd.DataFrame({
            'valid_column': [1, 2, 3, 4, 5]
        })
        
        config = ChartConfig(
            chart_type=ChartType.BAR,
            title="Invalid Column Test",
            x_column="invalid_column",
            aggregation="count"
        )
        
        # Should not crash and return fallback chart
        plotly_chart = self.plotly_engine.create_chart(config, data)
        chartjs_chart = self.chartjs_engine.create_chart(config, data)
        
        assert isinstance(plotly_chart, Chart)
        assert isinstance(chartjs_chart, Chart)
    
    def test_missing_data_handling(self):
        """Test handling of data with missing values."""
        data_with_nulls = pd.DataFrame({
            'category': ['A', 'B', None, 'C', 'A'],
            'value': [10, None, 15, 20, 12]
        })
        
        config = ChartConfig(
            chart_type=ChartType.BAR,
            title="Missing Data Test",
            x_column="category",
            y_column="value",
            aggregation="sum"
        )
        
        # Should handle missing values gracefully
        plotly_chart = self.plotly_engine.create_chart(config, data_with_nulls)
        chartjs_chart = self.chartjs_engine.create_chart(config, data_with_nulls)
        
        assert isinstance(plotly_chart, Chart)
        assert isinstance(chartjs_chart, Chart)


class TestChartEngineIntegration:
    """Integration tests for chart engines."""
    
    def test_end_to_end_chart_generation(self):
        """Test complete chart generation workflow."""
        # Create realistic test data
        np.random.seed(42)  # For reproducible results
        
        data = pd.DataFrame({
            'product': ['A', 'B', 'C', 'D'] * 25,
            'sales': np.random.normal(1000, 200, 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
        })
        
        # Test different chart types with both engines
        chart_configs = [
            ChartConfig(ChartType.BAR, "Sales by Product", "product", "sales", aggregation="sum"),
            ChartConfig(ChartType.LINE, "Sales Trend", "date", "sales", aggregation="avg"),
            ChartConfig(ChartType.PIE, "Region Distribution", "region", aggregation="count"),
            ChartConfig(ChartType.SCATTER, "Sales vs Date", "date", "sales", aggregation="none"),
            ChartConfig(ChartType.HISTOGRAM, "Sales Distribution", "sales", aggregation="count")
        ]
        
        engines = [PlotlyChartEngine(), ChartJSEngine()]
        
        for engine in engines:
            for config in chart_configs:
                chart = engine.create_chart(config, data)
                
                assert isinstance(chart, Chart)
                assert chart.config.chart_type == config.chart_type
                assert chart.data is not None
                assert len(chart.data) > 0
    
    def test_large_dataset_performance(self):
        """Test chart generation with large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000),
            'value': np.random.normal(100, 25, 10000)
        })
        
        config = ChartConfig(
            chart_type=ChartType.BAR,
            title="Large Dataset Test",
            x_column="category",
            y_column="value",
            aggregation="mean"
        )
        
        # Should handle large datasets without issues
        plotly_engine = PlotlyChartEngine()
        chart = plotly_engine.create_chart(config, large_data)
        
        assert isinstance(chart, Chart)
        assert chart.data is not None


if __name__ == '__main__':
    pytest.main([__file__])