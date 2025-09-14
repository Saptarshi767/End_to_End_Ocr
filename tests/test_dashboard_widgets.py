"""
Tests for dashboard widgets and interactive components.

This module tests advanced dashboard widgets, KPI calculations,
interactive chart management, and theme management.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from src.visualization.dashboard_widgets import (
    WidgetConfig, FilterWidget, KPIWidget, ChartWidget,
    AdvancedFilterManager, KPICalculator, InteractiveChartManager,
    DashboardThemeManager, DashboardExportManager
)
from src.core.models import Filter, KPI, ChartConfig, ChartType


class TestAdvancedFilterManager:
    """Test cases for AdvancedFilterManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.filter_manager = AdvancedFilterManager()
        
        # Create test filter widget
        filter_config = Filter(
            column="category",
            filter_type="select",
            values=["A", "B", "C"],
            default_value="A"
        )
        
        widget_config = WidgetConfig(
            widget_id="filter_1",
            widget_type="filter",
            title="Category Filter"
        )
        
        self.filter_widget = FilterWidget(
            config=widget_config,
            filter_config=filter_config
        )
    
    def test_add_filter_widget(self):
        """Test adding filter widget."""
        self.filter_manager.add_filter_widget(self.filter_widget)
        
        assert "filter_1" in self.filter_manager.filter_widgets
        assert self.filter_manager.filter_widgets["filter_1"] == self.filter_widget
    
    def test_update_filter_value(self):
        """Test updating filter value."""
        self.filter_manager.add_filter_widget(self.filter_widget)
        
        # Update filter value
        updated_widgets = self.filter_manager.update_filter_value("filter_1", ["A", "B"])
        
        assert "filter_1" in updated_widgets
        assert self.filter_widget.current_value == ["A", "B"]
        assert self.filter_widget.is_active is True
    
    def test_filter_dependencies(self):
        """Test filter dependencies and cascading updates."""
        # Create dependent filter
        dependent_filter_config = Filter(
            column="subcategory",
            filter_type="select",
            values=["A1", "A2", "B1", "B2"],
            default_value=None
        )
        
        dependent_widget_config = WidgetConfig(
            widget_id="filter_2",
            widget_type="filter",
            title="Subcategory Filter"
        )
        
        dependent_widget = FilterWidget(
            config=dependent_widget_config,
            filter_config=dependent_filter_config,
            dependencies=["filter_1"]
        )
        
        # Add both widgets
        self.filter_manager.add_filter_widget(self.filter_widget)
        self.filter_manager.add_filter_widget(dependent_widget)
        
        # Update parent filter
        updated_widgets = self.filter_manager.update_filter_value("filter_1", ["A"])
        
        # Should update both parent and dependent
        assert len(updated_widgets) >= 1
        assert "filter_1" in updated_widgets
    
    def test_get_active_filters(self):
        """Test getting active filter values."""
        self.filter_manager.add_filter_widget(self.filter_widget)
        self.filter_manager.update_filter_value("filter_1", ["A", "B"])
        
        active_filters = self.filter_manager.get_active_filters()
        
        assert "filter_1" in active_filters
        assert active_filters["filter_1"] == ["A", "B"]
    
    def test_clear_all_filters(self):
        """Test clearing all filters."""
        self.filter_manager.add_filter_widget(self.filter_widget)
        self.filter_manager.update_filter_value("filter_1", ["A", "B"])
        
        # Clear all filters
        self.filter_manager.clear_all_filters()
        
        assert self.filter_widget.current_value is None
        assert self.filter_widget.is_active is False
    
    def test_filter_callbacks(self):
        """Test filter change callbacks."""
        self.filter_manager.add_filter_widget(self.filter_widget)
        
        # Add callback
        callback_mock = Mock()
        self.filter_manager.add_filter_callback("filter_1", callback_mock)
        
        # Update filter
        self.filter_manager.update_filter_value("filter_1", ["A", "B"])
        
        # Verify callback was called
        callback_mock.assert_called_once()


class TestKPICalculator:
    """Test cases for KPICalculator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.kpi_calculator = KPICalculator()
        
        # Sample data
        self.sample_data = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250],
            "profit": [20, 40, 30, 60, 50],
            "category": ["A", "B", "A", "C", "B"]
        })
    
    def test_register_kpi(self):
        """Test registering KPI calculation function."""
        def total_sales(data):
            return data["sales"].sum()
        
        self.kpi_calculator.register_kpi(
            "total_sales",
            "Total Sales",
            total_sales,
            "currency",
            "Sum of all sales"
        )
        
        assert "total_sales" in self.kpi_calculator.kpi_definitions
        assert self.kpi_calculator.kpi_definitions["total_sales"]["name"] == "Total Sales"
    
    def test_calculate_kpi(self):
        """Test KPI calculation."""
        def total_sales(data):
            return data["sales"].sum()
        
        self.kpi_calculator.register_kpi(
            "total_sales",
            "Total Sales",
            total_sales,
            "currency"
        )
        
        kpi = self.kpi_calculator.calculate_kpi("total_sales", self.sample_data)
        
        assert isinstance(kpi, KPI)
        assert kpi.name == "Total Sales"
        assert kpi.value == 1000  # Sum of sales
        assert kpi.format_type == "currency"
    
    def test_kpi_trend_calculation(self):
        """Test KPI trend calculation."""
        def avg_sales(data):
            return data["sales"].mean()
        
        self.kpi_calculator.register_kpi("avg_sales", "Average Sales", avg_sales)
        
        # Calculate KPI multiple times to test trend
        kpi1 = self.kpi_calculator.calculate_kpi("avg_sales", self.sample_data)
        assert kpi1.trend is None  # First calculation
        
        # Modify data and calculate again
        modified_data = self.sample_data.copy()
        modified_data["sales"] = modified_data["sales"] * 1.1
        
        kpi2 = self.kpi_calculator.calculate_kpi("avg_sales", modified_data)
        assert kpi2.trend is not None
        assert kpi2.trend > 0  # Should show positive trend
    
    def test_kpi_error_handling(self):
        """Test KPI calculation error handling."""
        def failing_calculation(data):
            raise ValueError("Calculation failed")
        
        self.kpi_calculator.register_kpi("failing_kpi", "Failing KPI", failing_calculation)
        
        kpi = self.kpi_calculator.calculate_kpi("failing_kpi", self.sample_data)
        
        assert kpi.value == 0
        assert "Error:" in kpi.description


class TestInteractiveChartManager:
    """Test cases for InteractiveChartManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.chart_manager = InteractiveChartManager()
        
        # Create test chart widget
        chart_config = ChartConfig(
            chart_type=ChartType.BAR,
            title="Test Chart",
            x_column="category",
            y_column="sales"
        )
        
        widget_config = WidgetConfig(
            widget_id="chart_1",
            widget_type="chart",
            title="Test Chart Widget"
        )
        
        self.chart_widget = ChartWidget(
            config=widget_config,
            chart_config=chart_config
        )
    
    def test_add_chart_widget(self):
        """Test adding chart widget."""
        self.chart_manager.add_chart_widget(self.chart_widget)
        
        assert "chart_1" in self.chart_manager.chart_widgets
        assert self.chart_manager.chart_widgets["chart_1"] == self.chart_widget
    
    def test_register_interaction_handler(self):
        """Test registering interaction handler."""
        self.chart_manager.add_chart_widget(self.chart_widget)
        
        # Register click handler
        click_handler = Mock(return_value={"status": "success"})
        self.chart_manager.register_interaction_handler("chart_1", "click", click_handler)
        
        assert "chart_1" in self.chart_manager.interaction_handlers
        assert "click" in self.chart_manager.interaction_handlers["chart_1"]
    
    def test_handle_chart_interaction(self):
        """Test handling chart interaction."""
        self.chart_manager.add_chart_widget(self.chart_widget)
        
        # Register and test interaction handler
        click_handler = Mock(return_value={"clicked_value": "A"})
        self.chart_manager.register_interaction_handler("chart_1", "click", click_handler)
        
        interaction_data = {"x": 100, "y": 200, "value": "A"}
        result = self.chart_manager.handle_chart_interaction("chart_1", "click", interaction_data)
        
        click_handler.assert_called_once_with(interaction_data)
        assert result["clicked_value"] == "A"
    
    def test_get_chart_interactions(self):
        """Test getting available chart interactions."""
        self.chart_manager.add_chart_widget(self.chart_widget)
        
        interactions = self.chart_manager.get_chart_interactions("chart_1")
        
        assert "available_interactions" in interactions
        assert "click" in interactions["available_interactions"]
        assert "hover" in interactions["available_interactions"]
    
    def test_update_chart_data(self):
        """Test updating chart data."""
        self.chart_manager.add_chart_widget(self.chart_widget)
        
        new_data = {"data": [{"x": "A", "y": 100}]}
        self.chart_manager.update_chart_data("chart_1", new_data)
        
        assert self.chart_widget.chart_data == new_data


class TestDashboardThemeManager:
    """Test cases for DashboardThemeManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.theme_manager = DashboardThemeManager()
    
    def test_default_themes(self):
        """Test default theme availability."""
        themes = self.theme_manager.get_available_themes()
        
        theme_ids = [theme["id"] for theme in themes]
        assert "default" in theme_ids
        assert "dark" in theme_ids
        assert "corporate" in theme_ids
    
    def test_set_theme(self):
        """Test setting current theme."""
        result = self.theme_manager.set_theme("dark")
        
        assert result is True
        assert self.theme_manager.current_theme == "dark"
        
        # Test invalid theme
        result = self.theme_manager.set_theme("nonexistent")
        assert result is False
    
    def test_get_theme(self):
        """Test getting theme configuration."""
        theme = self.theme_manager.get_theme("dark")
        
        assert "colors" in theme
        assert "chart_colors" in theme
        assert "background" in theme
        assert theme["name"] == "Dark"
    
    def test_add_custom_theme(self):
        """Test adding custom theme."""
        custom_theme = {
            "name": "Custom Theme",
            "colors": {"primary": "#ff0000"},
            "chart_colors": ["#ff0000", "#00ff00"],
            "background": "#ffffff",
            "text": "#000000",
            "border": "#cccccc"
        }
        
        self.theme_manager.add_custom_theme("custom", custom_theme)
        
        themes = self.theme_manager.get_available_themes()
        theme_ids = [theme["id"] for theme in themes]
        assert "custom" in theme_ids
    
    def test_apply_theme_to_chart(self):
        """Test applying theme to chart configuration."""
        chart_data = {
            "data": [{"type": "bar", "marker": {}}],
            "layout": {}
        }
        
        themed_chart = self.theme_manager.apply_theme_to_chart(chart_data)
        
        assert "paper_bgcolor" in themed_chart["layout"]
        assert "font" in themed_chart["layout"]
        assert "color" in themed_chart["data"][0]["marker"]


class TestDashboardExportManager:
    """Test cases for DashboardExportManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.export_manager = DashboardExportManager()
        
        self.sample_dashboard = {
            "title": "Test Dashboard",
            "charts": [
                {"id": "chart1", "type": "bar", "data": {"x": [1, 2, 3], "y": [4, 5, 6]}}
            ],
            "filters": [
                {"column": "category", "type": "select", "values": ["A", "B", "C"]}
            ],
            "kpis": [
                {"name": "Total Sales", "value": 1000, "format": "currency"}
            ]
        }
    
    def test_get_supported_formats(self):
        """Test getting supported export formats."""
        formats = self.export_manager.get_supported_formats()
        
        assert "json" in formats
        assert "html" in formats
        assert "pdf" in formats
        assert "png" in formats
    
    def test_get_format_options(self):
        """Test getting format-specific options."""
        pdf_options = self.export_manager.get_format_options("pdf")
        
        assert "page_size" in pdf_options
        assert "orientation" in pdf_options
        assert "include_filters" in pdf_options
    
    @patch("builtins.open", create=True)
    def test_export_json(self, mock_open):
        """Test JSON export functionality."""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = self.export_manager.export_dashboard(
            self.sample_dashboard, "json", "test.json"
        )
        
        assert result is True
        mock_open.assert_called_once_with("test.json", 'w', encoding='utf-8')
        mock_file.write.assert_called()
    
    @patch("builtins.open", create=True)
    def test_export_html(self, mock_open):
        """Test HTML export functionality."""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = self.export_manager.export_dashboard(
            self.sample_dashboard, "html", "test.html"
        )
        
        assert result is True
        mock_open.assert_called_once_with("test.html", 'w', encoding='utf-8')
    
    def test_export_unsupported_format(self):
        """Test export with unsupported format."""
        result = self.export_manager.export_dashboard(
            self.sample_dashboard, "unsupported", "test.xyz"
        )
        
        assert result is False


class TestWidgetSerialization:
    """Test cases for widget serialization."""
    
    def test_filter_widget_to_dict(self):
        """Test FilterWidget serialization."""
        filter_config = Filter(
            column="category",
            filter_type="select",
            values=["A", "B", "C"]
        )
        
        widget_config = WidgetConfig(
            widget_id="filter_1",
            widget_type="filter",
            title="Category Filter",
            position={"x": 0, "y": 0},
            size={"width": 4, "height": 1}
        )
        
        widget = FilterWidget(
            config=widget_config,
            filter_config=filter_config,
            current_value=["A"],
            is_active=True
        )
        
        widget_dict = widget.to_dict()
        
        assert widget_dict["id"] == "filter_1"
        assert widget_dict["type"] == "filter"
        assert widget_dict["filter_type"] == "select"
        assert widget_dict["current_value"] == ["A"]
        assert widget_dict["is_active"] is True
    
    def test_kpi_widget_to_dict(self):
        """Test KPIWidget serialization."""
        kpi = KPI(
            name="Total Sales",
            value=1000,
            format_type="currency",
            description="Sum of all sales",
            trend=5.2
        )
        
        widget_config = WidgetConfig(
            widget_id="kpi_1",
            widget_type="kpi",
            title="Sales KPI"
        )
        
        widget = KPIWidget(
            config=widget_config,
            kpi=kpi,
            trend_data=[900, 950, 1000],
            target_value=1200
        )
        
        widget_dict = widget.to_dict()
        
        assert widget_dict["id"] == "kpi_1"
        assert widget_dict["type"] == "kpi"
        assert widget_dict["name"] == "Total Sales"
        assert widget_dict["value"] == 1000
        assert widget_dict["trend"] == 5.2
        assert widget_dict["target_value"] == 1200
    
    def test_chart_widget_to_dict(self):
        """Test ChartWidget serialization."""
        chart_config = ChartConfig(
            chart_type=ChartType.BAR,
            title="Sales by Category",
            x_column="category",
            y_column="sales"
        )
        
        widget_config = WidgetConfig(
            widget_id="chart_1",
            widget_type="chart",
            title="Sales Chart"
        )
        
        widget = ChartWidget(
            config=widget_config,
            chart_config=chart_config,
            chart_data={"data": [{"x": "A", "y": 100}]}
        )
        
        widget_dict = widget.to_dict()
        
        assert widget_dict["id"] == "chart_1"
        assert widget_dict["type"] == "chart"
        assert widget_dict["chart_type"] == "bar"
        assert widget_dict["chart_config"]["x_column"] == "category"
        assert widget_dict["chart_data"]["data"][0]["x"] == "A"


if __name__ == "__main__":
    pytest.main([__file__])