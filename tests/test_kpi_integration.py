"""
Integration tests for KPI functionality with dashboard framework.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.visualization.dashboard_framework import InteractiveDashboard
from src.visualization.kpi_engine import EnhancedKPIEngine
from src.visualization.kpi_widgets import KPIWidgetManager


class TestKPIIntegration:
    """Test KPI integration with dashboard framework."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.dashboard = InteractiveDashboard()
        
        # Sample business data
        self.sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'revenue': np.random.normal(10000, 1000, 30),
            'cost': np.random.normal(7000, 800, 30),
            'orders': np.random.randint(100, 300, 30),
            'customers': np.random.randint(50, 150, 30),
            'category': np.random.choice(['A', 'B', 'C'], 30),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 30)
        })
        
        # Ensure positive values
        self.sample_data['revenue'] = np.abs(self.sample_data['revenue'])
        self.sample_data['cost'] = np.abs(self.sample_data['cost'])
    
    def test_dashboard_kpi_generation(self):
        """Test KPI generation in dashboard."""
        dashboard_config = self.dashboard.generate_dashboard(self.sample_data)
        
        # Should have KPIs
        assert len(dashboard_config.kpis) > 0
        
        # Check KPI properties
        for kpi in dashboard_config.kpis:
            assert kpi.name is not None
            assert kpi.value is not None
            assert kpi.format_type is not None
    
    def test_kpi_widgets_creation(self):
        """Test KPI widgets are created."""
        self.dashboard.generate_dashboard(self.sample_data)
        
        # Get KPI widgets
        kpi_widgets = self.dashboard.get_kpi_widgets()
        
        assert "widgets" in kpi_widgets
        assert "layout" in kpi_widgets
        assert kpi_widgets["total_widgets"] > 0
    
    def test_kpi_target_update(self):
        """Test updating KPI targets."""
        dashboard_config = self.dashboard.generate_dashboard(self.sample_data)
        
        if len(dashboard_config.kpis) > 0:
            kpi_name = dashboard_config.kpis[0].name
            target_value = 15000.0
            
            success = self.dashboard.update_kpi_target(kpi_name, target_value)
            
            # Should succeed if KPI exists
            assert isinstance(success, bool)
    
    def test_kpi_insights_generation(self):
        """Test KPI insights generation."""
        self.dashboard.generate_dashboard(self.sample_data)
        
        insights = self.dashboard.get_kpi_insights()
        
        # Should return list (may be empty)
        assert isinstance(insights, list)
    
    def test_kpi_alerts(self):
        """Test KPI alerts."""
        self.dashboard.generate_dashboard(self.sample_data)
        
        alerts = self.dashboard.get_kpi_alerts()
        
        # Should return list (may be empty)
        assert isinstance(alerts, list)
    
    def test_kpi_configuration_export(self):
        """Test KPI configuration export."""
        self.dashboard.generate_dashboard(self.sample_data)
        
        config = self.dashboard.export_kpi_configuration()
        
        assert "widgets" in config
        assert "export_timestamp" in config
    
    def test_enhanced_kpi_engine_integration(self):
        """Test enhanced KPI engine integration."""
        # Test that the dashboard uses the enhanced KPI engine
        assert hasattr(self.dashboard, 'kpi_engine')
        assert isinstance(self.dashboard.kpi_engine, EnhancedKPIEngine)
        
        # Test that KPI widget manager is available
        assert hasattr(self.dashboard, 'kpi_widget_manager')
        assert isinstance(self.dashboard.kpi_widget_manager, KPIWidgetManager)
    
    def test_kpi_with_business_data(self):
        """Test KPI detection with business-relevant data."""
        business_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50, freq='D'),
            'total_revenue': np.random.normal(50000, 5000, 50),
            'total_cost': np.random.normal(35000, 3500, 50),
            'profit_margin': np.random.normal(0.15, 0.05, 50),
            'customer_count': np.random.randint(200, 500, 50),
            'order_value': np.random.normal(150, 30, 50),
            'conversion_rate': np.random.normal(0.05, 0.01, 50)
        })
        
        # Ensure positive values
        business_data['total_revenue'] = np.abs(business_data['total_revenue'])
        business_data['total_cost'] = np.abs(business_data['total_cost'])
        business_data['profit_margin'] = np.abs(business_data['profit_margin'])
        business_data['order_value'] = np.abs(business_data['order_value'])
        business_data['conversion_rate'] = np.abs(business_data['conversion_rate'])
        
        dashboard_config = self.dashboard.generate_dashboard(business_data)
        
        # Should detect business-relevant KPIs
        kpi_names = [kpi.name.lower() for kpi in dashboard_config.kpis]
        
        # Check for business metrics
        business_keywords = ['revenue', 'cost', 'profit', 'customer', 'order', 'conversion']
        found_business_kpis = any(
            any(keyword in name for keyword in business_keywords)
            for name in kpi_names
        )
        
        assert found_business_kpis, f"No business KPIs found in: {kpi_names}"
    
    def test_kpi_format_types(self):
        """Test KPI format type detection."""
        financial_data = pd.DataFrame({
            'revenue_usd': [10000, 11000, 12000],
            'cost_usd': [7000, 7500, 8000],
            'conversion_rate': [0.05, 0.06, 0.07],
            'profit_margin_pct': [15.5, 16.2, 17.1],
            'order_count': [100, 110, 120]
        })
        
        dashboard_config = self.dashboard.generate_dashboard(financial_data)
        
        # Check that different format types are detected
        format_types = [kpi.format_type for kpi in dashboard_config.kpis]
        
        # Should have variety of format types
        assert len(set(format_types)) > 1
    
    def test_kpi_error_handling(self):
        """Test KPI error handling with problematic data."""
        # Data with issues
        problematic_data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [np.inf, -np.inf, 0, 1, 2]
        })
        
        # Should not crash
        try:
            dashboard_config = self.dashboard.generate_dashboard(problematic_data)
            assert isinstance(dashboard_config.kpis, list)
        except Exception as e:
            pytest.fail(f"Dashboard generation failed with problematic data: {e}")
    
    def test_kpi_performance_with_large_data(self):
        """Test KPI performance with larger datasets."""
        # Create larger dataset
        large_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'value1': np.random.normal(1000, 100, 1000),
            'value2': np.random.normal(500, 50, 1000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 1000)
        })
        
        import time
        start_time = time.time()
        
        dashboard_config = self.dashboard.generate_dashboard(large_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 10.0, f"KPI generation took too long: {processing_time}s"
        assert len(dashboard_config.kpis) > 0


def test_kpi_engine_standalone():
    """Test KPI engine functionality standalone."""
    engine = EnhancedKPIEngine()
    
    data = pd.DataFrame({
        'sales': [1000, 1100, 1200, 1300, 1400],
        'profit': [200, 220, 240, 260, 280],
        'customers': [50, 55, 60, 65, 70]
    })
    
    # Auto-detect KPIs
    kpi_definitions = engine.auto_detect_kpis(data, max_kpis=10)
    assert len(kpi_definitions) > 0
    
    # Calculate KPIs
    calculated_kpis = engine.calculate_kpis(data)
    assert len(calculated_kpis) > 0
    
    # Get summary
    summary = engine.get_kpi_summary(data)
    assert "kpis" in summary
    assert "insights" in summary
    assert "summary" in summary


def test_kpi_widgets_standalone():
    """Test KPI widgets functionality standalone."""
    from src.visualization.kpi_widgets import (
        KPIWidgetManager, KPIWidgetConfig, KPIWidgetData, 
        KPIWidgetType, KPIDisplayFormat
    )
    from src.core.models import KPI
    
    manager = KPIWidgetManager()
    
    # Create test KPI
    kpi = KPI(
        name="Test Revenue",
        value=25000,
        format_type="currency",
        description="Test revenue KPI"
    )
    
    # Create widget config
    config = KPIWidgetConfig(
        widget_id="test_widget",
        widget_type=KPIWidgetType.CARD,
        title="Revenue Widget",
        display_format=KPIDisplayFormat.CURRENCY
    )
    
    # Create widget data
    data = KPIWidgetData(kpi=kpi)
    
    # Add widget
    widget_id = manager.add_widget(config, data)
    assert widget_id == "test_widget"
    
    # Render dashboard
    rendered = manager.render_dashboard()
    assert rendered["total_widgets"] == 1
    assert "test_widget" in rendered["widgets"]


if __name__ == "__main__":
    pytest.main([__file__])