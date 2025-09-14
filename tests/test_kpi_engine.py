"""
Tests for KPI engine functionality.

This module tests automatic KPI detection, calculation, trend analysis,
and comparison features.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.visualization.kpi_engine import (
    AutoKPIDetector, TrendAnalyzer, KPIComparator, KPIInsightGenerator,
    EnhancedKPIEngine, KPIType, TrendDirection, KPIDefinition
)
from src.core.models import KPI


class TestAutoKPIDetector:
    """Test cases for AutoKPIDetector."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = AutoKPIDetector()
        
        # Sample business data
        self.business_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'revenue': np.random.normal(10000, 2000, 100),
            'cost': np.random.normal(7000, 1500, 100),
            'quantity': np.random.randint(50, 200, 100),
            'customer_id': np.random.randint(1, 50, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
        })
        
        # Ensure positive revenue and cost
        self.business_data['revenue'] = np.abs(self.business_data['revenue'])
        self.business_data['cost'] = np.abs(self.business_data['cost'])
    
    def test_detect_numeric_summary_kpis(self):
        """Test detection of numeric summary KPIs."""
        kpis = self.detector._detect_numeric_summary_kpis(self.business_data)
        
        # Should detect KPIs for numeric columns
        kpi_names = [kpi.name for kpi in kpis]
        
        assert any('Total Revenue' in name for name in kpi_names)
        assert any('Average Revenue' in name for name in kpi_names)
        assert any('Total Cost' in name for name in kpi_names)
        assert any('Average Quantity' in name for name in kpi_names)
    
    def test_detect_categorical_kpis(self):
        """Test detection of categorical KPIs."""
        kpis = self.detector._detect_categorical_kpis(self.business_data)
        
        kpi_names = [kpi.name for kpi in kpis]
        
        assert any('Top Category' in name for name in kpi_names)
        assert any('Unique Category' in name for name in kpi_names)
        assert any('Top Region' in name for name in kpi_names)
    
    def test_detect_time_series_kpis(self):
        """Test detection of time-series KPIs."""
        kpis = self.detector._detect_time_series_kpis(self.business_data)
        
        kpi_names = [kpi.name for kpi in kpis]
        
        assert any('Records per Day' in name for name in kpi_names)
        assert any('Recent Records' in name for name in kpi_names)
    
    def test_detect_business_kpis(self):
        """Test detection of business-specific KPIs."""
        kpis = self.detector._detect_business_kpis(self.business_data)
        
        kpi_names = [kpi.name for kpi in kpis]
        
        # Should detect revenue and cost related KPIs
        assert any('revenue' in name.lower() for name in kpi_names)
        assert len(kpis) > 0
    
    def test_detect_data_quality_kpis(self):
        """Test detection of data quality KPIs."""
        # Add some missing values
        test_data = self.business_data.copy()
        test_data.loc[0:5, 'revenue'] = np.nan
        
        kpis = self.detector._detect_data_quality_kpis(test_data)
        
        kpi_names = [kpi.name for kpi in kpis]
        
        assert 'Data Completeness' in kpi_names
        assert 'Total Records' in kpi_names
        assert 'Duplicate Rate' in kpi_names
    
    def test_detect_kpis_integration(self):
        """Test full KPI detection integration."""
        kpis = self.detector.detect_kpis(self.business_data, max_kpis=15)
        
        assert len(kpis) <= 15
        assert len(kpis) > 0
        
        # Check that KPIs have required attributes
        for kpi in kpis:
            assert hasattr(kpi, 'kpi_id')
            assert hasattr(kpi, 'name')
            assert hasattr(kpi, 'kpi_type')
            assert hasattr(kpi, 'calculation_func')
    
    def test_infer_format_type(self):
        """Test format type inference."""
        # Test currency detection
        revenue_series = pd.Series([1000, 2000, 3000])
        format_type = self.detector._infer_format_type('revenue', revenue_series)
        assert format_type == "currency"
        
        # Test percentage detection
        rate_series = pd.Series([0.1, 0.2, 0.3])
        format_type = self.detector._infer_format_type('conversion_rate', rate_series)
        assert format_type == "percentage"
        
        # Test number detection
        count_series = pd.Series([10, 20, 30])
        format_type = self.detector._infer_format_type('count', count_series)
        assert format_type == "number"
    
    def test_score_kpis(self):
        """Test KPI scoring and ranking."""
        # Create test KPIs with different categories
        kpis = [
            KPIDefinition("kpi1", "Revenue", KPIType.SUM, lambda x: x.sum(), category="business"),
            KPIDefinition("kpi2", "Count", KPIType.COUNT, lambda x: len(x), category="quality"),
            KPIDefinition("kpi3", "Average", KPIType.AVERAGE, lambda x: x.mean(), category="summary")
        ]
        
        scored_kpis = self.detector._score_kpis(kpis, self.business_data)
        
        # Business KPIs should be scored higher
        assert scored_kpis[0].category == "business"
        assert hasattr(scored_kpis[0], 'score')


class TestTrendAnalyzer:
    """Test cases for TrendAnalyzer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = TrendAnalyzer()
    
    def test_analyze_upward_trend(self):
        """Test analysis of upward trend."""
        values = [100, 110, 120, 130, 140, 150]
        trend = self.analyzer.analyze_trend(values)
        
        assert trend.direction == TrendDirection.UP
        assert trend.percentage_change > 0
        assert trend.absolute_change > 0
        assert trend.trend_strength > 0.8  # Strong upward trend
    
    def test_analyze_downward_trend(self):
        """Test analysis of downward trend."""
        values = [150, 140, 130, 120, 110, 100]
        trend = self.analyzer.analyze_trend(values)
        
        assert trend.direction == TrendDirection.DOWN
        assert trend.percentage_change < 0
        assert trend.absolute_change < 0
        assert trend.trend_strength > 0.8  # Strong downward trend
    
    def test_analyze_stable_trend(self):
        """Test analysis of stable trend."""
        values = [100, 102, 98, 101, 99, 103]
        trend = self.analyzer.analyze_trend(values)
        
        assert trend.direction == TrendDirection.STABLE
        assert abs(trend.percentage_change) < 5
    
    def test_analyze_volatile_trend(self):
        """Test analysis of volatile trend."""
        values = [100, 200, 50, 180, 30, 170, 40, 190]  # More volatile pattern
        trend = self.analyzer.analyze_trend(values)
        
        # Should be volatile due to high coefficient of variation
        assert trend.direction == TrendDirection.VOLATILE or trend.volatility > 0.5
        assert trend.volatility > 0.3
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        values = [100, 110]  # Less than minimum required
        trend = self.analyzer.analyze_trend(values)
        
        assert trend.direction == TrendDirection.STABLE
        assert trend.percentage_change == 0
        assert trend.trend_strength == 0
    
    def test_calculate_trend_strength(self):
        """Test trend strength calculation."""
        # Perfect linear trend
        values = [1, 2, 3, 4, 5]
        strength = self.analyzer._calculate_trend_strength(values)
        assert strength > 0.95  # Should be very strong
        
        # Random values
        values = [1, 5, 2, 4, 3]
        strength = self.analyzer._calculate_trend_strength(values)
        assert strength < 0.5  # Should be weak
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Low volatility
        stable_values = [100, 101, 99, 102, 98]
        volatility = self.analyzer._calculate_volatility(stable_values)
        assert volatility < 0.1
        
        # High volatility
        volatile_values = [100, 200, 50, 180, 30]
        volatility = self.analyzer._calculate_volatility(volatile_values)
        assert volatility > 0.5


class TestKPIComparator:
    """Test cases for KPIComparator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.comparator = KPIComparator()
        
        self.current_kpi = KPI(
            name="Revenue",
            value=10000,
            format_type="currency",
            description="Total revenue"
        )
        
        self.previous_kpi = KPI(
            name="Revenue",
            value=8000,
            format_type="currency",
            description="Total revenue"
        )
    
    def test_compare_periods_improvement(self):
        """Test period comparison with improvement."""
        comparison = self.comparator.compare_periods(self.current_kpi, self.previous_kpi)
        
        assert comparison.kpi_name == "Revenue"
        assert comparison.current_value == 10000
        assert comparison.comparison_value == 8000
        assert comparison.difference == 2000
        assert comparison.percentage_difference == 25.0
        assert comparison.is_improvement == True
        assert comparison.comparison_type == "previous_period"
    
    def test_compare_periods_decline(self):
        """Test period comparison with decline."""
        declined_kpi = KPI(name="Revenue", value=6000, format_type="currency")
        comparison = self.comparator.compare_periods(declined_kpi, self.previous_kpi)
        
        assert comparison.difference == -2000
        assert comparison.percentage_difference == -25.0
        assert comparison.is_improvement == False
    
    def test_compare_to_target_exceeded(self):
        """Test target comparison when target is exceeded."""
        target_value = 9000
        comparison = self.comparator.compare_to_target(self.current_kpi, target_value)
        
        assert comparison.comparison_type == "target"
        assert comparison.current_value == 10000
        assert comparison.comparison_value == 9000
        assert comparison.percentage_difference > 0
        assert comparison.is_improvement == True
    
    def test_compare_to_target_missed(self):
        """Test target comparison when target is missed."""
        target_value = 12000
        comparison = self.comparator.compare_to_target(self.current_kpi, target_value)
        
        assert comparison.percentage_difference < 0
        assert comparison.is_improvement == False
    
    def test_compare_to_benchmark(self):
        """Test benchmark comparison."""
        benchmark_value = 9500
        comparison = self.comparator.compare_to_benchmark(self.current_kpi, benchmark_value)
        
        assert comparison.comparison_type == "benchmark"
        assert comparison.current_value == 10000
        assert comparison.comparison_value == 9500
        assert comparison.is_improvement == True
    
    def test_zero_division_handling(self):
        """Test handling of zero values in comparisons."""
        zero_kpi = KPI(name="Test", value=0, format_type="number")
        current_kpi = KPI(name="Test", value=100, format_type="number")
        
        comparison = self.comparator.compare_periods(current_kpi, zero_kpi)
        assert comparison.percentage_difference == 0  # Should handle division by zero


class TestKPIInsightGenerator:
    """Test cases for KPIInsightGenerator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = KPIInsightGenerator()
        
        self.kpi = KPI(
            name="Revenue",
            value=10000,
            format_type="currency",
            description="Total revenue"
        )
    
    def test_trend_insights_upward(self):
        """Test trend insights for upward trend."""
        from src.visualization.kpi_engine import TrendAnalysis
        
        trend = TrendAnalysis(
            direction=TrendDirection.UP,
            percentage_change=25.0,
            absolute_change=2000,
            trend_strength=0.8,
            volatility=0.1,
            data_points=[8000, 9000, 10000]
        )
        
        insights = self.generator._trend_insights(self.kpi, trend, [])
        
        assert len(insights) > 0
        assert insights[0].insight_type == "trend"
        assert insights[0].severity == "success"
        assert "upward trend" in insights[0].message.lower()
    
    def test_trend_insights_downward(self):
        """Test trend insights for downward trend."""
        from src.visualization.kpi_engine import TrendAnalysis
        
        trend = TrendAnalysis(
            direction=TrendDirection.DOWN,
            percentage_change=-25.0,
            absolute_change=-2000,
            trend_strength=0.8,
            volatility=0.1,
            data_points=[12000, 11000, 10000]
        )
        
        insights = self.generator._trend_insights(self.kpi, trend, [])
        
        assert len(insights) > 0
        assert insights[0].severity == "warning"
        assert "downward trend" in insights[0].message.lower()
    
    def test_anomaly_insights(self):
        """Test anomaly detection insights."""
        zero_kpi = KPI(name="Revenue", value=0, format_type="currency")
        insights = self.generator._anomaly_insights(zero_kpi, None, [])
        
        assert len(insights) > 0
        assert insights[0].insight_type == "anomaly"
        assert insights[0].severity == "warning"
    
    def test_target_achievement_insights(self):
        """Test target achievement insights."""
        from src.visualization.kpi_engine import KPIComparison
        
        # Target exceeded
        comparison = KPIComparison(
            kpi_name="Revenue",
            current_value=10000,
            comparison_value=9000,
            difference=1000,
            percentage_difference=11.1,
            is_improvement=True,
            comparison_type="target"
        )
        
        insights = self.generator._target_achievement_insights(self.kpi, None, [comparison])
        
        assert len(insights) > 0
        assert insights[0].insight_type == "target_achievement"
        assert insights[0].severity == "success"
        assert "exceeded target" in insights[0].message.lower()
    
    def test_comparison_insights(self):
        """Test comparison insights."""
        from src.visualization.kpi_engine import KPIComparison
        
        comparison = KPIComparison(
            kpi_name="Revenue",
            current_value=10000,
            comparison_value=8000,
            difference=2000,
            percentage_difference=25.0,
            is_improvement=True,
            comparison_type="previous_period"
        )
        
        insights = self.generator._comparison_insights(self.kpi, None, [comparison])
        
        assert len(insights) > 0
        assert insights[0].insight_type == "comparison"
        assert "increased by" in insights[0].message.lower()
    
    def test_generate_insights_integration(self):
        """Test full insight generation."""
        from src.visualization.kpi_engine import TrendAnalysis, KPIComparison
        
        trend = TrendAnalysis(
            direction=TrendDirection.UP,
            percentage_change=15.0,
            absolute_change=1500,
            trend_strength=0.9,
            volatility=0.05,
            data_points=[8500, 9250, 10000]
        )
        
        comparison = KPIComparison(
            kpi_name="Revenue",
            current_value=10000,
            comparison_value=9000,
            difference=1000,
            percentage_difference=11.1,
            is_improvement=True,
            comparison_type="target"
        )
        
        insights = self.generator.generate_insights(self.kpi, trend, [comparison])
        
        assert len(insights) >= 2  # Should have trend and target insights
        insight_types = [insight.insight_type for insight in insights]
        assert "trend" in insight_types
        assert "target_achievement" in insight_types


class TestEnhancedKPIEngine:
    """Test cases for EnhancedKPIEngine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = EnhancedKPIEngine()
        
        # Sample data
        self.sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50, freq='D'),
            'revenue': np.random.normal(10000, 1000, 50),
            'cost': np.random.normal(7000, 800, 50),
            'orders': np.random.randint(100, 300, 50),
            'customers': np.random.randint(50, 150, 50)
        })
        
        # Ensure positive values
        self.sample_data['revenue'] = np.abs(self.sample_data['revenue'])
        self.sample_data['cost'] = np.abs(self.sample_data['cost'])
    
    def test_auto_detect_kpis(self):
        """Test automatic KPI detection."""
        kpis = self.engine.auto_detect_kpis(self.sample_data, max_kpis=8)
        
        assert len(kpis) <= 8
        assert len(kpis) > 0
        
        # Check that KPIs are registered
        assert len(self.engine.kpi_definitions) > 0
    
    def test_calculate_kpis(self):
        """Test KPI calculation."""
        # First detect KPIs
        self.engine.auto_detect_kpis(self.sample_data)
        
        # Calculate KPIs
        calculated_kpis = self.engine.calculate_kpis(self.sample_data)
        
        assert len(calculated_kpis) > 0
        
        for kpi in calculated_kpis:
            assert isinstance(kpi, KPI)
            assert kpi.name is not None
            assert kpi.value is not None
    
    def test_analyze_kpi_trends(self):
        """Test KPI trend analysis."""
        # Setup KPI with history - ascending values
        kpi_id = "test_kpi"
        self.engine.kpi_history[kpi_id] = [
            (datetime.now() - timedelta(days=i), 1000 + (10-i) * 100)  # Fixed: ascending order
            for i in range(10, 0, -1)
        ]
        
        trend = self.engine.analyze_kpi_trends(kpi_id)
        
        assert trend is not None
        assert trend.direction == TrendDirection.UP
        assert trend.percentage_change > 0
    
    def test_compare_kpis(self):
        """Test KPI comparison."""
        current_kpis = [
            KPI(name="Revenue", value=10000, format_type="currency"),
            KPI(name="Orders", value=200, format_type="number")
        ]
        
        previous_kpis = [
            KPI(name="Revenue", value=9000, format_type="currency"),
            KPI(name="Orders", value=180, format_type="number")
        ]
        
        comparisons = self.engine.compare_kpis(current_kpis, previous_kpis)
        
        assert len(comparisons) == 2
        assert all(comp.comparison_type == "previous_period" for comp in comparisons)
    
    def test_set_kpi_target(self):
        """Test setting KPI targets."""
        kpi_id = "revenue_kpi"
        target_value = 12000
        
        self.engine.set_kpi_target(kpi_id, target_value)
        
        assert self.engine.targets[kpi_id] == target_value
    
    def test_set_kpi_benchmark(self):
        """Test setting KPI benchmarks."""
        kpi_id = "revenue_kpi"
        benchmark_value = 11000
        
        self.engine.set_kpi_benchmark(kpi_id, benchmark_value)
        
        assert self.engine.benchmarks[kpi_id] == benchmark_value
    
    def test_get_kpi_summary(self):
        """Test comprehensive KPI summary."""
        summary = self.engine.get_kpi_summary(self.sample_data)
        
        assert "kpis" in summary
        assert "comparisons" in summary
        assert "insights" in summary
        assert "summary" in summary
        
        assert isinstance(summary["kpis"], list)
        assert isinstance(summary["summary"], dict)
        assert "total_kpis" in summary["summary"]
    
    def test_kpi_history_management(self):
        """Test KPI history storage and management."""
        # Detect and calculate KPIs multiple times
        self.engine.auto_detect_kpis(self.sample_data)
        
        # First calculation
        kpis1 = self.engine.calculate_kpis(self.sample_data)
        
        # Modify data and calculate again
        modified_data = self.sample_data.copy()
        modified_data['revenue'] *= 1.1
        
        kpis2 = self.engine.calculate_kpis(modified_data)
        
        # Check that history is stored
        assert len(self.engine.kpi_history) > 0
        
        # Check that each KPI has history
        for kpi_id in self.engine.kpi_definitions.keys():
            if kpi_id in self.engine.kpi_history:
                assert len(self.engine.kpi_history[kpi_id]) >= 1
    
    def test_error_handling(self):
        """Test error handling in KPI calculations."""
        # Create a KPI definition that will fail
        def failing_calculation(df):
            raise ValueError("Calculation failed")
        
        from src.visualization.kpi_engine import KPIDefinition, KPIType
        
        failing_kpi = KPIDefinition(
            kpi_id="failing_kpi",
            name="Failing KPI",
            kpi_type=KPIType.SUM,
            calculation_func=failing_calculation
        )
        
        self.engine.kpi_definitions["failing_kpi"] = failing_kpi
        
        # Should not raise exception, but log error
        kpis = self.engine.calculate_kpis(self.sample_data, ["failing_kpi"])
        
        # Should return empty list or handle gracefully
        assert isinstance(kpis, list)


@pytest.fixture
def sample_dataframe():
    """Fixture providing sample dataframe for tests."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'revenue': np.random.normal(5000, 500, 30),
        'cost': np.random.normal(3000, 300, 30),
        'quantity': np.random.randint(10, 100, 30),
        'category': np.random.choice(['A', 'B', 'C'], 30),
        'region': np.random.choice(['North', 'South'], 30)
    })


def test_kpi_engine_performance(sample_dataframe):
    """Test KPI engine performance with larger datasets."""
    # Create larger dataset
    large_data = pd.concat([sample_dataframe] * 100, ignore_index=True)
    
    engine = EnhancedKPIEngine()
    
    # Measure time for KPI detection and calculation
    import time
    
    start_time = time.time()
    kpis = engine.auto_detect_kpis(large_data, max_kpis=20)
    detection_time = time.time() - start_time
    
    start_time = time.time()
    calculated_kpis = engine.calculate_kpis(large_data)
    calculation_time = time.time() - start_time
    
    # Performance assertions (adjust thresholds as needed)
    assert detection_time < 5.0  # Should complete within 5 seconds
    assert calculation_time < 2.0  # Should complete within 2 seconds
    assert len(calculated_kpis) > 0


def test_kpi_engine_memory_usage(sample_dataframe):
    """Test KPI engine memory usage."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    engine = EnhancedKPIEngine()
    
    # Perform multiple operations
    for _ in range(10):
        engine.auto_detect_kpis(sample_dataframe)
        engine.calculate_kpis(sample_dataframe)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 100MB)
    assert memory_increase < 100 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__])