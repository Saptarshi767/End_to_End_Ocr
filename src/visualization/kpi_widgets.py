"""
Customizable KPI dashboard widgets.

This module provides customizable KPI widgets with various display formats,
interactive features, and real-time updates.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

from ..core.models import KPI
from .kpi_engine import TrendAnalysis, KPIComparison, KPIInsight, TrendDirection


logger = logging.getLogger(__name__)


class KPIWidgetType(Enum):
    """Types of KPI widgets."""
    SIMPLE_VALUE = "simple_value"
    CARD = "card"
    GAUGE = "gauge"
    PROGRESS_BAR = "progress_bar"
    TREND_CHART = "trend_chart"
    COMPARISON = "comparison"
    SPARKLINE = "sparkline"


class KPIDisplayFormat(Enum):
    """Display formats for KPI values."""
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DECIMAL = "decimal"
    SCIENTIFIC = "scientific"
    CUSTOM = "custom"


@dataclass
class KPIWidgetConfig:
    """Configuration for KPI widgets."""
    widget_id: str
    widget_type: KPIWidgetType
    title: str
    display_format: KPIDisplayFormat = KPIDisplayFormat.NUMBER
    show_trend: bool = True
    show_comparison: bool = True
    show_target: bool = False
    color_scheme: str = "default"
    size: str = "medium"  # small, medium, large
    refresh_interval: Optional[int] = None
    custom_format: Optional[str] = None
    thresholds: Dict[str, float] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)


@dataclass
class KPIWidgetData:
    """Data for KPI widgets."""
    kpi: KPI
    trend: Optional[TrendAnalysis] = None
    comparison: Optional[KPIComparison] = None
    insights: List[KPIInsight] = field(default_factory=list)
    historical_data: List[float] = field(default_factory=list)
    target_value: Optional[float] = None
    benchmark_value: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)


class KPIValueFormatter:
    """Formats KPI values for display."""
    
    @staticmethod
    def format_value(value: Union[int, float, str], format_type: KPIDisplayFormat, 
                    custom_format: Optional[str] = None, precision: int = 2) -> str:
        """Format KPI value according to display format."""
        if isinstance(value, str):
            return value
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        if format_type == KPIDisplayFormat.NUMBER:
            if numeric_value >= 1000000:
                return f"{numeric_value/1000000:.1f}M"
            elif numeric_value >= 1000:
                return f"{numeric_value/1000:.1f}K"
            else:
                return f"{numeric_value:,.{precision}f}"
        
        elif format_type == KPIDisplayFormat.CURRENCY:
            if numeric_value >= 1000000:
                return f"${numeric_value/1000000:.1f}M"
            elif numeric_value >= 1000:
                return f"${numeric_value/1000:.1f}K"
            else:
                return f"${numeric_value:,.{precision}f}"
        
        elif format_type == KPIDisplayFormat.PERCENTAGE:
            return f"{numeric_value:.{precision}f}%"
        
        elif format_type == KPIDisplayFormat.DECIMAL:
            return f"{numeric_value:.{precision}f}"
        
        elif format_type == KPIDisplayFormat.SCIENTIFIC:
            return f"{numeric_value:.{precision}e}"
        
        elif format_type == KPIDisplayFormat.CUSTOM and custom_format:
            try:
                return custom_format.format(value=numeric_value)
            except:
                return str(numeric_value)
        
        return str(numeric_value)
    
    @staticmethod
    def format_trend(trend: Optional[TrendAnalysis]) -> Dict[str, Any]:
        """Format trend data for display."""
        if not trend:
            return {"direction": "stable", "change": 0, "icon": "→", "color": "gray"}
        
        direction_map = {
            TrendDirection.UP: {"icon": "↗", "color": "green", "text": "increasing"},
            TrendDirection.DOWN: {"icon": "↘", "color": "red", "text": "decreasing"},
            TrendDirection.STABLE: {"icon": "→", "color": "gray", "text": "stable"},
            TrendDirection.VOLATILE: {"icon": "↕", "color": "orange", "text": "volatile"}
        }
        
        direction_info = direction_map.get(trend.direction, direction_map[TrendDirection.STABLE])
        
        return {
            "direction": trend.direction.value,
            "change": trend.percentage_change,
            "icon": direction_info["icon"],
            "color": direction_info["color"],
            "text": direction_info["text"],
            "strength": trend.trend_strength
        }
    
    @staticmethod
    def format_comparison(comparison: Optional[KPIComparison]) -> Dict[str, Any]:
        """Format comparison data for display."""
        if not comparison:
            return {"type": "none", "change": 0, "text": "No comparison available"}
        
        change_text = "increased" if comparison.difference > 0 else "decreased"
        comparison_text = f"{change_text} by {abs(comparison.percentage_difference):.1f}%"
        
        if comparison.comparison_type == "target":
            if comparison.is_improvement:
                comparison_text = f"Target exceeded by {comparison.percentage_difference:.1f}%"
            else:
                comparison_text = f"Below target by {abs(comparison.percentage_difference):.1f}%"
        
        return {
            "type": comparison.comparison_type,
            "change": comparison.percentage_difference,
            "text": comparison_text,
            "is_improvement": comparison.is_improvement,
            "color": "green" if comparison.is_improvement else "red"
        }


class KPIWidget:
    """Base class for KPI widgets."""
    
    def __init__(self, config: KPIWidgetConfig, data: KPIWidgetData):
        self.config = config
        self.data = data
        self.formatter = KPIValueFormatter()
    
    def render(self) -> Dict[str, Any]:
        """Render widget as dictionary for JSON serialization."""
        base_data = {
            "id": self.config.widget_id,
            "type": self.config.widget_type.value,
            "title": self.config.title,
            "kpi_name": self.data.kpi.name,
            "value": self.formatter.format_value(
                self.data.kpi.value, 
                self.config.display_format,
                self.config.custom_format
            ),
            "raw_value": self.data.kpi.value,
            "description": self.data.kpi.description,
            "last_updated": self.data.last_updated.isoformat(),
            "position": self.config.position,
            "size": self.config.size,
            "color_scheme": self.config.color_scheme
        }
        
        # Add trend information
        if self.config.show_trend and self.data.trend:
            base_data["trend"] = self.formatter.format_trend(self.data.trend)
        
        # Add comparison information
        if self.config.show_comparison and self.data.comparison:
            base_data["comparison"] = self.formatter.format_comparison(self.data.comparison)
        
        # Add target information
        if self.config.show_target and self.data.target_value:
            base_data["target"] = {
                "value": self.data.target_value,
                "formatted_value": self.formatter.format_value(
                    self.data.target_value, self.config.display_format
                ),
                "achievement": (float(self.data.kpi.value) / self.data.target_value * 100) if self.data.target_value != 0 else 0
            }
        
        # Add insights
        if self.data.insights:
            base_data["insights"] = [
                {
                    "type": insight.insight_type,
                    "message": insight.message,
                    "severity": insight.severity,
                    "confidence": insight.confidence
                }
                for insight in self.data.insights
            ]
        
        return base_data
    
    def update_data(self, new_data: KPIWidgetData) -> None:
        """Update widget data."""
        self.data = new_data
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get alert status based on thresholds."""
        if not self.config.thresholds:
            return {"status": "normal", "message": ""}
        
        try:
            value = float(self.data.kpi.value)
        except (ValueError, TypeError):
            return {"status": "unknown", "message": "Cannot evaluate non-numeric value"}
        
        # Check critical threshold
        if "critical_low" in self.config.thresholds and value < self.config.thresholds["critical_low"]:
            return {"status": "critical", "message": f"Value below critical threshold ({self.config.thresholds['critical_low']})"}
        
        if "critical_high" in self.config.thresholds and value > self.config.thresholds["critical_high"]:
            return {"status": "critical", "message": f"Value above critical threshold ({self.config.thresholds['critical_high']})"}
        
        # Check warning threshold
        if "warning_low" in self.config.thresholds and value < self.config.thresholds["warning_low"]:
            return {"status": "warning", "message": f"Value below warning threshold ({self.config.thresholds['warning_low']})"}
        
        if "warning_high" in self.config.thresholds and value > self.config.thresholds["warning_high"]:
            return {"status": "warning", "message": f"Value above warning threshold ({self.config.thresholds['warning_high']})"}
        
        return {"status": "normal", "message": "Value within normal range"}


class SimpleValueWidget(KPIWidget):
    """Simple value display widget."""
    
    def render(self) -> Dict[str, Any]:
        """Render simple value widget."""
        data = super().render()
        data.update({
            "widget_class": "simple-value-widget",
            "display_style": "minimal"
        })
        return data


class CardWidget(KPIWidget):
    """Card-style KPI widget with rich information."""
    
    def render(self) -> Dict[str, Any]:
        """Render card widget."""
        data = super().render()
        
        # Add card-specific styling
        data.update({
            "widget_class": "card-widget",
            "display_style": "card",
            "show_border": True,
            "show_shadow": True
        })
        
        # Add alert status
        alert_status = self.get_alert_status()
        data["alert_status"] = alert_status
        
        return data


class GaugeWidget(KPIWidget):
    """Gauge-style KPI widget."""
    
    def render(self) -> Dict[str, Any]:
        """Render gauge widget."""
        data = super().render()
        
        # Calculate gauge parameters
        gauge_data = self._calculate_gauge_data()
        
        data.update({
            "widget_class": "gauge-widget",
            "display_style": "gauge",
            "gauge": gauge_data
        })
        
        return data
    
    def _calculate_gauge_data(self) -> Dict[str, Any]:
        """Calculate gauge display parameters."""
        try:
            value = float(self.data.kpi.value)
        except (ValueError, TypeError):
            return {"percentage": 0, "color": "gray"}
        
        # Determine gauge range
        if self.data.target_value:
            max_value = self.data.target_value * 1.2  # 120% of target
            percentage = min((value / max_value) * 100, 100)
        elif "max" in self.config.thresholds:
            max_value = self.config.thresholds["max"]
            percentage = min((value / max_value) * 100, 100)
        else:
            # Use historical data to determine range
            if self.data.historical_data:
                max_value = max(self.data.historical_data) * 1.1
                percentage = min((value / max_value) * 100, 100) if max_value > 0 else 0
            else:
                percentage = 50  # Default to middle
        
        # Determine color based on percentage and thresholds
        color = self._get_gauge_color(percentage)
        
        return {
            "percentage": percentage,
            "color": color,
            "min_value": 0,
            "max_value": max_value if 'max_value' in locals() else 100,
            "sectors": self._get_gauge_sectors()
        }
    
    def _get_gauge_color(self, percentage: float) -> str:
        """Get gauge color based on percentage."""
        if percentage >= 80:
            return "green"
        elif percentage >= 60:
            return "yellow"
        elif percentage >= 40:
            return "orange"
        else:
            return "red"
    
    def _get_gauge_sectors(self) -> List[Dict[str, Any]]:
        """Get gauge sectors for color coding."""
        return [
            {"range": [0, 40], "color": "red", "label": "Low"},
            {"range": [40, 60], "color": "orange", "label": "Below Average"},
            {"range": [60, 80], "color": "yellow", "label": "Average"},
            {"range": [80, 100], "color": "green", "label": "Good"}
        ]


class ProgressBarWidget(KPIWidget):
    """Progress bar KPI widget."""
    
    def render(self) -> Dict[str, Any]:
        """Render progress bar widget."""
        data = super().render()
        
        # Calculate progress
        progress_data = self._calculate_progress()
        
        data.update({
            "widget_class": "progress-bar-widget",
            "display_style": "progress_bar",
            "progress": progress_data
        })
        
        return data
    
    def _calculate_progress(self) -> Dict[str, Any]:
        """Calculate progress bar data."""
        try:
            value = float(self.data.kpi.value)
        except (ValueError, TypeError):
            return {"percentage": 0, "color": "gray", "status": "unknown"}
        
        if self.data.target_value:
            percentage = min((value / self.data.target_value) * 100, 100)
            status = "on_track" if percentage >= 80 else "behind" if percentage >= 50 else "critical"
        else:
            percentage = 50  # Default
            status = "unknown"
        
        color = "green" if percentage >= 80 else "yellow" if percentage >= 50 else "red"
        
        return {
            "percentage": percentage,
            "color": color,
            "status": status,
            "target_value": self.data.target_value,
            "current_value": value
        }


class TrendChartWidget(KPIWidget):
    """Trend chart KPI widget."""
    
    def render(self) -> Dict[str, Any]:
        """Render trend chart widget."""
        data = super().render()
        
        # Prepare chart data
        chart_data = self._prepare_chart_data()
        
        data.update({
            "widget_class": "trend-chart-widget",
            "display_style": "trend_chart",
            "chart": chart_data
        })
        
        return data
    
    def _prepare_chart_data(self) -> Dict[str, Any]:
        """Prepare data for trend chart."""
        if not self.data.historical_data:
            return {"type": "line", "data": [], "message": "No historical data available"}
        
        # Create time series data
        chart_points = []
        for i, value in enumerate(self.data.historical_data):
            chart_points.append({
                "x": i,
                "y": value,
                "label": f"Period {i+1}"
            })
        
        return {
            "type": "line",
            "data": chart_points,
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        }


class ComparisonWidget(KPIWidget):
    """Comparison KPI widget."""
    
    def render(self) -> Dict[str, Any]:
        """Render comparison widget."""
        data = super().render()
        
        # Add comparison-specific data
        comparison_data = self._prepare_comparison_data()
        
        data.update({
            "widget_class": "comparison-widget",
            "display_style": "comparison",
            "comparison_details": comparison_data
        })
        
        return data
    
    def _prepare_comparison_data(self) -> Dict[str, Any]:
        """Prepare detailed comparison data."""
        if not self.data.comparison:
            return {"available": False, "message": "No comparison data available"}
        
        comparison = self.data.comparison
        
        return {
            "available": True,
            "current_value": comparison.current_value,
            "comparison_value": comparison.comparison_value,
            "difference": comparison.difference,
            "percentage_difference": comparison.percentage_difference,
            "is_improvement": comparison.is_improvement,
            "comparison_type": comparison.comparison_type,
            "visual_indicator": {
                "arrow": "↗" if comparison.difference > 0 else "↘" if comparison.difference < 0 else "→",
                "color": "green" if comparison.is_improvement else "red" if comparison.difference != 0 else "gray"
            }
        }


class SparklineWidget(KPIWidget):
    """Sparkline KPI widget."""
    
    def render(self) -> Dict[str, Any]:
        """Render sparkline widget."""
        data = super().render()
        
        # Prepare sparkline data
        sparkline_data = self._prepare_sparkline_data()
        
        data.update({
            "widget_class": "sparkline-widget",
            "display_style": "sparkline",
            "sparkline": sparkline_data
        })
        
        return data
    
    def _prepare_sparkline_data(self) -> Dict[str, Any]:
        """Prepare sparkline chart data."""
        if not self.data.historical_data:
            return {"data": [], "message": "No data for sparkline"}
        
        # Normalize data for sparkline (0-100 scale)
        min_val = min(self.data.historical_data)
        max_val = max(self.data.historical_data)
        
        if max_val == min_val:
            normalized_data = [50] * len(self.data.historical_data)
        else:
            normalized_data = [
                ((val - min_val) / (max_val - min_val)) * 100
                for val in self.data.historical_data
            ]
        
        return {
            "data": normalized_data,
            "min_value": min_val,
            "max_value": max_val,
            "current_value": self.data.historical_data[-1] if self.data.historical_data else 0,
            "trend_color": self._get_sparkline_color()
        }
    
    def _get_sparkline_color(self) -> str:
        """Get sparkline color based on trend."""
        if self.data.trend:
            if self.data.trend.direction == TrendDirection.UP:
                return "green"
            elif self.data.trend.direction == TrendDirection.DOWN:
                return "red"
            elif self.data.trend.direction == TrendDirection.VOLATILE:
                return "orange"
        return "blue"


class KPIWidgetFactory:
    """Factory for creating KPI widgets."""
    
    @staticmethod
    def create_widget(config: KPIWidgetConfig, data: KPIWidgetData) -> KPIWidget:
        """Create KPI widget based on configuration."""
        widget_classes = {
            KPIWidgetType.SIMPLE_VALUE: SimpleValueWidget,
            KPIWidgetType.CARD: CardWidget,
            KPIWidgetType.GAUGE: GaugeWidget,
            KPIWidgetType.PROGRESS_BAR: ProgressBarWidget,
            KPIWidgetType.TREND_CHART: TrendChartWidget,
            KPIWidgetType.COMPARISON: ComparisonWidget,
            KPIWidgetType.SPARKLINE: SparklineWidget
        }
        
        widget_class = widget_classes.get(config.widget_type, CardWidget)
        return widget_class(config, data)


class KPIWidgetManager:
    """Manager for KPI widgets on a dashboard."""
    
    def __init__(self):
        self.widgets: Dict[str, KPIWidget] = {}
        self.widget_configs: Dict[str, KPIWidgetConfig] = {}
        self.update_callbacks: List[Callable] = []
    
    def add_widget(self, config: KPIWidgetConfig, data: KPIWidgetData) -> str:
        """Add KPI widget to dashboard."""
        widget = KPIWidgetFactory.create_widget(config, data)
        self.widgets[config.widget_id] = widget
        self.widget_configs[config.widget_id] = config
        
        logger.info(f"Added KPI widget: {config.widget_id} ({config.widget_type.value})")
        return config.widget_id
    
    def update_widget_data(self, widget_id: str, new_data: KPIWidgetData) -> bool:
        """Update data for a specific widget."""
        if widget_id not in self.widgets:
            return False
        
        self.widgets[widget_id].update_data(new_data)
        
        # Trigger update callbacks
        for callback in self.update_callbacks:
            try:
                callback(widget_id, new_data)
            except Exception as e:
                logger.error(f"Error in widget update callback: {e}")
        
        return True
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove widget from dashboard."""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
            del self.widget_configs[widget_id]
            logger.info(f"Removed KPI widget: {widget_id}")
            return True
        return False
    
    def get_widget(self, widget_id: str) -> Optional[KPIWidget]:
        """Get widget by ID."""
        return self.widgets.get(widget_id)
    
    def get_all_widgets(self) -> Dict[str, KPIWidget]:
        """Get all widgets."""
        return self.widgets.copy()
    
    def render_dashboard(self) -> Dict[str, Any]:
        """Render all widgets for dashboard display."""
        rendered_widgets = {}
        
        for widget_id, widget in self.widgets.items():
            try:
                rendered_widgets[widget_id] = widget.render()
            except Exception as e:
                logger.error(f"Error rendering widget {widget_id}: {e}")
                rendered_widgets[widget_id] = {
                    "id": widget_id,
                    "error": f"Rendering error: {str(e)}"
                }
        
        return {
            "widgets": rendered_widgets,
            "layout": self._generate_layout(),
            "total_widgets": len(rendered_widgets),
            "last_updated": datetime.now().isoformat()
        }
    
    def _generate_layout(self) -> Dict[str, Any]:
        """Generate layout information for widgets."""
        layout_grid = []
        
        for widget_id, config in self.widget_configs.items():
            if config.position:
                layout_grid.append({
                    "widget_id": widget_id,
                    "x": config.position.get("x", 0),
                    "y": config.position.get("y", 0),
                    "width": self._get_width_for_size(config.size),
                    "height": self._get_height_for_size(config.size, config.widget_type)
                })
        
        return {
            "grid": layout_grid,
            "columns": 12,  # Bootstrap-style 12-column grid
            "responsive": True
        }
    
    def _get_width_for_size(self, size: str) -> int:
        """Get grid width for widget size."""
        size_map = {
            "small": 3,
            "medium": 4,
            "large": 6,
            "extra_large": 12
        }
        return size_map.get(size, 4)
    
    def _get_height_for_size(self, size: str, widget_type: KPIWidgetType) -> int:
        """Get grid height for widget size and type."""
        base_heights = {
            KPIWidgetType.SIMPLE_VALUE: 2,
            KPIWidgetType.CARD: 3,
            KPIWidgetType.GAUGE: 4,
            KPIWidgetType.PROGRESS_BAR: 2,
            KPIWidgetType.TREND_CHART: 5,
            KPIWidgetType.COMPARISON: 3,
            KPIWidgetType.SPARKLINE: 2
        }
        
        base_height = base_heights.get(widget_type, 3)
        
        size_multipliers = {
            "small": 0.8,
            "medium": 1.0,
            "large": 1.3,
            "extra_large": 1.5
        }
        
        multiplier = size_multipliers.get(size, 1.0)
        return max(2, int(base_height * multiplier))
    
    def add_update_callback(self, callback: Callable) -> None:
        """Add callback for widget updates."""
        self.update_callbacks.append(callback)
    
    def get_widget_alerts(self) -> List[Dict[str, Any]]:
        """Get all widget alerts."""
        alerts = []
        
        for widget_id, widget in self.widgets.items():
            alert_status = widget.get_alert_status()
            if alert_status["status"] != "normal":
                alerts.append({
                    "widget_id": widget_id,
                    "kpi_name": widget.data.kpi.name,
                    "status": alert_status["status"],
                    "message": alert_status["message"],
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export widget configuration for saving/loading."""
        return {
            "widgets": {
                widget_id: {
                    "config": {
                        "widget_id": config.widget_id,
                        "widget_type": config.widget_type.value,
                        "title": config.title,
                        "display_format": config.display_format.value,
                        "show_trend": config.show_trend,
                        "show_comparison": config.show_comparison,
                        "show_target": config.show_target,
                        "color_scheme": config.color_scheme,
                        "size": config.size,
                        "custom_format": config.custom_format,
                        "thresholds": config.thresholds,
                        "position": config.position
                    }
                }
                for widget_id, config in self.widget_configs.items()
            },
            "export_timestamp": datetime.now().isoformat()
        }