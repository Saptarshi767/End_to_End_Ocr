"""
Dashboard widgets and interactive components.

This module provides specialized dashboard widgets for enhanced interactivity,
including advanced filters, KPI widgets, and custom chart components.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, date
import pandas as pd
import json
import logging

from ..core.models import Filter, KPI, ChartConfig, ChartType


logger = logging.getLogger(__name__)


@dataclass
class WidgetConfig:
    """Configuration for dashboard widgets."""
    widget_id: str
    widget_type: str
    title: str
    position: Dict[str, int] = field(default_factory=dict)
    size: Dict[str, int] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    data_source: Optional[str] = None
    refresh_interval: Optional[int] = None


@dataclass
class FilterWidget:
    """Advanced filter widget with enhanced functionality."""
    config: WidgetConfig
    filter_config: Filter
    current_value: Any = None
    is_active: bool = False
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary for JSON serialization."""
        return {
            "id": self.config.widget_id,
            "type": "filter",
            "title": self.config.title,
            "filter_type": self.filter_config.filter_type,
            "column": self.filter_config.column,
            "values": self.filter_config.values,
            "current_value": self.current_value,
            "is_active": self.is_active,
            "position": self.config.position,
            "size": self.config.size,
            "options": self.config.options
        }


@dataclass
class KPIWidget:
    """KPI display widget with trend indicators."""
    config: WidgetConfig
    kpi: KPI
    trend_data: List[float] = field(default_factory=list)
    target_value: Optional[float] = None
    alert_threshold: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary for JSON serialization."""
        return {
            "id": self.config.widget_id,
            "type": "kpi",
            "title": self.config.title,
            "name": self.kpi.name,
            "value": self.kpi.value,
            "format_type": self.kpi.format_type,
            "description": self.kpi.description,
            "trend": self.kpi.trend,
            "trend_data": self.trend_data,
            "target_value": self.target_value,
            "alert_threshold": self.alert_threshold,
            "position": self.config.position,
            "size": self.config.size,
            "options": self.config.options
        }


@dataclass
class ChartWidget:
    """Chart widget with interactive capabilities."""
    config: WidgetConfig
    chart_config: ChartConfig
    chart_data: Dict[str, Any] = field(default_factory=dict)
    interactions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary for JSON serialization."""
        return {
            "id": self.config.widget_id,
            "type": "chart",
            "title": self.config.title,
            "chart_type": self.chart_config.chart_type.value,
            "chart_config": {
                "x_column": self.chart_config.x_column,
                "y_column": self.chart_config.y_column,
                "color_column": self.chart_config.color_column,
                "aggregation": self.chart_config.aggregation,
                "options": self.chart_config.options
            },
            "chart_data": self.chart_data,
            "interactions": self.interactions,
            "position": self.config.position,
            "size": self.config.size,
            "options": self.config.options
        }


class AdvancedFilterManager:
    """Advanced filter manager with cascading filters and dependencies."""
    
    def __init__(self):
        self.filter_widgets: Dict[str, FilterWidget] = {}
        self.filter_dependencies: Dict[str, List[str]] = {}
        self.filter_callbacks: Dict[str, List[Callable]] = {}
    
    def add_filter_widget(self, widget: FilterWidget) -> None:
        """Add filter widget to manager."""
        self.filter_widgets[widget.config.widget_id] = widget
        
        # Setup dependencies
        if widget.dependencies:
            for dep in widget.dependencies:
                if dep not in self.filter_dependencies:
                    self.filter_dependencies[dep] = []
                self.filter_dependencies[dep].append(widget.config.widget_id)
        
        logger.info(f"Added filter widget: {widget.config.widget_id}")
    
    def update_filter_value(self, widget_id: str, value: Any) -> List[str]:
        """Update filter value and cascade to dependent filters."""
        if widget_id not in self.filter_widgets:
            return []
        
        widget = self.filter_widgets[widget_id]
        old_value = widget.current_value
        widget.current_value = value
        widget.is_active = value is not None and value != ""
        
        # Update dependent filters
        updated_widgets = [widget_id]
        if widget_id in self.filter_dependencies:
            for dependent_id in self.filter_dependencies[widget_id]:
                if dependent_id in self.filter_widgets:
                    self._update_dependent_filter(dependent_id, widget_id, value)
                    updated_widgets.append(dependent_id)
        
        # Trigger callbacks
        if widget_id in self.filter_callbacks:
            for callback in self.filter_callbacks[widget_id]:
                try:
                    callback(widget_id, old_value, value)
                except Exception as e:
                    logger.error(f"Error in filter callback: {e}")
        
        return updated_widgets
    
    def _update_dependent_filter(self, dependent_id: str, parent_id: str, parent_value: Any) -> None:
        """Update dependent filter based on parent filter change."""
        dependent_widget = self.filter_widgets[dependent_id]
        
        # This is a simplified implementation - in practice, you'd implement
        # specific logic based on the relationship between filters
        if dependent_widget.filter_config.filter_type == "select":
            # Filter available values based on parent selection
            # This would require access to the underlying data
            pass
    
    def get_active_filters(self) -> Dict[str, Any]:
        """Get all active filter values."""
        return {
            widget_id: widget.current_value
            for widget_id, widget in self.filter_widgets.items()
            if widget.is_active
        }
    
    def clear_all_filters(self) -> None:
        """Clear all filter values."""
        for widget in self.filter_widgets.values():
            widget.current_value = None
            widget.is_active = False
    
    def add_filter_callback(self, widget_id: str, callback: Callable) -> None:
        """Add callback for filter changes."""
        if widget_id not in self.filter_callbacks:
            self.filter_callbacks[widget_id] = []
        self.filter_callbacks[widget_id].append(callback)


class KPICalculator:
    """Advanced KPI calculation engine."""
    
    def __init__(self):
        self.kpi_definitions: Dict[str, Dict[str, Any]] = {}
        self.calculation_cache: Dict[str, Any] = {}
    
    def register_kpi(self, kpi_id: str, name: str, calculation_func: Callable, 
                    format_type: str = "number", description: str = "") -> None:
        """Register a KPI calculation function."""
        self.kpi_definitions[kpi_id] = {
            "name": name,
            "calculation_func": calculation_func,
            "format_type": format_type,
            "description": description
        }
    
    def calculate_kpi(self, kpi_id: str, data: pd.DataFrame, **kwargs) -> KPI:
        """Calculate KPI value from data."""
        if kpi_id not in self.kpi_definitions:
            raise ValueError(f"Unknown KPI: {kpi_id}")
        
        definition = self.kpi_definitions[kpi_id]
        
        try:
            value = definition["calculation_func"](data, **kwargs)
            
            # Calculate trend if historical data is available
            trend = self._calculate_trend(kpi_id, value)
            
            return KPI(
                name=definition["name"],
                value=value,
                format_type=definition["format_type"],
                description=definition["description"],
                trend=trend
            )
        except Exception as e:
            logger.error(f"Error calculating KPI {kpi_id}: {e}")
            return KPI(
                name=definition["name"],
                value=0,
                format_type=definition["format_type"],
                description=f"Error: {str(e)}"
            )
    
    def _calculate_trend(self, kpi_id: str, current_value: float) -> Optional[float]:
        """Calculate trend percentage from previous value."""
        cache_key = f"{kpi_id}_history"
        
        if cache_key not in self.calculation_cache:
            self.calculation_cache[cache_key] = []
        
        history = self.calculation_cache[cache_key]
        
        if len(history) > 0:
            previous_value = history[-1]
            if previous_value != 0:
                trend = ((current_value - previous_value) / previous_value) * 100
            else:
                trend = 0.0
        else:
            trend = None
        
        # Store current value for next calculation
        history.append(current_value)
        
        # Keep only last 10 values
        if len(history) > 10:
            history.pop(0)
        
        return trend
    
    def get_available_kpis(self) -> List[str]:
        """Get list of available KPI IDs."""
        return list(self.kpi_definitions.keys())


class InteractiveChartManager:
    """Manager for interactive chart behaviors and events."""
    
    def __init__(self):
        self.chart_widgets: Dict[str, ChartWidget] = {}
        self.interaction_handlers: Dict[str, Dict[str, Callable]] = {}
    
    def add_chart_widget(self, widget: ChartWidget) -> None:
        """Add chart widget to manager."""
        self.chart_widgets[widget.config.widget_id] = widget
        logger.info(f"Added chart widget: {widget.config.widget_id}")
    
    def register_interaction_handler(self, widget_id: str, interaction_type: str, 
                                   handler: Callable) -> None:
        """Register interaction handler for chart widget."""
        if widget_id not in self.interaction_handlers:
            self.interaction_handlers[widget_id] = {}
        
        self.interaction_handlers[widget_id][interaction_type] = handler
    
    def handle_chart_interaction(self, widget_id: str, interaction_type: str, 
                               interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chart interaction event."""
        if (widget_id in self.interaction_handlers and 
            interaction_type in self.interaction_handlers[widget_id]):
            
            handler = self.interaction_handlers[widget_id][interaction_type]
            try:
                return handler(interaction_data)
            except Exception as e:
                logger.error(f"Error handling chart interaction: {e}")
                return {"error": str(e)}
        
        return {"message": "No handler registered for this interaction"}
    
    def update_chart_data(self, widget_id: str, new_data: Dict[str, Any]) -> None:
        """Update chart data for widget."""
        if widget_id in self.chart_widgets:
            self.chart_widgets[widget_id].chart_data = new_data
    
    def get_chart_interactions(self, widget_id: str) -> Dict[str, Any]:
        """Get available interactions for chart widget."""
        if widget_id not in self.chart_widgets:
            return {}
        
        widget = self.chart_widgets[widget_id]
        chart_type = widget.chart_config.chart_type
        
        # Define available interactions based on chart type
        interactions = {
            ChartType.BAR: ["click", "hover", "select"],
            ChartType.LINE: ["click", "hover", "zoom", "pan"],
            ChartType.PIE: ["click", "hover", "select"],
            ChartType.SCATTER: ["click", "hover", "select", "zoom", "pan"],
            ChartType.HISTOGRAM: ["click", "hover", "zoom"],
            ChartType.HEATMAP: ["click", "hover"],
            ChartType.TABLE: ["click", "sort", "filter"]
        }
        
        return {
            "available_interactions": interactions.get(chart_type, []),
            "registered_handlers": list(self.interaction_handlers.get(widget_id, {}).keys())
        }


class DashboardThemeManager:
    """Manager for dashboard themes and styling."""
    
    def __init__(self):
        self.themes: Dict[str, Dict[str, Any]] = {
            "default": {
                "name": "Default",
                "colors": {
                    "primary": "#007bff",
                    "secondary": "#6c757d",
                    "success": "#28a745",
                    "danger": "#dc3545",
                    "warning": "#ffc107",
                    "info": "#17a2b8",
                    "light": "#f8f9fa",
                    "dark": "#343a40"
                },
                "chart_colors": [
                    "#007bff", "#28a745", "#ffc107", "#dc3545",
                    "#17a2b8", "#6f42c1", "#e83e8c", "#fd7e14"
                ],
                "background": "#ffffff",
                "text": "#212529",
                "border": "#dee2e6"
            },
            "dark": {
                "name": "Dark",
                "colors": {
                    "primary": "#0d6efd",
                    "secondary": "#6c757d",
                    "success": "#198754",
                    "danger": "#dc3545",
                    "warning": "#ffc107",
                    "info": "#0dcaf0",
                    "light": "#f8f9fa",
                    "dark": "#212529"
                },
                "chart_colors": [
                    "#0d6efd", "#198754", "#ffc107", "#dc3545",
                    "#0dcaf0", "#6f42c1", "#e83e8c", "#fd7e14"
                ],
                "background": "#212529",
                "text": "#ffffff",
                "border": "#495057"
            },
            "corporate": {
                "name": "Corporate",
                "colors": {
                    "primary": "#2c3e50",
                    "secondary": "#95a5a6",
                    "success": "#27ae60",
                    "danger": "#e74c3c",
                    "warning": "#f39c12",
                    "info": "#3498db",
                    "light": "#ecf0f1",
                    "dark": "#34495e"
                },
                "chart_colors": [
                    "#2c3e50", "#27ae60", "#f39c12", "#e74c3c",
                    "#3498db", "#9b59b6", "#e91e63", "#ff5722"
                ],
                "background": "#ffffff",
                "text": "#2c3e50",
                "border": "#bdc3c7"
            }
        }
        self.current_theme = "default"
    
    def set_theme(self, theme_name: str) -> bool:
        """Set current dashboard theme."""
        if theme_name in self.themes:
            self.current_theme = theme_name
            return True
        return False
    
    def get_theme(self, theme_name: Optional[str] = None) -> Dict[str, Any]:
        """Get theme configuration."""
        theme_name = theme_name or self.current_theme
        return self.themes.get(theme_name, self.themes["default"])
    
    def get_available_themes(self) -> List[Dict[str, str]]:
        """Get list of available themes."""
        return [
            {"id": theme_id, "name": theme_config["name"]}
            for theme_id, theme_config in self.themes.items()
        ]
    
    def add_custom_theme(self, theme_id: str, theme_config: Dict[str, Any]) -> None:
        """Add custom theme configuration."""
        self.themes[theme_id] = theme_config
    
    def apply_theme_to_chart(self, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply current theme to chart configuration."""
        theme = self.get_theme()
        
        # Apply theme colors to chart
        if "data" in chart_data:
            if isinstance(chart_data["data"], list):
                for i, dataset in enumerate(chart_data["data"]):
                    if "marker" not in dataset:
                        dataset["marker"] = {}
                    
                    color_index = i % len(theme["chart_colors"])
                    dataset["marker"]["color"] = theme["chart_colors"][color_index]
        
        # Apply theme to layout
        if "layout" in chart_data:
            layout = chart_data["layout"]
            layout["paper_bgcolor"] = theme["background"]
            layout["plot_bgcolor"] = theme["background"]
            layout["font"] = {"color": theme["text"]}
        
        return chart_data


class DashboardExportManager:
    """Manager for dashboard export functionality."""
    
    def __init__(self):
        self.export_formats = ["pdf", "png", "html", "json"]
        self.export_options = {
            "pdf": {
                "page_size": "A4",
                "orientation": "landscape",
                "include_filters": True,
                "include_kpis": True
            },
            "png": {
                "width": 1920,
                "height": 1080,
                "dpi": 300,
                "background": "white"
            },
            "html": {
                "include_css": True,
                "include_js": True,
                "standalone": True
            },
            "json": {
                "include_data": True,
                "include_config": True,
                "pretty_print": True
            }
        }
    
    def export_dashboard(self, dashboard_data: Dict[str, Any], format_type: str, 
                        file_path: str, options: Optional[Dict[str, Any]] = None) -> bool:
        """Export dashboard to specified format."""
        if format_type not in self.export_formats:
            logger.error(f"Unsupported export format: {format_type}")
            return False
        
        export_options = self.export_options[format_type].copy()
        if options:
            export_options.update(options)
        
        try:
            if format_type == "json":
                return self._export_json(dashboard_data, file_path, export_options)
            elif format_type == "html":
                return self._export_html(dashboard_data, file_path, export_options)
            elif format_type == "pdf":
                return self._export_pdf(dashboard_data, file_path, export_options)
            elif format_type == "png":
                return self._export_png(dashboard_data, file_path, export_options)
            
        except Exception as e:
            logger.error(f"Error exporting dashboard to {format_type}: {e}")
            return False
        
        return False
    
    def _export_json(self, dashboard_data: Dict[str, Any], file_path: str, 
                    options: Dict[str, Any]) -> bool:
        """Export dashboard as JSON."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if options.get("pretty_print", True):
                    json.dump(dashboard_data, f, indent=2, default=str)
                else:
                    json.dump(dashboard_data, f, default=str)
            return True
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            return False
    
    def _export_html(self, dashboard_data: Dict[str, Any], file_path: str, 
                    options: Dict[str, Any]) -> bool:
        """Export dashboard as HTML."""
        # This would generate an HTML representation of the dashboard
        # For now, return a placeholder implementation
        html_content = self._generate_html_template(dashboard_data, options)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return True
        except Exception as e:
            logger.error(f"Error exporting HTML: {e}")
            return False
    
    def _export_pdf(self, dashboard_data: Dict[str, Any], file_path: str, 
                   options: Dict[str, Any]) -> bool:
        """Export dashboard as PDF."""
        # This would require a PDF generation library like reportlab or weasyprint
        # For now, return a placeholder implementation
        logger.info(f"PDF export to {file_path} - implementation pending")
        return True
    
    def _export_png(self, dashboard_data: Dict[str, Any], file_path: str, 
                   options: Dict[str, Any]) -> bool:
        """Export dashboard as PNG image."""
        # This would require image generation capabilities
        # For now, return a placeholder implementation
        logger.info(f"PNG export to {file_path} - implementation pending")
        return True
    
    def _generate_html_template(self, dashboard_data: Dict[str, Any], 
                              options: Dict[str, Any]) -> str:
        """Generate HTML template for dashboard."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard_data.get('title', 'Dashboard')}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; gap: 20px; }}
                .widget {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .kpi {{ text-align: center; }}
                .kpi-value {{ font-size: 2em; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>{dashboard_data.get('title', 'Dashboard')}</h1>
            <div class="dashboard">
                <!-- Dashboard content would be generated here -->
                <p>Dashboard exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        return self.export_formats.copy()
    
    def get_format_options(self, format_type: str) -> Dict[str, Any]:
        """Get available options for export format."""
        return self.export_options.get(format_type, {}).copy()