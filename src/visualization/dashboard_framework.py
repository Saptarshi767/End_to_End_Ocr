"""
Interactive dashboard framework for OCR table analytics.

This module provides the core dashboard framework with layout management,
interactive filters, real-time updates, and user interaction capabilities.
"""

from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import json
import logging
from abc import ABC, abstractmethod

from ..core.models import (
    Dashboard, Chart, Filter, KPI, DashboardLayout, ChartConfig, ChartType
)
from ..core.interfaces import DashboardGeneratorInterface
from .chart_selector import ChartTypeSelector
from .chart_engines import ChartEngineFactory
from .kpi_engine import EnhancedKPIEngine
from .kpi_widgets import KPIWidgetManager, KPIWidgetConfig, KPIWidgetData, KPIWidgetType, KPIDisplayFormat


logger = logging.getLogger(__name__)


@dataclass
class FilterState:
    """Current state of a dashboard filter."""
    column: str
    filter_type: str
    active_values: List[Any] = field(default_factory=list)
    is_active: bool = False
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DashboardState:
    """Current state of the dashboard."""
    filters: Dict[str, FilterState] = field(default_factory=dict)
    selected_charts: Set[str] = field(default_factory=set)
    refresh_interval: Optional[int] = None  # seconds
    last_refresh: datetime = field(default_factory=datetime.now)
    is_auto_refresh: bool = False


class DashboardEventType:
    """Dashboard event types for real-time updates."""
    FILTER_CHANGED = "filter_changed"
    CHART_SELECTED = "chart_selected"
    DATA_UPDATED = "data_updated"
    LAYOUT_CHANGED = "layout_changed"
    REFRESH_REQUESTED = "refresh_requested"


@dataclass
class DashboardEvent:
    """Dashboard event for real-time updates."""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "dashboard"


class DashboardEventHandler(ABC):
    """Abstract base class for dashboard event handlers."""
    
    @abstractmethod
    def handle_event(self, event: DashboardEvent) -> None:
        """Handle dashboard event."""
        pass


class FilterManager:
    """Manages dashboard filters and their interactions."""
    
    def __init__(self):
        self.filters: Dict[str, Filter] = {}
        self.filter_states: Dict[str, FilterState] = {}
        self.event_handlers: List[DashboardEventHandler] = []
    
    def add_filter(self, filter_config: Filter) -> None:
        """Add a new filter to the dashboard."""
        self.filters[filter_config.column] = filter_config
        self.filter_states[filter_config.column] = FilterState(
            column=filter_config.column,
            filter_type=filter_config.filter_type,
            active_values=[filter_config.default_value] if filter_config.default_value else []
        )
        logger.info(f"Added filter for column: {filter_config.column}")
    
    def update_filter(self, column: str, values: List[Any]) -> None:
        """Update filter values and trigger events."""
        if column not in self.filter_states:
            logger.warning(f"Filter not found for column: {column}")
            return
        
        old_values = self.filter_states[column].active_values.copy()
        self.filter_states[column].active_values = values
        self.filter_states[column].is_active = len(values) > 0
        self.filter_states[column].last_updated = datetime.now()
        
        # Trigger filter change event
        event = DashboardEvent(
            event_type=DashboardEventType.FILTER_CHANGED,
            data={
                "column": column,
                "old_values": old_values,
                "new_values": values,
                "filter_type": self.filter_states[column].filter_type
            }
        )
        self._trigger_event(event)
    
    def clear_filter(self, column: str) -> None:
        """Clear a specific filter."""
        if column in self.filter_states:
            self.update_filter(column, [])
    
    def clear_all_filters(self) -> None:
        """Clear all active filters."""
        for column in self.filter_states:
            self.clear_filter(column)
    
    def get_active_filters(self) -> Dict[str, FilterState]:
        """Get all currently active filters."""
        return {
            col: state for col, state in self.filter_states.items()
            if state.is_active
        }
    
    def apply_filters(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply all active filters to a dataframe."""
        filtered_df = dataframe.copy()
        
        for column, state in self.get_active_filters().items():
            if column not in filtered_df.columns:
                continue
            
            if state.filter_type == "select":
                filtered_df = filtered_df[filtered_df[column].isin(state.active_values)]
            elif state.filter_type == "range":
                if len(state.active_values) >= 2:
                    min_val, max_val = state.active_values[0], state.active_values[1]
                    filtered_df = filtered_df[
                        (filtered_df[column] >= min_val) & 
                        (filtered_df[column] <= max_val)
                    ]
            elif state.filter_type == "search":
                if state.active_values:
                    search_term = str(state.active_values[0]).lower()
                    filtered_df = filtered_df[
                        filtered_df[column].astype(str).str.lower().str.contains(search_term, na=False)
                    ]
        
        return filtered_df
    
    def add_event_handler(self, handler: DashboardEventHandler) -> None:
        """Add event handler for filter changes."""
        self.event_handlers.append(handler)
    
    def _trigger_event(self, event: DashboardEvent) -> None:
        """Trigger event to all registered handlers."""
        for handler in self.event_handlers:
            try:
                handler.handle_event(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")


class LayoutManager:
    """Manages dashboard layout and chart positioning."""
    
    def __init__(self, grid_columns: int = 12):
        self.grid_columns = grid_columns
        self.chart_positions: Dict[str, Dict[str, int]] = {}
        self.responsive = True
    
    def add_chart_position(self, chart_id: str, x: int, y: int, width: int, height: int) -> None:
        """Add or update chart position in the layout."""
        self.chart_positions[chart_id] = {
            "x": x,
            "y": y,
            "width": min(width, self.grid_columns),
            "height": height
        }
    
    def auto_layout_charts(self, chart_ids: List[str]) -> None:
        """Automatically layout charts in a responsive grid."""
        charts_per_row = min(3, len(chart_ids))  # Max 3 charts per row
        chart_width = self.grid_columns // charts_per_row
        
        for i, chart_id in enumerate(chart_ids):
            row = i // charts_per_row
            col = i % charts_per_row
            
            self.add_chart_position(
                chart_id=chart_id,
                x=col * chart_width,
                y=row * 4,  # 4 units height per row
                width=chart_width,
                height=4
            )
    
    def get_layout_config(self) -> DashboardLayout:
        """Get current layout configuration."""
        return DashboardLayout(
            grid_columns=self.grid_columns,
            chart_positions=self.chart_positions.copy(),
            responsive=self.responsive
        )
    
    def update_layout(self, layout: DashboardLayout) -> None:
        """Update layout configuration."""
        self.grid_columns = layout.grid_columns
        self.chart_positions = layout.chart_positions.copy()
        self.responsive = layout.responsive


class RefreshManager:
    """Manages real-time data updates and refresh capabilities."""
    
    def __init__(self):
        self.refresh_callbacks: List[Callable[[], pd.DataFrame]] = []
        self.auto_refresh_interval: Optional[int] = None
        self.is_auto_refresh_active: bool = False
        self.last_refresh: datetime = datetime.now()
        self.event_handlers: List[DashboardEventHandler] = []
    
    def add_data_source(self, callback: Callable[[], pd.DataFrame]) -> None:
        """Add a data source callback for refresh operations."""
        self.refresh_callbacks.append(callback)
    
    def set_auto_refresh(self, interval_seconds: int) -> None:
        """Enable auto-refresh with specified interval."""
        self.auto_refresh_interval = interval_seconds
        self.is_auto_refresh_active = True
        logger.info(f"Auto-refresh enabled with {interval_seconds}s interval")
    
    def disable_auto_refresh(self) -> None:
        """Disable auto-refresh."""
        self.is_auto_refresh_active = False
        self.auto_refresh_interval = None
        logger.info("Auto-refresh disabled")
    
    def manual_refresh(self) -> List[pd.DataFrame]:
        """Manually trigger data refresh."""
        refreshed_data = []
        
        for callback in self.refresh_callbacks:
            try:
                data = callback()
                refreshed_data.append(data)
            except Exception as e:
                logger.error(f"Error refreshing data source: {e}")
        
        self.last_refresh = datetime.now()
        
        # Trigger refresh event
        event = DashboardEvent(
            event_type=DashboardEventType.DATA_UPDATED,
            data={
                "refresh_type": "manual",
                "data_sources_count": len(refreshed_data),
                "timestamp": self.last_refresh
            }
        )
        self._trigger_event(event)
        
        return refreshed_data
    
    def should_auto_refresh(self) -> bool:
        """Check if auto-refresh should be triggered."""
        if not self.is_auto_refresh_active or not self.auto_refresh_interval:
            return False
        
        time_since_refresh = (datetime.now() - self.last_refresh).total_seconds()
        return time_since_refresh >= self.auto_refresh_interval
    
    def add_event_handler(self, handler: DashboardEventHandler) -> None:
        """Add event handler for refresh events."""
        self.event_handlers.append(handler)
    
    def _trigger_event(self, event: DashboardEvent) -> None:
        """Trigger event to all registered handlers."""
        for handler in self.event_handlers:
            try:
                handler.handle_event(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")


class InteractiveDashboard(DashboardGeneratorInterface):
    """
    Interactive dashboard framework with real-time updates and user controls.
    
    Provides comprehensive dashboard functionality including layout management,
    interactive filters, real-time data updates, and event handling.
    """
    
    def __init__(self, chart_engine: str = "plotly"):
        self.chart_selector = ChartTypeSelector()
        self.chart_engine = ChartEngineFactory.create_engine(chart_engine)
        self.filter_manager = FilterManager()
        self.layout_manager = LayoutManager()
        self.refresh_manager = RefreshManager()
        self.dashboard_state = DashboardState()
        self.current_data: Optional[pd.DataFrame] = None
        self.event_handlers: List[DashboardEventHandler] = []
        
        # Enhanced KPI functionality
        self.kpi_engine = EnhancedKPIEngine()
        self.kpi_widget_manager = KPIWidgetManager()
    
    def generate_dashboard(self, dataframe: pd.DataFrame) -> Dashboard:
        """Generate interactive dashboard from structured data."""
        logger.info(f"Generating dashboard for data with shape {dataframe.shape}")
        
        self.current_data = dataframe.copy()
        
        # Generate charts
        charts = self._generate_charts(dataframe)
        
        # Generate filters
        filters = self.create_filters(dataframe)
        
        # Generate KPIs
        kpis = self._generate_kpis(dataframe)
        
        # Setup layout
        chart_ids = [chart.id for chart in charts]
        self.layout_manager.auto_layout_charts(chart_ids)
        layout = self.layout_manager.get_layout_config()
        
        # Create dashboard
        dashboard = Dashboard(
            title=f"Data Analysis Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            charts=charts,
            filters=filters,
            kpis=kpis,
            layout=layout
        )
        
        # Setup filters in filter manager
        for filter_config in filters:
            self.filter_manager.add_filter(filter_config)
        
        logger.info(f"Generated dashboard with {len(charts)} charts, {len(filters)} filters, {len(kpis)} KPIs")
        return dashboard
    
    def auto_select_visualizations(self, dataframe: pd.DataFrame) -> List[ChartConfig]:
        """Automatically select appropriate chart types."""
        return self.chart_selector.get_chart_recommendations(dataframe, max_recommendations=5)
    
    def create_filters(self, dataframe: pd.DataFrame) -> List[Filter]:
        """Create interactive filters based on data columns."""
        filters = []
        
        for column in dataframe.columns:
            filter_config = self._create_column_filter(dataframe, column)
            if filter_config:
                filters.append(filter_config)
        
        return filters
    
    def _create_column_filter(self, dataframe: pd.DataFrame, column: str) -> Optional[Filter]:
        """Create filter configuration for a specific column."""
        series = dataframe[column]
        unique_count = series.nunique()
        
        # Skip columns with too many unique values or mostly null
        if unique_count > 50 or series.isnull().sum() / len(series) > 0.8:
            return None
        
        # Determine filter type based on data characteristics
        if pd.api.types.is_numeric_dtype(series):
            if unique_count <= 10:
                # Categorical numeric filter
                return Filter(
                    column=column,
                    filter_type="select",
                    values=sorted(series.dropna().unique().tolist()),
                    default_value=None
                )
            else:
                # Range filter for continuous numeric data
                min_val, max_val = series.min(), series.max()
                return Filter(
                    column=column,
                    filter_type="range",
                    values=[min_val, max_val],
                    default_value=None
                )
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Date range filter
            min_date, max_date = series.min(), series.max()
            return Filter(
                column=column,
                filter_type="range",
                values=[min_date, max_date],
                default_value=None
            )
        
        else:
            # Categorical filter
            if unique_count <= 20:
                return Filter(
                    column=column,
                    filter_type="select",
                    values=series.value_counts().index.tolist()[:20],
                    default_value=None
                )
            else:
                # Search filter for high-cardinality text
                return Filter(
                    column=column,
                    filter_type="search",
                    values=[],
                    default_value=""
                )
    
    def _generate_charts(self, dataframe: pd.DataFrame) -> List[Chart]:
        """Generate charts for the dashboard."""
        chart_configs = self.auto_select_visualizations(dataframe)
        charts = []
        
        for config in chart_configs:
            try:
                chart = self.chart_engine.create_chart(config, dataframe)
                charts.append(chart)
            except Exception as e:
                logger.error(f"Error creating chart {config.title}: {e}")
        
        return charts
    
    def _generate_kpis(self, dataframe: pd.DataFrame) -> List[KPI]:
        """Generate KPIs for the dashboard using enhanced KPI engine."""
        try:
            # Auto-detect and calculate KPIs
            detected_kpis = self.kpi_engine.auto_detect_kpis(dataframe, max_kpis=8)
            calculated_kpis = self.kpi_engine.calculate_kpis(dataframe)
            
            # Generate KPI widgets
            self._create_kpi_widgets(calculated_kpis, dataframe)
            
            logger.info(f"Generated {len(calculated_kpis)} KPIs using enhanced engine")
            return calculated_kpis
            
        except Exception as e:
            logger.error(f"Error generating enhanced KPIs: {e}")
            # Fallback to basic KPIs
            return self._generate_basic_kpis(dataframe)
    
    def _generate_basic_kpis(self, dataframe: pd.DataFrame) -> List[KPI]:
        """Generate basic KPIs as fallback."""
        kpis = []
        
        # Basic KPIs
        kpis.append(KPI(
            name="Total Records",
            value=len(dataframe),
            format_type="number",
            description="Total number of records in the dataset"
        ))
        
        kpis.append(KPI(
            name="Columns",
            value=len(dataframe.columns),
            format_type="number",
            description="Number of columns in the dataset"
        ))
        
        # Numeric column KPIs
        numeric_columns = dataframe.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            for col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                try:
                    mean_val = dataframe[col].mean()
                    kpis.append(KPI(
                        name=f"Avg {col}",
                        value=round(mean_val, 2),
                        format_type="number",
                        description=f"Average value of {col}"
                    ))
                except Exception:
                    continue
        
        return kpis
    
    def _create_kpi_widgets(self, kpis: List[KPI], dataframe: pd.DataFrame) -> None:
        """Create KPI widgets for the dashboard."""
        # Clear existing widgets
        self.kpi_widget_manager = KPIWidgetManager()
        
        # Get KPI summary with trends and comparisons
        kpi_summary = self.kpi_engine.get_kpi_summary(dataframe)
        
        for i, kpi in enumerate(kpis):
            # Determine widget type based on KPI characteristics
            widget_type = self._determine_widget_type(kpi, i)
            
            # Create widget configuration
            config = KPIWidgetConfig(
                widget_id=f"kpi_widget_{i}",
                widget_type=widget_type,
                title=kpi.name,
                display_format=self._get_display_format(kpi.format_type),
                show_trend=True,
                show_comparison=True,
                show_target=False,
                size="medium",
                position={"x": (i % 4) * 3, "y": 0}  # 4 widgets per row
            )
            
            # Find corresponding trend and comparison data
            trend = None
            comparison = None
            insights = []
            
            for comp in kpi_summary.get("comparisons", []):
                if comp.get("kpi_name") == kpi.name:
                    from .kpi_engine import KPIComparison
                    comparison = KPIComparison(
                        kpi_name=comp["kpi_name"],
                        current_value=comp["current_value"],
                        comparison_value=comp["comparison_value"],
                        difference=comp["difference"],
                        percentage_difference=comp["percentage_difference"],
                        is_improvement=comp["is_improvement"],
                        comparison_type=comp["comparison_type"]
                    )
                    break
            
            for insight_data in kpi_summary.get("insights", []):
                if insight_data.get("kpi_name") == kpi.name:
                    from .kpi_engine import KPIInsight
                    insight = KPIInsight(
                        kpi_name=insight_data["kpi_name"],
                        insight_type=insight_data["insight_type"],
                        message=insight_data["message"],
                        severity=insight_data["severity"],
                        confidence=insight_data["confidence"],
                        supporting_data=insight_data.get("supporting_data", {})
                    )
                    insights.append(insight)
            
            # Create widget data
            widget_data = KPIWidgetData(
                kpi=kpi,
                trend=trend,
                comparison=comparison,
                insights=insights,
                historical_data=[],  # Would be populated with actual historical data
                target_value=None
            )
            
            # Add widget to manager
            self.kpi_widget_manager.add_widget(config, widget_data)
    
    def _determine_widget_type(self, kpi: KPI, index: int) -> KPIWidgetType:
        """Determine appropriate widget type for KPI."""
        # Use different widget types for variety
        widget_types = [
            KPIWidgetType.CARD,
            KPIWidgetType.SIMPLE_VALUE,
            KPIWidgetType.GAUGE,
            KPIWidgetType.PROGRESS_BAR
        ]
        
        # Business metrics get more prominent widgets
        if any(keyword in kpi.name.lower() for keyword in ['revenue', 'profit', 'sales', 'total']):
            return KPIWidgetType.CARD
        
        # Use cycling pattern for others
        return widget_types[index % len(widget_types)]
    
    def _get_display_format(self, format_type: str) -> KPIDisplayFormat:
        """Convert KPI format type to widget display format."""
        format_map = {
            "currency": KPIDisplayFormat.CURRENCY,
            "percentage": KPIDisplayFormat.PERCENTAGE,
            "number": KPIDisplayFormat.NUMBER,
            "decimal": KPIDisplayFormat.DECIMAL
        }
        return format_map.get(format_type, KPIDisplayFormat.NUMBER)
    
    def update_filter(self, column: str, values: List[Any]) -> Dashboard:
        """Update filter and regenerate affected charts."""
        self.filter_manager.update_filter(column, values)
        
        if self.current_data is not None:
            # Apply filters to data
            filtered_data = self.filter_manager.apply_filters(self.current_data)
            
            # Regenerate dashboard with filtered data
            return self.generate_dashboard(filtered_data)
        
        return self.generate_dashboard(pd.DataFrame())
    
    def refresh_data(self, new_data: pd.DataFrame) -> Dashboard:
        """Refresh dashboard with new data."""
        self.current_data = new_data.copy()
        
        # Apply current filters to new data
        if self.filter_manager.get_active_filters():
            filtered_data = self.filter_manager.apply_filters(new_data)
        else:
            filtered_data = new_data
        
        # Regenerate dashboard
        dashboard = self.generate_dashboard(filtered_data)
        
        # Trigger refresh event
        event = DashboardEvent(
            event_type=DashboardEventType.DATA_UPDATED,
            data={
                "refresh_type": "data_update",
                "new_shape": new_data.shape,
                "filtered_shape": filtered_data.shape
            }
        )
        self._trigger_event(event)
        
        return dashboard
    
    def set_auto_refresh(self, interval_seconds: int, data_callback: Callable[[], pd.DataFrame]) -> None:
        """Enable auto-refresh with data callback."""
        self.refresh_manager.add_data_source(data_callback)
        self.refresh_manager.set_auto_refresh(interval_seconds)
        self.dashboard_state.refresh_interval = interval_seconds
        self.dashboard_state.is_auto_refresh = True
    
    def disable_auto_refresh(self) -> None:
        """Disable auto-refresh."""
        self.refresh_manager.disable_auto_refresh()
        self.dashboard_state.is_auto_refresh = False
        self.dashboard_state.refresh_interval = None
    
    def get_dashboard_state(self) -> DashboardState:
        """Get current dashboard state."""
        return self.dashboard_state
    
    def add_event_handler(self, handler: DashboardEventHandler) -> None:
        """Add event handler for dashboard events."""
        self.event_handlers.append(handler)
        self.filter_manager.add_event_handler(handler)
        self.refresh_manager.add_event_handler(handler)
    
    def _trigger_event(self, event: DashboardEvent) -> None:
        """Trigger event to all registered handlers."""
        for handler in self.event_handlers:
            try:
                handler.handle_event(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def get_kpi_widgets(self) -> Dict[str, Any]:
        """Get rendered KPI widgets."""
        return self.kpi_widget_manager.render_dashboard()
    
    def update_kpi_target(self, kpi_name: str, target_value: float) -> bool:
        """Update target value for a KPI."""
        try:
            # Find KPI ID by name
            kpi_id = None
            for kid, kpi_def in self.kpi_engine.kpi_definitions.items():
                if kpi_def.name == kpi_name:
                    kpi_id = kid
                    break
            
            if kpi_id:
                self.kpi_engine.set_kpi_target(kpi_id, target_value)
                
                # Update widget if it exists
                for widget_id, widget in self.kpi_widget_manager.widgets.items():
                    if widget.data.kpi.name == kpi_name:
                        widget.data.target_value = target_value
                        break
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating KPI target: {e}")
            return False
    
    def get_kpi_insights(self) -> List[Dict[str, Any]]:
        """Get KPI insights and alerts."""
        if self.current_data is None:
            return []
        
        try:
            kpi_summary = self.kpi_engine.get_kpi_summary(self.current_data)
            return kpi_summary.get("insights", [])
        except Exception as e:
            logger.error(f"Error getting KPI insights: {e}")
            return []
    
    def get_kpi_alerts(self) -> List[Dict[str, Any]]:
        """Get KPI alerts from widgets."""
        return self.kpi_widget_manager.get_widget_alerts()
    
    def export_kpi_configuration(self) -> Dict[str, Any]:
        """Export KPI widget configuration."""
        return self.kpi_widget_manager.export_configuration()


class DashboardController:
    """
    Controller for managing dashboard interactions and state.
    
    Provides high-level interface for dashboard operations and coordinates
    between different dashboard components.
    """
    
    def __init__(self, dashboard_framework: InteractiveDashboard):
        self.dashboard = dashboard_framework
        self.current_dashboard_config: Optional[Dashboard] = None
        self.interaction_history: List[DashboardEvent] = []
    
    def initialize_dashboard(self, dataframe: pd.DataFrame) -> Dashboard:
        """Initialize dashboard with data."""
        self.current_dashboard_config = self.dashboard.generate_dashboard(dataframe)
        return self.current_dashboard_config
    
    def apply_filter(self, column: str, values: List[Any]) -> Dashboard:
        """Apply filter and update dashboard."""
        updated_dashboard = self.dashboard.update_filter(column, values)
        self.current_dashboard_config = updated_dashboard
        
        # Record interaction
        event = DashboardEvent(
            event_type=DashboardEventType.FILTER_CHANGED,
            data={"column": column, "values": values}
        )
        self.interaction_history.append(event)
        
        return updated_dashboard
    
    def clear_filters(self) -> Dashboard:
        """Clear all filters and refresh dashboard."""
        self.dashboard.filter_manager.clear_all_filters()
        
        if self.dashboard.current_data is not None:
            updated_dashboard = self.dashboard.generate_dashboard(self.dashboard.current_data)
            self.current_dashboard_config = updated_dashboard
            return updated_dashboard
        
        return self.current_dashboard_config or Dashboard()
    
    def refresh_dashboard(self, new_data: Optional[pd.DataFrame] = None) -> Dashboard:
        """Refresh dashboard with optional new data."""
        if new_data is not None:
            updated_dashboard = self.dashboard.refresh_data(new_data)
        else:
            # Manual refresh with existing data
            refreshed_data = self.dashboard.refresh_manager.manual_refresh()
            if refreshed_data and len(refreshed_data) > 0:
                updated_dashboard = self.dashboard.refresh_data(refreshed_data[0])
            else:
                updated_dashboard = self.current_dashboard_config or Dashboard()
        
        self.current_dashboard_config = updated_dashboard
        return updated_dashboard
    
    def get_dashboard_json(self) -> str:
        """Get dashboard configuration as JSON."""
        if not self.current_dashboard_config:
            return "{}"
        
        # Convert dashboard to JSON-serializable format
        dashboard_dict = {
            "id": self.current_dashboard_config.id,
            "title": self.current_dashboard_config.title,
            "charts": [
                {
                    "id": chart.id,
                    "config": {
                        "chart_type": chart.config.chart_type.value,
                        "title": chart.config.title,
                        "x_column": chart.config.x_column,
                        "y_column": chart.config.y_column,
                        "aggregation": chart.config.aggregation
                    },
                    "data": chart.data
                }
                for chart in self.current_dashboard_config.charts
            ],
            "filters": [
                {
                    "column": f.column,
                    "filter_type": f.filter_type,
                    "values": f.values,
                    "default_value": f.default_value
                }
                for f in self.current_dashboard_config.filters
            ],
            "kpis": [
                {
                    "name": kpi.name,
                    "value": kpi.value,
                    "format_type": kpi.format_type,
                    "description": kpi.description
                }
                for kpi in self.current_dashboard_config.kpis
            ],
            "layout": {
                "grid_columns": self.current_dashboard_config.layout.grid_columns,
                "chart_positions": self.current_dashboard_config.layout.chart_positions,
                "responsive": self.current_dashboard_config.layout.responsive
            }
        }
        
        return json.dumps(dashboard_dict, indent=2, default=str)
    
    def get_interaction_history(self) -> List[DashboardEvent]:
        """Get history of dashboard interactions."""
        return self.interaction_history.copy()