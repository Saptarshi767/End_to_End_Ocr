"""
Embed service for creating embeddable dashboard widgets

Provides functionality to create embeddable widgets that can be
integrated into external websites and applications.
"""

import json
import secrets
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..core.models import Dashboard, Chart, KPI


class WidgetType(Enum):
    """Types of embeddable widgets"""
    CHART = "chart"
    KPI = "kpi"
    TABLE = "table"
    DASHBOARD = "dashboard"
    FILTER = "filter"


class EmbedTheme(Enum):
    """Embed widget themes"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    CUSTOM = "custom"


@dataclass
class EmbedWidget:
    """Embeddable widget configuration"""
    widget_id: str
    dashboard_id: str
    widget_type: WidgetType
    source_id: str  # ID of chart, KPI, etc.
    title: str
    theme: EmbedTheme
    width: Optional[int] = None
    height: Optional[int] = None
    auto_refresh: bool = False
    refresh_interval: int = 300  # seconds
    show_title: bool = True
    show_legend: bool = True
    show_toolbar: bool = False
    custom_css: Optional[str] = None
    allowed_domains: List[str] = None
    created_at: Optional[str] = None


class EmbedService:
    """Service for creating and managing embeddable widgets"""
    
    def __init__(self, base_url: str = "https://app.example.com"):
        self.base_url = base_url.rstrip('/')
        
        # In-memory storage for demo (use database in production)
        self.embed_widgets: Dict[str, EmbedWidget] = {}
    
    def create_chart_embed(
        self,
        dashboard_id: str,
        chart_id: str,
        title: str = None,
        theme: EmbedTheme = EmbedTheme.LIGHT,
        width: int = 600,
        height: int = 400,
        customization: Dict[str, Any] = None
    ) -> EmbedWidget:
        """
        Create embeddable chart widget
        
        Args:
            dashboard_id: Dashboard containing the chart
            chart_id: Chart to embed
            title: Widget title
            theme: Visual theme
            width: Widget width in pixels
            height: Widget height in pixels
            customization: Custom styling options
            
        Returns:
            EmbedWidget configuration
        """
        
        widget_id = self._generate_widget_id()
        
        widget = EmbedWidget(
            widget_id=widget_id,
            dashboard_id=dashboard_id,
            widget_type=WidgetType.CHART,
            source_id=chart_id,
            title=title or "Chart Widget",
            theme=theme,
            width=width,
            height=height,
            **self._apply_customization(customization or {})
        )
        
        self.embed_widgets[widget_id] = widget
        
        return widget
    
    def create_kpi_embed(
        self,
        dashboard_id: str,
        kpi_id: str,
        title: str = None,
        theme: EmbedTheme = EmbedTheme.LIGHT,
        width: int = 300,
        height: int = 150,
        customization: Dict[str, Any] = None
    ) -> EmbedWidget:
        """
        Create embeddable KPI widget
        
        Args:
            dashboard_id: Dashboard containing the KPI
            kpi_id: KPI to embed
            title: Widget title
            theme: Visual theme
            width: Widget width in pixels
            height: Widget height in pixels
            customization: Custom styling options
            
        Returns:
            EmbedWidget configuration
        """
        
        widget_id = self._generate_widget_id()
        
        widget = EmbedWidget(
            widget_id=widget_id,
            dashboard_id=dashboard_id,
            widget_type=WidgetType.KPI,
            source_id=kpi_id,
            title=title or "KPI Widget",
            theme=theme,
            width=width,
            height=height,
            **self._apply_customization(customization or {})
        )
        
        self.embed_widgets[widget_id] = widget
        
        return widget
    
    def create_dashboard_embed(
        self,
        dashboard_id: str,
        title: str = None,
        theme: EmbedTheme = EmbedTheme.LIGHT,
        width: int = 1200,
        height: int = 800,
        customization: Dict[str, Any] = None
    ) -> EmbedWidget:
        """
        Create embeddable full dashboard widget
        
        Args:
            dashboard_id: Dashboard to embed
            title: Widget title
            theme: Visual theme
            width: Widget width in pixels
            height: Widget height in pixels
            customization: Custom styling options
            
        Returns:
            EmbedWidget configuration
        """
        
        widget_id = self._generate_widget_id()
        
        widget = EmbedWidget(
            widget_id=widget_id,
            dashboard_id=dashboard_id,
            widget_type=WidgetType.DASHBOARD,
            source_id=dashboard_id,
            title=title or "Dashboard Widget",
            theme=theme,
            width=width,
            height=height,
            **self._apply_customization(customization or {})
        )
        
        self.embed_widgets[widget_id] = widget
        
        return widget
    
    def generate_embed_code(
        self,
        widget_id: str,
        responsive: bool = True,
        lazy_load: bool = True
    ) -> str:
        """
        Generate HTML embed code for widget
        
        Args:
            widget_id: Widget ID to embed
            responsive: Make widget responsive
            lazy_load: Enable lazy loading
            
        Returns:
            HTML embed code
        """
        
        widget = self.embed_widgets.get(widget_id)
        if not widget:
            raise ValueError(f"Widget {widget_id} not found")
        
        # Generate iframe URL
        iframe_url = f"{self.base_url}/embed/{widget_id}"
        
        # Build iframe attributes
        iframe_attrs = [
            f'src="{iframe_url}"',
            f'title="{widget.title}"',
            'frameborder="0"',
            'allowtransparency="true"',
            'scrolling="no"'
        ]
        
        # Add dimensions
        if responsive:
            iframe_attrs.extend([
                'width="100%"',
                f'height="{widget.height or 400}"',
                'style="max-width: 100%; min-height: 300px;"'
            ])
        else:
            iframe_attrs.extend([
                f'width="{widget.width or 600}"',
                f'height="{widget.height or 400}"'
            ])
        
        # Add lazy loading
        if lazy_load:
            iframe_attrs.append('loading="lazy"')
        
        # Generate embed code
        embed_code = f'''
<!-- OCR Analytics Embed Widget -->
<div class="ocr-analytics-embed" data-widget-id="{widget_id}">
    <iframe {' '.join(iframe_attrs)}></iframe>
</div>
<script>
    // Auto-resize iframe if needed
    window.addEventListener('message', function(event) {{
        if (event.origin !== '{self.base_url}') return;
        if (event.data.type === 'resize' && event.data.widgetId === '{widget_id}') {{
            const iframe = document.querySelector('[data-widget-id="{widget_id}"] iframe');
            if (iframe) {{
                iframe.style.height = event.data.height + 'px';
            }}
        }}
    }});
</script>
'''.strip()
        
        return embed_code
    
    def generate_javascript_embed(
        self,
        widget_id: str,
        container_id: str = None
    ) -> str:
        """
        Generate JavaScript embed code for dynamic loading
        
        Args:
            widget_id: Widget ID to embed
            container_id: Container element ID
            
        Returns:
            JavaScript embed code
        """
        
        widget = self.embed_widgets.get(widget_id)
        if not widget:
            raise ValueError(f"Widget {widget_id} not found")
        
        container_id = container_id or f"ocr-widget-{widget_id}"
        
        js_code = f'''
<div id="{container_id}"></div>
<script>
(function() {{
    const widgetConfig = {{
        widgetId: '{widget_id}',
        apiUrl: '{self.base_url}/api/embed/{widget_id}',
        containerId: '{container_id}',
        theme: '{widget.theme.value}',
        autoRefresh: {str(widget.auto_refresh).lower()},
        refreshInterval: {widget.refresh_interval}
    }};
    
    // Load widget data and render
    function loadWidget() {{
        fetch(widgetConfig.apiUrl)
            .then(response => response.json())
            .then(data => {{
                renderWidget(data);
            }})
            .catch(error => {{
                console.error('Failed to load widget:', error);
                document.getElementById(widgetConfig.containerId).innerHTML = 
                    '<div style="padding: 20px; text-align: center; color: #666;">Failed to load widget</div>';
            }});
    }}
    
    function renderWidget(data) {{
        const container = document.getElementById(widgetConfig.containerId);
        
        // Basic rendering based on widget type
        if (data.type === 'chart') {{
            container.innerHTML = `
                <div style="width: {widget.width or 600}px; height: {widget.height or 400}px; border: 1px solid #ddd; border-radius: 4px; padding: 10px;">
                    <h3>${{data.title}}</h3>
                    <div id="chart-${{widgetConfig.widgetId}}"></div>
                </div>
            `;
            // Render chart using your preferred charting library
        }} else if (data.type === 'kpi') {{
            container.innerHTML = `
                <div style="width: {widget.width or 300}px; height: {widget.height or 150}px; border: 1px solid #ddd; border-radius: 4px; padding: 20px; text-align: center;">
                    <h4>${{data.title}}</h4>
                    <div style="font-size: 2em; font-weight: bold; color: #2196F3;">${{data.value}} ${{data.unit || ''}}</div>
                    ${{data.trend ? `<div style="color: ${{data.trend === 'up' ? '#4CAF50' : '#F44336'}};">${{data.change || ''}}</div>` : ''}}
                </div>
            `;
        }}
    }}
    
    // Auto-refresh if enabled
    if (widgetConfig.autoRefresh) {{
        setInterval(loadWidget, widgetConfig.refreshInterval * 1000);
    }}
    
    // Initial load
    loadWidget();
}})();
</script>
'''.strip()
        
        return js_code
    
    def get_widget_data(self, widget_id: str) -> Dict[str, Any]:
        """
        Get widget data for rendering
        
        Args:
            widget_id: Widget ID
            
        Returns:
            Widget data for rendering
        """
        
        widget = self.embed_widgets.get(widget_id)
        if not widget:
            raise ValueError(f"Widget {widget_id} not found")
        
        # Get dashboard data (implement based on your dashboard service)
        from ..core.services import DashboardService
        
        dashboard_service = DashboardService()
        dashboard = dashboard_service.get_dashboard(widget.dashboard_id)
        
        if not dashboard:
            raise ValueError(f"Dashboard {widget.dashboard_id} not found")
        
        # Extract widget-specific data
        if widget.widget_type == WidgetType.CHART:
            chart = next((c for c in dashboard.charts if c.id == widget.source_id), None)
            if chart:
                return {
                    "type": "chart",
                    "title": widget.title,
                    "chart_type": chart.chart_type,
                    "data": chart.data,
                    "options": chart.options,
                    "theme": widget.theme.value
                }
        
        elif widget.widget_type == WidgetType.KPI:
            kpi = next((k for k in dashboard.kpis if k.id == widget.source_id), None)
            if kpi:
                return {
                    "type": "kpi",
                    "title": widget.title,
                    "value": kpi.value,
                    "unit": kpi.unit,
                    "trend": getattr(kpi, 'trend', None),
                    "change": getattr(kpi, 'change', None),
                    "theme": widget.theme.value
                }
        
        elif widget.widget_type == WidgetType.DASHBOARD:
            return {
                "type": "dashboard",
                "title": widget.title,
                "charts": [chart.__dict__ for chart in dashboard.charts],
                "kpis": [kpi.__dict__ for kpi in dashboard.kpis],
                "layout": dashboard.layout.__dict__ if dashboard.layout else None,
                "theme": widget.theme.value
            }
        
        return {"type": "unknown", "title": widget.title}
    
    def update_widget(
        self,
        widget_id: str,
        title: str = None,
        theme: EmbedTheme = None,
        width: int = None,
        height: int = None,
        customization: Dict[str, Any] = None
    ) -> bool:
        """
        Update widget configuration
        
        Args:
            widget_id: Widget ID to update
            title: New title
            theme: New theme
            width: New width
            height: New height
            customization: New customization options
            
        Returns:
            True if updated successfully
        """
        
        widget = self.embed_widgets.get(widget_id)
        if not widget:
            return False
        
        if title is not None:
            widget.title = title
        
        if theme is not None:
            widget.theme = theme
        
        if width is not None:
            widget.width = width
        
        if height is not None:
            widget.height = height
        
        if customization:
            custom_attrs = self._apply_customization(customization)
            for key, value in custom_attrs.items():
                setattr(widget, key, value)
        
        return True
    
    def delete_widget(self, widget_id: str) -> bool:
        """
        Delete embed widget
        
        Args:
            widget_id: Widget ID to delete
            
        Returns:
            True if deleted successfully
        """
        
        if widget_id in self.embed_widgets:
            del self.embed_widgets[widget_id]
            return True
        
        return False
    
    def list_dashboard_widgets(self, dashboard_id: str) -> List[EmbedWidget]:
        """
        List all embed widgets for a dashboard
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            List of embed widgets
        """
        
        widgets = []
        for widget in self.embed_widgets.values():
            if widget.dashboard_id == dashboard_id:
                widgets.append(widget)
        
        return widgets
    
    def _generate_widget_id(self) -> str:
        """Generate unique widget ID"""
        return f"widget_{secrets.token_urlsafe(16)}"
    
    def _apply_customization(self, customization: Dict[str, Any]) -> Dict[str, Any]:
        """Apply customization options to widget"""
        
        result = {}
        
        # Map customization options to widget attributes
        if 'auto_refresh' in customization:
            result['auto_refresh'] = bool(customization['auto_refresh'])
        
        if 'refresh_interval' in customization:
            result['refresh_interval'] = max(30, int(customization['refresh_interval']))
        
        if 'show_title' in customization:
            result['show_title'] = bool(customization['show_title'])
        
        if 'show_legend' in customization:
            result['show_legend'] = bool(customization['show_legend'])
        
        if 'show_toolbar' in customization:
            result['show_toolbar'] = bool(customization['show_toolbar'])
        
        if 'custom_css' in customization:
            result['custom_css'] = str(customization['custom_css'])
        
        if 'allowed_domains' in customization:
            result['allowed_domains'] = list(customization['allowed_domains'])
        
        return result