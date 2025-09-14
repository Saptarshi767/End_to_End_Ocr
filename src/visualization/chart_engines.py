"""
Chart generation engines for multiple visualization libraries.

This module provides implementations for different chart libraries including
Plotly and Chart.js, with a unified interface for chart generation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import json
from abc import ABC, abstractmethod
import logging

from ..core.models import Chart, ChartConfig, ChartType
from ..core.interfaces import ChartEngineInterface


logger = logging.getLogger(__name__)


class BaseChartEngine(ChartEngineInterface):
    """Base class for chart generation engines."""
    
    def __init__(self, library_name: str):
        self.library_name = library_name
        self.supported_types = [
            ChartType.BAR, ChartType.LINE, ChartType.PIE, 
            ChartType.SCATTER, ChartType.HISTOGRAM, ChartType.HEATMAP, ChartType.TABLE
        ]
    
    def get_supported_chart_types(self) -> List[str]:
        """Get list of supported chart types."""
        return [chart_type.value for chart_type in self.supported_types]
    
    def auto_select_chart_type(self, data: pd.DataFrame, columns: List[str]) -> str:
        """Automatically select appropriate chart type for data."""
        # This is a simple implementation - the ChartTypeSelector provides more sophisticated logic
        if len(columns) == 1:
            if data[columns[0]].dtype in ['object', 'category']:
                return ChartType.BAR.value
            else:
                return ChartType.HISTOGRAM.value
        elif len(columns) == 2:
            if data[columns[0]].dtype in ['object', 'category']:
                return ChartType.BAR.value
            else:
                return ChartType.SCATTER.value
        else:
            return ChartType.TABLE.value


class PlotlyChartEngine(BaseChartEngine):
    """Chart engine using Plotly library."""
    
    def __init__(self):
        super().__init__("plotly")
    
    def create_chart(self, config: ChartConfig, data: pd.DataFrame) -> Chart:
        """Create chart using Plotly library."""
        try:
            chart_data = self._generate_plotly_chart(config, data)
            
            return Chart(
                config=config,
                data=chart_data
            )
        except Exception as e:
            logger.error(f"Error creating Plotly chart: {e}")
            # Return a fallback table chart
            return self._create_fallback_table(config, data)
    
    def _generate_plotly_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate Plotly chart configuration."""
        
        if config.chart_type == ChartType.BAR:
            return self._create_bar_chart(config, data)
        elif config.chart_type == ChartType.LINE:
            return self._create_line_chart(config, data)
        elif config.chart_type == ChartType.PIE:
            return self._create_pie_chart(config, data)
        elif config.chart_type == ChartType.SCATTER:
            return self._create_scatter_chart(config, data)
        elif config.chart_type == ChartType.HISTOGRAM:
            return self._create_histogram_chart(config, data)
        elif config.chart_type == ChartType.HEATMAP:
            return self._create_heatmap_chart(config, data)
        else:  # TABLE
            return self._create_table_chart(config, data)
    
    def _create_bar_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Plotly bar chart."""
        if config.y_column:
            # Grouped bar chart
            grouped_data = data.groupby(config.x_column)[config.y_column].agg(config.aggregation).reset_index()
            x_values = grouped_data[config.x_column].tolist()
            y_values = grouped_data[config.y_column].tolist()
        else:
            # Count bar chart
            value_counts = data[config.x_column].value_counts()
            x_values = value_counts.index.tolist()
            y_values = value_counts.values.tolist()
        
        return {
            "data": [{
                "type": "bar",
                "x": x_values,
                "y": y_values,
                "name": config.y_column or "Count"
            }],
            "layout": {
                "title": config.title,
                "xaxis": {"title": config.x_column},
                "yaxis": {"title": config.y_column or "Count"},
                "showlegend": False
            }
        }
    
    def _create_line_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Plotly line chart."""
        if config.y_column:
            # Sort by x column for proper line connection
            sorted_data = data.sort_values(config.x_column)
            x_values = sorted_data[config.x_column].tolist()
            y_values = sorted_data[config.y_column].tolist()
        else:
            # Time series count
            grouped_data = data.groupby(config.x_column).size().reset_index(name='count')
            grouped_data = grouped_data.sort_values(config.x_column)
            x_values = grouped_data[config.x_column].tolist()
            y_values = grouped_data['count'].tolist()
        
        return {
            "data": [{
                "type": "scatter",
                "mode": "lines+markers",
                "x": x_values,
                "y": y_values,
                "name": config.y_column or "Count"
            }],
            "layout": {
                "title": config.title,
                "xaxis": {"title": config.x_column},
                "yaxis": {"title": config.y_column or "Count"}
            }
        }
    
    def _create_pie_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Plotly pie chart."""
        value_counts = data[config.x_column].value_counts()
        
        return {
            "data": [{
                "type": "pie",
                "labels": value_counts.index.tolist(),
                "values": value_counts.values.tolist(),
                "textinfo": "label+percent"
            }],
            "layout": {
                "title": config.title,
                "showlegend": True
            }
        }
    
    def _create_scatter_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Plotly scatter chart."""
        scatter_data = {
            "type": "scatter",
            "mode": "markers",
            "x": data[config.x_column].tolist(),
            "y": data[config.y_column].tolist() if config.y_column else [],
            "name": f"{config.y_column} vs {config.x_column}"
        }
        
        # Add color coding if specified
        if config.color_column:
            scatter_data["marker"] = {
                "color": data[config.color_column].tolist(),
                "colorscale": "Viridis",
                "showscale": True
            }
        
        return {
            "data": [scatter_data],
            "layout": {
                "title": config.title,
                "xaxis": {"title": config.x_column},
                "yaxis": {"title": config.y_column or "Value"}
            }
        }
    
    def _create_histogram_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Plotly histogram chart."""
        bins = config.options.get("bins", 20)
        
        return {
            "data": [{
                "type": "histogram",
                "x": data[config.x_column].tolist(),
                "nbinsx": bins,
                "name": config.x_column
            }],
            "layout": {
                "title": config.title,
                "xaxis": {"title": config.x_column},
                "yaxis": {"title": "Frequency"},
                "showlegend": False
            }
        }
    
    def _create_heatmap_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Plotly heatmap chart."""
        # Create pivot table for heatmap
        if config.y_column:
            pivot_data = data.pivot_table(
                values=config.y_column if config.aggregation != "count" else None,
                index=config.x_column,
                columns=config.y_column if config.aggregation == "count" else config.color_column,
                aggfunc=config.aggregation if config.aggregation != "count" else "size",
                fill_value=0
            )
        else:
            # Simple frequency heatmap
            pivot_data = pd.crosstab(data[config.x_column], data.index // 10)  # Group by row chunks
        
        return {
            "data": [{
                "type": "heatmap",
                "z": pivot_data.values.tolist(),
                "x": pivot_data.columns.tolist(),
                "y": pivot_data.index.tolist(),
                "colorscale": "Viridis"
            }],
            "layout": {
                "title": config.title,
                "xaxis": {"title": config.y_column or "Categories"},
                "yaxis": {"title": config.x_column}
            }
        }
    
    def _create_table_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Plotly table chart."""
        # Limit to first 100 rows for performance
        display_data = data.head(100)
        
        return {
            "data": [{
                "type": "table",
                "header": {
                    "values": display_data.columns.tolist(),
                    "fill": {"color": "#C2D4FF"},
                    "align": "left"
                },
                "cells": {
                    "values": [display_data[col].tolist() for col in display_data.columns],
                    "fill": {"color": "#F5F8FF"},
                    "align": "left"
                }
            }],
            "layout": {
                "title": config.title,
                "margin": {"l": 0, "r": 0, "t": 30, "b": 0}
            }
        }
    
    def _create_fallback_table(self, config: ChartConfig, data: pd.DataFrame) -> Chart:
        """Create fallback table chart when other chart types fail."""
        table_config = ChartConfig(
            chart_type=ChartType.TABLE,
            title=f"Data Table - {config.title}",
            x_column=config.x_column,
            aggregation="none"
        )
        
        table_data = self._create_table_chart(table_config, data)
        
        return Chart(
            config=table_config,
            data=table_data
        )


class ChartJSEngine(BaseChartEngine):
    """Chart engine using Chart.js library."""
    
    def __init__(self):
        super().__init__("chartjs")
    
    def create_chart(self, config: ChartConfig, data: pd.DataFrame) -> Chart:
        """Create chart using Chart.js library."""
        try:
            chart_data = self._generate_chartjs_chart(config, data)
            
            return Chart(
                config=config,
                data=chart_data
            )
        except Exception as e:
            logger.error(f"Error creating Chart.js chart: {e}")
            # Return a fallback table chart
            return self._create_fallback_table(config, data)
    
    def _generate_chartjs_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate Chart.js chart configuration."""
        
        if config.chart_type == ChartType.BAR:
            return self._create_bar_chart(config, data)
        elif config.chart_type == ChartType.LINE:
            return self._create_line_chart(config, data)
        elif config.chart_type == ChartType.PIE:
            return self._create_pie_chart(config, data)
        elif config.chart_type == ChartType.SCATTER:
            return self._create_scatter_chart(config, data)
        elif config.chart_type == ChartType.HISTOGRAM:
            return self._create_histogram_chart(config, data)
        else:  # TABLE or HEATMAP (not directly supported by Chart.js)
            return self._create_table_chart(config, data)
    
    def _create_bar_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Chart.js bar chart."""
        if config.y_column:
            grouped_data = data.groupby(config.x_column)[config.y_column].agg(config.aggregation).reset_index()
            labels = grouped_data[config.x_column].tolist()
            values = grouped_data[config.y_column].tolist()
        else:
            value_counts = data[config.x_column].value_counts()
            labels = value_counts.index.tolist()
            values = value_counts.values.tolist()
        
        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": config.y_column or "Count",
                    "data": values,
                    "backgroundColor": "rgba(54, 162, 235, 0.6)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": config.title
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": config.x_column
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": config.y_column or "Count"
                        }
                    }
                }
            }
        }
    
    def _create_line_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Chart.js line chart."""
        if config.y_column:
            sorted_data = data.sort_values(config.x_column)
            labels = sorted_data[config.x_column].tolist()
            values = sorted_data[config.y_column].tolist()
        else:
            grouped_data = data.groupby(config.x_column).size().reset_index(name='count')
            grouped_data = grouped_data.sort_values(config.x_column)
            labels = grouped_data[config.x_column].tolist()
            values = grouped_data['count'].tolist()
        
        return {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": config.y_column or "Count",
                    "data": values,
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "tension": 0.1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": config.title
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": config.x_column
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": config.y_column or "Count"
                        }
                    }
                }
            }
        }
    
    def _create_pie_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Chart.js pie chart."""
        value_counts = data[config.x_column].value_counts()
        
        # Generate colors for pie slices
        colors = [
            f"hsl({i * 360 / len(value_counts)}, 70%, 60%)" 
            for i in range(len(value_counts))
        ]
        
        return {
            "type": "pie",
            "data": {
                "labels": value_counts.index.tolist(),
                "datasets": [{
                    "data": value_counts.values.tolist(),
                    "backgroundColor": colors,
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": config.title
                    },
                    "legend": {
                        "position": "right"
                    }
                }
            }
        }
    
    def _create_scatter_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Chart.js scatter chart."""
        scatter_data = []
        for _, row in data.iterrows():
            scatter_data.append({
                "x": row[config.x_column],
                "y": row[config.y_column] if config.y_column else 0
            })
        
        return {
            "type": "scatter",
            "data": {
                "datasets": [{
                    "label": f"{config.y_column} vs {config.x_column}",
                    "data": scatter_data,
                    "backgroundColor": "rgba(255, 99, 132, 0.6)",
                    "borderColor": "rgba(255, 99, 132, 1)"
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": config.title
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": config.x_column
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": config.y_column or "Value"
                        }
                    }
                }
            }
        }
    
    def _create_histogram_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create Chart.js histogram (using bar chart with binned data)."""
        # Create bins for histogram
        bins = config.options.get("bins", 20)
        values = data[config.x_column].dropna()
        
        if len(values) == 0:
            return self._create_table_chart(config, data)
        
        # Calculate histogram bins
        hist, bin_edges = pd.cut(values, bins=bins, retbins=True)
        hist_counts = hist.value_counts().sort_index()
        
        # Create labels from bin edges
        labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
        
        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Frequency",
                    "data": hist_counts.values.tolist(),
                    "backgroundColor": "rgba(153, 102, 255, 0.6)",
                    "borderColor": "rgba(153, 102, 255, 1)",
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": config.title
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": config.x_column
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Frequency"
                        }
                    }
                }
            }
        }
    
    def _create_table_chart(self, config: ChartConfig, data: pd.DataFrame) -> Dict[str, Any]:
        """Create table representation (Chart.js doesn't support tables natively)."""
        # Return data in a format suitable for HTML table rendering
        display_data = data.head(100)
        
        return {
            "type": "table",
            "data": {
                "headers": display_data.columns.tolist(),
                "rows": display_data.values.tolist()
            },
            "options": {
                "title": config.title,
                "pagination": True,
                "pageSize": 10
            }
        }
    
    def _create_fallback_table(self, config: ChartConfig, data: pd.DataFrame) -> Chart:
        """Create fallback table chart when other chart types fail."""
        table_config = ChartConfig(
            chart_type=ChartType.TABLE,
            title=f"Data Table - {config.title}",
            x_column=config.x_column,
            aggregation="none"
        )
        
        table_data = self._create_table_chart(table_config, data)
        
        return Chart(
            config=table_config,
            data=table_data
        )


class ChartEngineFactory:
    """Factory for creating chart engines."""
    
    _engines = {
        "plotly": PlotlyChartEngine,
        "chartjs": ChartJSEngine
    }
    
    @classmethod
    def create_engine(cls, engine_type: str = "plotly") -> ChartEngineInterface:
        """
        Create chart engine instance.
        
        Args:
            engine_type: Type of chart engine ("plotly" or "chartjs")
            
        Returns:
            ChartEngineInterface instance
        """
        if engine_type not in cls._engines:
            logger.warning(f"Unknown engine type {engine_type}, defaulting to plotly")
            engine_type = "plotly"
        
        return cls._engines[engine_type]()
    
    @classmethod
    def get_available_engines(cls) -> List[str]:
        """Get list of available chart engines."""
        return list(cls._engines.keys())