"""
Chart type selection engine for automatic visualization generation.

This module implements intelligent chart type selection based on data patterns,
supporting multiple visualization libraries and automatic configuration generation.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
import logging

from ..core.models import ChartType, ChartConfig, DataType, ColumnInfo
from ..core.interfaces import ChartEngineInterface


logger = logging.getLogger(__name__)


class DataPattern(Enum):
    """Data pattern types for chart selection."""
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    TIME_SERIES = "time_series"
    CATEGORICAL_NUMERICAL = "categorical_numerical"
    NUMERICAL_NUMERICAL = "numerical_numerical"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"


@dataclass
class ColumnAnalysis:
    """Analysis results for a single column."""
    name: str
    data_type: DataType
    unique_count: int
    null_count: int
    is_categorical: bool
    is_numerical: bool
    is_temporal: bool
    cardinality_ratio: float  # unique_count / total_count
    sample_values: List[Any]
    statistics: Dict[str, Any]


@dataclass
class DataAnalysis:
    """Complete data analysis for chart selection."""
    columns: List[ColumnAnalysis]
    row_count: int
    patterns: List[DataPattern]
    recommended_charts: List[Tuple[ChartType, float]]  # (chart_type, confidence)
    column_relationships: Dict[str, List[str]]


class ChartTypeSelector:
    """
    Intelligent chart type selection engine.
    
    Analyzes data patterns and automatically selects appropriate chart types
    based on data characteristics, column types, and visualization best practices.
    """
    
    def __init__(self):
        self.categorical_threshold = 0.1  # Max cardinality ratio for categorical
        self.time_series_keywords = ['date', 'time', 'timestamp', 'year', 'month', 'day']
        
        # Chart type scoring weights
        self.chart_weights = {
            ChartType.BAR: {
                DataPattern.CATEGORICAL: 0.9,
                DataPattern.CATEGORICAL_NUMERICAL: 0.8,
                DataPattern.DISTRIBUTION: 0.6
            },
            ChartType.LINE: {
                DataPattern.TIME_SERIES: 0.9,
                DataPattern.NUMERICAL_NUMERICAL: 0.7,
                DataPattern.CORRELATION: 0.6
            },
            ChartType.PIE: {
                DataPattern.CATEGORICAL: 0.7,
                DataPattern.DISTRIBUTION: 0.8
            },
            ChartType.SCATTER: {
                DataPattern.NUMERICAL_NUMERICAL: 0.9,
                DataPattern.CORRELATION: 0.8
            },
            ChartType.HISTOGRAM: {
                DataPattern.DISTRIBUTION: 0.9,
                DataPattern.NUMERICAL: 0.8
            },
            ChartType.HEATMAP: {
                DataPattern.CORRELATION: 0.9,
                DataPattern.CATEGORICAL_NUMERICAL: 0.6
            }
        }
    
    def analyze_dataframe(self, df: pd.DataFrame) -> DataAnalysis:
        """
        Analyze dataframe to understand data patterns and characteristics.
        
        Args:
            df: Input dataframe to analyze
            
        Returns:
            DataAnalysis object containing complete analysis results
        """
        logger.info(f"Analyzing dataframe with shape {df.shape}")
        
        # Analyze each column
        columns = []
        for col_name in df.columns:
            column_analysis = self._analyze_column(df, col_name)
            columns.append(column_analysis)
        
        # Detect data patterns
        patterns = self._detect_patterns(columns, df)
        
        # Generate chart recommendations
        recommended_charts = self._recommend_charts(patterns, columns)
        
        # Analyze column relationships
        relationships = self._analyze_relationships(columns, df)
        
        return DataAnalysis(
            columns=columns,
            row_count=len(df),
            patterns=patterns,
            recommended_charts=recommended_charts,
            column_relationships=relationships
        )
    
    def _analyze_column(self, df: pd.DataFrame, col_name: str) -> ColumnAnalysis:
        """Analyze individual column characteristics."""
        series = df[col_name]
        
        # Basic statistics
        unique_count = series.nunique()
        null_count = series.isnull().sum()
        total_count = len(series)
        cardinality_ratio = unique_count / total_count if total_count > 0 else 0
        
        # Data type detection
        data_type = self._detect_data_type(series)
        
        # Pattern detection
        # For numerical data, only consider categorical if very low cardinality
        if data_type == DataType.NUMBER:
            is_categorical = unique_count <= 10 and cardinality_ratio <= 0.5
        elif data_type == DataType.DATE:
            # Date columns should not be considered categorical for chart selection
            is_categorical = False
        else:
            is_categorical = (
                cardinality_ratio <= self.categorical_threshold or
                data_type in [DataType.TEXT, DataType.BOOLEAN] or
                unique_count <= 20
            )
        
        is_numerical = data_type == DataType.NUMBER
        
        is_temporal = (
            data_type == DataType.DATE or
            any(keyword in col_name.lower() for keyword in self.time_series_keywords)
        )
        
        # Sample values
        sample_values = series.dropna().head(10).tolist()
        
        # Statistical summary
        statistics = {}
        if is_numerical:
            try:
                statistics = {
                    'mean': float(series.mean()) if not series.empty else 0,
                    'median': float(series.median()) if not series.empty else 0,
                    'std': float(series.std()) if not series.empty else 0,
                    'min': float(series.min()) if not series.empty else 0,
                    'max': float(series.max()) if not series.empty else 0
                }
            except (TypeError, ValueError):
                statistics = {}
        elif is_categorical:
            try:
                value_counts = series.value_counts().head(10)
                statistics = {
                    'top_values': value_counts.to_dict(),
                    'mode': series.mode().iloc[0] if not series.mode().empty else None
                }
            except (TypeError, ValueError):
                statistics = {}
        
        return ColumnAnalysis(
            name=col_name,
            data_type=data_type,
            unique_count=unique_count,
            null_count=null_count,
            is_categorical=is_categorical,
            is_numerical=is_numerical,
            is_temporal=is_temporal,
            cardinality_ratio=cardinality_ratio,
            sample_values=sample_values,
            statistics=statistics
        )
    
    def _detect_data_type(self, series: pd.Series) -> DataType:
        """Detect the most appropriate data type for a series."""
        # Try datetime first
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATE
        
        # Try numeric types
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            return DataType.NUMBER
        
        # Try boolean
        if pd.api.types.is_bool_dtype(series):
            return DataType.BOOLEAN
        
        # Check if string can be converted to datetime
        if series.dtype == 'object':
            sample = series.dropna().head(100)
            if not sample.empty:
                try:
                    pd.to_datetime(sample, errors='raise')
                    return DataType.DATE
                except:
                    pass
                
                # Check if it's numeric stored as string
                try:
                    numeric_sample = pd.to_numeric(sample, errors='raise')
                    return DataType.NUMBER
                except:
                    pass
        
        return DataType.TEXT
    
    def _detect_patterns(self, columns: List[ColumnAnalysis], df: pd.DataFrame) -> List[DataPattern]:
        """Detect data patterns for chart selection."""
        patterns = []
        
        categorical_cols = [col for col in columns if col.is_categorical]
        numerical_cols = [col for col in columns if col.is_numerical]
        temporal_cols = [col for col in columns if col.is_temporal]
        
        # Time series pattern
        if temporal_cols and numerical_cols:
            patterns.append(DataPattern.TIME_SERIES)
        
        # Categorical pattern
        if categorical_cols:
            patterns.append(DataPattern.CATEGORICAL)
        
        # Numerical pattern
        if numerical_cols:
            patterns.append(DataPattern.NUMERICAL)
        
        # Mixed patterns
        if categorical_cols and numerical_cols:
            patterns.append(DataPattern.CATEGORICAL_NUMERICAL)
        
        if len(numerical_cols) >= 2:
            patterns.append(DataPattern.NUMERICAL_NUMERICAL)
            patterns.append(DataPattern.CORRELATION)
        
        # Distribution pattern (single numerical column)
        if len(numerical_cols) == 1 and not categorical_cols:
            patterns.append(DataPattern.DISTRIBUTION)
        
        return patterns
    
    def _recommend_charts(self, patterns: List[DataPattern], columns: List[ColumnAnalysis]) -> List[Tuple[ChartType, float]]:
        """Recommend chart types based on detected patterns."""
        chart_scores = {}
        
        # Score each chart type based on patterns
        for chart_type, pattern_weights in self.chart_weights.items():
            score = 0.0
            pattern_count = 0
            for pattern in patterns:
                if pattern in pattern_weights:
                    score += pattern_weights[pattern]
                    pattern_count += 1
            
            if score > 0:
                # Use max score instead of average for better prioritization
                chart_scores[chart_type] = score / pattern_count if pattern_count > 0 else 0
        
        # Apply pattern-specific boosts
        if DataPattern.TIME_SERIES in patterns:
            # Boost line charts for time series data
            if ChartType.LINE in chart_scores:
                chart_scores[ChartType.LINE] *= 1.5
        
        # Apply additional heuristics
        chart_scores = self._apply_heuristics(chart_scores, columns)
        
        # Sort by score and return top recommendations
        sorted_charts = sorted(chart_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_charts[:5]  # Return top 5 recommendations
    
    def _apply_heuristics(self, chart_scores: Dict[ChartType, float], columns: List[ColumnAnalysis]) -> Dict[ChartType, float]:
        """Apply additional heuristics to refine chart scores."""
        categorical_cols = [col for col in columns if col.is_categorical]
        numerical_cols = [col for col in columns if col.is_numerical]
        
        # Boost pie chart for small categorical datasets
        if categorical_cols and ChartType.PIE in chart_scores:
            for col in categorical_cols:
                if col.unique_count <= 8:  # Good for pie charts
                    chart_scores[ChartType.PIE] *= 1.2
        
        # Penalize pie chart for too many categories
        if categorical_cols and ChartType.PIE in chart_scores:
            for col in categorical_cols:
                if col.unique_count > 10:
                    chart_scores[ChartType.PIE] *= 0.5
        
        # Boost histogram for single numerical column
        if len(numerical_cols) == 1 and len(categorical_cols) == 0:
            if ChartType.HISTOGRAM in chart_scores:
                chart_scores[ChartType.HISTOGRAM] *= 1.3
        
        # Boost scatter plot for two numerical columns
        if len(numerical_cols) >= 2 and ChartType.SCATTER in chart_scores:
            chart_scores[ChartType.SCATTER] *= 1.2
        
        return chart_scores
    
    def _analyze_relationships(self, columns: List[ColumnAnalysis], df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze relationships between columns."""
        relationships = {}
        
        numerical_cols = [col.name for col in columns if col.is_numerical]
        
        # Calculate correlations for numerical columns
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr()
            
            for col in numerical_cols:
                related_cols = []
                for other_col in numerical_cols:
                    if col != other_col and abs(corr_matrix.loc[col, other_col]) > 0.5:
                        related_cols.append(other_col)
                
                if related_cols:
                    relationships[col] = related_cols
        
        return relationships
    
    def select_chart_type(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> ChartType:
        """
        Select the best chart type for given data and columns.
        
        Args:
            df: Input dataframe
            columns: Specific columns to consider (if None, uses all columns)
            
        Returns:
            Recommended ChartType
        """
        if columns:
            df_subset = df[columns]
        else:
            df_subset = df
        
        analysis = self.analyze_dataframe(df_subset)
        
        if analysis.recommended_charts:
            return analysis.recommended_charts[0][0]
        
        # Fallback to table if no clear recommendation
        return ChartType.TABLE
    
    def generate_chart_config(self, df: pd.DataFrame, chart_type: ChartType, 
                            columns: Optional[List[str]] = None) -> ChartConfig:
        """
        Generate chart configuration for specified chart type and data.
        
        Args:
            df: Input dataframe
            chart_type: Type of chart to generate
            columns: Specific columns to use
            
        Returns:
            ChartConfig object with appropriate configuration
        """
        if columns:
            df_subset = df[columns]
        else:
            df_subset = df
        
        analysis = self.analyze_dataframe(df_subset)
        
        # Find appropriate columns for chart type
        categorical_cols = [col.name for col in analysis.columns if col.is_categorical]
        numerical_cols = [col.name for col in analysis.columns if col.is_numerical]
        temporal_cols = [col.name for col in analysis.columns if col.is_temporal]
        
        # Generate configuration based on chart type
        config = self._generate_config_for_type(
            chart_type, categorical_cols, numerical_cols, temporal_cols, analysis
        )
        
        return config
    
    def _generate_config_for_type(self, chart_type: ChartType, categorical_cols: List[str],
                                numerical_cols: List[str], temporal_cols: List[str],
                                analysis: DataAnalysis) -> ChartConfig:
        """Generate configuration for specific chart type."""
        
        if chart_type == ChartType.BAR:
            x_col = categorical_cols[0] if categorical_cols else numerical_cols[0]
            y_col = numerical_cols[0] if numerical_cols else None
            title = f"{y_col or 'Count'} by {x_col}" if y_col else f"Distribution of {x_col}"
            
            return ChartConfig(
                chart_type=chart_type,
                title=title,
                x_column=x_col,
                y_column=y_col,
                aggregation="sum" if y_col else "count"
            )
        
        elif chart_type == ChartType.LINE:
            x_col = temporal_cols[0] if temporal_cols else numerical_cols[0]
            y_col = numerical_cols[0] if numerical_cols else numerical_cols[1] if len(numerical_cols) > 1 else None
            title = f"{y_col} over {x_col}" if y_col else f"Trend of {x_col}"
            
            return ChartConfig(
                chart_type=chart_type,
                title=title,
                x_column=x_col,
                y_column=y_col,
                aggregation="avg" if y_col else "count"
            )
        
        elif chart_type == ChartType.PIE:
            x_col = categorical_cols[0] if categorical_cols else numerical_cols[0]
            title = f"Distribution of {x_col}"
            
            return ChartConfig(
                chart_type=chart_type,
                title=title,
                x_column=x_col,
                aggregation="count"
            )
        
        elif chart_type == ChartType.SCATTER:
            x_col = numerical_cols[0] if numerical_cols else categorical_cols[0]
            y_col = numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0]
            color_col = categorical_cols[0] if categorical_cols else None
            title = f"{y_col} vs {x_col}"
            
            return ChartConfig(
                chart_type=chart_type,
                title=title,
                x_column=x_col,
                y_column=y_col,
                color_column=color_col,
                aggregation="none"
            )
        
        elif chart_type == ChartType.HISTOGRAM:
            x_col = numerical_cols[0] if numerical_cols else categorical_cols[0]
            title = f"Distribution of {x_col}"
            
            return ChartConfig(
                chart_type=chart_type,
                title=title,
                x_column=x_col,
                aggregation="count",
                options={"bins": 20}
            )
        
        elif chart_type == ChartType.HEATMAP:
            # For heatmap, we need at least 2 dimensions
            x_col = categorical_cols[0] if categorical_cols else numerical_cols[0]
            y_col = categorical_cols[1] if len(categorical_cols) > 1 else numerical_cols[0]
            title = f"Heatmap of {x_col} vs {y_col}"
            
            return ChartConfig(
                chart_type=chart_type,
                title=title,
                x_column=x_col,
                y_column=y_col,
                aggregation="count"
            )
        
        else:  # TABLE or fallback
            return ChartConfig(
                chart_type=ChartType.TABLE,
                title="Data Table",
                x_column=analysis.columns[0].name if analysis.columns else "data",
                aggregation="none"
            )
    
    def get_chart_recommendations(self, df: pd.DataFrame, max_recommendations: int = 3) -> List[ChartConfig]:
        """
        Get multiple chart recommendations for a dataframe.
        
        Args:
            df: Input dataframe
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of ChartConfig objects with recommended configurations
        """
        analysis = self.analyze_dataframe(df)
        recommendations = []
        
        # Get top chart type recommendations
        top_charts = analysis.recommended_charts[:max_recommendations]
        
        for chart_type, confidence in top_charts:
            config = self.generate_chart_config(df, chart_type)
            recommendations.append(config)
        
        return recommendations