"""
Advanced KPI calculation and display engine.

This module provides automatic KPI detection, calculation, trend analysis,
and comparison features for dashboard analytics.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

from ..core.models import KPI, DataType


logger = logging.getLogger(__name__)


class KPIType(Enum):
    """Types of KPIs that can be automatically detected."""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    GROWTH_RATE = "growth_rate"
    VARIANCE = "variance"
    STANDARD_DEVIATION = "standard_deviation"


class TrendDirection(Enum):
    """Trend direction indicators."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class KPIDefinition:
    """Definition of a KPI calculation."""
    kpi_id: str
    name: str
    kpi_type: KPIType
    calculation_func: Callable[[pd.DataFrame], Union[int, float]]
    format_type: str = "number"
    description: str = ""
    target_value: Optional[float] = None
    alert_threshold: Optional[float] = None
    is_higher_better: bool = True
    category: str = "general"


@dataclass
class TrendAnalysis:
    """Trend analysis results for a KPI."""
    direction: TrendDirection
    percentage_change: float
    absolute_change: float
    trend_strength: float  # 0-1 scale
    volatility: float
    data_points: List[float]
    time_period: str = "recent"


@dataclass
class KPIComparison:
    """Comparison between KPI values."""
    kpi_name: str
    current_value: float
    comparison_value: float
    difference: float
    percentage_difference: float
    is_improvement: bool
    comparison_type: str  # "previous_period", "target", "benchmark"


@dataclass
class KPIInsight:
    """Insight generated from KPI analysis."""
    kpi_name: str
    insight_type: str  # "trend", "anomaly", "target_achievement", "comparison"
    message: str
    severity: str  # "info", "warning", "critical", "success"
    confidence: float
    supporting_data: Dict[str, Any] = field(default_factory=dict)


class AutoKPIDetector:
    """Automatically detects relevant KPIs from data."""
    
    def __init__(self):
        self.detection_rules: Dict[str, Callable] = {
            "numeric_summary": self._detect_numeric_summary_kpis,
            "categorical_distribution": self._detect_categorical_kpis,
            "time_series": self._detect_time_series_kpis,
            "business_metrics": self._detect_business_kpis,
            "data_quality": self._detect_data_quality_kpis
        }
    
    def detect_kpis(self, dataframe: pd.DataFrame, max_kpis: int = 10) -> List[KPIDefinition]:
        """Detect relevant KPIs from dataframe."""
        logger.info(f"Detecting KPIs for dataframe with shape {dataframe.shape}")
        
        detected_kpis = []
        
        # Run all detection rules
        for rule_name, rule_func in self.detection_rules.items():
            try:
                rule_kpis = rule_func(dataframe)
                detected_kpis.extend(rule_kpis)
                logger.debug(f"Rule '{rule_name}' detected {len(rule_kpis)} KPIs")
            except Exception as e:
                logger.error(f"Error in detection rule '{rule_name}': {e}")
        
        # Score and rank KPIs
        scored_kpis = self._score_kpis(detected_kpis, dataframe)
        
        # Return top KPIs
        return scored_kpis[:max_kpis]
    
    def _detect_numeric_summary_kpis(self, dataframe: pd.DataFrame) -> List[KPIDefinition]:
        """Detect basic numeric summary KPIs."""
        kpis = []
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Skip columns with too many nulls
            if dataframe[col].isnull().sum() / len(dataframe) > 0.5:
                continue
            
            # Total/Sum KPI
            if dataframe[col].min() >= 0:  # Only for non-negative values
                kpis.append(KPIDefinition(
                    kpi_id=f"total_{col}",
                    name=f"Total {col.title()}",
                    kpi_type=KPIType.SUM,
                    calculation_func=lambda df, column=col: df[column].sum(),
                    format_type=self._infer_format_type(col, dataframe[col]),
                    description=f"Sum of all {col} values",
                    category="summary"
                ))
            
            # Average KPI
            kpis.append(KPIDefinition(
                kpi_id=f"avg_{col}",
                name=f"Average {col.title()}",
                kpi_type=KPIType.AVERAGE,
                calculation_func=lambda df, column=col: df[column].mean(),
                format_type=self._infer_format_type(col, dataframe[col]),
                description=f"Average value of {col}",
                category="summary"
            ))
            
            # Max KPI for relevant columns
            if dataframe[col].nunique() > 5:  # Only if there's variation
                kpis.append(KPIDefinition(
                    kpi_id=f"max_{col}",
                    name=f"Maximum {col.title()}",
                    kpi_type=KPIType.MAX,
                    calculation_func=lambda df, column=col: df[column].max(),
                    format_type=self._infer_format_type(col, dataframe[col]),
                    description=f"Maximum value of {col}",
                    category="summary"
                ))
        
        return kpis
    
    def _detect_categorical_kpis(self, dataframe: pd.DataFrame) -> List[KPIDefinition]:
        """Detect KPIs for categorical data."""
        kpis = []
        categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            # Skip high-cardinality columns
            if dataframe[col].nunique() > 20:
                continue
            
            # Most common category
            most_common = dataframe[col].value_counts().index[0]
            kpis.append(KPIDefinition(
                kpi_id=f"top_{col}",
                name=f"Top {col.title()}",
                kpi_type=KPIType.COUNT,
                calculation_func=lambda df, column=col, value=most_common: (df[column] == value).sum(),
                format_type="number",
                description=f"Count of most common {col} ({most_common})",
                category="categorical"
            ))
            
            # Unique count
            kpis.append(KPIDefinition(
                kpi_id=f"unique_{col}",
                name=f"Unique {col.title()}",
                kpi_type=KPIType.COUNT,
                calculation_func=lambda df, column=col: df[column].nunique(),
                format_type="number",
                description=f"Number of unique {col} values",
                category="categorical"
            ))
        
        return kpis
    
    def _detect_time_series_kpis(self, dataframe: pd.DataFrame) -> List[KPIDefinition]:
        """Detect time-series related KPIs."""
        kpis = []
        date_columns = dataframe.select_dtypes(include=['datetime64']).columns
        
        if len(date_columns) == 0:
            # Try to detect date columns by name
            potential_date_cols = [col for col in dataframe.columns 
                                 if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated'])]
            for col in potential_date_cols:
                try:
                    pd.to_datetime(dataframe[col])
                    date_columns = [col]
                    break
                except:
                    continue
        
        if len(date_columns) > 0:
            date_col = date_columns[0]
            
            # Records per time period
            kpis.append(KPIDefinition(
                kpi_id="records_per_day",
                name="Records per Day",
                kpi_type=KPIType.AVERAGE,
                calculation_func=lambda df: len(df) / max(1, (pd.to_datetime(df[date_col]).max() - pd.to_datetime(df[date_col]).min()).days),
                format_type="number",
                description="Average number of records per day",
                category="time_series"
            ))
            
            # Recent activity (last 7 days)
            kpis.append(KPIDefinition(
                kpi_id="recent_records",
                name="Recent Records (7d)",
                kpi_type=KPIType.COUNT,
                calculation_func=lambda df: len(df[pd.to_datetime(df[date_col]) >= pd.to_datetime(df[date_col]).max() - timedelta(days=7)]),
                format_type="number",
                description="Number of records in the last 7 days",
                category="time_series"
            ))
        
        return kpis
    
    def _detect_business_kpis(self, dataframe: pd.DataFrame) -> List[KPIDefinition]:
        """Detect business-relevant KPIs based on column names."""
        kpis = []
        
        # Common business metrics patterns
        business_patterns = {
            'revenue': ['revenue', 'sales', 'income', 'earnings'],
            'cost': ['cost', 'expense', 'spend', 'expenditure'],
            'profit': ['profit', 'margin', 'net'],
            'quantity': ['quantity', 'qty', 'amount', 'volume'],
            'price': ['price', 'rate', 'fee', 'charge'],
            'customer': ['customer', 'client', 'user'],
            'order': ['order', 'transaction', 'purchase']
        }
        
        for metric_type, patterns in business_patterns.items():
            matching_cols = []
            for col in dataframe.columns:
                if any(pattern in col.lower() for pattern in patterns):
                    matching_cols.append(col)
            
            for col in matching_cols:
                if dataframe[col].dtype in ['int64', 'float64']:
                    # Revenue/Sales KPIs
                    if metric_type in ['revenue', 'profit']:
                        kpis.append(KPIDefinition(
                            kpi_id=f"total_{metric_type}",
                            name=f"Total {metric_type.title()}",
                            kpi_type=KPIType.SUM,
                            calculation_func=lambda df, column=col: df[column].sum(),
                            format_type="currency",
                            description=f"Total {metric_type} amount",
                            category="business",
                            is_higher_better=True
                        ))
                    
                    # Average transaction value
                    if metric_type in ['revenue', 'order']:
                        kpis.append(KPIDefinition(
                            kpi_id=f"avg_{metric_type}_value",
                            name=f"Average {metric_type.title()} Value",
                            kpi_type=KPIType.AVERAGE,
                            calculation_func=lambda df, column=col: df[column].mean(),
                            format_type="currency" if metric_type == 'revenue' else "number",
                            description=f"Average {metric_type} value per record",
                            category="business"
                        ))
        
        return kpis
    
    def _detect_data_quality_kpis(self, dataframe: pd.DataFrame) -> List[KPIDefinition]:
        """Detect data quality KPIs."""
        kpis = []
        
        # Completeness KPI
        kpis.append(KPIDefinition(
            kpi_id="data_completeness",
            name="Data Completeness",
            kpi_type=KPIType.PERCENTAGE,
            calculation_func=lambda df: (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            format_type="percentage",
            description="Percentage of non-null values in the dataset",
            category="quality",
            is_higher_better=True
        ))
        
        # Record count
        kpis.append(KPIDefinition(
            kpi_id="total_records",
            name="Total Records",
            kpi_type=KPIType.COUNT,
            calculation_func=lambda df: len(df),
            format_type="number",
            description="Total number of records in the dataset",
            category="quality"
        ))
        
        # Duplicate rate
        kpis.append(KPIDefinition(
            kpi_id="duplicate_rate",
            name="Duplicate Rate",
            kpi_type=KPIType.PERCENTAGE,
            calculation_func=lambda df: (df.duplicated().sum() / len(df)) * 100,
            format_type="percentage",
            description="Percentage of duplicate records",
            category="quality",
            is_higher_better=False
        ))
        
        return kpis
    
    def _score_kpis(self, kpis: List[KPIDefinition], dataframe: pd.DataFrame) -> List[KPIDefinition]:
        """Score and rank KPIs by relevance."""
        scored_kpis = []
        
        for kpi in kpis:
            score = 0
            
            # Base score by category
            category_scores = {
                "business": 10,
                "summary": 8,
                "time_series": 7,
                "categorical": 6,
                "quality": 5
            }
            score += category_scores.get(kpi.category, 3)
            
            # Bonus for business-relevant KPIs
            if any(keyword in kpi.name.lower() for keyword in ['revenue', 'profit', 'sales', 'customer']):
                score += 5
            
            # Bonus for commonly used metrics
            if kpi.kpi_type in [KPIType.SUM, KPIType.AVERAGE, KPIType.COUNT]:
                score += 3
            
            # Store score for sorting
            kpi.score = score
            scored_kpis.append(kpi)
        
        # Sort by score (descending)
        return sorted(scored_kpis, key=lambda x: getattr(x, 'score', 0), reverse=True)
    
    def _infer_format_type(self, column_name: str, series: pd.Series) -> str:
        """Infer the appropriate format type for a column."""
        col_lower = column_name.lower()
        
        # Currency indicators
        if any(keyword in col_lower for keyword in ['revenue', 'sales', 'price', 'cost', 'profit', 'income', 'fee']):
            return "currency"
        
        # Percentage indicators
        if any(keyword in col_lower for keyword in ['rate', 'percent', 'ratio']) or series.max() <= 1.0:
            return "percentage"
        
        # Default to number
        return "number"


class TrendAnalyzer:
    """Analyzes trends in KPI values over time."""
    
    def __init__(self, min_data_points: int = 3):
        self.min_data_points = min_data_points
    
    def analyze_trend(self, values: List[float], time_period: str = "recent") -> TrendAnalysis:
        """Analyze trend in a series of values."""
        if len(values) < self.min_data_points:
            return TrendAnalysis(
                direction=TrendDirection.STABLE,
                percentage_change=0.0,
                absolute_change=0.0,
                trend_strength=0.0,
                volatility=0.0,
                data_points=values,
                time_period=time_period
            )
        
        # Calculate basic trend metrics
        first_value = values[0]
        last_value = values[-1]
        absolute_change = last_value - first_value
        percentage_change = (absolute_change / first_value * 100) if first_value != 0 else 0
        
        # Determine trend direction
        direction = self._determine_direction(values, percentage_change)
        
        # Calculate trend strength using linear regression
        trend_strength = self._calculate_trend_strength(values)
        
        # Calculate volatility
        volatility = self._calculate_volatility(values)
        
        return TrendAnalysis(
            direction=direction,
            percentage_change=percentage_change,
            absolute_change=absolute_change,
            trend_strength=trend_strength,
            volatility=volatility,
            data_points=values.copy(),
            time_period=time_period
        )
    
    def _determine_direction(self, values: List[float], percentage_change: float) -> TrendDirection:
        """Determine the overall trend direction."""
        # Check for high volatility
        if self._calculate_volatility(values) > 0.3:
            return TrendDirection.VOLATILE
        
        # Determine direction based on percentage change
        if abs(percentage_change) < 5:  # Less than 5% change
            return TrendDirection.STABLE
        elif percentage_change > 0:
            return TrendDirection.UP
        else:
            return TrendDirection.DOWN
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate trend strength using correlation coefficient."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Return absolute correlation as strength (0-1)
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility as coefficient of variation."""
        if len(values) < 2:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Coefficient of variation
        return (std_val / mean_val) if mean_val != 0 else 0.0


class KPIComparator:
    """Compares KPI values across different dimensions."""
    
    def compare_periods(self, current_kpi: KPI, previous_kpi: KPI) -> KPIComparison:
        """Compare KPI values between two time periods."""
        current_val = float(current_kpi.value)
        previous_val = float(previous_kpi.value)
        
        difference = current_val - previous_val
        percentage_diff = (difference / previous_val * 100) if previous_val != 0 else 0
        
        # Determine if this is an improvement (depends on KPI type)
        is_improvement = difference > 0  # Default assumption
        
        return KPIComparison(
            kpi_name=current_kpi.name,
            current_value=current_val,
            comparison_value=previous_val,
            difference=difference,
            percentage_difference=percentage_diff,
            is_improvement=is_improvement,
            comparison_type="previous_period"
        )
    
    def compare_to_target(self, kpi: KPI, target_value: float) -> KPIComparison:
        """Compare KPI value to a target."""
        current_val = float(kpi.value)
        difference = current_val - target_value
        percentage_diff = (difference / target_value * 100) if target_value != 0 else 0
        
        is_improvement = current_val >= target_value
        
        return KPIComparison(
            kpi_name=kpi.name,
            current_value=current_val,
            comparison_value=target_value,
            difference=difference,
            percentage_difference=percentage_diff,
            is_improvement=is_improvement,
            comparison_type="target"
        )
    
    def compare_to_benchmark(self, kpi: KPI, benchmark_value: float) -> KPIComparison:
        """Compare KPI value to a benchmark."""
        current_val = float(kpi.value)
        difference = current_val - benchmark_value
        percentage_diff = (difference / benchmark_value * 100) if benchmark_value != 0 else 0
        
        is_improvement = current_val >= benchmark_value
        
        return KPIComparison(
            kpi_name=kpi.name,
            current_value=current_val,
            comparison_value=benchmark_value,
            difference=difference,
            percentage_difference=percentage_diff,
            is_improvement=is_improvement,
            comparison_type="benchmark"
        )


class KPIInsightGenerator:
    """Generates insights from KPI analysis."""
    
    def __init__(self):
        self.insight_rules = [
            self._trend_insights,
            self._anomaly_insights,
            self._target_achievement_insights,
            self._comparison_insights
        ]
    
    def generate_insights(self, kpi: KPI, trend: Optional[TrendAnalysis] = None, 
                         comparisons: List[KPIComparison] = None) -> List[KPIInsight]:
        """Generate insights for a KPI."""
        insights = []
        
        for rule in self.insight_rules:
            try:
                rule_insights = rule(kpi, trend, comparisons or [])
                insights.extend(rule_insights)
            except Exception as e:
                logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def _trend_insights(self, kpi: KPI, trend: Optional[TrendAnalysis], 
                       comparisons: List[KPIComparison]) -> List[KPIInsight]:
        """Generate trend-based insights."""
        insights = []
        
        if trend is None:
            return insights
        
        if trend.direction == TrendDirection.UP and trend.trend_strength > 0.7:
            insights.append(KPIInsight(
                kpi_name=kpi.name,
                insight_type="trend",
                message=f"{kpi.name} shows a strong upward trend ({trend.percentage_change:.1f}% increase)",
                severity="success",
                confidence=trend.trend_strength,
                supporting_data={"trend_strength": trend.trend_strength, "change": trend.percentage_change}
            ))
        elif trend.direction == TrendDirection.DOWN and trend.trend_strength > 0.7:
            insights.append(KPIInsight(
                kpi_name=kpi.name,
                insight_type="trend",
                message=f"{kpi.name} shows a concerning downward trend ({trend.percentage_change:.1f}% decrease)",
                severity="warning",
                confidence=trend.trend_strength,
                supporting_data={"trend_strength": trend.trend_strength, "change": trend.percentage_change}
            ))
        elif trend.direction == TrendDirection.VOLATILE:
            insights.append(KPIInsight(
                kpi_name=kpi.name,
                insight_type="trend",
                message=f"{kpi.name} shows high volatility (CV: {trend.volatility:.2f})",
                severity="info",
                confidence=0.8,
                supporting_data={"volatility": trend.volatility}
            ))
        
        return insights
    
    def _anomaly_insights(self, kpi: KPI, trend: Optional[TrendAnalysis], 
                         comparisons: List[KPIComparison]) -> List[KPIInsight]:
        """Generate anomaly-based insights."""
        insights = []
        
        # Check for extreme values (placeholder logic)
        if isinstance(kpi.value, (int, float)):
            if kpi.value == 0:
                insights.append(KPIInsight(
                    kpi_name=kpi.name,
                    insight_type="anomaly",
                    message=f"{kpi.name} has a value of zero, which may indicate an issue",
                    severity="warning",
                    confidence=0.9
                ))
        
        return insights
    
    def _target_achievement_insights(self, kpi: KPI, trend: Optional[TrendAnalysis], 
                                   comparisons: List[KPIComparison]) -> List[KPIInsight]:
        """Generate target achievement insights."""
        insights = []
        
        target_comparisons = [c for c in comparisons if c.comparison_type == "target"]
        
        for comparison in target_comparisons:
            if comparison.is_improvement:
                insights.append(KPIInsight(
                    kpi_name=kpi.name,
                    insight_type="target_achievement",
                    message=f"{kpi.name} exceeded target by {comparison.percentage_difference:.1f}%",
                    severity="success",
                    confidence=0.95,
                    supporting_data={"target_value": comparison.comparison_value, "achievement": comparison.percentage_difference}
                ))
            else:
                insights.append(KPIInsight(
                    kpi_name=kpi.name,
                    insight_type="target_achievement",
                    message=f"{kpi.name} is {abs(comparison.percentage_difference):.1f}% below target",
                    severity="warning",
                    confidence=0.95,
                    supporting_data={"target_value": comparison.comparison_value, "shortfall": comparison.percentage_difference}
                ))
        
        return insights
    
    def _comparison_insights(self, kpi: KPI, trend: Optional[TrendAnalysis], 
                           comparisons: List[KPIComparison]) -> List[KPIInsight]:
        """Generate comparison-based insights."""
        insights = []
        
        period_comparisons = [c for c in comparisons if c.comparison_type == "previous_period"]
        
        for comparison in period_comparisons:
            if abs(comparison.percentage_difference) > 20:  # Significant change
                severity = "success" if comparison.is_improvement else "warning"
                direction = "increased" if comparison.difference > 0 else "decreased"
                
                insights.append(KPIInsight(
                    kpi_name=kpi.name,
                    insight_type="comparison",
                    message=f"{kpi.name} {direction} by {abs(comparison.percentage_difference):.1f}% compared to previous period",
                    severity=severity,
                    confidence=0.9,
                    supporting_data={"previous_value": comparison.comparison_value, "change": comparison.percentage_difference}
                ))
        
        return insights


class EnhancedKPIEngine:
    """
    Enhanced KPI engine with automatic detection, trend analysis, and insights.
    
    This is the main class that orchestrates all KPI functionality including
    automatic detection, calculation, trend analysis, comparisons, and insights.
    """
    
    def __init__(self):
        self.detector = AutoKPIDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.comparator = KPIComparator()
        self.insight_generator = KPIInsightGenerator()
        self.kpi_definitions: Dict[str, KPIDefinition] = {}
        self.kpi_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.targets: Dict[str, float] = {}
        self.benchmarks: Dict[str, float] = {}
    
    def auto_detect_kpis(self, dataframe: pd.DataFrame, max_kpis: int = 10) -> List[KPIDefinition]:
        """Automatically detect relevant KPIs from data."""
        detected_kpis = self.detector.detect_kpis(dataframe, max_kpis)
        
        # Register detected KPIs
        for kpi_def in detected_kpis:
            self.kpi_definitions[kpi_def.kpi_id] = kpi_def
        
        logger.info(f"Auto-detected {len(detected_kpis)} KPIs")
        return detected_kpis
    
    def calculate_kpis(self, dataframe: pd.DataFrame, kpi_ids: Optional[List[str]] = None) -> List[KPI]:
        """Calculate KPI values from data."""
        if kpi_ids is None:
            kpi_ids = list(self.kpi_definitions.keys())
        
        calculated_kpis = []
        
        for kpi_id in kpi_ids:
            if kpi_id not in self.kpi_definitions:
                logger.warning(f"Unknown KPI ID: {kpi_id}")
                continue
            
            try:
                kpi_def = self.kpi_definitions[kpi_id]
                value = kpi_def.calculation_func(dataframe)
                
                # Calculate trend if historical data exists
                trend = None
                if kpi_id in self.kpi_history:
                    historical_values = [val for _, val in self.kpi_history[kpi_id]]
                    historical_values.append(value)
                    trend = self.trend_analyzer.analyze_trend(historical_values[-10:])  # Last 10 values
                
                kpi = KPI(
                    name=kpi_def.name,
                    value=value,
                    format_type=kpi_def.format_type,
                    description=kpi_def.description,
                    trend=trend.percentage_change if trend else None
                )
                
                calculated_kpis.append(kpi)
                
                # Store in history
                if kpi_id not in self.kpi_history:
                    self.kpi_history[kpi_id] = []
                self.kpi_history[kpi_id].append((datetime.now(), value))
                
                # Keep only last 50 historical values
                if len(self.kpi_history[kpi_id]) > 50:
                    self.kpi_history[kpi_id] = self.kpi_history[kpi_id][-50:]
                
            except Exception as e:
                logger.error(f"Error calculating KPI {kpi_id}: {e}")
        
        return calculated_kpis
    
    def analyze_kpi_trends(self, kpi_id: str, time_period: str = "recent") -> Optional[TrendAnalysis]:
        """Analyze trends for a specific KPI."""
        if kpi_id not in self.kpi_history:
            return None
        
        values = [val for _, val in self.kpi_history[kpi_id]]
        return self.trend_analyzer.analyze_trend(values, time_period)
    
    def compare_kpis(self, current_kpis: List[KPI], previous_kpis: Optional[List[KPI]] = None) -> List[KPIComparison]:
        """Compare current KPIs with previous values or targets."""
        comparisons = []
        
        # Compare with previous period
        if previous_kpis:
            current_dict = {kpi.name: kpi for kpi in current_kpis}
            previous_dict = {kpi.name: kpi for kpi in previous_kpis}
            
            for name in current_dict:
                if name in previous_dict:
                    comparison = self.comparator.compare_periods(
                        current_dict[name], previous_dict[name]
                    )
                    comparisons.append(comparison)
        
        # Compare with targets
        for kpi in current_kpis:
            kpi_id = self._get_kpi_id_by_name(kpi.name)
            if kpi_id and kpi_id in self.targets:
                comparison = self.comparator.compare_to_target(kpi, self.targets[kpi_id])
                comparisons.append(comparison)
        
        # Compare with benchmarks
        for kpi in current_kpis:
            kpi_id = self._get_kpi_id_by_name(kpi.name)
            if kpi_id and kpi_id in self.benchmarks:
                comparison = self.comparator.compare_to_benchmark(kpi, self.benchmarks[kpi_id])
                comparisons.append(comparison)
        
        return comparisons
    
    def generate_kpi_insights(self, kpis: List[KPI], comparisons: Optional[List[KPIComparison]] = None) -> List[KPIInsight]:
        """Generate insights for KPIs."""
        all_insights = []
        
        for kpi in kpis:
            kpi_id = self._get_kpi_id_by_name(kpi.name)
            trend = self.analyze_kpi_trends(kpi_id) if kpi_id else None
            
            kpi_comparisons = [c for c in (comparisons or []) if c.kpi_name == kpi.name]
            
            insights = self.insight_generator.generate_insights(kpi, trend, kpi_comparisons)
            all_insights.extend(insights)
        
        return all_insights
    
    def set_kpi_target(self, kpi_id: str, target_value: float) -> None:
        """Set target value for a KPI."""
        self.targets[kpi_id] = target_value
        logger.info(f"Set target for {kpi_id}: {target_value}")
    
    def set_kpi_benchmark(self, kpi_id: str, benchmark_value: float) -> None:
        """Set benchmark value for a KPI."""
        self.benchmarks[kpi_id] = benchmark_value
        logger.info(f"Set benchmark for {kpi_id}: {benchmark_value}")
    
    def get_kpi_summary(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive KPI summary."""
        # Auto-detect and calculate KPIs
        detected_kpis = self.auto_detect_kpis(dataframe)
        calculated_kpis = self.calculate_kpis(dataframe)
        
        # Generate comparisons and insights
        comparisons = self.compare_kpis(calculated_kpis)
        insights = self.generate_kpi_insights(calculated_kpis, comparisons)
        
        return {
            "kpis": [self._kpi_to_dict(kpi) for kpi in calculated_kpis],
            "comparisons": [self._comparison_to_dict(comp) for comp in comparisons],
            "insights": [self._insight_to_dict(insight) for insight in insights],
            "summary": {
                "total_kpis": len(calculated_kpis),
                "total_insights": len(insights),
                "critical_insights": len([i for i in insights if i.severity == "critical"]),
                "warning_insights": len([i for i in insights if i.severity == "warning"])
            }
        }
    
    def _get_kpi_id_by_name(self, name: str) -> Optional[str]:
        """Get KPI ID by name."""
        for kpi_id, kpi_def in self.kpi_definitions.items():
            if kpi_def.name == name:
                return kpi_id
        return None
    
    def _kpi_to_dict(self, kpi: KPI) -> Dict[str, Any]:
        """Convert KPI to dictionary."""
        return {
            "name": kpi.name,
            "value": kpi.value,
            "format_type": kpi.format_type,
            "description": kpi.description,
            "trend": kpi.trend
        }
    
    def _comparison_to_dict(self, comparison: KPIComparison) -> Dict[str, Any]:
        """Convert KPI comparison to dictionary."""
        return {
            "kpi_name": comparison.kpi_name,
            "current_value": comparison.current_value,
            "comparison_value": comparison.comparison_value,
            "difference": comparison.difference,
            "percentage_difference": comparison.percentage_difference,
            "is_improvement": comparison.is_improvement,
            "comparison_type": comparison.comparison_type
        }
    
    def _insight_to_dict(self, insight: KPIInsight) -> Dict[str, Any]:
        """Convert KPI insight to dictionary."""
        return {
            "kpi_name": insight.kpi_name,
            "insight_type": insight.insight_type,
            "message": insight.message,
            "severity": insight.severity,
            "confidence": insight.confidence,
            "supporting_data": insight.supporting_data
        }