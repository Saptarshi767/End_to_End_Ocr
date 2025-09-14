"""
Enhanced missing value handling system for OCR extracted table data.

This module provides configurable strategies for handling missing data,
data quality assessment metrics, and user-configurable policies.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from sklearn.impute import KNNImputer
import warnings

from ..core.models import DataType, ValidationResult
from ..core.exceptions import DataProcessingError


logger = logging.getLogger(__name__)


class MissingValueStrategy(Enum):
    """Available strategies for handling missing values."""
    REMOVE = "remove"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    ZERO = "zero"
    EMPTY_STRING = "empty_string"
    FALSE = "false"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE_LINEAR = "interpolate_linear"
    INTERPOLATE_POLYNOMIAL = "interpolate_polynomial"
    KNN_IMPUTE = "knn_impute"
    CUSTOM_VALUE = "custom_value"
    STATISTICAL_OUTLIER = "statistical_outlier"


class DataQualityMetric(Enum):
    """Data quality metrics for assessment."""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"


@dataclass
class MissingValuePolicy:
    """Policy configuration for handling missing values in a specific column."""
    column_name: str
    data_type: DataType
    strategy: MissingValueStrategy
    custom_value: Any = None
    threshold: float = 0.5  # Threshold for applying strategy
    fallback_strategy: Optional[MissingValueStrategy] = None
    interpolation_method: str = "linear"  # For interpolation strategies
    knn_neighbors: int = 5  # For KNN imputation
    polynomial_order: int = 2  # For polynomial interpolation


@dataclass
class DataQualityAssessment:
    """Assessment of data quality for a column or dataset."""
    column_name: str
    total_rows: int
    missing_count: int
    missing_percentage: float
    completeness_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    quality_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MissingValueReport:
    """Report on missing value handling operations."""
    original_missing_count: int
    final_missing_count: int
    strategies_applied: Dict[str, str]
    rows_removed: int
    values_imputed: int
    quality_improvement: float
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: int = 0


@dataclass
class MissingValueConfig:
    """Configuration for missing value handling system."""
    default_policies: Dict[str, MissingValuePolicy] = field(default_factory=dict)
    missing_indicators: List[str] = field(default_factory=lambda: [
        '', 'null', 'NULL', 'None', 'N/A', 'n/a', 'NA', 'na',
        '-', '--', '?', 'unknown', 'Unknown', 'UNKNOWN', 'NaN',
        'nan', '#N/A', '#NULL!', '#DIV/0!', 'nil', 'NIL'
    ])
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'completeness': 0.8,
        'consistency': 0.9,
        'validity': 0.85,
        'uniqueness': 0.95
    })
    auto_detect_outliers: bool = True
    outlier_z_threshold: float = 3.0
    max_missing_percentage: float = 0.7  # Remove columns with >70% missing
    enable_advanced_imputation: bool = True


class DataQualityAssessor:
    """Assesses data quality and provides recommendations."""
    
    def __init__(self, config: MissingValueConfig):
        self.config = config
    
    def assess_column_quality(self, series: pd.Series, column_name: str, 
                            data_type: DataType) -> DataQualityAssessment:
        """
        Assess data quality for a single column.
        
        Args:
            series: Pandas Series to assess
            column_name: Name of the column
            data_type: Detected data type
            
        Returns:
            DataQualityAssessment object
        """
        total_rows = len(series)
        missing_count = series.isna().sum()
        missing_percentage = missing_count / total_rows if total_rows > 0 else 0.0
        
        # Calculate quality scores
        completeness_score = 1.0 - missing_percentage
        consistency_score = self._calculate_consistency_score(series, data_type)
        validity_score = self._calculate_validity_score(series, data_type)
        uniqueness_score = self._calculate_uniqueness_score(series)
        
        # Identify quality issues
        quality_issues = []
        recommendations = []
        
        if missing_percentage > self.config.quality_thresholds.get('completeness', 0.8):
            quality_issues.append(f"High missing data rate: {missing_percentage:.1%}")
            recommendations.append("Consider data collection improvement or imputation")
        
        if consistency_score < self.config.quality_thresholds.get('consistency', 0.9):
            quality_issues.append("Inconsistent data formatting detected")
            recommendations.append("Apply data standardization and cleaning")
        
        if validity_score < self.config.quality_thresholds.get('validity', 0.85):
            quality_issues.append("Invalid values detected for data type")
            recommendations.append("Review and correct invalid entries")
        
        if uniqueness_score < self.config.quality_thresholds.get('uniqueness', 0.95):
            quality_issues.append("Potential duplicate values detected")
            recommendations.append("Review for duplicate entries")
        
        return DataQualityAssessment(
            column_name=column_name,
            total_rows=total_rows,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            uniqueness_score=uniqueness_score,
            quality_issues=quality_issues,
            recommendations=recommendations
        )
    
    def _calculate_consistency_score(self, series: pd.Series, data_type: DataType) -> float:
        """Calculate consistency score based on data type patterns."""
        if series.empty or series.isna().all():
            return 0.0
        
        valid_series = series.dropna()
        if valid_series.empty:
            return 0.0
        
        try:
            if data_type == DataType.NUMBER:
                # Check if numeric values are consistently formatted
                numeric_count = pd.to_numeric(valid_series, errors='coerce').notna().sum()
                return numeric_count / len(valid_series)
            
            elif data_type == DataType.DATE:
                # Check if date values follow consistent patterns
                date_count = pd.to_datetime(valid_series, errors='coerce').notna().sum()
                return date_count / len(valid_series)
            
            elif data_type == DataType.BOOLEAN:
                # Check if boolean values are consistently formatted
                bool_patterns = ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n']
                consistent_count = sum(1 for val in valid_series 
                                     if str(val).lower().strip() in bool_patterns)
                return consistent_count / len(valid_series)
            
            else:
                # For text, check for consistent casing and formatting
                if len(valid_series) <= 1:
                    return 1.0
                
                # Simple consistency check based on string patterns
                str_series = valid_series.astype(str)
                unique_patterns = len(set(str_series.str.len()))
                max_patterns = min(10, len(str_series))  # Reasonable threshold
                return 1.0 - (unique_patterns / max_patterns)
        
        except Exception as e:
            logger.warning(f"Error calculating consistency score: {e}")
            return 0.5  # Default moderate score
    
    def _calculate_validity_score(self, series: pd.Series, data_type: DataType) -> float:
        """Calculate validity score based on data type constraints."""
        if series.empty or series.isna().all():
            return 0.0
        
        valid_series = series.dropna()
        if valid_series.empty:
            return 0.0
        
        try:
            if data_type == DataType.NUMBER:
                # Check if values can be converted to numbers
                numeric_series = pd.to_numeric(valid_series, errors='coerce')
                valid_count = numeric_series.notna().sum()
                return valid_count / len(valid_series)
            
            elif data_type == DataType.CURRENCY:
                # Check if values follow currency patterns
                currency_pattern = r'^[\$€£¥₹]?\s*[\d,]+\.?\d*$'
                valid_count = sum(1 for val in valid_series 
                                if pd.notna(val) and 
                                re.match(currency_pattern, str(val).strip()))
                return valid_count / len(valid_series)
            
            elif data_type == DataType.PERCENTAGE:
                # Check if values are valid percentages
                valid_count = 0
                for val in valid_series:
                    val_str = str(val).strip()
                    if val_str.endswith('%'):
                        try:
                            float(val_str[:-1])
                            valid_count += 1
                        except ValueError:
                            continue
                    else:
                        try:
                            num_val = float(val_str)
                            if 0 <= num_val <= 1:  # Decimal percentage
                                valid_count += 1
                        except ValueError:
                            continue
                return valid_count / len(valid_series)
            
            else:
                # For other types, assume valid if not empty
                return 1.0
        
        except Exception as e:
            logger.warning(f"Error calculating validity score: {e}")
            return 0.5
    
    def _calculate_uniqueness_score(self, series: pd.Series) -> float:
        """Calculate uniqueness score (1 - duplicate rate)."""
        if series.empty:
            return 1.0
        
        try:
            total_count = len(series)
            unique_count = series.nunique()
            return unique_count / total_count if total_count > 0 else 1.0
        except (TypeError, ValueError) as e:
            # Handle unhashable types (like lists, dicts) by converting to string
            logger.warning(f"Error calculating uniqueness, using string conversion: {e}")
            try:
                str_series = series.astype(str)
                total_count = len(str_series)
                unique_count = str_series.nunique()
                return unique_count / total_count if total_count > 0 else 1.0
            except Exception as e2:
                logger.warning(f"Failed to calculate uniqueness score: {e2}")
                return 0.5  # Default moderate score
    
    def assess_dataset_quality(self, dataframe: pd.DataFrame, 
                             data_types: Dict[str, DataType]) -> Dict[str, DataQualityAssessment]:
        """
        Assess data quality for entire dataset.
        
        Args:
            dataframe: DataFrame to assess
            data_types: Mapping of column names to data types
            
        Returns:
            Dictionary mapping column names to quality assessments
        """
        assessments = {}
        
        for column in dataframe.columns:
            data_type = data_types.get(column, DataType.TEXT)
            assessment = self.assess_column_quality(
                dataframe[column], column, data_type
            )
            assessments[column] = assessment
        
        return assessments


class MissingValueHandler:
    """Enhanced missing value handling system."""
    
    def __init__(self, config: MissingValueConfig = None):
        self.config = config or MissingValueConfig()
        self.quality_assessor = DataQualityAssessor(self.config)
        self._strategy_functions = self._initialize_strategy_functions()
    
    def _initialize_strategy_functions(self) -> Dict[MissingValueStrategy, Callable]:
        """Initialize mapping of strategies to implementation functions."""
        return {
            MissingValueStrategy.REMOVE: self._apply_remove_strategy,
            MissingValueStrategy.MEAN: self._apply_mean_strategy,
            MissingValueStrategy.MEDIAN: self._apply_median_strategy,
            MissingValueStrategy.MODE: self._apply_mode_strategy,
            MissingValueStrategy.ZERO: self._apply_zero_strategy,
            MissingValueStrategy.EMPTY_STRING: self._apply_empty_string_strategy,
            MissingValueStrategy.FALSE: self._apply_false_strategy,
            MissingValueStrategy.FORWARD_FILL: self._apply_forward_fill_strategy,
            MissingValueStrategy.BACKWARD_FILL: self._apply_backward_fill_strategy,
            MissingValueStrategy.INTERPOLATE_LINEAR: self._apply_linear_interpolation,
            MissingValueStrategy.INTERPOLATE_POLYNOMIAL: self._apply_polynomial_interpolation,
            MissingValueStrategy.KNN_IMPUTE: self._apply_knn_imputation,
            MissingValueStrategy.CUSTOM_VALUE: self._apply_custom_value_strategy,
            MissingValueStrategy.STATISTICAL_OUTLIER: self._apply_statistical_outlier_strategy
        }
    
    def handle_missing_values(self, dataframe: pd.DataFrame, 
                            policies: Optional[Dict[str, MissingValuePolicy]] = None,
                            data_types: Optional[Dict[str, DataType]] = None) -> Tuple[pd.DataFrame, MissingValueReport]:
        """
        Handle missing values in DataFrame using configured policies.
        
        Args:
            dataframe: Input DataFrame
            policies: Column-specific policies (optional)
            data_types: Data types for columns (optional)
            
        Returns:
            Tuple of (processed DataFrame, processing report)
        """
        import time
        start_time = time.time()
        
        result_df = dataframe.copy()
        original_missing = result_df.isna().sum().sum()
        
        # Standardize missing value indicators
        result_df = self._standardize_missing_indicators(result_df)
        
        # Use provided policies or generate default ones
        if policies is None:
            policies = self._generate_default_policies(result_df, data_types or {})
        
        strategies_applied = {}
        warnings_list = []
        rows_removed = 0
        values_imputed = 0
        
        # Apply policies column by column
        for column in result_df.columns:
            if column in policies:
                policy = policies[column]
                try:
                    original_rows = len(result_df)
                    original_missing_col = result_df[column].isna().sum()
                    
                    result_df, column_report = self._apply_policy_to_column(
                        result_df, column, policy
                    )
                    
                    strategies_applied[column] = policy.strategy.value
                    rows_removed += original_rows - len(result_df)
                    values_imputed += original_missing_col - result_df[column].isna().sum()
                    
                    if column_report.get('warnings'):
                        warnings_list.extend(column_report['warnings'])
                
                except Exception as e:
                    logger.error(f"Error applying policy to column '{column}': {e}")
                    warnings_list.append(f"Failed to process column '{column}': {str(e)}")
        
        final_missing = result_df.isna().sum().sum()
        quality_improvement = (original_missing - final_missing) / original_missing if original_missing > 0 else 0.0
        
        processing_time = int((time.time() - start_time) * 1000)
        
        report = MissingValueReport(
            original_missing_count=original_missing,
            final_missing_count=final_missing,
            strategies_applied=strategies_applied,
            rows_removed=rows_removed,
            values_imputed=values_imputed,
            quality_improvement=quality_improvement,
            warnings=warnings_list,
            processing_time_ms=processing_time
        )
        
        return result_df, report  
  
    def _standardize_missing_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Standardize missing value indicators to NaN."""
        result_df = dataframe.copy()
        
        for column in result_df.columns:
            # Convert to string for pattern matching
            str_series = result_df[column].astype(str)
            
            # Replace missing indicators with NaN
            for indicator in self.config.missing_indicators:
                mask = str_series == indicator
                result_df.loc[mask, column] = np.nan
        
        return result_df
    
    def _generate_default_policies(self, dataframe: pd.DataFrame, 
                                 data_types: Dict[str, DataType]) -> Dict[str, MissingValuePolicy]:
        """Generate default policies based on data types and quality assessment."""
        policies = {}
        
        for column in dataframe.columns:
            data_type = data_types.get(column, DataType.TEXT)
            
            # Assess column quality to inform policy selection
            assessment = self.quality_assessor.assess_column_quality(
                dataframe[column], column, data_type
            )
            
            # Select strategy based on data type and quality
            strategy = self._select_default_strategy(data_type, assessment)
            
            policies[column] = MissingValuePolicy(
                column_name=column,
                data_type=data_type,
                strategy=strategy,
                fallback_strategy=MissingValueStrategy.REMOVE
            )
        
        return policies
    
    def _select_default_strategy(self, data_type: DataType, 
                               assessment: DataQualityAssessment) -> MissingValueStrategy:
        """Select default strategy based on data type and quality assessment."""
        missing_percentage = assessment.missing_percentage
        
        # If too much missing data, recommend removal
        if missing_percentage > self.config.max_missing_percentage:
            return MissingValueStrategy.REMOVE
        
        # Strategy selection based on data type
        if data_type == DataType.NUMBER:
            if missing_percentage < 0.1:
                return MissingValueStrategy.MEAN
            elif missing_percentage < 0.3:
                return MissingValueStrategy.MEDIAN
            else:
                return MissingValueStrategy.INTERPOLATE_LINEAR
        
        elif data_type == DataType.CURRENCY:
            return MissingValueStrategy.ZERO if missing_percentage < 0.2 else MissingValueStrategy.MEDIAN
        
        elif data_type == DataType.PERCENTAGE:
            return MissingValueStrategy.ZERO if missing_percentage < 0.2 else MissingValueStrategy.MEAN
        
        elif data_type == DataType.DATE:
            if missing_percentage < 0.1:
                return MissingValueStrategy.FORWARD_FILL
            else:
                return MissingValueStrategy.REMOVE
        
        elif data_type == DataType.BOOLEAN:
            return MissingValueStrategy.MODE if missing_percentage < 0.3 else MissingValueStrategy.FALSE
        
        else:  # TEXT
            if missing_percentage < 0.2:
                return MissingValueStrategy.MODE
            elif missing_percentage < 0.4:
                return MissingValueStrategy.EMPTY_STRING
            else:
                return MissingValueStrategy.REMOVE
    
    def _apply_policy_to_column(self, dataframe: pd.DataFrame, column: str, 
                              policy: MissingValuePolicy) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply missing value policy to a specific column."""
        result_df = dataframe.copy()
        report = {'warnings': []}
        
        # Check if strategy is applicable
        missing_count = result_df[column].isna().sum()
        if missing_count == 0:
            return result_df, report
        
        missing_percentage = missing_count / len(result_df)
        
        # Apply threshold check
        if missing_percentage > policy.threshold and policy.fallback_strategy:
            report['warnings'].append(
                f"Column '{column}' missing percentage ({missing_percentage:.1%}) "
                f"exceeds threshold ({policy.threshold:.1%}), using fallback strategy"
            )
            strategy = policy.fallback_strategy
        else:
            strategy = policy.strategy
        
        # Apply the strategy
        try:
            strategy_func = self._strategy_functions.get(strategy)
            if strategy_func:
                result_df = strategy_func(result_df, column, policy)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        except Exception as e:
            logger.error(f"Error applying strategy {strategy} to column {column}: {e}")
            report['warnings'].append(f"Strategy failed, removing missing values: {str(e)}")
            # Fallback to removal
            result_df = self._apply_remove_strategy(result_df, column, policy)
        
        return result_df, report
    
    # Strategy implementation methods
    def _apply_remove_strategy(self, dataframe: pd.DataFrame, column: str, 
                             policy: MissingValuePolicy) -> pd.DataFrame:
        """Remove rows with missing values in the specified column."""
        return dataframe.dropna(subset=[column])
    
    def _apply_mean_strategy(self, dataframe: pd.DataFrame, column: str, 
                           policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values with column mean."""
        result_df = dataframe.copy()
        numeric_series = pd.to_numeric(result_df[column], errors='coerce')
        mean_value = numeric_series.mean()
        
        if pd.notna(mean_value):
            result_df[column] = result_df[column].fillna(mean_value)
        
        return result_df
    
    def _apply_median_strategy(self, dataframe: pd.DataFrame, column: str, 
                             policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values with column median."""
        result_df = dataframe.copy()
        numeric_series = pd.to_numeric(result_df[column], errors='coerce')
        median_value = numeric_series.median()
        
        if pd.notna(median_value):
            result_df[column] = result_df[column].fillna(median_value)
        
        return result_df
    
    def _apply_mode_strategy(self, dataframe: pd.DataFrame, column: str, 
                           policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values with column mode."""
        result_df = dataframe.copy()
        mode_values = result_df[column].mode()
        
        if not mode_values.empty:
            result_df[column] = result_df[column].fillna(mode_values.iloc[0])
        
        return result_df
    
    def _apply_zero_strategy(self, dataframe: pd.DataFrame, column: str, 
                           policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values with zero."""
        result_df = dataframe.copy()
        result_df[column] = result_df[column].fillna(0)
        return result_df
    
    def _apply_empty_string_strategy(self, dataframe: pd.DataFrame, column: str, 
                                   policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values with empty string."""
        result_df = dataframe.copy()
        result_df[column] = result_df[column].fillna('')
        return result_df
    
    def _apply_false_strategy(self, dataframe: pd.DataFrame, column: str, 
                            policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values with False."""
        result_df = dataframe.copy()
        result_df[column] = result_df[column].fillna(False)
        return result_df
    
    def _apply_forward_fill_strategy(self, dataframe: pd.DataFrame, column: str, 
                                   policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values using forward fill."""
        result_df = dataframe.copy()
        result_df[column] = result_df[column].ffill()
        return result_df
    
    def _apply_backward_fill_strategy(self, dataframe: pd.DataFrame, column: str, 
                                    policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values using backward fill."""
        result_df = dataframe.copy()
        result_df[column] = result_df[column].bfill()
        return result_df
    
    def _apply_linear_interpolation(self, dataframe: pd.DataFrame, column: str, 
                                  policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values using linear interpolation."""
        result_df = dataframe.copy()
        numeric_series = pd.to_numeric(result_df[column], errors='coerce')
        
        if numeric_series.notna().sum() >= 2:  # Need at least 2 points for interpolation
            interpolated = numeric_series.interpolate(method='linear')
            result_df[column] = interpolated
        
        return result_df
    
    def _apply_polynomial_interpolation(self, dataframe: pd.DataFrame, column: str, 
                                      policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values using polynomial interpolation."""
        result_df = dataframe.copy()
        numeric_series = pd.to_numeric(result_df[column], errors='coerce')
        
        min_points = policy.polynomial_order + 1
        if numeric_series.notna().sum() >= min_points:
            try:
                interpolated = numeric_series.interpolate(
                    method='polynomial', 
                    order=policy.polynomial_order
                )
                result_df[column] = interpolated
            except Exception as e:
                logger.warning(f"Polynomial interpolation failed, using linear: {e}")
                # Fallback to linear interpolation
                interpolated = numeric_series.interpolate(method='linear')
                result_df[column] = interpolated
        
        return result_df
    
    def _apply_knn_imputation(self, dataframe: pd.DataFrame, column: str, 
                            policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values using KNN imputation."""
        if not self.config.enable_advanced_imputation:
            logger.warning("Advanced imputation disabled, using mean instead")
            return self._apply_mean_strategy(dataframe, column, policy)
        
        result_df = dataframe.copy()
        
        try:
            # Select numeric columns for KNN imputation
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if column not in numeric_columns:
                # Convert target column to numeric if possible
                numeric_series = pd.to_numeric(result_df[column], errors='coerce')
                if numeric_series.notna().sum() > 0:
                    result_df[column] = numeric_series
                    numeric_columns.append(column)
            
            if len(numeric_columns) >= 2:  # Need at least 2 columns for KNN
                imputer = KNNImputer(n_neighbors=policy.knn_neighbors)
                
                # Apply KNN imputation
                imputed_data = imputer.fit_transform(result_df[numeric_columns])
                
                # Update the target column
                column_index = numeric_columns.index(column)
                result_df[column] = imputed_data[:, column_index]
            else:
                logger.warning("Insufficient numeric columns for KNN, using mean instead")
                return self._apply_mean_strategy(result_df, column, policy)
        
        except Exception as e:
            logger.warning(f"KNN imputation failed, using mean instead: {e}")
            return self._apply_mean_strategy(result_df, column, policy)
        
        return result_df
    
    def _apply_custom_value_strategy(self, dataframe: pd.DataFrame, column: str, 
                                   policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values with custom value."""
        result_df = dataframe.copy()
        
        if policy.custom_value is not None:
            result_df[column] = result_df[column].fillna(policy.custom_value)
        else:
            logger.warning(f"No custom value specified for column {column}, using zero")
            result_df[column] = result_df[column].fillna(0)
        
        return result_df
    
    def _apply_statistical_outlier_strategy(self, dataframe: pd.DataFrame, column: str, 
                                          policy: MissingValuePolicy) -> pd.DataFrame:
        """Fill missing values while handling statistical outliers."""
        result_df = dataframe.copy()
        numeric_series = pd.to_numeric(result_df[column], errors='coerce')
        
        if numeric_series.notna().sum() < 3:
            # Not enough data for outlier detection
            return self._apply_median_strategy(result_df, column, policy)
        
        # Calculate z-scores to identify outliers
        z_scores = np.abs(stats.zscore(numeric_series.dropna()))
        outlier_threshold = self.config.outlier_z_threshold
        
        # Use median of non-outlier values for imputation
        non_outlier_values = numeric_series.dropna()[z_scores < outlier_threshold]
        
        if len(non_outlier_values) > 0:
            fill_value = non_outlier_values.median()
            result_df[column] = result_df[column].fillna(fill_value)
        else:
            # All values are outliers, use overall median
            result_df[column] = result_df[column].fillna(numeric_series.median())
        
        return result_df
    
    def create_user_policy(self, column_name: str, data_type: DataType, 
                          strategy: MissingValueStrategy, **kwargs) -> MissingValuePolicy:
        """
        Create a user-configurable missing value policy.
        
        Args:
            column_name: Name of the column
            data_type: Data type of the column
            strategy: Strategy to apply
            **kwargs: Additional policy parameters
            
        Returns:
            MissingValuePolicy object
        """
        return MissingValuePolicy(
            column_name=column_name,
            data_type=data_type,
            strategy=strategy,
            custom_value=kwargs.get('custom_value'),
            threshold=kwargs.get('threshold', 0.5),
            fallback_strategy=kwargs.get('fallback_strategy'),
            interpolation_method=kwargs.get('interpolation_method', 'linear'),
            knn_neighbors=kwargs.get('knn_neighbors', 5),
            polynomial_order=kwargs.get('polynomial_order', 2)
        )
    
    def get_quality_assessment(self, dataframe: pd.DataFrame, 
                             data_types: Dict[str, DataType]) -> Dict[str, DataQualityAssessment]:
        """
        Get comprehensive data quality assessment.
        
        Args:
            dataframe: DataFrame to assess
            data_types: Data types for columns
            
        Returns:
            Dictionary of quality assessments by column
        """
        return self.quality_assessor.assess_dataset_quality(dataframe, data_types)
    
    def recommend_strategies(self, dataframe: pd.DataFrame, 
                           data_types: Dict[str, DataType]) -> Dict[str, MissingValueStrategy]:
        """
        Recommend missing value strategies based on data analysis.
        
        Args:
            dataframe: DataFrame to analyze
            data_types: Data types for columns
            
        Returns:
            Dictionary mapping column names to recommended strategies
        """
        recommendations = {}
        quality_assessments = self.get_quality_assessment(dataframe, data_types)
        
        for column, assessment in quality_assessments.items():
            data_type = data_types.get(column, DataType.TEXT)
            strategy = self._select_default_strategy(data_type, assessment)
            recommendations[column] = strategy
        
        return recommendations


# Convenience functions for direct usage
def handle_missing_values_simple(dataframe: pd.DataFrame, 
                                strategy: str = 'auto',
                                data_types: Optional[Dict[str, DataType]] = None) -> pd.DataFrame:
    """
    Simple convenience function for handling missing values.
    
    Args:
        dataframe: Input DataFrame
        strategy: Strategy to use ('auto', 'remove', 'mean', 'median', 'mode')
        data_types: Optional data types mapping
        
    Returns:
        DataFrame with missing values handled
    """
    handler = MissingValueHandler()
    
    if strategy == 'auto':
        result_df, _ = handler.handle_missing_values(dataframe, data_types=data_types)
    else:
        # Apply single strategy to all columns
        try:
            strategy_enum = MissingValueStrategy(strategy)
            policies = {}
            
            for column in dataframe.columns:
                data_type = data_types.get(column, DataType.TEXT) if data_types else DataType.TEXT
                policies[column] = MissingValuePolicy(
                    column_name=column,
                    data_type=data_type,
                    strategy=strategy_enum
                )
            
            result_df, _ = handler.handle_missing_values(dataframe, policies, data_types)
        
        except ValueError:
            logger.error(f"Unknown strategy: {strategy}")
            result_df = dataframe.dropna()  # Fallback to removal
    
    return result_df


def assess_data_quality(dataframe: pd.DataFrame, 
                       data_types: Optional[Dict[str, DataType]] = None) -> Dict[str, DataQualityAssessment]:
    """
    Convenience function for data quality assessment.
    
    Args:
        dataframe: DataFrame to assess
        data_types: Optional data types mapping
        
    Returns:
        Dictionary of quality assessments by column
    """
    config = MissingValueConfig()
    assessor = DataQualityAssessor(config)
    
    if data_types is None:
        data_types = {col: DataType.TEXT for col in dataframe.columns}
    
    return assessor.assess_dataset_quality(dataframe, data_types)