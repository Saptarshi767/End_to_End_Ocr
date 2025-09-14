"""
Schema detection and management system for OCR extracted table data.

This module provides comprehensive schema inference, validation, compatibility checking,
and versioning capabilities for structured data extracted from documents.
"""

import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import re
from collections import Counter, defaultdict

from ..core.models import DataType, DataSchema, ColumnInfo, ValidationResult
from ..core.interfaces import DataCleaningInterface
from ..core.exceptions import DataProcessingError


logger = logging.getLogger(__name__)


class SchemaCompatibility(Enum):
    """Schema compatibility levels."""
    IDENTICAL = "identical"
    COMPATIBLE = "compatible"
    MINOR_CHANGES = "minor_changes"
    MAJOR_CHANGES = "major_changes"
    INCOMPATIBLE = "incompatible"


class SchemaChangeType(Enum):
    """Types of schema changes."""
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_RENAMED = "column_renamed"
    TYPE_CHANGED = "type_changed"
    NULLABLE_CHANGED = "nullable_changed"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"


@dataclass
class SchemaChange:
    """Represents a change between two schemas."""
    change_type: SchemaChangeType
    column_name: str
    old_value: Any = None
    new_value: Any = None
    description: str = ""
    severity: str = "minor"  # minor, major, breaking


@dataclass
class SchemaVersion:
    """Represents a versioned schema."""
    version: str
    schema: DataSchema
    created_at: datetime
    description: str = ""
    hash: str = ""
    parent_version: Optional[str] = None
    changes: List[SchemaChange] = field(default_factory=list)


@dataclass
class SchemaCompatibilityResult:
    """Result of schema compatibility check."""
    compatibility: SchemaCompatibility
    changes: List[SchemaChange]
    can_migrate: bool
    migration_strategy: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class SchemaInferenceConfig:
    """Configuration for schema inference."""
    sample_size: int = 1000
    confidence_threshold: float = 0.8
    null_threshold: float = 0.95  # Column is nullable if >95% values are non-null
    unique_threshold: float = 0.95  # Column is unique if >95% values are unique
    categorical_threshold: int = 20  # Max unique values for categorical detection
    date_formats: List[str] = field(default_factory=lambda: [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%m-%d-%Y', '%B %d, %Y', '%d %B %Y',
        '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S'
    ])
    currency_symbols: List[str] = field(default_factory=lambda: ['$', '€', '£', '¥', '₹'])


class SchemaInferenceEngine:
    """Engine for inferring data schemas from structured data."""
    
    def __init__(self, config: SchemaInferenceConfig = None):
        self.config = config or SchemaInferenceConfig()
    
    def infer_schema(self, dataframe: pd.DataFrame, table_name: str = "table") -> DataSchema:
        """
        Infer comprehensive schema from a pandas DataFrame.
        
        Args:
            dataframe: Input DataFrame to analyze
            table_name: Name of the table for schema identification
            
        Returns:
            DataSchema object with inferred column information
        """
        if dataframe.empty:
            return DataSchema(
                columns=[],
                row_count=0,
                data_types={},
                sample_data={},
                created_at=datetime.now()
            )
        
        # Sample data for analysis if dataset is large
        sample_df = self._sample_dataframe(dataframe)
        
        # Infer column information
        columns = []
        data_types = {}
        sample_data = {}
        
        for column_name in dataframe.columns:
            column_info = self._infer_column_info(
                dataframe[column_name], 
                sample_df[column_name], 
                column_name
            )
            columns.append(column_info)
            data_types[column_name] = column_info.data_type
            sample_data[column_name] = column_info.sample_values
        
        schema = DataSchema(
            columns=columns,
            row_count=len(dataframe),
            data_types=data_types,
            sample_data=sample_data,
            created_at=datetime.now()
        )
        
        # Add additional metadata
        schema.metadata = {
            'table_name': table_name,
            'inference_config': asdict(self.config),
            'column_count': len(columns),
            'total_cells': len(dataframe) * len(dataframe.columns),
            'memory_usage': dataframe.memory_usage(deep=True).sum()
        }
        
        return schema
    
    def _sample_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Sample DataFrame for analysis if it's too large."""
        if len(dataframe) <= self.config.sample_size:
            return dataframe
        
        # Use stratified sampling to maintain data distribution
        return dataframe.sample(n=self.config.sample_size, random_state=42)
    
    def _infer_column_info(self, full_series: pd.Series, sample_series: pd.Series, column_name: str) -> ColumnInfo:
        """
        Infer detailed information about a column.
        
        Args:
            full_series: Complete column data
            sample_series: Sampled column data for analysis
            column_name: Name of the column
            
        Returns:
            ColumnInfo object with inferred metadata
        """
        # Basic statistics
        total_count = len(full_series)
        null_count = full_series.isnull().sum()
        non_null_count = total_count - null_count
        
        # Determine if column is nullable (if more than 5% are null, consider nullable)
        nullable = (null_count / total_count) > (1 - self.config.null_threshold) if total_count > 0 else True
        
        # Get unique value count
        unique_values = full_series.nunique()
        
        # Infer data type from sample
        data_type = self._infer_data_type(sample_series.dropna())
        
        # Get sample values (non-null) and ensure JSON serializable
        sample_values = []
        for val in sample_series.dropna().head(5):
            if isinstance(val, (np.integer, np.floating)):
                sample_values.append(float(val))
            elif isinstance(val, np.bool_):
                sample_values.append(bool(val))
            else:
                sample_values.append(str(val))
        
        # Create column info
        column_info = ColumnInfo(
            name=column_name,
            data_type=data_type,
            nullable=nullable,
            unique_values=unique_values,
            sample_values=sample_values
        )
        
        # Add extended metadata
        column_info.metadata = {
            'total_count': total_count,
            'null_count': null_count,
            'non_null_count': non_null_count,
            'unique_values': unique_values,
            'null_percentage': (null_count / total_count * 100) if total_count > 0 else 0,
            'unique_percentage': (unique_values / non_null_count * 100) if non_null_count > 0 else 0,
            'is_primary_key_candidate': unique_values == non_null_count and non_null_count > 0,
            'is_categorical': unique_values <= self.config.categorical_threshold and unique_values < non_null_count,
            'memory_usage': full_series.memory_usage(deep=True)
        }
        
        # Add type-specific metadata
        if data_type == DataType.NUMBER:
            self._add_numeric_metadata(column_info, sample_series.dropna())
        elif data_type == DataType.TEXT:
            self._add_text_metadata(column_info, sample_series.dropna())
        elif data_type == DataType.DATE:
            self._add_date_metadata(column_info, sample_series.dropna())
        
        return column_info
    
    def _infer_data_type(self, series: pd.Series) -> DataType:
        """
        Infer the most appropriate data type for a series.
        
        Args:
            series: Non-null pandas Series to analyze
            
        Returns:
            Inferred DataType
        """
        if series.empty:
            return DataType.TEXT
        
        # Convert to string for pattern matching
        str_series = series.astype(str).str.strip()
        
        # Test in order of specificity
        type_tests = [
            (DataType.BOOLEAN, self._is_boolean_type),
            (DataType.CURRENCY, self._is_currency_type),
            (DataType.PERCENTAGE, self._is_percentage_type),
            (DataType.DATE, self._is_date_type),
            (DataType.NUMBER, self._is_numeric_type),
        ]
        
        for data_type, test_func in type_tests:
            if test_func(str_series):
                return data_type
        
        return DataType.TEXT
    
    def _is_boolean_type(self, str_series: pd.Series) -> bool:
        """Check if series contains boolean values."""
        boolean_values = {
            'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 
            'on', 'off', 'enabled', 'disabled', 'active', 'inactive'
        }
        
        unique_lower = set(str_series.str.lower().unique())
        return len(unique_lower) <= 2 and unique_lower.issubset(boolean_values)
    
    def _is_currency_type(self, str_series: pd.Series) -> bool:
        """Check if series contains currency values."""
        currency_pattern = r'^[' + ''.join(re.escape(s) for s in self.config.currency_symbols) + r']\s*[\d,]+\.?\d*$'
        matches = str_series.str.match(currency_pattern, na=False).sum()
        return (matches / len(str_series)) >= self.config.confidence_threshold
    
    def _is_percentage_type(self, str_series: pd.Series) -> bool:
        """Check if series contains percentage values."""
        percentage_pattern = r'^[\d,]+\.?\d*\s*%$'
        matches = str_series.str.match(percentage_pattern, na=False).sum()
        return (matches / len(str_series)) >= self.config.confidence_threshold
    
    def _is_date_type(self, str_series: pd.Series) -> bool:
        """Check if series contains date values."""
        for date_format in self.config.date_formats:
            try:
                parsed_count = 0
                for value in str_series.head(min(50, len(str_series))):
                    try:
                        datetime.strptime(value, date_format)
                        parsed_count += 1
                    except ValueError:
                        continue
                
                if parsed_count / min(50, len(str_series)) >= self.config.confidence_threshold:
                    return True
            except Exception:
                continue
        
        return False
    
    def _is_numeric_type(self, str_series: pd.Series) -> bool:
        """Check if series contains numeric values."""
        try:
            # Check if values look like identifiers (all same length digits)
            if len(str_series) > 0:
                first_val = str(str_series.iloc[0])
                if first_val.isdigit() and len(first_val) >= 4:
                    # Check if all values have same length (likely identifiers like zip codes)
                    lengths = [len(str(val)) for val in str_series]
                    if len(set(lengths)) == 1 and all(str(val).isdigit() for val in str_series):
                        return False  # Treat as text (identifiers)
            
            # Try to convert to numeric, counting successful conversions
            numeric_series = pd.to_numeric(str_series.str.replace(',', ''), errors='coerce')
            non_null_count = numeric_series.notna().sum()
            
            # Additional check: if all values are integers and look like codes, treat as text
            if non_null_count == len(str_series):
                # Check if values look like codes/IDs (all integers, similar patterns)
                try:
                    int_values = [int(float(val)) for val in str_series if pd.notna(val)]
                    if all(val == int(val) for val in numeric_series.dropna()):
                        # Check for patterns that suggest identifiers
                        str_lengths = [len(str(val).replace('.0', '')) for val in str_series]
                        if len(set(str_lengths)) == 1 and str_lengths[0] >= 4:
                            return False  # Likely identifiers
                except:
                    pass
            
            return (non_null_count / len(str_series)) >= self.config.confidence_threshold
        except Exception:
            return False
    
    def _add_numeric_metadata(self, column_info: ColumnInfo, series: pd.Series):
        """Add numeric-specific metadata to column info."""
        try:
            numeric_series = pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')
            numeric_series = numeric_series.dropna()
            
            if not numeric_series.empty:
                column_info.metadata.update({
                    'min_value': float(numeric_series.min()),
                    'max_value': float(numeric_series.max()),
                    'mean_value': float(numeric_series.mean()),
                    'median_value': float(numeric_series.median()),
                    'std_deviation': float(numeric_series.std()),
                    'has_decimals': any('.' in str(val) for val in series.head(100)),
                    'is_integer': all(float(val).is_integer() for val in numeric_series.head(100) if pd.notna(val))
                })
        except Exception as e:
            logger.debug(f"Error adding numeric metadata: {e}")
    
    def _add_text_metadata(self, column_info: ColumnInfo, series: pd.Series):
        """Add text-specific metadata to column info."""
        try:
            str_series = series.astype(str)
            lengths = str_series.str.len()
            
            column_info.metadata.update({
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'avg_length': float(lengths.mean()),
                'contains_special_chars': any(bool(re.search(r'[^a-zA-Z0-9\s]', val)) for val in str_series.head(100)),
                'is_uppercase': all(val.isupper() for val in str_series.head(100) if val.strip()),
                'is_lowercase': all(val.islower() for val in str_series.head(100) if val.strip()),
                'common_patterns': self._find_common_patterns(str_series.head(100))
            })
        except Exception as e:
            logger.debug(f"Error adding text metadata: {e}")
    
    def _add_date_metadata(self, column_info: ColumnInfo, series: pd.Series):
        """Add date-specific metadata to column info."""
        try:
            # Try to parse dates with different formats
            parsed_dates = []
            detected_format = None
            
            for date_format in self.config.date_formats:
                try:
                    sample_dates = []
                    for value in series.head(20):
                        try:
                            parsed_date = datetime.strptime(str(value), date_format)
                            sample_dates.append(parsed_date)
                        except ValueError:
                            continue
                    
                    if len(sample_dates) >= len(series.head(20)) * 0.8:  # 80% success rate
                        parsed_dates = sample_dates
                        detected_format = date_format
                        break
                except Exception:
                    continue
            
            if parsed_dates:
                column_info.metadata.update({
                    'detected_format': detected_format,
                    'min_date': min(parsed_dates).isoformat(),
                    'max_date': max(parsed_dates).isoformat(),
                    'date_range_days': (max(parsed_dates) - min(parsed_dates)).days,
                    'has_time_component': 'H' in detected_format if detected_format else False
                })
        except Exception as e:
            logger.debug(f"Error adding date metadata: {e}")
    
    def _find_common_patterns(self, str_series: pd.Series) -> List[str]:
        """Find common patterns in text data."""
        patterns = []
        
        # Check for common patterns
        pattern_tests = [
            (r'^[A-Z]{2,3}-\d+$', 'Code pattern (e.g., ABC-123)'),
            (r'^\d{3}-\d{2}-\d{4}$', 'SSN pattern'),
            (r'^\d{3}-\d{3}-\d{4}$', 'Phone pattern'),
            (r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', 'Email pattern'),
            (r'^\d{5}(-\d{4})?$', 'ZIP code pattern'),
            (r'^[A-Z][a-z]+ [A-Z][a-z]+$', 'Full name pattern')
        ]
        
        for pattern, description in pattern_tests:
            matches = str_series.str.match(pattern, na=False).sum()
            if matches / len(str_series) >= 0.5:  # 50% match rate
                patterns.append(description)
        
        return patterns


class SchemaValidator:
    """Validates schemas and checks compatibility between schema versions."""
    
    def validate_schema(self, schema: DataSchema) -> ValidationResult:
        """
        Validate a schema for consistency and completeness.
        
        Args:
            schema: DataSchema to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        # Check basic schema structure
        if not schema.columns:
            errors.append("Schema has no columns defined")
        
        if schema.row_count < 0:
            errors.append("Row count cannot be negative")
        
        # Validate each column
        column_names = set()
        for column in schema.columns:
            # Check for duplicate column names
            if column.name in column_names:
                errors.append(f"Duplicate column name: {column.name}")
            column_names.add(column.name)
            
            # Validate column properties
            if not column.name or not column.name.strip():
                errors.append("Column name cannot be empty")
            
            if column.unique_values < 0:
                errors.append(f"Column {column.name}: unique_values cannot be negative")
            
            if column.unique_values > schema.row_count:
                warnings.append(f"Column {column.name}: unique_values ({column.unique_values}) exceeds row_count ({schema.row_count})")
        
        # Check data_types consistency
        for column_name, data_type in schema.data_types.items():
            if column_name not in column_names:
                warnings.append(f"data_types contains column '{column_name}' not found in columns list")
        
        # Check sample_data consistency
        for column_name, sample_values in schema.sample_data.items():
            if column_name not in column_names:
                warnings.append(f"sample_data contains column '{column_name}' not found in columns list")
            
            if len(sample_values) > 10:
                warnings.append(f"Column {column_name}: sample_data has {len(sample_values)} values, consider limiting to 10")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=1.0 if len(errors) == 0 else 0.0
        )
    
    def check_compatibility(self, old_schema: DataSchema, new_schema: DataSchema) -> SchemaCompatibilityResult:
        """
        Check compatibility between two schemas and identify changes.
        
        Args:
            old_schema: Previous schema version
            new_schema: New schema version
            
        Returns:
            SchemaCompatibilityResult with compatibility assessment
        """
        changes = []
        warnings = []
        errors = []
        
        # Create column mappings
        old_columns = {col.name: col for col in old_schema.columns}
        new_columns = {col.name: col for col in new_schema.columns}
        
        # Check for removed columns
        for col_name in old_columns:
            if col_name not in new_columns:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.COLUMN_REMOVED,
                    column_name=col_name,
                    old_value=old_columns[col_name].data_type,
                    description=f"Column '{col_name}' was removed",
                    severity="major"
                ))
        
        # Check for added columns
        for col_name in new_columns:
            if col_name not in old_columns:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.COLUMN_ADDED,
                    column_name=col_name,
                    new_value=new_columns[col_name].data_type,
                    description=f"Column '{col_name}' was added",
                    severity="minor"
                ))
        
        # Check for changed columns
        for col_name in old_columns:
            if col_name in new_columns:
                old_col = old_columns[col_name]
                new_col = new_columns[col_name]
                
                # Check data type changes
                if old_col.data_type != new_col.data_type:
                    severity = self._assess_type_change_severity(old_col.data_type, new_col.data_type)
                    changes.append(SchemaChange(
                        change_type=SchemaChangeType.TYPE_CHANGED,
                        column_name=col_name,
                        old_value=old_col.data_type,
                        new_value=new_col.data_type,
                        description=f"Column '{col_name}' type changed from {old_col.data_type.value} to {new_col.data_type.value}",
                        severity=severity
                    ))
                
                # Check nullable changes
                if old_col.nullable != new_col.nullable:
                    severity = "major" if new_col.nullable and not old_col.nullable else "minor"
                    changes.append(SchemaChange(
                        change_type=SchemaChangeType.NULLABLE_CHANGED,
                        column_name=col_name,
                        old_value=old_col.nullable,
                        new_value=new_col.nullable,
                        description=f"Column '{col_name}' nullable changed from {old_col.nullable} to {new_col.nullable}",
                        severity=severity
                    ))
        
        # Determine overall compatibility
        compatibility = self._determine_compatibility(changes)
        
        # Determine if migration is possible
        can_migrate = compatibility != SchemaCompatibility.INCOMPATIBLE
        migration_strategy = self._suggest_migration_strategy(changes) if can_migrate else None
        
        return SchemaCompatibilityResult(
            compatibility=compatibility,
            changes=changes,
            can_migrate=can_migrate,
            migration_strategy=migration_strategy,
            warnings=warnings,
            errors=errors
        )
    
    def _assess_type_change_severity(self, old_type: DataType, new_type: DataType) -> str:
        """Assess the severity of a data type change."""
        # Define compatibility matrix
        compatible_changes = {
            DataType.TEXT: [DataType.TEXT],  # Text can only stay text
            DataType.NUMBER: [DataType.NUMBER, DataType.CURRENCY, DataType.PERCENTAGE],
            DataType.CURRENCY: [DataType.NUMBER, DataType.CURRENCY],
            DataType.PERCENTAGE: [DataType.NUMBER, DataType.PERCENTAGE],
            DataType.DATE: [DataType.DATE, DataType.TEXT],
            DataType.BOOLEAN: [DataType.BOOLEAN, DataType.TEXT]
        }
        
        if new_type in compatible_changes.get(old_type, []):
            return "minor"
        elif new_type == DataType.TEXT:  # Any type can become text, but it's a major change for structured data
            return "major"
        else:
            return "major"
    
    def _determine_compatibility(self, changes: List[SchemaChange]) -> SchemaCompatibility:
        """Determine overall compatibility based on changes."""
        if not changes:
            return SchemaCompatibility.IDENTICAL
        
        breaking_changes = [c for c in changes if c.severity == "breaking"]
        major_changes = [c for c in changes if c.severity == "major"]
        minor_changes = [c for c in changes if c.severity == "minor"]
        
        if breaking_changes:
            return SchemaCompatibility.INCOMPATIBLE
        elif major_changes:
            return SchemaCompatibility.MAJOR_CHANGES
        elif minor_changes:
            return SchemaCompatibility.MINOR_CHANGES
        else:
            return SchemaCompatibility.COMPATIBLE
    
    def _suggest_migration_strategy(self, changes: List[SchemaChange]) -> str:
        """Suggest a migration strategy based on schema changes."""
        strategies = []
        
        for change in changes:
            if change.change_type == SchemaChangeType.COLUMN_ADDED:
                strategies.append(f"Add default value for new column '{change.column_name}'")
            elif change.change_type == SchemaChangeType.COLUMN_REMOVED:
                strategies.append(f"Archive data from removed column '{change.column_name}'")
            elif change.change_type == SchemaChangeType.TYPE_CHANGED:
                strategies.append(f"Convert column '{change.column_name}' from {change.old_value.value} to {change.new_value.value}")
            elif change.change_type == SchemaChangeType.NULLABLE_CHANGED:
                if change.new_value:  # Became nullable
                    strategies.append(f"Allow null values in column '{change.column_name}'")
                else:  # Became non-nullable
                    strategies.append(f"Ensure no null values in column '{change.column_name}' before migration")
        
        return "; ".join(strategies) if strategies else "No migration required"


class SchemaVersionManager:
    """Manages schema versions and evolution over time."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("schemas")
        self.storage_path.mkdir(exist_ok=True)
        self.versions: Dict[str, SchemaVersion] = {}
        self.validator = SchemaValidator()
        self._load_versions()
    
    def create_version(self, schema: DataSchema, description: str = "", parent_version: Optional[str] = None) -> str:
        """
        Create a new schema version.
        
        Args:
            schema: DataSchema to version
            description: Description of changes
            parent_version: Parent version identifier
            
        Returns:
            Version identifier
        """
        # Validate schema first
        validation_result = self.validator.validate_schema(schema)
        if not validation_result.is_valid:
            raise DataProcessingError(f"Invalid schema: {', '.join(validation_result.errors)}")
        
        # Generate version identifier with microseconds to ensure uniqueness
        schema_hash = self._calculate_schema_hash(schema)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        version_id = f"v{timestamp}_{schema_hash[:8]}"
        
        # Ensure uniqueness by adding counter if needed
        counter = 1
        original_version_id = version_id
        while version_id in self.versions:
            version_id = f"{original_version_id}_{counter}"
            counter += 1
        
        # Check for changes if parent version exists
        changes = []
        if parent_version and parent_version in self.versions:
            parent_schema = self.versions[parent_version].schema
            compatibility_result = self.validator.check_compatibility(parent_schema, schema)
            changes = compatibility_result.changes
        
        # Create version
        version = SchemaVersion(
            version=version_id,
            schema=schema,
            created_at=datetime.now(),
            description=description,
            hash=schema_hash,
            parent_version=parent_version,
            changes=changes
        )
        
        # Store version
        self.versions[version_id] = version
        self._save_version(version)
        
        logger.info(f"Created schema version {version_id} with {len(changes)} changes")
        return version_id
    
    def get_version(self, version_id: str) -> Optional[SchemaVersion]:
        """Get a specific schema version."""
        return self.versions.get(version_id)
    
    def get_latest_version(self) -> Optional[SchemaVersion]:
        """Get the most recent schema version."""
        if not self.versions:
            return None
        
        return max(self.versions.values(), key=lambda v: v.created_at)
    
    def list_versions(self) -> List[SchemaVersion]:
        """List all schema versions sorted by creation time."""
        return sorted(self.versions.values(), key=lambda v: v.created_at, reverse=True)
    
    def get_version_history(self, version_id: str) -> List[SchemaVersion]:
        """Get the history of a schema version including all parent versions."""
        history = []
        current_version = self.versions.get(version_id)
        
        while current_version:
            history.append(current_version)
            parent_id = current_version.parent_version
            current_version = self.versions.get(parent_id) if parent_id else None
        
        return history
    
    def compare_versions(self, version1_id: str, version2_id: str) -> SchemaCompatibilityResult:
        """Compare two schema versions."""
        version1 = self.versions.get(version1_id)
        version2 = self.versions.get(version2_id)
        
        if not version1:
            raise ValueError(f"Version {version1_id} not found")
        if not version2:
            raise ValueError(f"Version {version2_id} not found")
        
        return self.validator.check_compatibility(version1.schema, version2.schema)
    
    def _calculate_schema_hash(self, schema: DataSchema) -> str:
        """Calculate a hash for the schema structure."""
        # Create a normalized representation of the schema
        schema_dict = {
            'columns': [
                {
                    'name': col.name,
                    'data_type': col.data_type.value,
                    'nullable': bool(col.nullable)
                }
                for col in sorted(schema.columns, key=lambda c: c.name)
            ],
            'data_types': {k: v.value for k, v in sorted(schema.data_types.items())}
        }
        
        schema_json = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-serializable types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _save_version(self, version: SchemaVersion):
        """Save a schema version to storage."""
        version_file = self.storage_path / f"{version.version}.json"
        
        # Convert to serializable format
        version_data = {
            'version': version.version,
            'created_at': version.created_at.isoformat(),
            'description': version.description,
            'hash': version.hash,
            'parent_version': version.parent_version,
            'schema': {
                'columns': [
                    {
                        'name': col.name,
                        'data_type': col.data_type.value,
                        'nullable': bool(col.nullable),  # Ensure it's a Python bool
                        'unique_values': int(col.unique_values),  # Ensure it's a Python int
                        'sample_values': self._make_json_serializable(col.sample_values),
                        'metadata': self._make_json_serializable(getattr(col, 'metadata', {}))
                    }
                    for col in version.schema.columns
                ],
                'row_count': int(version.schema.row_count),
                'data_types': {k: v.value for k, v in version.schema.data_types.items()},
                'sample_data': self._make_json_serializable(version.schema.sample_data),
                'created_at': version.schema.created_at.isoformat(),
                'metadata': self._make_json_serializable(getattr(version.schema, 'metadata', {}))
            },
            'changes': [
                {
                    'change_type': change.change_type.value,
                    'column_name': change.column_name,
                    'old_value': change.old_value.value if hasattr(change.old_value, 'value') else change.old_value,
                    'new_value': change.new_value.value if hasattr(change.new_value, 'value') else change.new_value,
                    'description': change.description,
                    'severity': change.severity
                }
                for change in version.changes
            ]
        }
        
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)
    
    def _load_versions(self):
        """Load all schema versions from storage."""
        if not self.storage_path.exists():
            return
        
        for version_file in self.storage_path.glob("*.json"):
            try:
                with open(version_file, 'r') as f:
                    version_data = json.load(f)
                
                # Reconstruct schema
                columns = []
                for col_data in version_data['schema']['columns']:
                    column = ColumnInfo(
                        name=col_data['name'],
                        data_type=DataType(col_data['data_type']),
                        nullable=col_data['nullable'],
                        unique_values=col_data['unique_values'],
                        sample_values=col_data['sample_values']
                    )
                    column.metadata = col_data.get('metadata', {})
                    columns.append(column)
                
                schema = DataSchema(
                    columns=columns,
                    row_count=version_data['schema']['row_count'],
                    data_types={k: DataType(v) for k, v in version_data['schema']['data_types'].items()},
                    sample_data=version_data['schema']['sample_data'],
                    created_at=datetime.fromisoformat(version_data['schema']['created_at'])
                )
                schema.metadata = version_data['schema'].get('metadata', {})
                
                # Reconstruct changes
                changes = []
                for change_data in version_data['changes']:
                    change = SchemaChange(
                        change_type=SchemaChangeType(change_data['change_type']),
                        column_name=change_data['column_name'],
                        old_value=DataType(change_data['old_value']) if change_data['old_value'] and isinstance(change_data['old_value'], str) and change_data['old_value'] in [dt.value for dt in DataType] else change_data['old_value'],
                        new_value=DataType(change_data['new_value']) if change_data['new_value'] and isinstance(change_data['new_value'], str) and change_data['new_value'] in [dt.value for dt in DataType] else change_data['new_value'],
                        description=change_data['description'],
                        severity=change_data['severity']
                    )
                    changes.append(change)
                
                # Reconstruct version
                version = SchemaVersion(
                    version=version_data['version'],
                    schema=schema,
                    created_at=datetime.fromisoformat(version_data['created_at']),
                    description=version_data['description'],
                    hash=version_data['hash'],
                    parent_version=version_data['parent_version'],
                    changes=changes
                )
                
                self.versions[version.version] = version
                
            except Exception as e:
                logger.error(f"Error loading schema version from {version_file}: {e}")


class SchemaManager:
    """Main interface for schema detection and management."""
    
    def __init__(self, storage_path: Optional[str] = None, config: SchemaInferenceConfig = None):
        self.inference_engine = SchemaInferenceEngine(config)
        self.validator = SchemaValidator()
        self.version_manager = SchemaVersionManager(storage_path)
    
    def detect_schema(self, dataframe: pd.DataFrame, table_name: str = "table") -> DataSchema:
        """
        Detect schema from a DataFrame.
        
        Args:
            dataframe: Input DataFrame
            table_name: Name of the table
            
        Returns:
            Detected DataSchema
        """
        return self.inference_engine.infer_schema(dataframe, table_name)
    
    def validate_schema(self, schema: DataSchema) -> ValidationResult:
        """Validate a schema."""
        return self.validator.validate_schema(schema)
    
    def create_schema_version(self, schema: DataSchema, description: str = "", parent_version: Optional[str] = None) -> str:
        """Create a new schema version."""
        return self.version_manager.create_version(schema, description, parent_version)
    
    def get_schema_version(self, version_id: str) -> Optional[SchemaVersion]:
        """Get a specific schema version."""
        return self.version_manager.get_version(version_id)
    
    def check_schema_compatibility(self, old_schema: DataSchema, new_schema: DataSchema) -> SchemaCompatibilityResult:
        """Check compatibility between two schemas."""
        return self.validator.check_compatibility(old_schema, new_schema)
    
    def get_latest_schema(self) -> Optional[DataSchema]:
        """Get the latest schema version."""
        latest_version = self.version_manager.get_latest_version()
        return latest_version.schema if latest_version else None
    
    def list_schema_versions(self) -> List[SchemaVersion]:
        """List all schema versions."""
        return self.version_manager.list_versions()
    
    def export_schema_for_ai(self, schema: DataSchema) -> Dict[str, Any]:
        """
        Export schema information in a format optimized for AI/LLM consumption.
        
        This method formats schema information to help conversational AI systems
        understand the data structure and available columns for query generation.
        
        Args:
            schema: DataSchema to export
            
        Returns:
            Dictionary with AI-friendly schema information
        """
        ai_schema = {
            'table_info': {
                'total_rows': schema.row_count,
                'total_columns': len(schema.columns),
                'created_at': schema.created_at.isoformat() if schema.created_at else None
            },
            'columns': [],
            'data_patterns': {},
            'query_suggestions': []
        }
        
        # Process each column for AI consumption
        for column in schema.columns:
            column_info = {
                'name': column.name,
                'type': column.data_type.value,
                'description': self._generate_column_description(column),
                'nullable': column.nullable,
                'unique_values': column.unique_values,
                'sample_values': column.sample_values[:3],  # Limit to 3 samples
                'analysis_hints': self._generate_analysis_hints(column)
            }
            
            # Add type-specific metadata
            if hasattr(column, 'metadata') and column.metadata:
                if column.data_type == DataType.NUMBER:
                    column_info['numeric_range'] = {
                        'min': column.metadata.get('min_value'),
                        'max': column.metadata.get('max_value'),
                        'mean': column.metadata.get('mean_value')
                    }
                elif column.data_type == DataType.TEXT:
                    column_info['text_patterns'] = column.metadata.get('common_patterns', [])
                elif column.data_type == DataType.DATE:
                    column_info['date_range'] = {
                        'min_date': column.metadata.get('min_date'),
                        'max_date': column.metadata.get('max_date'),
                        'format': column.metadata.get('detected_format')
                    }
            
            ai_schema['columns'].append(column_info)
        
        # Generate data patterns summary
        ai_schema['data_patterns'] = self._analyze_data_patterns(schema)
        
        # Generate query suggestions
        ai_schema['query_suggestions'] = self._generate_query_suggestions(schema)
        
        return ai_schema
    
    def _generate_column_description(self, column: ColumnInfo) -> str:
        """Generate a human-readable description of a column."""
        descriptions = []
        
        # Basic type description
        type_descriptions = {
            DataType.NUMBER: "numeric values",
            DataType.TEXT: "text values", 
            DataType.DATE: "date values",
            DataType.BOOLEAN: "true/false values",
            DataType.CURRENCY: "monetary amounts",
            DataType.PERCENTAGE: "percentage values"
        }
        
        desc = f"Column containing {type_descriptions.get(column.data_type, 'values')}"
        
        # Add nullability info
        if column.nullable:
            desc += " (may contain missing values)"
        else:
            desc += " (required field)"
        
        # Add uniqueness info
        if hasattr(column, 'metadata') and column.metadata:
            if column.metadata.get('is_primary_key_candidate'):
                desc += " - appears to be a unique identifier"
            elif column.metadata.get('is_categorical'):
                desc += " - categorical data with limited distinct values"
        
        return desc
    
    def _generate_analysis_hints(self, column: ColumnInfo) -> List[str]:
        """Generate hints for how this column can be analyzed."""
        hints = []
        
        if column.data_type == DataType.NUMBER:
            hints.extend([
                "Can be used for mathematical calculations",
                "Suitable for aggregations (sum, average, min, max)",
                "Can be used in comparisons and filtering"
            ])
        elif column.data_type == DataType.TEXT:
            hints.extend([
                "Can be used for grouping and categorization",
                "Suitable for text search and filtering",
                "Can be used for counting distinct values"
            ])
        elif column.data_type == DataType.DATE:
            hints.extend([
                "Can be used for time-based analysis",
                "Suitable for trend analysis and time series",
                "Can be used for date range filtering"
            ])
        elif column.data_type == DataType.BOOLEAN:
            hints.extend([
                "Can be used for binary analysis",
                "Suitable for counting true/false occurrences",
                "Can be used for conditional filtering"
            ])
        elif column.data_type in [DataType.CURRENCY, DataType.PERCENTAGE]:
            hints.extend([
                "Can be used for financial analysis",
                "Suitable for aggregations and comparisons",
                "Can be used for calculating totals and averages"
            ])
        
        # Add hints based on metadata
        if hasattr(column, 'metadata') and column.metadata:
            if column.metadata.get('is_categorical'):
                hints.append("Good for creating distribution charts")
            if column.metadata.get('is_primary_key_candidate'):
                hints.append("Can be used as a unique identifier for joins")
        
        return hints
    
    def _analyze_data_patterns(self, schema: DataSchema) -> Dict[str, Any]:
        """Analyze overall data patterns in the schema."""
        patterns = {
            'has_identifiers': False,
            'has_temporal_data': False,
            'has_numeric_measures': False,
            'has_categorical_data': False,
            'potential_relationships': []
        }
        
        numeric_columns = []
        text_columns = []
        date_columns = []
        
        for column in schema.columns:
            if column.data_type in [DataType.NUMBER, DataType.CURRENCY, DataType.PERCENTAGE]:
                numeric_columns.append(column.name)
                patterns['has_numeric_measures'] = True
            elif column.data_type == DataType.TEXT:
                text_columns.append(column.name)
                if hasattr(column, 'metadata') and column.metadata:
                    if column.metadata.get('is_primary_key_candidate'):
                        patterns['has_identifiers'] = True
                    if column.metadata.get('is_categorical'):
                        patterns['has_categorical_data'] = True
            elif column.data_type == DataType.DATE:
                date_columns.append(column.name)
                patterns['has_temporal_data'] = True
        
        # Suggest potential relationships
        if numeric_columns and text_columns:
            patterns['potential_relationships'].append(
                f"Numeric analysis by categories: {', '.join(numeric_columns)} grouped by {', '.join(text_columns[:2])}"
            )
        
        if date_columns and numeric_columns:
            patterns['potential_relationships'].append(
                f"Time series analysis: {', '.join(numeric_columns)} over time ({', '.join(date_columns)})"
            )
        
        return patterns
    
    def _generate_query_suggestions(self, schema: DataSchema) -> List[str]:
        """Generate example queries that can be asked about this data."""
        suggestions = []
        
        numeric_columns = [col.name for col in schema.columns 
                          if col.data_type in [DataType.NUMBER, DataType.CURRENCY, DataType.PERCENTAGE]]
        text_columns = [col.name for col in schema.columns if col.data_type == DataType.TEXT]
        date_columns = [col.name for col in schema.columns if col.data_type == DataType.DATE]
        
        # Basic aggregation suggestions
        if numeric_columns:
            suggestions.append(f"What is the average {numeric_columns[0]}?")
            suggestions.append(f"Show me the total {numeric_columns[0]}")
            if len(numeric_columns) > 1:
                suggestions.append(f"Compare {numeric_columns[0]} and {numeric_columns[1]}")
        
        # Grouping suggestions - prefer categorical columns over identifiers
        if numeric_columns and text_columns:
            # Find the best text column for grouping (prefer categorical over identifiers)
            best_text_col = text_columns[0]
            for col in schema.columns:
                if col.name in text_columns and hasattr(col, 'metadata') and col.metadata:
                    if col.metadata.get('is_categorical') and not col.metadata.get('is_primary_key_candidate'):
                        best_text_col = col.name
                        break
            
            suggestions.append(f"Show {numeric_columns[0]} by {best_text_col}")
            suggestions.append(f"Which {best_text_col} has the highest {numeric_columns[0]}?")
        
        # Time-based suggestions
        if date_columns and numeric_columns:
            suggestions.append(f"Show {numeric_columns[0]} trends over time")
            suggestions.append(f"What was the {numeric_columns[0]} in the last month?")
        
        # Filtering suggestions - prefer categorical columns
        if text_columns:
            best_filter_col = text_columns[0]
            for col in schema.columns:
                if col.name in text_columns and hasattr(col, 'metadata') and col.metadata:
                    if col.metadata.get('is_categorical') and not col.metadata.get('is_primary_key_candidate'):
                        best_filter_col = col.name
                        break
            suggestions.append(f"Filter data by {best_filter_col}")
        
        # Count suggestions
        suggestions.append("How many records are there?")
        if text_columns:
            # Prefer categorical columns for counting
            best_count_col = text_columns[0]
            for col in schema.columns:
                if col.name in text_columns and hasattr(col, 'metadata') and col.metadata:
                    if col.metadata.get('is_categorical') and not col.metadata.get('is_primary_key_candidate'):
                        best_count_col = col.name
                        break
            suggestions.append(f"How many unique {best_count_col} values are there?")
        
        return suggestions[:10]  # Limit to 10 suggestions
    
    def detect_schema_evolution(self, old_dataframe: pd.DataFrame, new_dataframe: pd.DataFrame, 
                               table_name: str = "table") -> Dict[str, Any]:
        """
        Detect schema evolution between two versions of the same dataset.
        
        This method compares schemas from two DataFrames and provides detailed
        analysis of changes, migration strategies, and compatibility assessment.
        
        Args:
            old_dataframe: Previous version of the data
            new_dataframe: New version of the data
            table_name: Name of the table for context
            
        Returns:
            Dictionary with evolution analysis and migration recommendations
        """
        # Detect schemas
        old_schema = self.detect_schema(old_dataframe, f"{table_name}_old")
        new_schema = self.detect_schema(new_dataframe, f"{table_name}_new")
        
        # Check compatibility
        compatibility_result = self.check_schema_compatibility(old_schema, new_schema)
        
        # Analyze data quality changes
        quality_analysis = self._analyze_data_quality_changes(old_dataframe, new_dataframe)
        
        # Generate migration recommendations
        migration_recommendations = self._generate_migration_recommendations(
            old_schema, new_schema, compatibility_result.changes
        )
        
        evolution_report = {
            'compatibility': {
                'level': compatibility_result.compatibility.value,
                'can_migrate': compatibility_result.can_migrate,
                'migration_strategy': compatibility_result.migration_strategy
            },
            'changes': [
                {
                    'type': change.change_type.value,
                    'column': change.column_name,
                    'description': change.description,
                    'severity': change.severity,
                    'old_value': change.old_value.value if hasattr(change.old_value, 'value') else change.old_value,
                    'new_value': change.new_value.value if hasattr(change.new_value, 'value') else change.new_value
                }
                for change in compatibility_result.changes
            ],
            'data_quality': quality_analysis,
            'migration_recommendations': migration_recommendations,
            'schemas': {
                'old': self.export_schema_for_ai(old_schema),
                'new': self.export_schema_for_ai(new_schema)
            }
        }
        
        return evolution_report
    
    def _analyze_data_quality_changes(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze changes in data quality between two DataFrames."""
        quality_changes = {
            'row_count_change': len(new_df) - len(old_df),
            'column_count_change': len(new_df.columns) - len(old_df.columns),
            'data_quality_metrics': {}
        }
        
        # Analyze common columns
        common_columns = set(old_df.columns) & set(new_df.columns)
        
        for col in common_columns:
            old_col = old_df[col]
            new_col = new_df[col]
            
            old_null_pct = (old_col.isnull().sum() / len(old_col)) * 100 if len(old_col) > 0 else 0
            new_null_pct = (new_col.isnull().sum() / len(new_col)) * 100 if len(new_col) > 0 else 0
            
            quality_changes['data_quality_metrics'][col] = {
                'null_percentage_change': new_null_pct - old_null_pct,
                'unique_values_change': new_col.nunique() - old_col.nunique(),
                'data_type_stable': str(old_col.dtype) == str(new_col.dtype)
            }
        
        return quality_changes
    
    def _generate_migration_recommendations(self, old_schema: DataSchema, new_schema: DataSchema, 
                                          changes: List[SchemaChange]) -> List[Dict[str, Any]]:
        """Generate detailed migration recommendations based on schema changes."""
        recommendations = []
        
        for change in changes:
            recommendation = {
                'change_type': change.change_type.value,
                'column': change.column_name,
                'priority': self._assess_migration_priority(change),
                'actions': [],
                'risks': [],
                'validation_steps': []
            }
            
            if change.change_type == SchemaChangeType.COLUMN_ADDED:
                recommendation['actions'] = [
                    f"Add new column '{change.column_name}' to existing data",
                    "Set default values for existing records",
                    "Update data ingestion processes"
                ]
                recommendation['risks'] = [
                    "Existing queries may need updates",
                    "Storage requirements will increase"
                ]
                recommendation['validation_steps'] = [
                    f"Verify default values for '{change.column_name}' are appropriate",
                    "Test existing applications with new schema"
                ]
                
            elif change.change_type == SchemaChangeType.COLUMN_REMOVED:
                recommendation['actions'] = [
                    f"Archive data from column '{change.column_name}'",
                    "Update dependent queries and reports",
                    "Remove column from data ingestion"
                ]
                recommendation['risks'] = [
                    "Data loss if not properly archived",
                    "Breaking changes for dependent systems"
                ]
                recommendation['validation_steps'] = [
                    f"Confirm '{change.column_name}' data is safely archived",
                    "Verify all dependent systems are updated"
                ]
                
            elif change.change_type == SchemaChangeType.TYPE_CHANGED:
                recommendation['actions'] = [
                    f"Convert '{change.column_name}' from {change.old_value} to {change.new_value}",
                    "Validate data conversion accuracy",
                    "Update type constraints and validations"
                ]
                recommendation['risks'] = [
                    "Data loss during type conversion",
                    "Performance impact during migration"
                ]
                recommendation['validation_steps'] = [
                    f"Test conversion of sample '{change.column_name}' data",
                    "Verify converted data maintains business meaning"
                ]
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _assess_migration_priority(self, change: SchemaChange) -> str:
        """Assess the priority level for a schema change migration."""
        if change.severity == "breaking":
            return "critical"
        elif change.severity == "major":
            return "high"
        elif change.change_type == SchemaChangeType.COLUMN_REMOVED:
            return "high"  # Data loss risk
        elif change.change_type == SchemaChangeType.TYPE_CHANGED:
            return "medium"
        else:
            return "low"