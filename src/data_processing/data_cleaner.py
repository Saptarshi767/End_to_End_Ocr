"""
Data cleaning and standardization service for OCR extracted table data.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from collections import Counter

from ..core.models import DataType, DataSchema, ColumnInfo, ValidationResult
from ..core.interfaces import DataCleaningInterface
from ..core.exceptions import DataProcessingError
from .missing_value_handler import (
    MissingValueHandler, MissingValueConfig, MissingValuePolicy, 
    MissingValueStrategy, handle_missing_values_simple
)


logger = logging.getLogger(__name__)


@dataclass
class DataTypeDetectionConfig:
    """Configuration for data type detection."""
    confidence_threshold: float = 0.7
    sample_size: int = 100
    date_formats: List[str] = field(default_factory=lambda: [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%m-%d-%Y', '%B %d, %Y', '%d %B %Y',
        '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S'
    ])
    currency_symbols: List[str] = field(default_factory=lambda: ['$', '€', '£', '¥', '₹'])
    percentage_threshold: float = 0.8
    missing_indicators: List[str] = field(default_factory=lambda: [
        '', 'null', 'NULL', 'None', 'N/A', 'n/a', 'NA', 'na',
        '-', '--', '?', 'unknown', 'Unknown', 'UNKNOWN'
    ])



@dataclass
class DuplicateDetectionConfig:
    """Configuration for duplicate detection."""
    similarity_threshold: float = 0.85
    fuzzy_match_columns: List[str] = field(default_factory=list)
    exact_match_columns: List[str] = field(default_factory=list)
    ignore_case: bool = True
    ignore_whitespace: bool = True
    header_similarity_threshold: float = 0.9
    enable_fuzzy_matching: bool = True
    max_edit_distance: int = 3


@dataclass
class DuplicateDetectionResult:
    """Result of duplicate detection process."""
    original_rows: int
    final_rows: int
    exact_duplicates_removed: int
    fuzzy_duplicates_removed: int
    headers_consolidated: int
    duplicate_groups: List[List[int]] = field(default_factory=list)
    consolidation_log: List[str] = field(default_factory=list)


@dataclass
class TypeDetectionResult:
    """Result of data type detection for a column."""
    detected_type: DataType
    confidence: float
    conversion_errors: int = 0
    sample_values: List[Any] = field(default_factory=list)
    conversion_function: Optional[callable] = None


class DuplicateDetector:
    """Detects and handles duplicate rows and headers with fuzzy matching capabilities."""
    
    def __init__(self, config: DuplicateDetectionConfig = None):
        self.config = config or DuplicateDetectionConfig()
    
    def detect_and_remove_duplicates(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, DuplicateDetectionResult]:
        """
        Detect and remove duplicate rows with both exact and fuzzy matching.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, detection result)
        """
        result_df = dataframe.copy()
        original_rows = len(result_df)
        
        # Initialize result tracking
        result = DuplicateDetectionResult(
            original_rows=original_rows,
            final_rows=original_rows,
            exact_duplicates_removed=0,
            fuzzy_duplicates_removed=0,
            headers_consolidated=0
        )
        
        # Step 1: Generate meaningful column names if missing
        result_df = self._generate_meaningful_column_names(result_df)
        
        # Step 2: Remove exact duplicates
        result_df, exact_removed = self._remove_exact_duplicates(result_df)
        result.exact_duplicates_removed = exact_removed
        
        # Step 3: Consolidate duplicate headers (for multi-page tables)
        result_df, headers_consolidated = self._consolidate_headers(result_df)
        result.headers_consolidated = headers_consolidated
        
        # Step 4: Apply fuzzy matching if enabled
        if self.config.enable_fuzzy_matching:
            result_df, fuzzy_removed, duplicate_groups = self._remove_fuzzy_duplicates(result_df)
            result.fuzzy_duplicates_removed = fuzzy_removed
            result.duplicate_groups = duplicate_groups
        
        result.final_rows = len(result_df)
        
        # Log consolidation actions
        if exact_removed > 0:
            result.consolidation_log.append(f"Removed {exact_removed} exact duplicate rows")
        if headers_consolidated > 0:
            result.consolidation_log.append(f"Consolidated {headers_consolidated} duplicate headers")
        if result.fuzzy_duplicates_removed > 0:
            result.consolidation_log.append(f"Removed {result.fuzzy_duplicates_removed} fuzzy duplicate rows")
        
        return result_df, result
    
    def _remove_exact_duplicates(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove exact duplicate rows."""
        initial_rows = len(dataframe)
        result_df = dataframe.drop_duplicates()
        removed = initial_rows - len(result_df)
        
        logger.info(f"Removed {removed} exact duplicate rows")
        return result_df, removed
    
    def _consolidate_headers(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Consolidate duplicate headers that may appear in multi-page tables.
        
        This identifies rows that are likely duplicate headers and removes them.
        """
        if len(dataframe) <= 1:
            return dataframe, 0
        
        result_df = dataframe.copy()
        headers_removed = 0
        
        # Get the actual column headers
        column_headers = set(dataframe.columns)
        
        # Find rows that match column headers (case-insensitive)
        rows_to_remove = []
        
        for idx, row in dataframe.iterrows():
            # Convert row values to strings and normalize
            row_values = set()
            for val in row.values:
                if pd.notna(val):
                    normalized_val = str(val).strip().lower() if self.config.ignore_case else str(val).strip()
                    if self.config.ignore_whitespace:
                        normalized_val = re.sub(r'\s+', ' ', normalized_val)
                    row_values.add(normalized_val)
            
            # Normalize column headers for comparison
            normalized_headers = set()
            for header in column_headers:
                normalized_header = header.strip().lower() if self.config.ignore_case else header.strip()
                if self.config.ignore_whitespace:
                    normalized_header = re.sub(r'\s+', ' ', normalized_header)
                normalized_headers.add(normalized_header)
            
            # Check if row values significantly overlap with headers
            if row_values and normalized_headers:
                overlap = len(row_values.intersection(normalized_headers))
                overlap_ratio = overlap / len(normalized_headers)
                
                if overlap_ratio >= self.config.header_similarity_threshold:
                    rows_to_remove.append(idx)
                    headers_removed += 1
        
        if rows_to_remove:
            result_df = result_df.drop(rows_to_remove)
            logger.info(f"Consolidated {headers_removed} duplicate header rows")
        
        return result_df, headers_removed
    
    def _remove_fuzzy_duplicates(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, int, List[List[int]]]:
        """
        Remove fuzzy duplicate rows using similarity matching.
        
        Returns:
            Tuple of (cleaned DataFrame, number removed, list of duplicate groups)
        """
        if len(dataframe) <= 1:
            return dataframe, 0, []
        
        # Convert DataFrame to list of normalized row strings for comparison
        normalized_rows = []
        for idx, row in dataframe.iterrows():
            normalized_row = self._normalize_row_for_comparison(row)
            normalized_rows.append((idx, normalized_row))
        
        # Find duplicate groups using fuzzy matching
        duplicate_groups = self._find_fuzzy_duplicate_groups(normalized_rows)
        
        # Keep only the first row from each duplicate group
        rows_to_remove = []
        for group in duplicate_groups:
            if len(group) > 1:
                # Keep the first row, remove the rest
                rows_to_remove.extend(group[1:])
        
        # Remove duplicate rows
        result_df = dataframe.drop(rows_to_remove) if rows_to_remove else dataframe
        fuzzy_removed = len(rows_to_remove)
        
        logger.info(f"Removed {fuzzy_removed} fuzzy duplicate rows in {len(duplicate_groups)} groups")
        
        return result_df, fuzzy_removed, duplicate_groups
    
    def _normalize_row_for_comparison(self, row: pd.Series) -> str:
        """
        Normalize a row for fuzzy comparison.
        
        Args:
            row: Pandas Series representing a row
            
        Returns:
            Normalized string representation of the row
        """
        values = []
        for val in row.values:
            if pd.notna(val):
                str_val = str(val)
                if self.config.ignore_case:
                    str_val = str_val.lower()
                if self.config.ignore_whitespace:
                    str_val = re.sub(r'\s+', ' ', str_val.strip())
                values.append(str_val)
            else:
                values.append('')
        
        return '|'.join(values)
    
    def _find_fuzzy_duplicate_groups(self, normalized_rows: List[Tuple[int, str]]) -> List[List[int]]:
        """
        Find groups of fuzzy duplicate rows.
        
        Args:
            normalized_rows: List of (index, normalized_string) tuples
            
        Returns:
            List of duplicate groups, where each group is a list of row indices
        """
        duplicate_groups = []
        processed_indices = set()
        
        for i, (idx1, row1) in enumerate(normalized_rows):
            if idx1 in processed_indices:
                continue
            
            current_group = [idx1]
            processed_indices.add(idx1)
            
            # Compare with remaining rows
            for j, (idx2, row2) in enumerate(normalized_rows[i+1:], i+1):
                if idx2 in processed_indices:
                    continue
                
                similarity = self._calculate_similarity(row1, row2)
                if similarity >= self.config.similarity_threshold:
                    current_group.append(idx2)
                    processed_indices.add(idx2)
            
            # Only add groups with more than one member
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
        
        return duplicate_groups
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using multiple methods.
        
        Args:
            str1, str2: Strings to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        if str1 == str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        # Use SequenceMatcher for similarity
        sequence_similarity = SequenceMatcher(None, str1, str2).ratio()
        
        # Also check edit distance for short strings
        if len(str1) <= 50 and len(str2) <= 50:
            edit_distance = self._levenshtein_distance(str1, str2)
            max_len = max(len(str1), len(str2))
            edit_similarity = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0
            
            # Use the higher of the two similarities
            return max(sequence_similarity, edit_similarity)
        
        return sequence_similarity
    
    def _levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            str1, str2: Strings to compare
            
        Returns:
            Edit distance between the strings
        """
        if len(str1) < len(str2):
            return self._levenshtein_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def detect_similar_entries(self, dataframe: pd.DataFrame, column: str = None) -> Dict[str, List[Tuple[int, str]]]:
        """
        Detect similar entries in a specific column or across all columns.
        
        Args:
            dataframe: Input DataFrame
            column: Specific column to analyze (None for all columns)
            
        Returns:
            Dictionary mapping similarity groups to lists of (index, value) tuples
        """
        similar_groups = {}
        
        if column:
            if column not in dataframe.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            columns_to_check = [column]
        else:
            columns_to_check = dataframe.columns
        
        for col in columns_to_check:
            col_groups = self._find_similar_values_in_column(dataframe[col])
            if col_groups:
                similar_groups[col] = col_groups
        
        return similar_groups
    
    def _find_similar_values_in_column(self, series: pd.Series) -> List[List[Tuple[int, str]]]:
        """
        Find similar values within a single column.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            List of similarity groups
        """
        # Get unique values with their indices
        value_indices = {}
        for idx, val in series.items():
            if pd.notna(val):
                str_val = str(val)
                if str_val not in value_indices:
                    value_indices[str_val] = []
                value_indices[str_val].append(idx)
        
        # Find similar groups
        similar_groups = []
        processed_values = set()
        
        for val1 in value_indices.keys():
            if val1 in processed_values:
                continue
            
            current_group = [(idx, val1) for idx in value_indices[val1]]
            processed_values.add(val1)
            
            for val2 in value_indices.keys():
                if val2 in processed_values:
                    continue
                
                similarity = self._calculate_similarity(val1, val2)
                if similarity >= self.config.similarity_threshold:
                    current_group.extend([(idx, val2) for idx in value_indices[val2]])
                    processed_values.add(val2)
            
            if len(current_group) > 1:
                similar_groups.append(current_group)
        
        return similar_groups
    
    def _generate_meaningful_column_names(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Generate meaningful column names based on data patterns when headers are missing.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with meaningful column names
        """
        result_df = dataframe.copy()
        columns_renamed = 0
        
        # Check if columns have generic names (like 'Unnamed: 0', '0', '1', etc.)
        generic_patterns = [
            r'^Unnamed:\s*\d+$',  # Pandas default unnamed columns
            r'^\d+$',             # Numeric column names
            r'^Column\s*\d+$',    # Generic column names
            r'^col\d+$',          # Generic col names
            r'^[A-Z]$'            # Single letter names
        ]
        
        new_column_names = {}
        
        for i, col in enumerate(result_df.columns):
            col_str = str(col)
            is_generic = any(re.match(pattern, col_str, re.IGNORECASE) for pattern in generic_patterns)
            
            if is_generic or col_str.strip() == '':
                # Generate meaningful name based on data patterns
                meaningful_name = self._infer_column_name_from_data(result_df[col], i)
                new_column_names[col] = meaningful_name
                columns_renamed += 1
        
        if new_column_names:
            result_df = result_df.rename(columns=new_column_names)
            logger.info(f"Generated meaningful names for {columns_renamed} columns")
        
        return result_df
    
    def _infer_column_name_from_data(self, series: pd.Series, column_index: int) -> str:
        """
        Infer a meaningful column name based on data patterns.
        
        Args:
            series: Column data to analyze
            column_index: Index of the column
            
        Returns:
            Meaningful column name
        """
        # Clean the series for analysis
        cleaned_series = series.dropna().astype(str).str.strip()
        
        if cleaned_series.empty:
            return f"Column_{column_index + 1}"
        
        # Sample some values for analysis
        sample_values = cleaned_series.head(10).tolist()
        
        # Check for common patterns
        
        # 1. Check for name patterns
        name_patterns = [
            (r'.*name.*', 'Name'),
            (r'.*first.*name.*', 'First_Name'),
            (r'.*last.*name.*', 'Last_Name'),
            (r'.*full.*name.*', 'Full_Name'),
            (r'.*customer.*', 'Customer'),
            (r'.*client.*', 'Client'),
            (r'.*user.*', 'User')
        ]
        
        # 2. Check for financial patterns
        financial_patterns = [
            (r'.*price.*', 'Price'),
            (r'.*cost.*', 'Cost'),
            (r'.*amount.*', 'Amount'),
            (r'.*total.*', 'Total'),
            (r'.*revenue.*', 'Revenue'),
            (r'.*profit.*', 'Profit'),
            (r'.*\$.*', 'Currency'),
            (r'.*€.*', 'Currency'),
            (r'.*£.*', 'Currency')
        ]
        
        # 3. Check for date patterns
        date_patterns = [
            (r'.*date.*', 'Date'),
            (r'.*time.*', 'Time'),
            (r'.*created.*', 'Created_Date'),
            (r'.*updated.*', 'Updated_Date'),
            (r'.*modified.*', 'Modified_Date'),
            (r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'Date'),
            (r'\d{4}-\d{2}-\d{2}', 'Date')
        ]
        
        # 4. Check for quantity patterns
        quantity_patterns = [
            (r'.*quantity.*', 'Quantity'),
            (r'.*qty.*', 'Quantity'),
            (r'.*count.*', 'Count'),
            (r'.*number.*', 'Number'),
            (r'.*num.*', 'Number'),
            (r'.*id.*', 'ID'),
            (r'.*identifier.*', 'ID')
        ]
        
        # 5. Check for location patterns
        location_patterns = [
            (r'.*address.*', 'Address'),
            (r'.*city.*', 'City'),
            (r'.*state.*', 'State'),
            (r'.*country.*', 'Country'),
            (r'.*zip.*', 'Zip_Code'),
            (r'.*postal.*', 'Postal_Code'),
            (r'.*location.*', 'Location')
        ]
        
        # 6. Check for contact patterns
        contact_patterns = [
            (r'.*email.*', 'Email'),
            (r'.*phone.*', 'Phone'),
            (r'.*mobile.*', 'Mobile'),
            (r'.*tel.*', 'Phone'),
            (r'.*contact.*', 'Contact'),
            (r'.*@.*\..*', 'Email')
        ]
        
        # 7. Check for percentage patterns
        percentage_patterns = [
            (r'.*%.*', 'Percentage'),
            (r'.*percent.*', 'Percentage'),
            (r'.*rate.*', 'Rate'),
            (r'.*ratio.*', 'Ratio')
        ]
        
        # Combine all patterns
        all_patterns = (
            name_patterns + financial_patterns + date_patterns + 
            quantity_patterns + location_patterns + contact_patterns + 
            percentage_patterns
        )
        
        # Check sample values against patterns
        for sample_value in sample_values:
            sample_lower = sample_value.lower()
            for pattern, suggested_name in all_patterns:
                if re.search(pattern, sample_lower, re.IGNORECASE):
                    return suggested_name
        
        # If no pattern matches, try to infer from data type
        try:
            # Try to detect if it's numeric
            numeric_count = 0
            for val in sample_values:
                try:
                    float(val.replace(',', '').replace('$', '').replace('%', ''))
                    numeric_count += 1
                except ValueError:
                    pass
            
            if numeric_count > len(sample_values) * 0.7:  # 70% numeric
                if any('%' in val for val in sample_values):
                    return 'Percentage'
                elif any('$' in val or '€' in val or '£' in val for val in sample_values):
                    return 'Currency'
                else:
                    return 'Number'
        except:
            pass
        
        # Check if it looks like a category/status
        unique_values = set(sample_values)
        if len(unique_values) <= 5 and len(sample_values) > 5:
            return 'Category'
        
        # Check if it looks like a description (long text)
        avg_length = sum(len(val) for val in sample_values) / len(sample_values)
        if avg_length > 50:
            return 'Description'
        elif avg_length > 20:
            return 'Text'
        
        # Default fallback
        return f"Column_{column_index + 1}"


class DataTypeDetector:
    """Detects and converts data types in extracted table data."""
    
    def __init__(self, config: DataTypeDetectionConfig = None):
        self.config = config or DataTypeDetectionConfig()
        
    def detect_column_type(self, series: pd.Series) -> TypeDetectionResult:
        """
        Detect the most appropriate data type for a pandas Series.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            TypeDetectionResult with detected type and metadata
        """
        # Clean the series first
        cleaned_series = self._clean_series(series)
        
        # Skip empty series
        if cleaned_series.empty or cleaned_series.isna().all():
            return TypeDetectionResult(
                detected_type=DataType.TEXT,
                confidence=0.0,
                sample_values=[]
            )
        
        # Sample data for analysis
        sample_size = min(len(cleaned_series), self.config.sample_size)
        sample = cleaned_series.dropna().head(sample_size)
        
        if sample.empty:
            return TypeDetectionResult(
                detected_type=DataType.TEXT,
                confidence=0.0,
                sample_values=[]
            )
        
        # Test different data types in order of specificity
        type_tests = [
            (DataType.BOOLEAN, self._test_boolean),
            (DataType.CURRENCY, self._test_currency),
            (DataType.PERCENTAGE, self._test_percentage),
            (DataType.DATE, self._test_date),
            (DataType.NUMBER, self._test_number),
            (DataType.TEXT, self._test_text)
        ]
        
        best_result = None
        best_confidence = 0.0
        
        for data_type, test_func in type_tests:
            try:
                confidence, conversion_func = test_func(sample)
                if confidence > best_confidence and confidence >= self.config.confidence_threshold:
                    best_confidence = confidence
                    best_result = TypeDetectionResult(
                        detected_type=data_type,
                        confidence=confidence,
                        sample_values=sample.head(5).tolist(),
                        conversion_function=conversion_func
                    )
            except Exception as e:
                logger.debug(f"Error testing {data_type} for column: {e}")
                continue
        
        # Default to text if no type meets confidence threshold
        if best_result is None:
            best_result = TypeDetectionResult(
                detected_type=DataType.TEXT,
                confidence=1.0,
                sample_values=sample.head(5).tolist(),
                conversion_function=str
            )
        
        return best_result
    
    def _clean_series(self, series: pd.Series) -> pd.Series:
        """Clean series by removing whitespace and standardizing missing values."""
        # Convert to string and strip whitespace
        cleaned = series.astype(str).str.strip()
        
        # Replace missing value indicators with NaN
        missing_indicators = self.config.missing_indicators
        for indicator in missing_indicators:
            cleaned = cleaned.replace(indicator, np.nan)
        
        return cleaned
    
    def _test_boolean(self, sample: pd.Series) -> Tuple[float, callable]:
        """Test if series contains boolean values."""
        boolean_patterns = [
            r'^(true|false)$',
            r'^(yes|no)$',
            r'^(y|n)$',
            r'^(1|0)$',
            r'^(on|off)$'
        ]
        
        matches = 0
        total = len(sample)
        
        for value in sample:
            value_str = str(value).lower().strip()
            for pattern in boolean_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matches += 1
                    break
        
        confidence = matches / total if total > 0 else 0.0
        
        def convert_boolean(val):
            val_str = str(val).lower().strip()
            if val_str in ['true', 'yes', 'y', '1', 'on']:
                return True
            elif val_str in ['false', 'no', 'n', '0', 'off']:
                return False
            else:
                return None
        
        return confidence, convert_boolean
    
    def _test_currency(self, sample: pd.Series) -> Tuple[float, callable]:
        """Test if series contains currency values."""
        currency_pattern = r'^[' + ''.join(re.escape(s) for s in self.config.currency_symbols) + r']\s*[\d,]+\.?\d*$'
        
        matches = 0
        total = len(sample)
        
        for value in sample:
            value_str = str(value).strip()
            if re.match(currency_pattern, value_str):
                matches += 1
        
        confidence = matches / total if total > 0 else 0.0
        
        def convert_currency(val):
            val_str = str(val).strip()
            # Remove currency symbols and commas
            for symbol in self.config.currency_symbols:
                val_str = val_str.replace(symbol, '')
            val_str = val_str.replace(',', '').strip()
            try:
                return float(val_str)
            except ValueError:
                return None
        
        return confidence, convert_currency
    
    def _test_percentage(self, sample: pd.Series) -> Tuple[float, callable]:
        """Test if series contains percentage values."""
        percentage_pattern = r'^[\d,]+\.?\d*\s*%$'
        
        matches = 0
        total = len(sample)
        
        for value in sample:
            value_str = str(value).strip()
            if re.match(percentage_pattern, value_str):
                matches += 1
        
        confidence = matches / total if total > 0 else 0.0
        
        def convert_percentage(val):
            val_str = str(val).strip()
            if val_str.endswith('%'):
                val_str = val_str[:-1].replace(',', '').strip()
                try:
                    return float(val_str) / 100.0
                except ValueError:
                    return None
            return None
        
        return confidence, convert_percentage
    
    def _test_date(self, sample: pd.Series) -> Tuple[float, callable]:
        """Test if series contains date values."""
        matches = 0
        total = len(sample)
        successful_format = None
        
        for date_format in self.config.date_formats:
            format_matches = 0
            for value in sample:
                try:
                    datetime.strptime(str(value).strip(), date_format)
                    format_matches += 1
                except ValueError:
                    continue
            
            if format_matches > matches:
                matches = format_matches
                successful_format = date_format
        
        confidence = matches / total if total > 0 else 0.0
        
        def convert_date(val):
            val_str = str(val).strip()
            for date_format in self.config.date_formats:
                try:
                    return datetime.strptime(val_str, date_format).date()
                except ValueError:
                    continue
            return None
        
        return confidence, convert_date
    
    def _test_number(self, sample: pd.Series) -> Tuple[float, callable]:
        """Test if series contains numeric values."""
        matches = 0
        total = len(sample)
        
        for value in sample:
            try:
                # Try to convert to float, handling commas
                val_str = str(value).strip().replace(',', '')
                float(val_str)
                matches += 1
            except ValueError:
                continue
        
        confidence = matches / total if total > 0 else 0.0
        
        def convert_number(val):
            try:
                val_str = str(val).strip().replace(',', '')
                # Try integer first, then float
                if '.' not in val_str:
                    return int(val_str)
                else:
                    return float(val_str)
            except ValueError:
                return None
        
        return confidence, convert_number
    
    def _test_text(self, sample: pd.Series) -> Tuple[float, callable]:
        """Test if series contains text values (always returns 1.0 as fallback)."""
        return 1.0, str
    
    def convert_column(self, series: pd.Series, target_type: DataType) -> pd.Series:
        """
        Convert a pandas Series to the specified data type.
        
        Args:
            series: Series to convert
            target_type: Target DataType
            
        Returns:
            Converted pandas Series
        """
        if target_type == DataType.TEXT:
            return series.astype(str)
        
        # Detect the conversion function
        detection_result = self.detect_column_type(series)
        
        if detection_result.detected_type == target_type and detection_result.conversion_function:
            # Use detected conversion function
            return series.apply(detection_result.conversion_function)
        else:
            # Use generic conversion based on target type
            return self._generic_convert(series, target_type)
    
    def _generic_convert(self, series: pd.Series, target_type: DataType) -> pd.Series:
        """Generic conversion function for data types."""
        try:
            if target_type == DataType.NUMBER:
                return pd.to_numeric(series, errors='coerce')
            elif target_type == DataType.DATE:
                return pd.to_datetime(series, errors='coerce')
            elif target_type == DataType.BOOLEAN:
                return series.map({'true': True, 'false': False, '1': True, '0': False, 
                                 'yes': True, 'no': False, 'y': True, 'n': False})
            elif target_type == DataType.CURRENCY:
                # Remove currency symbols and convert to float
                cleaned = series.astype(str).str.replace(r'[$€£¥₹,]', '', regex=True)
                return pd.to_numeric(cleaned, errors='coerce')
            elif target_type == DataType.PERCENTAGE:
                # Remove % and convert to decimal
                cleaned = series.astype(str).str.replace('%', '').str.replace(',', '')
                return pd.to_numeric(cleaned, errors='coerce') / 100.0
            else:
                return series.astype(str)
        except Exception as e:
            logger.warning(f"Generic conversion failed for {target_type}: {e}")
            return series


class DataCleaningService(DataCleaningInterface):
    """Main data cleaning and standardization service."""
    
    def __init__(self, 
                 type_detection_config: DataTypeDetectionConfig = None,
                 missing_value_config: MissingValueConfig = None,
                 duplicate_config: DuplicateDetectionConfig = None):
        self.type_detector = DataTypeDetector(type_detection_config)
        self.missing_config = missing_value_config or MissingValueConfig()
        self.duplicate_config = duplicate_config or DuplicateDetectionConfig()
        self.missing_value_handler = MissingValueHandler(self.missing_config)
    
    def standardize_data_types(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically detect and convert data types for all columns.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with standardized data types
        """
        result_df = dataframe.copy()
        
        for column in result_df.columns:
            try:
                detection_result = self.type_detector.detect_column_type(result_df[column])
                
                if detection_result.conversion_function:
                    # Apply the detected conversion function
                    result_df[column] = result_df[column].apply(
                        lambda x: detection_result.conversion_function(x) if pd.notna(x) else x
                    )
                
                logger.info(f"Column '{column}' detected as {detection_result.detected_type.value} "
                           f"with confidence {detection_result.confidence:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to standardize column '{column}': {e}")
                continue
        
        return result_df
    
    def handle_missing_values(self, dataframe: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing or corrupted values in the DataFrame.
        
        Args:
            dataframe: Input DataFrame
            strategy: Strategy to use ('auto' uses configured strategies per type)
            
        Returns:
            DataFrame with missing values handled
        """
        # Detect data types for all columns
        data_types = {}
        for column in dataframe.columns:
            detection_result = self.type_detector.detect_column_type(dataframe[column])
            data_types[column] = detection_result.detected_type
        
        if strategy == 'auto':
            # Use the enhanced missing value handler with automatic strategy selection
            result_df, report = self.missing_value_handler.handle_missing_values(
                dataframe, data_types=data_types
            )
            
            # Log the processing report
            logger.info(f"Missing value handling completed: "
                       f"Original missing: {report.original_missing_count}, "
                       f"Final missing: {report.final_missing_count}, "
                       f"Quality improvement: {report.quality_improvement:.2%}")
            
            if report.warnings:
                for warning in report.warnings:
                    logger.warning(warning)
            
            return result_df
        else:
            # Use simple strategy for all columns
            return handle_missing_values_simple(dataframe, strategy, data_types)
    

    def remove_duplicates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows and consolidate headers.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        duplicate_detector = DuplicateDetector(self.duplicate_config)
        result_df, detection_result = duplicate_detector.detect_and_remove_duplicates(dataframe)
        
        # Log the results
        logger.info(f"Duplicate detection completed: "
                   f"Original rows: {detection_result.original_rows}, "
                   f"Final rows: {detection_result.final_rows}, "
                   f"Exact duplicates removed: {detection_result.exact_duplicates_removed}, "
                   f"Fuzzy duplicates removed: {detection_result.fuzzy_duplicates_removed}, "
                   f"Headers consolidated: {detection_result.headers_consolidated}")
        
        for log_entry in detection_result.consolidation_log:
            logger.info(log_entry)
        
        return result_df
    
    def detect_schema(self, dataframe: pd.DataFrame) -> DataSchema:
        """
        Detect and return data schema information.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataSchema object with column information
        """
        columns = []
        data_types = {}
        sample_data = {}
        
        for column in dataframe.columns:
            try:
                detection_result = self.type_detector.detect_column_type(dataframe[column])
                
                column_info = ColumnInfo(
                    name=column,
                    data_type=detection_result.detected_type,
                    nullable=dataframe[column].isna().any(),
                    unique_values=dataframe[column].nunique(),
                    sample_values=detection_result.sample_values
                )
                
                columns.append(column_info)
                data_types[column] = detection_result.detected_type
                sample_data[column] = detection_result.sample_values
                
            except Exception as e:
                logger.warning(f"Failed to detect schema for column '{column}': {e}")
                # Default column info
                column_info = ColumnInfo(
                    name=column,
                    data_type=DataType.TEXT,
                    nullable=True,
                    unique_values=dataframe[column].nunique(),
                    sample_values=dataframe[column].head(5).tolist()
                )
                columns.append(column_info)
                data_types[column] = DataType.TEXT
                sample_data[column] = dataframe[column].head(5).tolist()
        
        return DataSchema(
            columns=columns,
            row_count=len(dataframe),
            data_types=data_types,
            sample_data=sample_data
        )
    
    def get_missing_value_report(self, dataframe: pd.DataFrame, 
                               policies: Optional[Dict[str, MissingValuePolicy]] = None) -> Dict[str, Any]:
        """
        Get detailed missing value analysis and recommendations.
        
        Args:
            dataframe: DataFrame to analyze
            policies: Optional custom policies
            
        Returns:
            Dictionary with missing value analysis and recommendations
        """
        # Detect data types
        data_types = {}
        for column in dataframe.columns:
            detection_result = self.type_detector.detect_column_type(dataframe[column])
            data_types[column] = detection_result.detected_type
        
        # Get quality assessment
        quality_assessments = self.missing_value_handler.get_quality_assessment(dataframe, data_types)
        
        # Get strategy recommendations
        strategy_recommendations = self.missing_value_handler.recommend_strategies(dataframe, data_types)
        
        # Calculate overall statistics
        total_cells = dataframe.size
        missing_cells = dataframe.isna().sum().sum()
        missing_percentage = missing_cells / total_cells if total_cells > 0 else 0.0
        
        return {
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': missing_percentage,
            'column_assessments': quality_assessments,
            'recommended_strategies': strategy_recommendations,
            'columns_with_high_missing': [
                col for col, assessment in quality_assessments.items()
                if assessment.missing_percentage > 0.5
            ]
        }
    
    def handle_missing_values_with_policies(self, dataframe: pd.DataFrame, 
                                          policies: Dict[str, MissingValuePolicy]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values using custom policies.
        
        Args:
            dataframe: Input DataFrame
            policies: Dictionary of column-specific policies
            
        Returns:
            Tuple of (processed DataFrame, processing report)
        """
        # Detect data types
        data_types = {}
        for column in dataframe.columns:
            detection_result = self.type_detector.detect_column_type(dataframe[column])
            data_types[column] = detection_result.detected_type
        
        return self.missing_value_handler.handle_missing_values(dataframe, policies, data_types)
    
    def create_missing_value_policy(self, column_name: str, data_type: DataType, 
                                  strategy: str, **kwargs) -> MissingValuePolicy:
        """
        Create a missing value policy for a column.
        
        Args:
            column_name: Name of the column
            data_type: Data type of the column
            strategy: Strategy name (string)
            **kwargs: Additional policy parameters
            
        Returns:
            MissingValuePolicy object
        """
        try:
            strategy_enum = MissingValueStrategy(strategy)
            return self.missing_value_handler.create_user_policy(
                column_name, data_type, strategy_enum, **kwargs
            )
        except ValueError:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available missing value strategies.
        
        Returns:
            List of strategy names
        """
        return [strategy.value for strategy in MissingValueStrategy]
    
    def detect_duplicates_detailed(self, dataframe: pd.DataFrame) -> DuplicateDetectionResult:
        """
        Perform detailed duplicate detection without removing duplicates.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DuplicateDetectionResult with detailed information
        """
        duplicate_detector = DuplicateDetector(self.duplicate_config)
        _, detection_result = duplicate_detector.detect_and_remove_duplicates(dataframe)
        return detection_result
    
    def find_similar_entries(self, dataframe: pd.DataFrame, column: str = None) -> Dict[str, List[Tuple[int, str]]]:
        """
        Find similar entries in the DataFrame using fuzzy matching.
        
        Args:
            dataframe: Input DataFrame
            column: Specific column to analyze (None for all columns)
            
        Returns:
            Dictionary mapping column names to lists of similar entry groups
        """
        duplicate_detector = DuplicateDetector(self.duplicate_config)
        return duplicate_detector.detect_similar_entries(dataframe, column)
    
    def consolidate_headers_only(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidate duplicate headers without removing row duplicates.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with consolidated headers
        """
        duplicate_detector = DuplicateDetector(self.duplicate_config)
        result_df, headers_consolidated = duplicate_detector._consolidate_headers(dataframe)
        
        if headers_consolidated > 0:
            logger.info(f"Consolidated {headers_consolidated} duplicate headers")
        
        return result_df
    
    def configure_duplicate_detection(self, 
                                    similarity_threshold: float = None,
                                    enable_fuzzy_matching: bool = None,
                                    header_similarity_threshold: float = None,
                                    ignore_case: bool = None,
                                    ignore_whitespace: bool = None) -> None:
        """
        Configure duplicate detection parameters.
        
        Args:
            similarity_threshold: Threshold for fuzzy matching (0.0 to 1.0)
            enable_fuzzy_matching: Whether to enable fuzzy matching
            header_similarity_threshold: Threshold for header consolidation
            ignore_case: Whether to ignore case in comparisons
            ignore_whitespace: Whether to ignore whitespace differences
        """
        if similarity_threshold is not None:
            self.duplicate_config.similarity_threshold = similarity_threshold
        if enable_fuzzy_matching is not None:
            self.duplicate_config.enable_fuzzy_matching = enable_fuzzy_matching
        if header_similarity_threshold is not None:
            self.duplicate_config.header_similarity_threshold = header_similarity_threshold
        if ignore_case is not None:
            self.duplicate_config.ignore_case = ignore_case
        if ignore_whitespace is not None:
            self.duplicate_config.ignore_whitespace = ignore_whitespace
    
    def generate_meaningful_column_names(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Generate meaningful column names based on data patterns when headers are missing.
        
        Args:
            dataframe: Input DataFrame with potentially generic column names
            
        Returns:
            DataFrame with meaningful column names
        """
        duplicate_detector = DuplicateDetector(self.duplicate_config)
        return duplicate_detector._generate_meaningful_column_names(dataframe)


# Convenience functions for direct usage
def detect_data_types(dataframe: pd.DataFrame, config: DataTypeDetectionConfig = None) -> Dict[str, TypeDetectionResult]:
    """
    Convenience function to detect data types for all columns in a DataFrame.
    
    Args:
        dataframe: Input DataFrame
        config: Optional configuration for detection
        
    Returns:
        Dictionary mapping column names to TypeDetectionResult
    """
    detector = DataTypeDetector(config)
    results = {}
    
    for column in dataframe.columns:
        results[column] = detector.detect_column_type(dataframe[column])
    
    return results


def standardize_dataframe(dataframe: pd.DataFrame, 
                         type_config: DataTypeDetectionConfig = None,
                         missing_config: MissingValueConfig = None) -> pd.DataFrame:
    """
    Convenience function to standardize a DataFrame with default settings.
    
    Args:
        dataframe: Input DataFrame
        type_config: Optional type detection configuration
        missing_config: Optional missing value configuration
        
    Returns:
        Standardized DataFrame
    """
    service = DataCleaningService(type_config, missing_config)
    
    # Apply standardization steps
    result = service.standardize_data_types(dataframe)
    result = service.handle_missing_values(result)
    result = service.remove_duplicates(result)
    
    return result