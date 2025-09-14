"""
Entity extraction component for identifying data column references and values.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..core.models import DataSchema, ColumnInfo, DataType


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from text."""
    text: str
    entity_type: str  # 'column', 'value', 'number', 'date', 'operator'
    confidence: float
    start_pos: int
    end_pos: int
    normalized_value: Any = None


class EntityExtractor:
    """Extract entities like column names, values, and operators from natural language."""
    
    def __init__(self):
        self.operators = {
            'equals': ['=', 'is', 'equals', 'equal to'],
            'greater_than': ['>', 'greater than', 'more than', 'above'],
            'less_than': ['<', 'less than', 'below', 'under'],
            'greater_equal': ['>=', 'at least', 'minimum'],
            'less_equal': ['<=', 'at most', 'maximum'],
            'not_equal': ['!=', 'not', 'not equal', 'different'],
            'contains': ['contains', 'includes', 'has'],
            'starts_with': ['starts with', 'begins with'],
            'ends_with': ['ends with', 'finishes with']
        }
        
        self.date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{2}-\d{2}-\d{4}\b',  # MM-DD-YYYY
            r'\b\w+ \d{1,2}, \d{4}\b'  # Month DD, YYYY
        ]
        
        self.number_patterns = [
            r'\b\d+\.\d+\b',  # Decimal numbers
            r'\b\d+\b',       # Integers
            r'\$\d+(?:\.\d{2})?\b',  # Currency
            r'\d+%\b'         # Percentages
        ]
    
    def extract_entities(self, text: str, schema: DataSchema) -> List[ExtractedEntity]:
        """
        Extract all entities from the given text.
        
        Args:
            text: Input text to analyze
            schema: Data schema for column matching
            
        Returns:
            List of extracted entities
        """
        entities = []
        text_lower = text.lower()
        
        # Extract column references
        entities.extend(self._extract_columns(text, text_lower, schema))
        
        # Extract numbers
        entities.extend(self._extract_numbers(text))
        
        # Extract dates
        entities.extend(self._extract_dates(text))
        
        # Extract operators
        entities.extend(self._extract_operators(text, text_lower))
        
        # Extract quoted values
        entities.extend(self._extract_quoted_values(text))
        
        # Sort by position in text
        entities.sort(key=lambda x: x.start_pos)
        
        return entities
    
    def _extract_columns(self, text: str, text_lower: str, schema: DataSchema) -> List[ExtractedEntity]:
        """Extract column references from text."""
        entities = []
        
        for column in schema.columns:
            column_name = column.name
            column_lower = column_name.lower()
            
            # Find exact matches
            for match in re.finditer(re.escape(column_lower), text_lower):
                confidence = 1.0
                entities.append(ExtractedEntity(
                    text=column_name,
                    entity_type='column',
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=column_name
                ))
            
            # Find matches with spaces instead of underscores
            column_spaced = column_lower.replace('_', ' ')
            if column_spaced != column_lower:
                for match in re.finditer(re.escape(column_spaced), text_lower):
                    confidence = 0.9
                    entities.append(ExtractedEntity(
                        text=column_name,
                        entity_type='column',
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        normalized_value=column_name
                    ))
            
            # Find fuzzy matches for multi-word columns
            if '_' in column_name:
                words = column_name.split('_')
                for word in words:
                    word_lower = word.lower()
                    if len(word_lower) > 2:  # Skip very short words
                        for match in re.finditer(r'\b' + re.escape(word_lower) + r'\b', text_lower):
                            confidence = 0.7  # Lower confidence for partial matches
                            entities.append(ExtractedEntity(
                                text=word,
                                entity_type='column',
                                confidence=confidence,
                                start_pos=match.start(),
                                end_pos=match.end(),
                                normalized_value=column_name
                            ))
        
        return entities
    
    def _extract_numbers(self, text: str) -> List[ExtractedEntity]:
        """Extract numeric values from text."""
        entities = []
        
        for pattern in self.number_patterns:
            for match in re.finditer(pattern, text):
                number_text = match.group()
                confidence = 1.0
                
                # Normalize the number
                normalized_value = self._normalize_number(number_text)
                
                entities.append(ExtractedEntity(
                    text=number_text,
                    entity_type='number',
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=normalized_value
                ))
        
        return entities
    
    def _extract_dates(self, text: str) -> List[ExtractedEntity]:
        """Extract date values from text."""
        entities = []
        
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text):
                date_text = match.group()
                confidence = 0.9  # High confidence for date patterns
                
                entities.append(ExtractedEntity(
                    text=date_text,
                    entity_type='date',
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=date_text
                ))
        
        return entities
    
    def _extract_operators(self, text: str, text_lower: str) -> List[ExtractedEntity]:
        """Extract comparison operators from text."""
        entities = []
        
        for op_type, keywords in self.operators.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                for match in re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                    confidence = 0.8
                    
                    entities.append(ExtractedEntity(
                        text=keyword,
                        entity_type='operator',
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        normalized_value=op_type
                    ))
        
        return entities
    
    def _extract_quoted_values(self, text: str) -> List[ExtractedEntity]:
        """Extract quoted string values from text."""
        entities = []
        
        # Single and double quotes
        quote_patterns = [
            r'"([^"]*)"',
            r"'([^']*)'"
        ]
        
        for pattern in quote_patterns:
            for match in re.finditer(pattern, text):
                quoted_text = match.group(1)  # Content without quotes
                confidence = 1.0
                
                entities.append(ExtractedEntity(
                    text=quoted_text,
                    entity_type='value',
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=quoted_text
                ))
        
        return entities
    
    def _normalize_number(self, number_text: str) -> float:
        """Normalize a number string to a float value."""
        # Remove currency symbols and percentage signs
        cleaned = re.sub(r'[$%,]', '', number_text)
        
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def find_column_matches(self, text: str, schema: DataSchema, 
                          threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find column matches using fuzzy string matching.
        
        Args:
            text: Input text
            schema: Data schema
            threshold: Minimum similarity threshold
            
        Returns:
            List of (column_name, similarity_score) tuples
        """
        matches = []
        text_lower = text.lower()
        
        for column in schema.columns:
            column_name = column.name
            column_lower = column_name.lower()
            
            # Calculate similarity
            similarity = SequenceMatcher(None, text_lower, column_lower).ratio()
            
            if similarity >= threshold:
                matches.append((column_name, similarity))
            
            # Also check individual words for multi-word columns
            if ' ' in column_name:
                words = column_name.split()
                for word in words:
                    if len(word) > 2:
                        word_similarity = SequenceMatcher(None, text_lower, word.lower()).ratio()
                        if word_similarity >= threshold:
                            matches.append((column_name, word_similarity * 0.8))  # Slightly lower confidence
        
        # Remove duplicates and sort by similarity
        unique_matches = {}
        for column, similarity in matches:
            if column not in unique_matches or similarity > unique_matches[column]:
                unique_matches[column] = similarity
        
        return sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)
    
    def extract_filter_conditions(self, text: str, schema: DataSchema) -> List[Dict[str, Any]]:
        """
        Extract filter conditions from natural language.
        
        Args:
            text: Input text
            schema: Data schema
            
        Returns:
            List of filter condition dictionaries
        """
        conditions = []
        entities = self.extract_entities(text, schema)
        
        # Group entities by type
        columns = [e for e in entities if e.entity_type == 'column']
        operators = [e for e in entities if e.entity_type == 'operator']
        values = [e for e in entities if e.entity_type in ['value', 'number', 'date']]
        
        # Try to match column-operator-value patterns
        for i, column in enumerate(columns):
            # Find the closest operator after this column
            closest_operator = None
            closest_value = None
            
            for op in operators:
                if op.start_pos > column.end_pos:
                    closest_operator = op
                    break
            
            if closest_operator:
                # Find the closest value after the operator
                for val in values:
                    if val.start_pos > closest_operator.end_pos:
                        closest_value = val
                        break
            
            if closest_operator and closest_value:
                conditions.append({
                    'column': column.normalized_value,
                    'operator': closest_operator.normalized_value,
                    'value': closest_value.normalized_value,
                    'confidence': min(column.confidence, closest_operator.confidence, closest_value.confidence)
                })
        
        return conditions
    
    def extract_aggregation_targets(self, text: str, schema: DataSchema) -> List[Dict[str, Any]]:
        """
        Extract columns that should be aggregated.
        
        Args:
            text: Input text
            schema: Data schema
            
        Returns:
            List of aggregation target dictionaries
        """
        targets = []
        entities = self.extract_entities(text, schema)
        
        # Find numeric columns that are likely aggregation targets
        numeric_columns = [col.name for col in schema.columns 
                          if col.data_type in [DataType.NUMBER, DataType.CURRENCY]]
        
        column_entities = [e for e in entities if e.entity_type == 'column']
        
        for entity in column_entities:
            column_name = entity.normalized_value
            
            # Higher confidence for numeric columns
            confidence = entity.confidence
            if column_name in numeric_columns:
                confidence *= 1.2
            
            targets.append({
                'column': column_name,
                'confidence': min(confidence, 1.0),
                'data_type': next((col.data_type for col in schema.columns 
                                 if col.name == column_name), DataType.TEXT)
            })
        
        return targets