"""
Natural Language Processing component for question parsing and intent recognition.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.models import DataSchema, ColumnInfo, DataType


class QuestionType(Enum):
    """Types of questions that can be asked."""
    AGGREGATION = "aggregation"  # sum, count, average, etc.
    FILTERING = "filtering"      # show records where...
    COMPARISON = "comparison"    # compare A vs B
    TREND = "trend"             # show trend over time
    DISTRIBUTION = "distribution"  # show distribution of values
    RANKING = "ranking"         # top/bottom N items
    STATISTICAL = "statistical"  # min, max, median, etc.
    DESCRIPTIVE = "descriptive"  # describe the data


class AggregationType(Enum):
    """Types of aggregation operations."""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STD = "std"


@dataclass
class ParsedIntent:
    """Parsed intent from natural language question."""
    question_type: QuestionType
    entities: Dict[str, Any]
    columns: List[str]
    aggregation: Optional[AggregationType] = None
    filters: List[Dict[str, Any]] = None
    confidence: float = 0.0
    original_question: str = ""


class NLPProcessor:
    """Natural Language Processing component for question understanding."""
    
    def __init__(self):
        self.aggregation_keywords = {
            'sum': AggregationType.SUM,
            'total': AggregationType.SUM,
            'add': AggregationType.SUM,
            'count': AggregationType.COUNT,
            'number': AggregationType.COUNT,
            'how many': AggregationType.COUNT,
            'average': AggregationType.AVERAGE,
            'avg': AggregationType.AVERAGE,
            'mean': AggregationType.AVERAGE,
            'minimum': AggregationType.MIN,
            'min': AggregationType.MIN,
            'lowest': AggregationType.MIN,
            'maximum': AggregationType.MAX,
            'max': AggregationType.MAX,
            'highest': AggregationType.MAX,
            'median': AggregationType.MEDIAN,
            'middle': AggregationType.MEDIAN,
            'standard deviation': AggregationType.STD,
            'std': AggregationType.STD
        }
        
        self.comparison_keywords = [
            'compare', 'versus', 'vs', 'against', 'difference', 'between'
        ]
        
        self.trend_keywords = [
            'trend', 'over time', 'change', 'growth', 'decline', 'pattern'
        ]
        
        self.filtering_keywords = [
            'where', 'filter', 'show', 'only', 'exclude', 'include'
        ]
        
        self.ranking_keywords = [
            'top', 'bottom', 'best', 'worst', 'highest', 'lowest', 'rank'
        ]
        
        self.distribution_keywords = [
            'distribution', 'spread', 'breakdown', 'proportion', 'percentage'
        ]
    
    def parse_question(self, question: str, schema: DataSchema) -> ParsedIntent:
        """
        Parse natural language question and extract intent.
        
        Args:
            question: Natural language question
            schema: Data schema for column reference
            
        Returns:
            ParsedIntent with extracted information
        """
        question_lower = question.lower().strip()
        
        # Extract entities (column references)
        entities = self._extract_entities(question_lower, schema)
        
        # Classify question type
        question_type = self._classify_question_type(question_lower)
        
        # Extract aggregation type if applicable
        aggregation = self._extract_aggregation(question_lower)
        
        # Extract filters
        filters = self._extract_filters(question_lower, schema)
        
        # Calculate confidence based on entity matches and keyword presence
        confidence = self._calculate_confidence(question_lower, entities, question_type)
        
        return ParsedIntent(
            question_type=question_type,
            entities=entities,
            columns=entities.get('columns', []),
            aggregation=aggregation,
            filters=filters,
            confidence=confidence,
            original_question=question
        )
    
    def _extract_entities(self, question: str, schema: DataSchema) -> Dict[str, Any]:
        """Extract entities like column names from the question."""
        entities = {
            'columns': [],
            'values': [],
            'numbers': []
        }
        
        question_lower = question.lower()
        
        # Extract column references
        for column in schema.columns:
            column_name = column.name
            column_lower = column_name.lower()
            
            # Check for exact matches
            if column_lower in question_lower:
                entities['columns'].append(column_name)
                continue
            
            # Check for matches with spaces replaced by underscores
            column_spaced = column_lower.replace('_', ' ')
            if column_spaced in question_lower:
                entities['columns'].append(column_name)
                continue
            
            # Check for partial word matches
            column_words = column_lower.split('_')
            question_words = question_lower.split()
            
            # If all words from column name appear in question
            if len(column_words) > 1 and all(word in question_words for word in column_words):
                entities['columns'].append(column_name)
                continue
            
            # Check for individual word matches (for single word columns)
            if len(column_words) == 1 and column_words[0] in question_words:
                entities['columns'].append(column_name)
        
        # Extract numeric values
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)
        entities['numbers'] = [float(n) for n in numbers]
        
        # Extract quoted values (potential filter values)
        quoted_values = re.findall(r'"([^"]*)"', question)
        quoted_values.extend(re.findall(r"'([^']*)'", question))
        entities['values'].extend(quoted_values)
        
        return entities
    
    def _classify_question_type(self, question: str) -> QuestionType:
        """Classify the type of question being asked."""
        
        # Check for aggregation keywords
        if any(keyword in question for keyword in self.aggregation_keywords.keys()):
            return QuestionType.AGGREGATION
        
        # Check for comparison keywords
        if any(keyword in question for keyword in self.comparison_keywords):
            return QuestionType.COMPARISON
        
        # Check for trend keywords
        if any(keyword in question for keyword in self.trend_keywords):
            return QuestionType.TREND
        
        # Check for filtering keywords
        if any(keyword in question for keyword in self.filtering_keywords):
            return QuestionType.FILTERING
        
        # Check for ranking keywords
        if any(keyword in question for keyword in self.ranking_keywords):
            return QuestionType.RANKING
        
        # Check for distribution keywords
        if any(keyword in question for keyword in self.distribution_keywords):
            return QuestionType.DISTRIBUTION
        
        # Check for statistical keywords
        if any(word in question for word in ['describe', 'summary', 'statistics', 'stats']):
            return QuestionType.STATISTICAL
        
        # Default to descriptive
        return QuestionType.DESCRIPTIVE
    
    def _extract_aggregation(self, question: str) -> Optional[AggregationType]:
        """Extract aggregation type from question."""
        for keyword, agg_type in self.aggregation_keywords.items():
            if keyword in question:
                return agg_type
        return None
    
    def _extract_filters(self, question: str, schema: DataSchema) -> List[Dict[str, Any]]:
        """Extract filter conditions from question."""
        filters = []
        
        # Look for common filter patterns
        # Pattern: "where column = value" or "where column_name is value"
        where_patterns = [
            r'where\s+(\w+)\s*(=|>|<|>=|<=|!=|is)\s*(["\']?[^"\']+["\']?)',
            r'where\s+([a-zA-Z_]+)\s+(is|equals?|greater\s+than|less\s+than)\s+(["\']?[^"\']+["\']?)'
        ]
        
        for pattern in where_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                column, operator, value = match
                # Clean up the value
                value = value.strip('"\'')
                
                # Map column name to actual schema column
                actual_column = self._find_matching_column(column, schema)
                if actual_column:
                    filters.append({
                        'column': actual_column,
                        'operator': operator,
                        'value': value
                    })
        
        # Pattern: "show only records with column > value"
        show_patterns = [
            r'show.*?(\w+)\s*(>|<|>=|<=|=)\s*(["\']?[^"\']+["\']?)',
            r'with\s+([a-zA-Z_]+)\s+(greater\s+than|less\s+than|equals?)\s+(["\']?[^"\']+["\']?)'
        ]
        
        for pattern in show_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                column, operator, value = match
                value = value.strip('"\'')
                
                actual_column = self._find_matching_column(column, schema)
                if actual_column:
                    filters.append({
                        'column': actual_column,
                        'operator': operator,
                        'value': value
                    })
        
        return filters
    
    def _find_matching_column(self, column_text: str, schema: DataSchema) -> Optional[str]:
        """Find the actual column name that matches the text."""
        column_lower = column_text.lower()
        
        for column in schema.columns:
            column_name = column.name
            column_name_lower = column_name.lower()
            
            # Exact match
            if column_lower == column_name_lower:
                return column_name
            
            # Match with underscores replaced by spaces
            if column_lower == column_name_lower.replace('_', ' '):
                return column_name
            
            # Match individual words
            if column_lower in column_name_lower.split('_'):
                return column_name
        
        return None
    
    def _calculate_confidence(self, question: str, entities: Dict[str, Any], 
                            question_type: QuestionType) -> float:
        """Calculate confidence score for the parsed intent."""
        confidence = 0.0
        
        # Base confidence for having entities
        if entities['columns']:
            confidence += 0.4
        
        # Bonus for question type classification
        if question_type != QuestionType.DESCRIPTIVE:
            confidence += 0.3
        
        # Bonus for having specific keywords
        keyword_lists = [
            self.aggregation_keywords.keys(),
            self.comparison_keywords,
            self.trend_keywords,
            self.filtering_keywords,
            self.ranking_keywords,
            self.distribution_keywords
        ]
        
        for keyword_list in keyword_lists:
            if any(keyword in question for keyword in keyword_list):
                confidence += 0.1
                break
        
        # Bonus for having numbers (often indicates specific queries)
        if entities['numbers']:
            confidence += 0.1
        
        # Bonus for having filter values
        if entities['values']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def extract_column_references(self, question: str, schema: DataSchema) -> List[Tuple[str, float]]:
        """
        Extract column references with confidence scores.
        
        Args:
            question: Natural language question
            schema: Data schema for column matching
            
        Returns:
            List of (column_name, confidence) tuples
        """
        question_lower = question.lower()
        column_matches = []
        
        for column in schema.columns:
            column_name = column.name
            column_name_lower = column_name.lower()
            confidence = 0.0
            
            # Exact match
            if column_name_lower in question_lower:
                confidence = 1.0
            else:
                # Match with spaces instead of underscores
                column_spaced = column_name_lower.replace('_', ' ')
                if column_spaced in question_lower:
                    confidence = 0.9
                else:
                    # Partial word matches
                    column_words = column_name_lower.split('_')
                    question_words = question_lower.split()
                    
                    matches = sum(1 for word in column_words if word in question_words)
                    if matches > 0:
                        confidence = matches / len(column_words)
            
            if confidence > 0:
                column_matches.append((column_name, confidence))
        
        # Sort by confidence
        column_matches.sort(key=lambda x: x[1], reverse=True)
        return column_matches
    
    def classify_question_complexity(self, intent: ParsedIntent) -> str:
        """
        Classify the complexity of the question.
        
        Args:
            intent: Parsed intent
            
        Returns:
            Complexity level: 'simple', 'medium', 'complex'
        """
        complexity_score = 0
        
        # Multiple columns increase complexity
        if len(intent.columns) > 1:
            complexity_score += 1
        
        # Filters increase complexity
        if intent.filters and len(intent.filters) > 0:
            complexity_score += 1
        
        # Certain question types are more complex
        complex_types = [QuestionType.COMPARISON, QuestionType.TREND, QuestionType.STATISTICAL]
        if intent.question_type in complex_types:
            complexity_score += 1
        
        # Multiple filters are very complex
        if intent.filters and len(intent.filters) > 1:
            complexity_score += 1
        
        if complexity_score == 0:
            return 'simple'
        elif complexity_score <= 2:
            return 'medium'
        else:
            return 'complex'
    
    def suggest_clarifications(self, intent: ParsedIntent, schema: DataSchema) -> List[str]:
        """
        Suggest clarifications for ambiguous questions.
        
        Args:
            intent: Parsed intent
            schema: Data schema
            
        Returns:
            List of clarification suggestions
        """
        suggestions = []
        
        # Low confidence suggests ambiguity
        if intent.confidence < 0.5:
            suggestions.append("Could you rephrase your question to be more specific?")
        
        # No columns detected
        if not intent.columns:
            available_columns = [col.name for col in schema.columns]
            suggestions.append(f"Available columns are: {', '.join(available_columns)}")
        
        # Ambiguous aggregation
        if intent.question_type == QuestionType.AGGREGATION and not intent.aggregation:
            suggestions.append("What type of calculation would you like? (sum, average, count, etc.)")
        
        # Missing comparison target
        if intent.question_type == QuestionType.COMPARISON and len(intent.columns) < 2:
            suggestions.append("What would you like to compare? Please specify two columns or values.")
        
        return suggestions