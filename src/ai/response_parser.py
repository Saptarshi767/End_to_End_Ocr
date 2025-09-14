"""
Response parsing and validation component for LLM outputs.
"""

import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.models import Query, ValidationResult, DataSchema


class ResponseType(Enum):
    """Types of responses from LLM."""
    SQL_QUERY = "sql_query"
    ANALYSIS = "analysis"
    EXPLANATION = "explanation"
    SUGGESTION = "suggestion"
    ERROR_DIAGNOSIS = "error_diagnosis"


@dataclass
class ParsedResponse:
    """Parsed and validated LLM response."""
    response_type: ResponseType
    content: str
    structured_data: Dict[str, Any]
    confidence: float
    validation_result: ValidationResult
    extracted_queries: List[Query] = None


class ResponseParser:
    """Parse and validate LLM responses for different tasks."""
    
    def __init__(self):
        self.sql_patterns = [
            r'```sql\s*(.*?)\s*```',  # SQL code blocks
            r'```\s*(SELECT.*?)\s*```',  # Generic code blocks with SELECT
            r'(SELECT\s+.*?(?:;|\n|$))',  # Standalone SELECT statements
            r'(WITH\s+.*?SELECT\s+.*?(?:;|\n|$))',  # CTE queries
            r'(INSERT\s+.*?(?:;|\n|$))',  # INSERT statements
            r'(UPDATE\s+.*?(?:;|\n|$))',  # UPDATE statements
            r'(DELETE\s+.*?(?:;|\n|$))'   # DELETE statements
        ]
        
        self.analysis_indicators = [
            'insights', 'analysis', 'findings', 'patterns', 'trends',
            'recommendations', 'conclusions', 'observations'
        ]
        
        self.explanation_indicators = [
            'explanation', 'because', 'due to', 'this means', 'in other words',
            'specifically', 'for example', 'this shows'
        ]
    
    def parse_response(self, response: str, expected_type: ResponseType, 
                      context: Optional[Dict[str, Any]] = None) -> ParsedResponse:
        """
        Parse LLM response based on expected type.
        
        Args:
            response: Raw LLM response
            expected_type: Expected response type
            context: Additional context for parsing
            
        Returns:
            ParsedResponse with structured data
        """
        if expected_type == ResponseType.SQL_QUERY:
            return self._parse_sql_response(response, context)
        elif expected_type == ResponseType.ANALYSIS:
            return self._parse_analysis_response(response, context)
        elif expected_type == ResponseType.EXPLANATION:
            return self._parse_explanation_response(response, context)
        elif expected_type == ResponseType.SUGGESTION:
            return self._parse_suggestion_response(response, context)
        elif expected_type == ResponseType.ERROR_DIAGNOSIS:
            return self._parse_error_diagnosis_response(response, context)
        else:
            return self._parse_generic_response(response, expected_type, context)
    
    def _parse_sql_response(self, response: str, context: Optional[Dict[str, Any]]) -> ParsedResponse:
        """Parse SQL query response."""
        queries = self.extract_sql_queries(response)
        
        if not queries:
            # Try to extract SQL without code blocks
            cleaned_response = response.strip()
            if self._looks_like_sql(cleaned_response):
                queries = [Query(sql=cleaned_response, columns=['*'])]
        
        # Validate queries if schema is provided
        validation_result = ValidationResult(is_valid=True, confidence=1.0)
        if context and 'schema' in context and queries:
            validation_result = self._validate_sql_queries(queries, context['schema'])
        
        confidence = self._calculate_sql_confidence(response, queries, validation_result)
        
        structured_data = {
            'query_count': len(queries),
            'primary_query': queries[0].sql if queries else None,
            'has_aggregation': any('SUM(' in q.sql or 'COUNT(' in q.sql or 'AVG(' in q.sql 
                                 for q in queries),
            'has_grouping': any('GROUP BY' in q.sql.upper() for q in queries),
            'has_filtering': any('WHERE' in q.sql.upper() for q in queries),
            'has_ordering': any('ORDER BY' in q.sql.upper() for q in queries)
        }
        
        return ParsedResponse(
            response_type=ResponseType.SQL_QUERY,
            content=response,
            structured_data=structured_data,
            confidence=confidence,
            validation_result=validation_result,
            extracted_queries=queries
        )
    
    def _parse_analysis_response(self, response: str, context: Optional[Dict[str, Any]]) -> ParsedResponse:
        """Parse data analysis response."""
        sections = self._extract_analysis_sections(response)
        
        structured_data = {
            'sections': sections,
            'has_insights': any('insight' in section.lower() for section in sections.values()) or 'insight' in response.lower(),
            'has_recommendations': any('recommend' in section.lower() for section in sections.values()) or 'recommend' in response.lower(),
            'has_numbers': bool(re.search(r'\d+(?:\.\d+)?%?', response)),
            'key_metrics': self._extract_metrics(response)
        }
        
        confidence = self._calculate_analysis_confidence(response, structured_data)
        
        validation_result = ValidationResult(
            is_valid=len(sections) > 0,
            confidence=confidence,
            warnings=[] if len(sections) > 0 else ["No clear analysis sections found"]
        )
        
        return ParsedResponse(
            response_type=ResponseType.ANALYSIS,
            content=response,
            structured_data=structured_data,
            confidence=confidence,
            validation_result=validation_result
        )
    
    def _parse_explanation_response(self, response: str, context: Optional[Dict[str, Any]]) -> ParsedResponse:
        """Parse explanation response."""
        explanation_elements = self._extract_explanation_elements(response)
        
        structured_data = {
            'elements': explanation_elements,
            'has_reasoning': any('because' in elem.lower() or 'due to' in elem.lower() 
                              for elem in explanation_elements),
            'has_examples': any('example' in elem.lower() or 'for instance' in elem.lower() 
                               for elem in explanation_elements),
            'clarity_score': self._calculate_clarity_score(response)
        }
        
        confidence = min(0.9, len(explanation_elements) * 0.2 + 0.3)
        
        validation_result = ValidationResult(
            is_valid=len(explanation_elements) > 0,
            confidence=confidence
        )
        
        return ParsedResponse(
            response_type=ResponseType.EXPLANATION,
            content=response,
            structured_data=structured_data,
            confidence=confidence,
            validation_result=validation_result
        )
    
    def _parse_suggestion_response(self, response: str, context: Optional[Dict[str, Any]]) -> ParsedResponse:
        """Parse suggestion response."""
        suggestions = self._extract_suggestions(response)
        
        structured_data = {
            'suggestions': suggestions,
            'suggestion_count': len(suggestions),
            'has_rationale': any('why:' in sugg.lower() or 'because' in sugg.lower() 
                                for sugg in suggestions),
            'actionable_count': sum(1 for sugg in suggestions if self._is_actionable(sugg))
        }
        
        confidence = min(0.9, len(suggestions) * 0.15 + 0.4)
        
        validation_result = ValidationResult(
            is_valid=len(suggestions) > 0,
            confidence=confidence
        )
        
        return ParsedResponse(
            response_type=ResponseType.SUGGESTION,
            content=response,
            structured_data=structured_data,
            confidence=confidence,
            validation_result=validation_result
        )
    
    def _parse_error_diagnosis_response(self, response: str, context: Optional[Dict[str, Any]]) -> ParsedResponse:
        """Parse error diagnosis response."""
        diagnosis_elements = self._extract_diagnosis_elements(response)
        
        structured_data = {
            'cause': diagnosis_elements.get('cause', ''),
            'solution': diagnosis_elements.get('solution', ''),
            'prevention': diagnosis_elements.get('prevention', ''),
            'has_specific_fix': bool(diagnosis_elements.get('solution')),
            'has_prevention_tips': bool(diagnosis_elements.get('prevention'))
        }
        
        confidence = self._calculate_diagnosis_confidence(diagnosis_elements)
        
        validation_result = ValidationResult(
            is_valid=bool(diagnosis_elements.get('cause') or diagnosis_elements.get('solution')),
            confidence=confidence
        )
        
        return ParsedResponse(
            response_type=ResponseType.ERROR_DIAGNOSIS,
            content=response,
            structured_data=structured_data,
            confidence=confidence,
            validation_result=validation_result
        )
    
    def _parse_generic_response(self, response: str, response_type: ResponseType, 
                               context: Optional[Dict[str, Any]]) -> ParsedResponse:
        """Parse generic response."""
        structured_data = {
            'length': len(response),
            'word_count': len(response.split()),
            'has_structure': bool(re.search(r'\d+\.|\-|\*', response))
        }
        
        confidence = 0.5  # Default confidence for generic responses
        
        validation_result = ValidationResult(
            is_valid=len(response.strip()) > 0,
            confidence=confidence
        )
        
        return ParsedResponse(
            response_type=response_type,
            content=response,
            structured_data=structured_data,
            confidence=confidence,
            validation_result=validation_result
        )
    
    def extract_sql_queries(self, text: str) -> List[Query]:
        """Extract SQL queries from text."""
        queries = []
        
        for pattern in self.sql_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Clean up the query
                sql = match.strip().rstrip(';')
                if sql and len(sql) > 10:  # Minimum length check
                    queries.append(Query(sql=sql, columns=['*']))
        
        # Remove duplicates
        unique_queries = []
        seen_sql = set()
        for query in queries:
            normalized_sql = ' '.join(query.sql.split()).upper()
            if normalized_sql not in seen_sql:
                seen_sql.add(normalized_sql)
                unique_queries.append(query)
        
        return unique_queries
    
    def _looks_like_sql(self, text: str) -> bool:
        """Check if text looks like a SQL query."""
        text_upper = text.upper()
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'INSERT', 'UPDATE', 'DELETE']
        
        # Must contain at least SELECT and FROM for basic queries
        if 'SELECT' in text_upper and 'FROM' in text_upper:
            return True
        
        # Or contain other SQL keywords
        keyword_count = sum(1 for keyword in sql_keywords if keyword in text_upper)
        return keyword_count >= 2
    
    def _validate_sql_queries(self, queries: List[Query], schema: DataSchema) -> ValidationResult:
        """Validate SQL queries against schema."""
        errors = []
        warnings = []
        
        schema_columns = {col.name.lower() for col in schema.columns}
        
        for i, query in enumerate(queries):
            sql_lower = query.sql.lower()
            
            # Check for basic SQL structure
            if 'select' not in sql_lower:
                errors.append(f"Query {i+1}: Missing SELECT clause")
                continue
            
            if 'from' not in sql_lower:
                errors.append(f"Query {i+1}: Missing FROM clause")
                continue
            
            # Extract column references (simple approach)
            # This is a basic implementation - a full parser would be more robust
            column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            potential_columns = re.findall(column_pattern, sql_lower)
            
            for col in potential_columns:
                if col not in ['select', 'from', 'where', 'group', 'by', 'order', 'having', 
                              'limit', 'distinct', 'as', 'and', 'or', 'not', 'in', 'like',
                              'data_table', 'count', 'sum', 'avg', 'min', 'max', 'desc', 'asc']:
                    if col not in schema_columns and col != '*':
                        warnings.append(f"Query {i+1}: Column '{col}' not found in schema")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=1.0 if len(errors) == 0 else max(0.0, 1.0 - len(errors) * 0.2)
        )
    
    def _calculate_sql_confidence(self, response: str, queries: List[Query], 
                                 validation: ValidationResult) -> float:
        """Calculate confidence score for SQL response."""
        confidence = 0.0
        
        # Base confidence from having queries
        if queries:
            confidence += 0.4
        
        # Bonus for validation
        confidence += validation.confidence * 0.3
        
        # Bonus for proper formatting
        if '```sql' in response or '```' in response:
            confidence += 0.1
        
        # Bonus for SQL keywords
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY']
        keyword_count = sum(1 for keyword in sql_keywords if keyword in response.upper())
        confidence += min(0.2, keyword_count * 0.05)
        
        return min(1.0, confidence)
    
    def _extract_analysis_sections(self, response: str) -> Dict[str, str]:
        """Extract analysis sections from response."""
        sections = {}
        
        # Look for numbered sections
        numbered_pattern = r'(\d+)\.\s*([^:\n]+):\s*([^\n]+)'
        matches = re.findall(numbered_pattern, response, re.MULTILINE)
        
        for num, title, content in matches:
            sections[title.strip()] = content.strip()
        
        # Look for bullet points
        if not sections:
            bullet_pattern = r'[-*]\s*([^:]+):\s*([^\n]+(?:\n(?![-*])[^\n]*)*)'
            matches = re.findall(bullet_pattern, response, re.MULTILINE)
            
            for title, content in matches:
                sections[title.strip()] = content.strip()
        
        # If no structured sections, try to identify key paragraphs
        if not sections:
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs):
                if any(indicator in paragraph.lower() for indicator in self.analysis_indicators):
                    sections[f"Analysis {i+1}"] = paragraph
        
        return sections
    
    def _extract_metrics(self, response: str) -> List[Dict[str, Any]]:
        """Extract numerical metrics from response."""
        metrics = []
        
        # Pattern for numbers with units/percentages
        metric_pattern = r'(\d+(?:\.\d+)?)\s*(%|dollars?|\$|units?|customers?|orders?)'
        matches = re.findall(metric_pattern, response, re.IGNORECASE)
        
        for value, unit in matches:
            metrics.append({
                'value': float(value),
                'unit': unit,
                'formatted': f"{value}{unit}"
            })
        
        return metrics
    
    def _calculate_analysis_confidence(self, response: str, structured_data: Dict[str, Any]) -> float:
        """Calculate confidence for analysis response."""
        confidence = 0.3  # Base confidence
        
        # Bonus for having sections
        if structured_data['sections']:
            confidence += 0.3
        
        # Bonus for insights and recommendations
        if structured_data['has_insights']:
            confidence += 0.2
        
        if structured_data['has_recommendations']:
            confidence += 0.2
        
        # Bonus for having numbers/metrics
        if structured_data['has_numbers']:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_explanation_elements(self, response: str) -> List[str]:
        """Extract explanation elements from response."""
        elements = []
        
        # Split by sentences and filter for explanatory content
        sentences = re.split(r'[.!?]+', response)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and any(indicator in sentence.lower() 
                                        for indicator in self.explanation_indicators):
                elements.append(sentence)
        
        return elements
    
    def _calculate_clarity_score(self, response: str) -> float:
        """Calculate clarity score for explanation."""
        # Simple heuristic based on sentence length and complexity
        sentences = re.split(r'[.!?]+', response)
        
        if not sentences:
            return 0.0
        
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Optimal sentence length is around 15-20 words
        if 10 <= avg_length <= 25:
            return 0.8
        elif 5 <= avg_length <= 35:
            return 0.6
        else:
            return 0.4
    
    def _extract_suggestions(self, response: str) -> List[str]:
        """Extract suggestions from response."""
        suggestions = []
        
        # Look for numbered suggestions
        numbered_pattern = r'(\d+)\.\s*([^\n]+)'
        matches = re.findall(numbered_pattern, response, re.MULTILINE)
        
        for num, content in matches:
            suggestions.append(content.strip())
        
        # Look for bullet points
        if not suggestions:
            bullet_pattern = r'[-*]\s*([^\n]+(?:\n(?![-*])[^\n]*)*)'
            matches = re.findall(bullet_pattern, response, re.MULTILINE)
            
            for content in matches:
                suggestions.append(content.strip())
        
        return suggestions
    
    def _is_actionable(self, suggestion: str) -> bool:
        """Check if suggestion is actionable."""
        actionable_words = ['analyze', 'investigate', 'compare', 'review', 'examine', 
                           'calculate', 'measure', 'track', 'monitor', 'implement']
        
        return any(word in suggestion.lower() for word in actionable_words)
    
    def _extract_diagnosis_elements(self, response: str) -> Dict[str, str]:
        """Extract diagnosis elements from response."""
        elements = {}
        
        # Look for structured sections
        patterns = {
            'cause': r'(?:cause|reason|problem):\s*([^\n]+(?:\n(?!(?:fix|solution|prevent))[^\n]*)*)',
            'solution': r'(?:fix|solution|resolve):\s*([^\n]+(?:\n(?!(?:cause|prevent))[^\n]*)*)',
            'prevention': r'(?:prevent|avoid|tips?):\s*([^\n]+(?:\n(?!(?:cause|fix))[^\n]*)*)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                elements[key] = match.group(1).strip()
        
        return elements
    
    def _calculate_diagnosis_confidence(self, elements: Dict[str, str]) -> float:
        """Calculate confidence for diagnosis response."""
        confidence = 0.2  # Base confidence
        
        if elements.get('cause'):
            confidence += 0.3
        
        if elements.get('solution'):
            confidence += 0.4
        
        if elements.get('prevention'):
            confidence += 0.1
        
        return min(1.0, confidence)