"""
Question classification component for categorizing user questions.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .nlp_processor import QuestionType, AggregationType
from ..core.models import DataSchema, DataType


@dataclass
class ClassificationResult:
    """Result of question classification."""
    question_type: QuestionType
    confidence: float
    features: Dict[str, Any]
    suggested_approach: str
    complexity_level: str


class QuestionClassifier:
    """Classify questions into different types for appropriate handling."""
    
    def __init__(self):
        # Define keyword patterns for each question type
        self.classification_patterns = {
            QuestionType.AGGREGATION: {
                'keywords': [
                    'sum', 'total', 'add up', 'count', 'how many', 'number of',
                    'average', 'mean', 'avg', 'minimum', 'min', 'maximum', 'max',
                    'median', 'standard deviation', 'std'
                ],
                'patterns': [
                    r'what.*(is|are).*(total|sum|count|average|mean)',
                    r'how many.*',
                    r'calculate.*(sum|total|average|mean)',
                    r'find.*(minimum|maximum|min|max)'
                ]
            },
            
            QuestionType.FILTERING: {
                'keywords': [
                    'where', 'filter', 'show', 'display', 'only', 'exclude',
                    'include', 'with', 'without', 'having'
                ],
                'patterns': [
                    r'show.*where.*',
                    r'filter.*by.*',
                    r'only.*with.*',
                    r'exclude.*',
                    r'records.*where.*'
                ]
            },
            
            QuestionType.COMPARISON: {
                'keywords': [
                    'compare', 'versus', 'vs', 'against', 'difference', 'between',
                    'higher', 'lower', 'better', 'worse', 'more', 'less'
                ],
                'patterns': [
                    r'compare.*with.*',
                    r'.*vs.*',
                    r'difference between.*',
                    r'which.*higher.*',
                    r'.*better than.*'
                ]
            },
            
            QuestionType.TREND: {
                'keywords': [
                    'trend', 'over time', 'change', 'growth', 'decline',
                    'increase', 'decrease', 'pattern', 'evolution',
                    'monthly', 'yearly', 'daily', 'weekly'
                ],
                'patterns': [
                    r'.*over time.*',
                    r'trend.*',
                    r'change.*over.*',
                    r'growth.*',
                    r'.*monthly.*',
                    r'.*yearly.*'
                ]
            },
            
            QuestionType.RANKING: {
                'keywords': [
                    'top', 'bottom', 'best', 'worst', 'highest', 'lowest',
                    'rank', 'first', 'last', 'leading', 'trailing'
                ],
                'patterns': [
                    r'top \d+.*',
                    r'bottom \d+.*',
                    r'best.*',
                    r'worst.*',
                    r'highest.*',
                    r'lowest.*'
                ]
            },
            
            QuestionType.DISTRIBUTION: {
                'keywords': [
                    'distribution', 'spread', 'breakdown', 'proportion',
                    'percentage', 'share', 'composition', 'split'
                ],
                'patterns': [
                    r'distribution.*',
                    r'breakdown.*',
                    r'what percentage.*',
                    r'proportion.*',
                    r'how.*distributed.*'
                ]
            },
            
            QuestionType.STATISTICAL: {
                'keywords': [
                    'describe', 'summary', 'statistics', 'stats', 'overview',
                    'analysis', 'insights', 'correlation', 'relationship'
                ],
                'patterns': [
                    r'describe.*',
                    r'summary.*',
                    r'statistics.*',
                    r'correlation.*',
                    r'relationship.*'
                ]
            }
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'simple': [
                'count', 'sum', 'show', 'list', 'what is'
            ],
            'medium': [
                'average', 'compare', 'filter', 'group by', 'where'
            ],
            'complex': [
                'correlation', 'trend', 'distribution', 'statistical',
                'multiple', 'join', 'relationship'
            ]
        }
    
    def classify_question(self, question: str, schema: DataSchema) -> ClassificationResult:
        """
        Classify a question into its type and complexity.
        
        Args:
            question: Natural language question
            schema: Data schema for context
            
        Returns:
            ClassificationResult with type and metadata
        """
        question_lower = question.lower().strip()
        
        # Calculate scores for each question type
        type_scores = {}
        features = {}
        
        for q_type, patterns in self.classification_patterns.items():
            score = self._calculate_type_score(question_lower, patterns)
            type_scores[q_type] = score
            
            # Store matching features
            matching_keywords = [kw for kw in patterns['keywords'] if kw in question_lower]
            matching_patterns = [p for p in patterns['patterns'] if re.search(p, question_lower)]
            
            features[q_type.value] = {
                'keywords': matching_keywords,
                'patterns': matching_patterns,
                'score': score
            }
        
        # Determine the best match
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        confidence = type_scores[best_type]
        
        # If no strong match, default to descriptive
        if confidence < 0.3:
            best_type = QuestionType.DESCRIPTIVE
            confidence = 0.5
        
        # Determine complexity
        complexity = self._determine_complexity(question_lower, schema)
        
        # Suggest approach
        approach = self._suggest_approach(best_type, complexity, schema)
        
        return ClassificationResult(
            question_type=best_type,
            confidence=confidence,
            features=features,
            suggested_approach=approach,
            complexity_level=complexity
        )
    
    def _calculate_type_score(self, question: str, patterns: Dict[str, List[str]]) -> float:
        """Calculate score for a specific question type."""
        score = 0.0
        
        # Keyword matching
        keyword_matches = sum(1 for kw in patterns['keywords'] if kw in question)
        keyword_score = min(keyword_matches * 0.2, 0.6)  # Max 0.6 from keywords
        
        # Pattern matching
        pattern_matches = sum(1 for p in patterns['patterns'] if re.search(p, question))
        pattern_score = min(pattern_matches * 0.3, 0.4)  # Max 0.4 from patterns
        
        score = keyword_score + pattern_score
        return min(score, 1.0)
    
    def _determine_complexity(self, question: str, schema: DataSchema) -> str:
        """Determine the complexity level of the question."""
        complexity_score = 0
        
        # Check for complexity indicators
        for level, indicators in self.complexity_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in question)
            if level == 'simple' and matches > 0:
                complexity_score -= 1
            elif level == 'medium' and matches > 0:
                complexity_score += 1
            elif level == 'complex' and matches > 0:
                complexity_score += 2
        
        # Multiple column references increase complexity
        column_count = sum(1 for col in schema.columns if col.name.lower() in question)
        if column_count > 2:
            complexity_score += 1
        
        # Multiple conditions increase complexity
        condition_indicators = ['and', 'or', 'where', 'having', 'group by']
        condition_count = sum(1 for indicator in condition_indicators if indicator in question)
        if condition_count > 1:
            complexity_score += 1
        
        # Determine final complexity
        if complexity_score <= 0:
            return 'simple'
        elif complexity_score <= 2:
            return 'medium'
        else:
            return 'complex'
    
    def _suggest_approach(self, question_type: QuestionType, complexity: str, 
                         schema: DataSchema) -> str:
        """Suggest the best approach for handling this question type."""
        
        approaches = {
            QuestionType.AGGREGATION: {
                'simple': 'Direct aggregation function',
                'medium': 'Grouped aggregation with filtering',
                'complex': 'Multi-level aggregation with joins'
            },
            QuestionType.FILTERING: {
                'simple': 'Simple WHERE clause',
                'medium': 'Multiple conditions with AND/OR',
                'complex': 'Subqueries and complex filtering'
            },
            QuestionType.COMPARISON: {
                'simple': 'Side-by-side comparison',
                'medium': 'Statistical comparison with metrics',
                'complex': 'Multi-dimensional comparison analysis'
            },
            QuestionType.TREND: {
                'simple': 'Time series line chart',
                'medium': 'Trend analysis with moving averages',
                'complex': 'Advanced time series analysis'
            },
            QuestionType.RANKING: {
                'simple': 'ORDER BY with LIMIT',
                'medium': 'Ranking with grouping',
                'complex': 'Multi-criteria ranking'
            },
            QuestionType.DISTRIBUTION: {
                'simple': 'Basic histogram or pie chart',
                'medium': 'Distribution analysis with statistics',
                'complex': 'Advanced statistical distribution'
            },
            QuestionType.STATISTICAL: {
                'simple': 'Basic descriptive statistics',
                'medium': 'Comprehensive statistical summary',
                'complex': 'Advanced statistical analysis'
            },
            QuestionType.DESCRIPTIVE: {
                'simple': 'Data overview and summary',
                'medium': 'Detailed data exploration',
                'complex': 'Comprehensive data analysis'
            }
        }
        
        return approaches.get(question_type, {}).get(complexity, 'General data analysis')
    
    def get_question_features(self, question: str) -> Dict[str, Any]:
        """
        Extract various features from the question for analysis.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary of extracted features
        """
        question_lower = question.lower()
        
        features = {
            'length': len(question),
            'word_count': len(question.split()),
            'has_numbers': bool(re.search(r'\d+', question)),
            'has_quotes': bool(re.search(r'["\']', question)),
            'question_words': [],
            'aggregation_words': [],
            'comparison_words': [],
            'time_words': []
        }
        
        # Extract question words
        question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who']
        features['question_words'] = [w for w in question_words if w in question_lower]
        
        # Extract aggregation words
        agg_words = ['sum', 'count', 'average', 'total', 'min', 'max']
        features['aggregation_words'] = [w for w in agg_words if w in question_lower]
        
        # Extract comparison words
        comp_words = ['compare', 'versus', 'vs', 'difference', 'higher', 'lower']
        features['comparison_words'] = [w for w in comp_words if w in question_lower]
        
        # Extract time-related words
        time_words = ['time', 'date', 'year', 'month', 'day', 'trend', 'over']
        features['time_words'] = [w for w in time_words if w in question_lower]
        
        return features
    
    def suggest_question_improvements(self, question: str, classification: ClassificationResult) -> List[str]:
        """
        Suggest improvements to make the question clearer.
        
        Args:
            question: Original question
            classification: Classification result
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Low confidence suggests unclear question
        if classification.confidence < 0.5:
            suggestions.append("Try to be more specific about what you want to know")
        
        # Suggest specific improvements based on question type
        if classification.question_type == QuestionType.AGGREGATION:
            if not any(word in question.lower() for word in ['sum', 'count', 'average', 'total']):
                suggestions.append("Specify the type of calculation (sum, count, average, etc.)")
        
        elif classification.question_type == QuestionType.COMPARISON:
            if 'compare' in question.lower() and 'with' not in question.lower():
                suggestions.append("Specify what you want to compare with what")
        
        elif classification.question_type == QuestionType.FILTERING:
            if 'where' not in question.lower() and 'filter' not in question.lower():
                suggestions.append("Use 'where' or 'filter by' to specify conditions")
        
        # General suggestions
        if len(question.split()) < 3:
            suggestions.append("Provide more context in your question")
        
        if not re.search(r'[?.]$', question):
            suggestions.append("End your question with a question mark")
        
        return suggestions