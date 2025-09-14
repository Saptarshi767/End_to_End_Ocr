"""
Response formatting system for conversational AI responses.
"""

import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .response_parser import ParsedResponse, ResponseType
from ..core.models import ConversationResponse, Chart, QueryResult, DataSchema
from ..visualization.chart_selector import ChartTypeSelector
from ..visualization.chart_engines import PlotlyChartEngine


class ResponseFormat(Enum):
    """Available response formats."""
    TEXT_ONLY = "text_only"
    TEXT_WITH_DATA = "text_with_data"
    TEXT_WITH_CHARTS = "text_with_charts"
    INTERACTIVE = "interactive"
    SUMMARY = "summary"


@dataclass
class FormattingContext:
    """Context for response formatting."""
    query_result: Optional[QueryResult] = None
    schema: Optional[DataSchema] = None
    original_question: str = ""
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    include_visualizations: bool = True
    include_suggestions: bool = True
    max_data_rows: int = 100


class ResponseFormatter:
    """Format conversational AI responses with text and visualizations."""
    
    def __init__(self):
        self.chart_selector = ChartTypeSelector()
        self.chart_engine = PlotlyChartEngine()
        
        # Response templates
        self.templates = {
            ResponseType.SQL_QUERY: self._format_query_response,
            ResponseType.ANALYSIS: self._format_analysis_response,
            ResponseType.EXPLANATION: self._format_explanation_response,
            ResponseType.SUGGESTION: self._format_suggestion_response,
            ResponseType.ERROR_DIAGNOSIS: self._format_error_response
        }
    
    def format_response(self, parsed_response: ParsedResponse, 
                       context: FormattingContext) -> ConversationResponse:
        """
        Format parsed LLM response into conversational response.
        
        Args:
            parsed_response: Parsed LLM response
            context: Formatting context
            
        Returns:
            ConversationResponse with formatted content
        """
        formatter = self.templates.get(parsed_response.response_type, self._format_generic_response)
        return formatter(parsed_response, context)
    
    def _format_query_response(self, parsed_response: ParsedResponse, 
                              context: FormattingContext) -> ConversationResponse:
        """Format SQL query response."""
        text_parts = []
        visualizations = []
        data_summary = {}
        
        # Add query execution results if available
        if context.query_result:
            result = context.query_result
            
            # Format text response
            if result.row_count == 0:
                text_parts.append("No data found matching your query.")
            elif result.row_count == 1:
                text_parts.append("Found 1 record:")
                text_parts.append(self._format_single_record(result.data[0]))
            else:
                text_parts.append(f"Found {result.row_count} records.")
                
                # Add summary statistics if numeric data
                summary_stats = self._calculate_summary_stats(result.data, result.columns)
                if summary_stats:
                    text_parts.append("\nSummary:")
                    for stat_name, stat_value in summary_stats.items():
                        text_parts.append(f"• {stat_name}: {stat_value}")
            
            # Create visualizations if requested
            if context.include_visualizations and result.row_count > 0:
                visualizations = self._create_visualizations_from_data(result, context)
            
            # Data summary
            data_summary = {
                'total_rows': result.row_count,
                'columns': result.columns,
                'execution_time_ms': result.execution_time_ms,
                'has_numeric_data': self._has_numeric_data(result.data),
                'sample_data': result.data[:min(5, len(result.data))]
            }
        else:
            # No query result, just show the generated query
            if parsed_response.extracted_queries:
                query = parsed_response.extracted_queries[0]
                text_parts.append("Generated SQL query:")
                text_parts.append(f"```sql\n{query.sql}\n```")
                
                if parsed_response.validation_result.warnings:
                    text_parts.append("\nWarnings:")
                    for warning in parsed_response.validation_result.warnings:
                        text_parts.append(f"• {warning}")
            else:
                text_parts.append("Could not generate a valid SQL query for your question.")
        
        # Generate follow-up suggestions
        suggestions = []
        if context.include_suggestions:
            suggestions = self._generate_query_suggestions(context)
        
        return ConversationResponse(
            text_response="\n".join(text_parts),
            visualizations=visualizations,
            data_summary=data_summary,
            suggested_questions=suggestions,
            confidence=parsed_response.confidence
        )
    
    def _format_analysis_response(self, parsed_response: ParsedResponse, 
                                 context: FormattingContext) -> ConversationResponse:
        """Format data analysis response with enhanced explanations."""
        text_parts = []
        visualizations = []
        data_summary = {}
        
        # Use the LLM's analysis as the main text
        text_parts.append(parsed_response.content)
        
        # Add data-driven insights if query result is available
        if context.query_result:
            result = context.query_result
            
            # Generate detailed explanation for complex queries
            explanation = self._generate_complex_query_explanation(result, context)
            if explanation:
                text_parts.append(f"\n**Detailed Analysis:**")
                text_parts.append(explanation)
            
            # Add quantitative insights
            insights = self._extract_data_insights(result)
            if insights:
                text_parts.append("\n**Key Insights:**")
                for insight in insights:
                    text_parts.append(f"• {insight}")
            
            # Add statistical summary for numeric data
            statistical_summary = self._generate_statistical_summary(result)
            if statistical_summary:
                text_parts.append(f"\n**Statistical Summary:**")
                text_parts.append(statistical_summary)
            
            # Create supporting visualizations
            if context.include_visualizations and result.row_count > 0:
                visualizations = self._create_visualizations_from_data(result, context)
                
                # Add explanation for visualizations
                if visualizations:
                    viz_explanation = self._explain_visualizations(visualizations, result)
                    if viz_explanation:
                        text_parts.append(f"\n**Visualization Insights:**")
                        text_parts.append(viz_explanation)
            
            data_summary = {
                'total_rows': result.row_count,
                'key_metrics': parsed_response.structured_data.get('key_metrics', []),
                'has_recommendations': parsed_response.structured_data.get('has_recommendations', False),
                'complexity_score': self._calculate_query_complexity(result, context),
                'data_quality_score': self._assess_data_quality(result)
            }
        
        # Generate contextual follow-up suggestions
        suggestions = []
        if context.include_suggestions:
            suggestions = self._generate_analysis_suggestions(parsed_response, context)
        
        return ConversationResponse(
            text_response="\n".join(text_parts),
            visualizations=visualizations,
            data_summary=data_summary,
            suggested_questions=suggestions,
            confidence=parsed_response.confidence
        )
    
    def _format_explanation_response(self, parsed_response: ParsedResponse, 
                                   context: FormattingContext) -> ConversationResponse:
        """Format explanation response."""
        # Explanations are primarily text-based
        text_response = self._enhance_explanation_text(parsed_response.content)
        
        # Generate related questions
        suggestions = []
        if context.include_suggestions:
            suggestions = self._generate_explanation_suggestions(context)
        
        return ConversationResponse(
            text_response=text_response,
            visualizations=[],
            data_summary={'clarity_score': parsed_response.structured_data.get('clarity_score', 0.5)},
            suggested_questions=suggestions,
            confidence=parsed_response.confidence
        )
    
    def _format_suggestion_response(self, parsed_response: ParsedResponse, 
                                  context: FormattingContext) -> ConversationResponse:
        """Format suggestion response."""
        text_parts = []
        
        # Format suggestions nicely
        suggestions_data = parsed_response.structured_data.get('suggestions', [])
        if suggestions_data:
            text_parts.append("Here are some suggestions for further analysis:")
            for i, suggestion in enumerate(suggestions_data, 1):
                text_parts.append(f"{i}. {suggestion}")
        else:
            text_parts.append(parsed_response.content)
        
        return ConversationResponse(
            text_response="\n".join(text_parts),
            visualizations=[],
            data_summary={'suggestion_count': len(suggestions_data)},
            suggested_questions=suggestions_data[:5],  # Use suggestions as follow-up questions
            confidence=parsed_response.confidence
        )
    
    def _format_error_response(self, parsed_response: ParsedResponse, 
                             context: FormattingContext) -> ConversationResponse:
        """Format error diagnosis response."""
        text_parts = []
        
        # Structure the error response
        elements = parsed_response.structured_data
        
        if elements.get('cause'):
            text_parts.append(f"**Problem:** {elements['cause']}")
        
        if elements.get('solution'):
            text_parts.append(f"**Solution:** {elements['solution']}")
        
        if elements.get('prevention'):
            text_parts.append(f"**Prevention:** {elements['prevention']}")
        
        # Fallback to raw content if no structured elements
        if not text_parts:
            text_parts.append(parsed_response.content)
        
        # Generate helpful suggestions
        suggestions = [
            "Try rephrasing your question",
            "Check if the column names are correct",
            "Simplify your query and try again"
        ]
        
        return ConversationResponse(
            text_response="\n\n".join(text_parts),
            visualizations=[],
            data_summary={'error_resolved': bool(elements.get('solution'))},
            suggested_questions=suggestions,
            confidence=parsed_response.confidence
        )
    
    def _format_generic_response(self, parsed_response: ParsedResponse, 
                               context: FormattingContext) -> ConversationResponse:
        """Format generic response."""
        return ConversationResponse(
            text_response=parsed_response.content,
            visualizations=[],
            data_summary={},
            suggested_questions=[],
            confidence=parsed_response.confidence
        )
    
    def _format_single_record(self, record: Dict[str, Any]) -> str:
        """Format a single data record."""
        lines = []
        for key, value in record.items():
            lines.append(f"• {key}: {value}")
        return "\n".join(lines)
    
    def _calculate_summary_stats(self, data: List[Dict[str, Any]], 
                                columns: List[str]) -> Dict[str, Any]:
        """Calculate summary statistics for numeric columns."""
        if not data:
            return {}
        
        stats = {}
        
        # Find numeric columns
        numeric_columns = []
        for col in columns:
            if data[0].get(col) is not None:
                try:
                    float(data[0][col])
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    pass
        
        # Calculate stats for numeric columns
        for col in numeric_columns:
            values = []
            for row in data:
                if row.get(col) is not None:
                    try:
                        values.append(float(row[col]))
                    except (ValueError, TypeError):
                        pass
            
            if values:
                stats[f"{col} (avg)"] = f"{sum(values) / len(values):.2f}"
                stats[f"{col} (total)"] = f"{sum(values):.2f}"
                stats[f"{col} (min)"] = f"{min(values):.2f}"
                stats[f"{col} (max)"] = f"{max(values):.2f}"
        
        return stats
    
    def _has_numeric_data(self, data: List[Dict[str, Any]]) -> bool:
        """Check if data contains numeric values."""
        if not data:
            return False
        
        for row in data:
            for value in row.values():
                try:
                    float(value)
                    return True
                except (ValueError, TypeError):
                    continue
        
        return False
    
    def _create_visualizations_from_data(self, query_result: QueryResult, 
                                       context: FormattingContext) -> List[Chart]:
        """Create appropriate visualizations from query result data."""
        visualizations = []
        
        if not query_result.data or query_result.row_count == 0:
            return visualizations
        
        try:
            # Convert to DataFrame-like structure for chart selector
            import pandas as pd
            df = pd.DataFrame(query_result.data)
            
            # Limit data size for performance
            if len(df) > context.max_data_rows:
                df = df.head(context.max_data_rows)
            
            # Auto-select appropriate chart types
            chart_configs = self.chart_selector.get_chart_recommendations(df, max_recommendations=3)
            
            # Create charts
            for config in chart_configs[:3]:  # Limit to 3 charts
                try:
                    chart = self.chart_engine.create_chart(config, df)
                    visualizations.append(chart)
                except Exception as e:
                    # Log error but continue with other charts
                    continue
        
        except Exception as e:
            # If visualization creation fails, continue without charts
            pass
        
        return visualizations
    
    def _extract_data_insights(self, query_result: QueryResult) -> List[str]:
        """Extract insights from query result data."""
        insights = []
        
        if not query_result.data:
            return insights
        
        # Basic insights
        insights.append(f"Dataset contains {query_result.row_count} records")
        
        # Column insights
        if query_result.columns:
            insights.append(f"Data includes {len(query_result.columns)} columns: {', '.join(query_result.columns[:5])}")
        
        # Numeric insights
        numeric_data = self._get_numeric_columns_data(query_result.data)
        for col, values in numeric_data.items():
            if values:
                avg_val = sum(values) / len(values)
                insights.append(f"Average {col}: {avg_val:.2f}")
        
        return insights
    
    def _get_numeric_columns_data(self, data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract numeric data by column."""
        numeric_data = {}
        
        for row in data:
            for key, value in row.items():
                try:
                    num_value = float(value)
                    if key not in numeric_data:
                        numeric_data[key] = []
                    numeric_data[key].append(num_value)
                except (ValueError, TypeError):
                    continue
        
        return numeric_data
    
    def _get_numeric_columns(self, data: List[Dict[str, Any]], columns: List[str]) -> List[str]:
        """Get list of numeric column names."""
        if not data:
            return []
        
        numeric_columns = []
        for col in columns:
            if data[0].get(col) is not None:
                try:
                    float(data[0][col])
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    pass
        
        return numeric_columns
    
    def _get_categorical_columns(self, data: List[Dict[str, Any]], columns: List[str]) -> List[str]:
        """Get list of categorical column names."""
        if not data:
            return []
        
        categorical_columns = []
        for col in columns:
            if col not in self._get_numeric_columns(data, columns):
                # Check if it's not a date column
                if not self._is_date_column(data, col):
                    categorical_columns.append(col)
        
        return categorical_columns
    
    def _get_date_columns(self, data: List[Dict[str, Any]], columns: List[str]) -> List[str]:
        """Get list of date column names."""
        if not data:
            return []
        
        date_columns = []
        for col in columns:
            if self._is_date_column(data, col):
                date_columns.append(col)
        
        return date_columns
    
    def _is_date_column(self, data: List[Dict[str, Any]], column: str) -> bool:
        """Check if column contains date values."""
        if not data or column not in data[0]:
            return False
        
        sample_value = str(data[0][column])
        
        # Simple date pattern matching
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        return any(re.match(pattern, sample_value) for pattern in date_patterns)
    
    def _generate_query_suggestions(self, context: FormattingContext) -> List[str]:
        """Generate contextual follow-up question suggestions for query responses."""
        suggestions = []
        
        if context.query_result and context.query_result.row_count > 0:
            result = context.query_result
            
            # Analyze data to generate contextual suggestions
            numeric_columns = self._get_numeric_columns(result.data, result.columns)
            categorical_columns = self._get_categorical_columns(result.data, result.columns)
            
            # Suggest aggregations based on numeric data
            if numeric_columns:
                for col in numeric_columns[:2]:  # Limit to 2 columns
                    suggestions.append(f"What is the average {col}?")
                    suggestions.append(f"Show me the total {col}")
                
                # Suggest comparisons
                if len(numeric_columns) > 1:
                    suggestions.append(f"Compare {numeric_columns[0]} vs {numeric_columns[1]}")
            
            # Suggest grouping based on categorical data
            if categorical_columns:
                for col in categorical_columns[:2]:
                    suggestions.append(f"Group results by {col}")
                    if numeric_columns:
                        suggestions.append(f"Show {numeric_columns[0]} breakdown by {col}")
            
            # Suggest filtering based on data patterns
            if result.row_count > 10:
                suggestions.append("Show me the top 10 results")
                suggestions.append("Filter by specific criteria")
            
            # Time-based suggestions if date columns exist
            date_columns = self._get_date_columns(result.data, result.columns)
            if date_columns:
                suggestions.append("Show trends over time")
                suggestions.append("Compare this month vs last month")
            
            # Suggest drill-down analysis
            if result.row_count > 1:
                suggestions.append("What are the key patterns in this data?")
                suggestions.append("Show me outliers or anomalies")
        else:
            # No results - suggest alternatives based on original question
            original_question = context.original_question.lower()
            
            if 'total' in original_question or 'sum' in original_question:
                suggestions.append("Try looking at individual records instead")
                suggestions.append("Check if data exists for a different time period")
            elif 'average' in original_question or 'mean' in original_question:
                suggestions.append("Look at the raw data distribution")
                suggestions.append("Try a different calculation method")
            else:
                suggestions.append("Try a different time period")
                suggestions.append("Check for data in related categories")
                suggestions.append("Broaden the search criteria")
        
        return suggestions[:5]
    
    def _generate_analysis_suggestions(self, parsed_response: ParsedResponse, 
                                     context: FormattingContext) -> List[str]:
        """Generate contextual suggestions for analysis responses."""
        suggestions = []
        
        # Based on analysis content
        if parsed_response.structured_data.get('has_recommendations'):
            suggestions.append("How can we implement these recommendations?")
            suggestions.append("What are the potential risks?")
            suggestions.append("What resources would be needed?")
        
        if parsed_response.structured_data.get('has_insights'):
            suggestions.append("What caused these patterns?")
            suggestions.append("How do these trends compare historically?")
            suggestions.append("What are the implications for the future?")
        
        # Context-specific suggestions based on data
        if context.query_result:
            result = context.query_result
            numeric_columns = self._get_numeric_columns(result.data, result.columns)
            categorical_columns = self._get_categorical_columns(result.data, result.columns)
            
            # Suggest deeper analysis based on data types
            if numeric_columns and categorical_columns:
                suggestions.append(f"How does {numeric_columns[0]} vary by {categorical_columns[0]}?")
                suggestions.append("What are the correlations between variables?")
            
            # Suggest time-based analysis if applicable
            date_columns = self._get_date_columns(result.data, result.columns)
            if date_columns and numeric_columns:
                suggestions.append("Show me trends over time")
                suggestions.append("What seasonal patterns exist?")
            
            # Suggest outlier analysis for large datasets
            if result.row_count > 50:
                suggestions.append("Identify outliers and anomalies")
                suggestions.append("What are the top and bottom performers?")
        
        # Generic analysis suggestions if no specific context
        if not suggestions:
            suggestions.extend([
                "Show me the detailed breakdown",
                "What are the key drivers?",
                "How can we improve performance?",
                "What should we focus on next?",
                "Are there any concerning trends?"
            ])
        
        return suggestions[:5]
    
    def _generate_explanation_suggestions(self, context: FormattingContext) -> List[str]:
        """Generate suggestions for explanation responses."""
        return [
            "Can you provide more details?",
            "What are the implications?",
            "How does this affect our strategy?",
            "What should we do next?",
            "Are there any alternatives?"
        ]
    
    def _generate_complex_query_explanation(self, query_result: QueryResult, 
                                          context: FormattingContext) -> str:
        """Generate detailed explanation for complex queries."""
        if not query_result.data:
            return ""
        
        explanation_parts = []
        
        # Explain data scope
        explanation_parts.append(f"This analysis covers {query_result.row_count} records")
        
        # Explain data patterns
        numeric_columns = self._get_numeric_columns(query_result.data, query_result.columns)
        categorical_columns = self._get_categorical_columns(query_result.data, query_result.columns)
        
        if numeric_columns:
            # Analyze numeric patterns
            for col in numeric_columns[:2]:  # Limit to avoid verbosity
                values = [float(row[col]) for row in query_result.data if row.get(col) is not None]
                if values:
                    avg_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    
                    explanation_parts.append(
                        f"The {col} ranges from {min_val:.2f} to {max_val:.2f} "
                        f"with an average of {avg_val:.2f}"
                    )
        
        if categorical_columns:
            # Analyze categorical distribution
            for col in categorical_columns[:2]:
                unique_values = set(row[col] for row in query_result.data if row.get(col) is not None)
                explanation_parts.append(
                    f"The {col} field has {len(unique_values)} distinct values"
                )
        
        # Explain relationships if multiple columns
        if len(query_result.columns) > 1:
            explanation_parts.append(
                "The data shows relationships between " + 
                ", ".join(query_result.columns[:3]) + 
                (" and others" if len(query_result.columns) > 3 else "")
            )
        
        return ". ".join(explanation_parts) + "."
    
    def _generate_statistical_summary(self, query_result: QueryResult) -> str:
        """Generate statistical summary for numeric data."""
        if not query_result.data:
            return ""
        
        numeric_columns = self._get_numeric_columns(query_result.data, query_result.columns)
        if not numeric_columns:
            return ""
        
        summary_parts = []
        
        for col in numeric_columns[:3]:  # Limit to 3 columns
            values = []
            for row in query_result.data:
                if row.get(col) is not None:
                    try:
                        values.append(float(row[col]))
                    except (ValueError, TypeError):
                        continue
            
            if values:
                # Calculate statistics
                mean_val = sum(values) / len(values)
                sorted_values = sorted(values)
                median_val = sorted_values[len(sorted_values) // 2]
                
                # Calculate standard deviation
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                std_dev = variance ** 0.5
                
                summary_parts.append(
                    f"{col}: Mean={mean_val:.2f}, Median={median_val:.2f}, "
                    f"Std Dev={std_dev:.2f}"
                )
        
        return "; ".join(summary_parts) if summary_parts else ""
    
    def _explain_visualizations(self, visualizations: List[Chart], 
                               query_result: QueryResult) -> str:
        """Generate explanations for created visualizations."""
        if not visualizations:
            return ""
        
        explanations = []
        
        for chart in visualizations:
            chart_type = chart.config.chart_type.value
            
            if chart_type == "bar":
                explanations.append(
                    f"The bar chart shows the distribution of {chart.config.y_column or 'values'} "
                    f"across different {chart.config.x_column} categories"
                )
            elif chart_type == "line":
                explanations.append(
                    f"The line chart reveals trends in {chart.config.y_column or 'values'} "
                    f"over {chart.config.x_column}"
                )
            elif chart_type == "pie":
                explanations.append(
                    f"The pie chart breaks down the composition of {chart.config.x_column} "
                    "showing relative proportions"
                )
            elif chart_type == "scatter":
                explanations.append(
                    f"The scatter plot explores the relationship between "
                    f"{chart.config.x_column} and {chart.config.y_column}"
                )
        
        return ". ".join(explanations) + "." if explanations else ""
    
    def _calculate_query_complexity(self, query_result: QueryResult, 
                                   context: FormattingContext) -> float:
        """Calculate complexity score for the query/analysis."""
        complexity = 0.0
        
        # Base complexity from data size
        if query_result.row_count > 1000:
            complexity += 0.3
        elif query_result.row_count > 100:
            complexity += 0.2
        else:
            complexity += 0.1
        
        # Complexity from number of columns
        if len(query_result.columns) > 5:
            complexity += 0.3
        elif len(query_result.columns) > 2:
            complexity += 0.2
        else:
            complexity += 0.1
        
        # Complexity from data types
        numeric_cols = self._get_numeric_columns(query_result.data, query_result.columns)
        categorical_cols = self._get_categorical_columns(query_result.data, query_result.columns)
        
        if len(numeric_cols) > 2 and len(categorical_cols) > 1:
            complexity += 0.4
        elif len(numeric_cols) > 1 or len(categorical_cols) > 1:
            complexity += 0.2
        
        return min(1.0, complexity)
    
    def _assess_data_quality(self, query_result: QueryResult) -> float:
        """Assess data quality score."""
        if not query_result.data:
            return 0.0
        
        quality_score = 1.0
        
        # Check for missing values
        total_cells = len(query_result.data) * len(query_result.columns)
        missing_cells = 0
        
        for row in query_result.data:
            for col in query_result.columns:
                if row.get(col) is None or row.get(col) == "":
                    missing_cells += 1
        
        if total_cells > 0:
            missing_ratio = missing_cells / total_cells
            quality_score -= missing_ratio * 0.5
        
        # Check for data consistency (simplified)
        for col in query_result.columns:
            values = [row.get(col) for row in query_result.data if row.get(col) is not None]
            if values:
                # Check type consistency
                first_type = type(values[0])
                type_consistency = sum(1 for v in values if type(v) == first_type) / len(values)
                if type_consistency < 0.9:
                    quality_score -= 0.1
        
        return max(0.0, quality_score)
    
    def _enhance_explanation_text(self, text: str) -> str:
        """Enhance explanation text with better formatting."""
        # Add bullet points for lists
        lines = text.split('\n')
        enhanced_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('•') and not line.startswith('-'):
                # Check if it looks like a list item
                if any(line.startswith(word) for word in ['First', 'Second', 'Third', 'Finally', 'Additionally']):
                    line = f"• {line}"
            enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def create_summary_response(self, responses: List[ConversationResponse], 
                              context: FormattingContext) -> ConversationResponse:
        """Create a summary response from multiple conversation responses."""
        text_parts = ["## Summary"]
        
        all_visualizations = []
        combined_data_summary = {}
        all_suggestions = []
        
        for i, response in enumerate(responses, 1):
            text_parts.append(f"\n### Response {i}")
            text_parts.append(response.text_response)
            
            all_visualizations.extend(response.visualizations)
            all_suggestions.extend(response.suggested_questions)
            
            # Combine data summaries
            for key, value in response.data_summary.items():
                if key in combined_data_summary:
                    if isinstance(value, (int, float)) and isinstance(combined_data_summary[key], (int, float)):
                        combined_data_summary[key] += value
                    else:
                        combined_data_summary[f"{key}_{i}"] = value
                else:
                    combined_data_summary[key] = value
        
        # Remove duplicate suggestions
        unique_suggestions = list(dict.fromkeys(all_suggestions))
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in responses) / len(responses) if responses else 0.0
        
        return ConversationResponse(
            text_response="\n".join(text_parts),
            visualizations=all_visualizations[:5],  # Limit visualizations
            data_summary=combined_data_summary,
            suggested_questions=unique_suggestions[:5],
            confidence=avg_confidence
        )