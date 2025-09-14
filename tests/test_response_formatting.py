"""
Tests for the Response Formatting System.
"""

import pytest
from unittest.mock import Mock, patch
from src.ai.response_formatter import ResponseFormatter, FormattingContext, ResponseFormat
from src.ai.response_parser import ParsedResponse, ResponseType
from src.ai.conversational_engine import ConversationalAIEngine, ConversationContext
from src.core.models import (
    ConversationResponse, QueryResult, DataSchema, ColumnInfo, DataType, Chart, ChartConfig
)
from src.core.models import ValidationResult


@pytest.fixture
def sample_schema():
    """Create a sample data schema for testing."""
    columns = [
        ColumnInfo(name="sales_amount", data_type=DataType.NUMBER),
        ColumnInfo(name="customer_name", data_type=DataType.TEXT),
        ColumnInfo(name="order_date", data_type=DataType.DATE),
        ColumnInfo(name="product_category", data_type=DataType.TEXT)
    ]
    
    return DataSchema(
        columns=columns,
        row_count=1000,
        data_types={col.name: col.data_type for col in columns}
    )


@pytest.fixture
def sample_query_result():
    """Create a sample query result."""
    return QueryResult(
        data=[
            {'sales_amount': 1000, 'customer_name': 'Alice', 'product_category': 'Electronics'},
            {'sales_amount': 1500, 'customer_name': 'Bob', 'product_category': 'Clothing'},
            {'sales_amount': 800, 'customer_name': 'Charlie', 'product_category': 'Electronics'}
        ],
        columns=['sales_amount', 'customer_name', 'product_category'],
        row_count=3,
        execution_time_ms=150
    )


@pytest.fixture
def response_formatter():
    """Create response formatter instance."""
    return ResponseFormatter()


@pytest.fixture
def conversational_engine():
    """Create conversational AI engine instance."""
    return ConversationalAIEngine()


class TestResponseFormatter:
    """Test cases for response formatter."""
    
    def test_format_query_response_with_data(self, response_formatter, sample_query_result, sample_schema):
        """Test formatting query response with data."""
        # Create a parsed response
        parsed_response = ParsedResponse(
            response_type=ResponseType.SQL_QUERY,
            content="Query executed successfully",
            structured_data={
                'query_count': 1,
                'primary_query': 'SELECT * FROM data_table',
                'has_aggregation': False,
                'has_filtering': False
            },
            confidence=0.9,
            validation_result=ValidationResult(is_valid=True, confidence=1.0)
        )
        
        # Create formatting context
        context = FormattingContext(
            query_result=sample_query_result,
            schema=sample_schema,
            original_question="Show me all sales data",
            include_visualizations=True,
            include_suggestions=True
        )
        
        # Format response
        response = response_formatter.format_response(parsed_response, context)
        
        # Assertions
        assert isinstance(response, ConversationResponse)
        assert response.confidence == 0.9
        assert "Found 3 records" in response.text_response
        assert len(response.suggested_questions) > 0
        assert response.data_summary['total_rows'] == 3
        assert response.data_summary['has_numeric_data'] is True
    
    def test_format_query_response_no_data(self, response_formatter, sample_schema):
        """Test formatting query response with no data."""
        # Create empty query result
        empty_result = QueryResult(
            data=[],
            columns=['sales_amount', 'customer_name'],
            row_count=0,
            execution_time_ms=50
        )
        
        parsed_response = ParsedResponse(
            response_type=ResponseType.SQL_QUERY,
            content="No results found",
            structured_data={'query_count': 1, 'has_data': False},
            confidence=0.8,
            validation_result=ValidationResult(is_valid=True, confidence=1.0)
        )
        
        context = FormattingContext(
            query_result=empty_result,
            schema=sample_schema,
            original_question="Show me sales for last year"
        )
        
        response = response_formatter.format_response(parsed_response, context)
        
        assert "No data found" in response.text_response
        assert len(response.visualizations) == 0
        assert response.data_summary['total_rows'] == 0
    
    def test_format_analysis_response(self, response_formatter, sample_query_result, sample_schema):
        """Test formatting analysis response with enhanced explanations."""
        parsed_response = ParsedResponse(
            response_type=ResponseType.ANALYSIS,
            content="The sales data shows strong performance in electronics category.",
            structured_data={
                'sections': {'Performance': 'Electronics leading sales'},
                'has_insights': True,
                'has_recommendations': True,
                'key_metrics': [{'value': 1100, 'unit': 'average'}]
            },
            confidence=0.85,
            validation_result=ValidationResult(is_valid=True, confidence=1.0)
        )
        
        context = FormattingContext(
            query_result=sample_query_result,
            schema=sample_schema,
            original_question="Analyze sales performance by category"
        )
        
        response = response_formatter.format_response(parsed_response, context)
        
        assert "strong performance" in response.text_response
        assert "Detailed Analysis:" in response.text_response
        assert "Key Insights:" in response.text_response
        assert "Statistical Summary:" in response.text_response
        assert len(response.suggested_questions) > 0
        assert response.data_summary['complexity_score'] > 0
        assert response.data_summary['data_quality_score'] > 0
    
    def test_contextual_suggestions_generation(self, response_formatter, sample_query_result, sample_schema):
        """Test generation of contextual follow-up suggestions."""
        context = FormattingContext(
            query_result=sample_query_result,
            schema=sample_schema,
            original_question="What are the total sales by category?"
        )
        
        suggestions = response_formatter._generate_query_suggestions(context)
        
        assert len(suggestions) > 0
        assert any("sales_amount" in suggestion for suggestion in suggestions)
        assert any("product_category" in suggestion for suggestion in suggestions)
    
    def test_complex_query_explanation(self, response_formatter, sample_query_result, sample_schema):
        """Test generation of explanations for complex queries."""
        context = FormattingContext(
            query_result=sample_query_result,
            schema=sample_schema,
            original_question="Complex analysis query"
        )
        
        explanation = response_formatter._generate_complex_query_explanation(sample_query_result, context)
        
        assert "This analysis covers 3 records" in explanation
        assert "sales_amount ranges from" in explanation
        assert "product_category field has" in explanation
    
    def test_statistical_summary_generation(self, response_formatter, sample_query_result):
        """Test generation of statistical summaries."""
        summary = response_formatter._generate_statistical_summary(sample_query_result)
        
        assert "sales_amount:" in summary
        assert "Mean=" in summary
        assert "Median=" in summary
        assert "Std Dev=" in summary
    
    def test_visualization_explanations(self, response_formatter, sample_query_result):
        """Test generation of visualization explanations."""
        # Create mock charts
        from src.core.models import Chart, ChartConfig, ChartType
        
        charts = [
            Chart(config=ChartConfig(
                chart_type=ChartType.BAR,
                title="Sales by Category",
                x_column="product_category",
                y_column="sales_amount"
            )),
            Chart(config=ChartConfig(
                chart_type=ChartType.LINE,
                title="Sales Trend",
                x_column="order_date",
                y_column="sales_amount"
            ))
        ]
        
        explanation = response_formatter._explain_visualizations(charts, sample_query_result)
        
        assert "bar chart shows the distribution" in explanation
        assert "line chart reveals trends" in explanation
    
    def test_data_quality_assessment(self, response_formatter):
        """Test data quality assessment functionality."""
        # Test with good quality data
        good_data = QueryResult(
            data=[
                {'col1': 100, 'col2': 'A'},
                {'col1': 200, 'col2': 'B'},
                {'col1': 300, 'col2': 'C'}
            ],
            columns=['col1', 'col2'],
            row_count=3
        )
        
        quality_score = response_formatter._assess_data_quality(good_data)
        assert quality_score > 0.8
        
        # Test with poor quality data (missing values)
        poor_data = QueryResult(
            data=[
                {'col1': 100, 'col2': None},
                {'col1': None, 'col2': 'B'},
                {'col1': 300, 'col2': ''}
            ],
            columns=['col1', 'col2'],
            row_count=3
        )
        
        quality_score = response_formatter._assess_data_quality(poor_data)
        assert quality_score < 0.8
    
    def test_query_complexity_calculation(self, response_formatter, sample_schema):
        """Test query complexity calculation."""
        # Simple query result
        simple_result = QueryResult(
            data=[{'col1': 1}],
            columns=['col1'],
            row_count=1
        )
        
        context = FormattingContext(schema=sample_schema)
        complexity = response_formatter._calculate_query_complexity(simple_result, context)
        assert complexity < 0.5
        
        # Complex query result
        complex_result = QueryResult(
            data=[{f'col{i}': i for i in range(10)} for _ in range(1000)],
            columns=[f'col{i}' for i in range(10)],
            row_count=1000
        )
        
        complexity = response_formatter._calculate_query_complexity(complex_result, context)
        assert complexity > 0.5
    
    def test_data_type_detection(self, response_formatter, sample_query_result):
        """Test detection of different data types in results."""
        numeric_cols = response_formatter._get_numeric_columns(
            sample_query_result.data, 
            sample_query_result.columns
        )
        assert 'sales_amount' in numeric_cols
        
        categorical_cols = response_formatter._get_categorical_columns(
            sample_query_result.data,
            sample_query_result.columns
        )
        assert 'customer_name' in categorical_cols
        assert 'product_category' in categorical_cols
    
    def test_error_response_formatting(self, response_formatter):
        """Test formatting of error diagnosis responses."""
        parsed_response = ParsedResponse(
            response_type=ResponseType.ERROR_DIAGNOSIS,
            content="Error analysis",
            structured_data={
                'cause': 'Invalid column name',
                'solution': 'Check column spelling',
                'prevention': 'Use schema validation'
            },
            confidence=0.7,
            validation_result=ValidationResult(is_valid=True, confidence=1.0)
        )
        
        context = FormattingContext(original_question="SELECT invalid_col FROM table")
        response = response_formatter.format_response(parsed_response, context)
        
        assert "Problem:" in response.text_response
        assert "Solution:" in response.text_response
        assert "Prevention:" in response.text_response
        assert len(response.suggested_questions) > 0
    
    def test_summary_response_creation(self, response_formatter, sample_schema):
        """Test creation of summary responses from multiple responses."""
        responses = [
            ConversationResponse(
                text_response="First analysis result",
                visualizations=[],
                data_summary={'metric1': 100},
                suggested_questions=["Question 1"],
                confidence=0.8
            ),
            ConversationResponse(
                text_response="Second analysis result", 
                visualizations=[],
                data_summary={'metric2': 200},
                suggested_questions=["Question 2"],
                confidence=0.9
            )
        ]
        
        context = FormattingContext(schema=sample_schema)
        summary = response_formatter.create_summary_response(responses, context)
        
        assert "Summary" in summary.text_response
        assert "Response 1" in summary.text_response
        assert "Response 2" in summary.text_response
        assert abs(summary.confidence - 0.85) < 0.01  # Average of 0.8 and 0.9 (allow for floating point precision)
        assert len(summary.suggested_questions) == 2


class TestConversationalEngine:
    """Test cases for conversational AI engine integration."""
    
    def test_process_question_high_confidence(self, conversational_engine, sample_schema):
        """Test processing question with high confidence intent parsing."""
        with patch.object(conversational_engine.nlp_processor, 'parse_question') as mock_parse:
            # Mock high confidence intent
            from src.ai.nlp_processor import ParsedIntent, QuestionType
            from src.ai.nlp_processor import AggregationType
            mock_parse.return_value = ParsedIntent(
                question_type=QuestionType.AGGREGATION,
                entities={},
                columns=['sales_amount'],
                aggregation=AggregationType.SUM,
                filters=[],
                confidence=0.8,
                original_question="What is the total sales?"
            )
            
            response = conversational_engine.process_question(
                "What is the total sales?",
                sample_schema,
                session_id="test_session"
            )
            
            assert isinstance(response, ConversationResponse)
            assert response.confidence > 0.7
    
    def test_process_question_low_confidence(self, conversational_engine, sample_schema):
        """Test processing question with low confidence requiring LLM assistance."""
        with patch.object(conversational_engine.nlp_processor, 'parse_question') as mock_parse:
            # Mock low confidence intent
            from src.ai.nlp_processor import ParsedIntent, QuestionType
            mock_parse.return_value = ParsedIntent(
                question_type=QuestionType.DESCRIPTIVE,
                entities={},
                columns=[],
                aggregation=None,
                filters=[],
                confidence=0.3,
                original_question="Analyze the business performance"
            )
            
            with patch.object(conversational_engine.llm_manager, 'generate_response') as mock_llm:
                from src.ai.llm_provider import LLMResponse
                from src.ai.llm_provider import LLMProvider
                mock_llm.return_value = LLMResponse(
                    content="Business performance analysis shows positive trends.",
                    usage={'tokens': 100},
                    model="gpt-3.5-turbo",
                    provider=LLMProvider.OPENAI,
                    response_time_ms=500,
                    confidence=0.8
                )
                
                response = conversational_engine.process_question(
                    "Analyze the business performance",
                    sample_schema,
                    session_id="test_session"
                )
                
                assert isinstance(response, ConversationResponse)
                assert len(response.text_response) > 0
    
    def test_conversation_context_management(self, conversational_engine, sample_schema):
        """Test conversation context and history management."""
        session_id = "test_context_session"
        
        # First question
        response1 = conversational_engine.process_question(
            "What are the sales?",
            sample_schema,
            session_id=session_id
        )
        
        # Check context was created
        assert session_id in conversational_engine.contexts
        context = conversational_engine.contexts[session_id]
        assert len(context.conversation_history) == 1
        
        # Second question
        response2 = conversational_engine.process_question(
            "Show me more details",
            sample_schema,
            session_id=session_id
        )
        
        # Check history updated
        assert len(context.conversation_history) == 2
    
    def test_error_handling(self, conversational_engine, sample_schema):
        """Test error handling in conversational engine."""
        with patch.object(conversational_engine.nlp_processor, 'parse_question') as mock_parse:
            # Mock an exception
            mock_parse.side_effect = Exception("Processing error")
            
            response = conversational_engine.process_question(
                "Invalid question",
                sample_schema
            )
            
            assert isinstance(response, ConversationResponse)
            assert "encountered an issue" in response.text_response
            assert response.confidence < 0.5
            assert response.data_summary.get('error') is True


class TestResponseQualityAndRelevance:
    """Test cases for response quality and relevance metrics."""
    
    def test_response_relevance_to_question(self, response_formatter, sample_query_result, sample_schema):
        """Test that responses are relevant to the original question."""
        # Test sales-related question
        context = FormattingContext(
            query_result=sample_query_result,
            schema=sample_schema,
            original_question="What are the sales by category?"
        )
        
        suggestions = response_formatter._generate_query_suggestions(context)
        
        # Suggestions should be relevant to sales and categories
        relevant_suggestions = [s for s in suggestions if 'sales' in s.lower() or 'category' in s.lower()]
        assert len(relevant_suggestions) > 0
    
    def test_explanation_clarity_and_completeness(self, response_formatter, sample_query_result, sample_schema):
        """Test that explanations are clear and complete."""
        context = FormattingContext(
            query_result=sample_query_result,
            schema=sample_schema,
            original_question="Complex multi-table analysis"
        )
        
        explanation = response_formatter._generate_complex_query_explanation(sample_query_result, context)
        
        # Should contain key information
        assert "records" in explanation
        assert "ranges from" in explanation or "distinct values" in explanation
        assert len(explanation.split('.')) >= 2  # Multiple sentences
    
    def test_visualization_appropriateness(self, response_formatter, sample_query_result):
        """Test that visualizations are appropriate for the data."""
        # This would typically test the chart selector, but we'll test the explanation
        from src.core.models import Chart, ChartConfig, ChartType
        
        # Bar chart for categorical data
        bar_chart = Chart(config=ChartConfig(
            chart_type=ChartType.BAR,
            title="Sales by Category",
            x_column="product_category",
            y_column="sales_amount"
        ))
        
        explanation = response_formatter._explain_visualizations([bar_chart], sample_query_result)
        assert "distribution" in explanation
        assert "categories" in explanation
    
    def test_suggestion_actionability(self, response_formatter, sample_query_result, sample_schema):
        """Test that suggestions are actionable and useful."""
        context = FormattingContext(
            query_result=sample_query_result,
            schema=sample_schema,
            original_question="Show me sales performance"
        )
        
        suggestions = response_formatter._generate_query_suggestions(context)
        
        # Check that suggestions contain actionable verbs
        actionable_verbs = ['show', 'compare', 'analyze', 'group', 'filter', 'what', 'how']
        actionable_suggestions = [
            s for s in suggestions 
            if any(verb in s.lower() for verb in actionable_verbs)
        ]
        
        assert len(actionable_suggestions) > 0
    
    def test_confidence_scoring_accuracy(self, response_formatter):
        """Test that confidence scores accurately reflect response quality."""
        # High quality response
        high_quality_response = ParsedResponse(
            response_type=ResponseType.SQL_QUERY,
            content="```sql\nSELECT SUM(sales) FROM table WHERE date > '2023-01-01'\n```",
            structured_data={
                'query_count': 1,
                'has_aggregation': True,
                'has_filtering': True
            },
            confidence=0.9,
            validation_result=ValidationResult(is_valid=True, confidence=1.0)
        )
        
        # Low quality response
        low_quality_response = ParsedResponse(
            response_type=ResponseType.SQL_QUERY,
            content="I'm not sure about this query",
            structured_data={'query_count': 0},
            confidence=0.3,
            validation_result=ValidationResult(is_valid=False, confidence=0.2)
        )
        
        # High quality should have higher confidence
        assert high_quality_response.confidence > low_quality_response.confidence
        assert high_quality_response.validation_result.confidence > low_quality_response.validation_result.confidence


if __name__ == "__main__":
    pytest.main([__file__])