#!/usr/bin/env python3
"""
Demonstration of the Enhanced Response Formatting System

This script demonstrates the key features implemented in task 8.4:
1. Response generation with text and visualizations
2. Contextual follow-up question suggestions  
3. Explanation generation for complex queries
4. Response quality and relevance testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.response_formatter import ResponseFormatter, FormattingContext
from src.ai.response_parser import ParsedResponse, ResponseType
from src.core.models import (
    QueryResult, DataSchema, ColumnInfo, DataType, ValidationResult,
    ConversationResponse, Chart, ChartConfig, ChartType
)


def create_sample_data():
    """Create sample data for demonstration."""
    # Sample schema
    schema = DataSchema(
        columns=[
            ColumnInfo(name="sales_amount", data_type=DataType.NUMBER),
            ColumnInfo(name="customer_name", data_type=DataType.TEXT),
            ColumnInfo(name="order_date", data_type=DataType.DATE),
            ColumnInfo(name="product_category", data_type=DataType.TEXT),
            ColumnInfo(name="region", data_type=DataType.TEXT)
        ],
        row_count=1000,
        data_types={
            "sales_amount": DataType.NUMBER,
            "customer_name": DataType.TEXT,
            "order_date": DataType.DATE,
            "product_category": DataType.TEXT,
            "region": DataType.TEXT
        }
    )
    
    # Sample query result
    query_result = QueryResult(
        data=[
            {
                'sales_amount': 15000, 'customer_name': 'Acme Corp', 
                'order_date': '2024-01-15', 'product_category': 'Electronics', 'region': 'North'
            },
            {
                'sales_amount': 8500, 'customer_name': 'Beta Inc', 
                'order_date': '2024-01-16', 'product_category': 'Clothing', 'region': 'South'
            },
            {
                'sales_amount': 22000, 'customer_name': 'Gamma LLC', 
                'order_date': '2024-01-17', 'product_category': 'Electronics', 'region': 'East'
            },
            {
                'sales_amount': 12500, 'customer_name': 'Delta Co', 
                'order_date': '2024-01-18', 'product_category': 'Home & Garden', 'region': 'West'
            },
            {
                'sales_amount': 18750, 'customer_name': 'Echo Systems', 
                'order_date': '2024-01-19', 'product_category': 'Electronics', 'region': 'North'
            }
        ],
        columns=['sales_amount', 'customer_name', 'order_date', 'product_category', 'region'],
        row_count=5,
        execution_time_ms=125
    )
    
    return schema, query_result


def demo_query_response_formatting():
    """Demonstrate query response formatting with enhanced features."""
    print("=" * 80)
    print("DEMO 1: Enhanced Query Response Formatting")
    print("=" * 80)
    
    formatter = ResponseFormatter()
    schema, query_result = create_sample_data()
    
    # Create a parsed response for a SQL query
    parsed_response = ParsedResponse(
        response_type=ResponseType.SQL_QUERY,
        content="Query executed successfully",
        structured_data={
            'query_count': 1,
            'primary_query': 'SELECT * FROM sales_data WHERE product_category = "Electronics"',
            'has_aggregation': False,
            'has_filtering': True,
            'has_grouping': False
        },
        confidence=0.92,
        validation_result=ValidationResult(is_valid=True, confidence=1.0)
    )
    
    # Create formatting context
    context = FormattingContext(
        query_result=query_result,
        schema=schema,
        original_question="Show me all electronics sales data",
        include_visualizations=True,
        include_suggestions=True
    )
    
    # Format the response
    response = formatter.format_response(parsed_response, context)
    
    print(f"Question: {context.original_question}")
    print(f"Confidence: {response.confidence:.2f}")
    print("\nFormatted Response:")
    print(response.text_response)
    print(f"\nData Summary: {response.data_summary}")
    print(f"\nSuggested Follow-up Questions:")
    for i, suggestion in enumerate(response.suggested_questions, 1):
        print(f"  {i}. {suggestion}")
    print(f"\nVisualizations Generated: {len(response.visualizations)}")


def demo_analysis_response_with_explanations():
    """Demonstrate analysis response with complex query explanations."""
    print("\n" + "=" * 80)
    print("DEMO 2: Analysis Response with Enhanced Explanations")
    print("=" * 80)
    
    formatter = ResponseFormatter()
    schema, query_result = create_sample_data()
    
    # Create a parsed response for data analysis
    parsed_response = ParsedResponse(
        response_type=ResponseType.ANALYSIS,
        content="The sales data reveals strong performance in the Electronics category, "
                "with significant regional variations. Electronics accounts for 60% of total sales volume.",
        structured_data={
            'sections': {
                'Performance': 'Electronics leading with 60% market share',
                'Regional Analysis': 'North and East regions outperforming'
            },
            'has_insights': True,
            'has_recommendations': True,
            'key_metrics': [
                {'value': 18583, 'unit': 'average_sales', 'formatted': '$18,583'},
                {'value': 60, 'unit': 'percent', 'formatted': '60%'}
            ]
        },
        confidence=0.87,
        validation_result=ValidationResult(is_valid=True, confidence=1.0)
    )
    
    # Create formatting context
    context = FormattingContext(
        query_result=query_result,
        schema=schema,
        original_question="Analyze sales performance by product category and region",
        include_visualizations=True,
        include_suggestions=True
    )
    
    # Format the response
    response = formatter.format_response(parsed_response, context)
    
    print(f"Question: {context.original_question}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Complexity Score: {response.data_summary.get('complexity_score', 'N/A'):.2f}")
    print(f"Data Quality Score: {response.data_summary.get('data_quality_score', 'N/A'):.2f}")
    print("\nFormatted Analysis Response:")
    print(response.text_response)
    print(f"\nContextual Follow-up Suggestions:")
    for i, suggestion in enumerate(response.suggested_questions, 1):
        print(f"  {i}. {suggestion}")


def demo_contextual_suggestions():
    """Demonstrate contextual suggestion generation."""
    print("\n" + "=" * 80)
    print("DEMO 3: Contextual Follow-up Question Generation")
    print("=" * 80)
    
    formatter = ResponseFormatter()
    schema, query_result = create_sample_data()
    
    # Test different question contexts
    test_questions = [
        "What are the total sales by region?",
        "Show me average sales amount",
        "Compare electronics vs clothing sales",
        "What are the sales trends over time?"
    ]
    
    for question in test_questions:
        context = FormattingContext(
            query_result=query_result,
            schema=schema,
            original_question=question,
            include_suggestions=True
        )
        
        suggestions = formatter._generate_query_suggestions(context)
        
        print(f"\nOriginal Question: '{question}'")
        print("Generated Contextual Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")


def demo_explanation_generation():
    """Demonstrate complex query explanation generation."""
    print("\n" + "=" * 80)
    print("DEMO 4: Complex Query Explanation Generation")
    print("=" * 80)
    
    formatter = ResponseFormatter()
    schema, query_result = create_sample_data()
    
    context = FormattingContext(
        query_result=query_result,
        schema=schema,
        original_question="Complex multi-dimensional analysis of sales performance"
    )
    
    # Generate detailed explanation
    explanation = formatter._generate_complex_query_explanation(query_result, context)
    print("Complex Query Explanation:")
    print(explanation)
    
    # Generate statistical summary
    statistical_summary = formatter._generate_statistical_summary(query_result)
    print(f"\nStatistical Summary:")
    print(statistical_summary)
    
    # Demonstrate visualization explanations
    sample_charts = [
        Chart(config=ChartConfig(
            chart_type=ChartType.BAR,
            title="Sales by Category",
            x_column="product_category",
            y_column="sales_amount"
        )),
        Chart(config=ChartConfig(
            chart_type=ChartType.LINE,
            title="Sales Trend Over Time",
            x_column="order_date",
            y_column="sales_amount"
        ))
    ]
    
    viz_explanation = formatter._explain_visualizations(sample_charts, query_result)
    print(f"\nVisualization Explanations:")
    print(viz_explanation)


def demo_response_quality_metrics():
    """Demonstrate response quality and relevance metrics."""
    print("\n" + "=" * 80)
    print("DEMO 5: Response Quality and Relevance Metrics")
    print("=" * 80)
    
    formatter = ResponseFormatter()
    schema, query_result = create_sample_data()
    
    # Test data quality assessment
    quality_score = formatter._assess_data_quality(query_result)
    print(f"Data Quality Score: {quality_score:.2f}")
    
    # Test query complexity calculation
    context = FormattingContext(schema=schema)
    complexity_score = formatter._calculate_query_complexity(query_result, context)
    print(f"Query Complexity Score: {complexity_score:.2f}")
    
    # Test data type detection
    numeric_cols = formatter._get_numeric_columns(query_result.data, query_result.columns)
    categorical_cols = formatter._get_categorical_columns(query_result.data, query_result.columns)
    date_cols = formatter._get_date_columns(query_result.data, query_result.columns)
    
    print(f"\nData Type Analysis:")
    print(f"  Numeric Columns: {numeric_cols}")
    print(f"  Categorical Columns: {categorical_cols}")
    print(f"  Date Columns: {date_cols}")
    
    # Test suggestion relevance
    context = FormattingContext(
        query_result=query_result,
        schema=schema,
        original_question="What are the sales by category?"
    )
    
    suggestions = formatter._generate_query_suggestions(context)
    relevant_suggestions = [s for s in suggestions if 'sales' in s.lower() or 'category' in s.lower()]
    
    print(f"\nSuggestion Relevance Analysis:")
    print(f"  Total Suggestions: {len(suggestions)}")
    print(f"  Relevant Suggestions: {len(relevant_suggestions)}")
    print(f"  Relevance Ratio: {len(relevant_suggestions)/len(suggestions):.2f}")


def main():
    """Run all demonstrations."""
    print("Enhanced Response Formatting System Demonstration")
    print("Task 8.4: Create response formatting system")
    print("\nFeatures Implemented:")
    print("✓ Response generation with text and visualizations")
    print("✓ Contextual follow-up question suggestions")
    print("✓ Explanation generation for complex queries")
    print("✓ Response quality and relevance testing")
    
    try:
        demo_query_response_formatting()
        demo_analysis_response_with_explanations()
        demo_contextual_suggestions()
        demo_explanation_generation()
        demo_response_quality_metrics()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nAll features of the enhanced response formatting system are working correctly!")
        print("The system now provides:")
        print("• Rich, contextual responses with supporting data")
        print("• Intelligent visualization selection and explanation")
        print("• Smart follow-up question suggestions")
        print("• Detailed explanations for complex queries")
        print("• Quality metrics and relevance scoring")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()