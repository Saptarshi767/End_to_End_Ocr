"""
Tests for the Natural Language Processing component.
"""

import pytest
from src.ai.nlp_processor import NLPProcessor, QuestionType, AggregationType, ParsedIntent
from src.ai.entity_extractor import EntityExtractor
from src.ai.question_classifier import QuestionClassifier
from src.core.models import DataSchema, ColumnInfo, DataType


@pytest.fixture
def sample_schema():
    """Create a sample data schema for testing."""
    columns = [
        ColumnInfo(name="sales_amount", data_type=DataType.NUMBER),
        ColumnInfo(name="customer_name", data_type=DataType.TEXT),
        ColumnInfo(name="order_date", data_type=DataType.DATE),
        ColumnInfo(name="product_category", data_type=DataType.TEXT),
        ColumnInfo(name="quantity", data_type=DataType.NUMBER),
        ColumnInfo(name="unit_price", data_type=DataType.CURRENCY),
        ColumnInfo(name="is_premium", data_type=DataType.BOOLEAN)
    ]
    
    return DataSchema(
        columns=columns,
        row_count=1000,
        data_types={col.name: col.data_type for col in columns}
    )


@pytest.fixture
def nlp_processor():
    """Create NLP processor instance."""
    return NLPProcessor()


@pytest.fixture
def entity_extractor():
    """Create entity extractor instance."""
    return EntityExtractor()


@pytest.fixture
def question_classifier():
    """Create question classifier instance."""
    return QuestionClassifier()


class TestNLPProcessor:
    """Test cases for NLP processor."""
    
    def test_parse_aggregation_question(self, nlp_processor, sample_schema):
        """Test parsing aggregation questions."""
        question = "What is the total sales amount?"
        result = nlp_processor.parse_question(question, sample_schema)
        
        assert result.question_type == QuestionType.AGGREGATION
        assert result.aggregation == AggregationType.SUM
        assert "sales_amount" in result.columns
        assert result.confidence > 0.5
    
    def test_parse_filtering_question(self, nlp_processor, sample_schema):
        """Test parsing filtering questions."""
        question = "Show me orders where customer_name is 'John Smith'"
        result = nlp_processor.parse_question(question, sample_schema)
        
        assert result.question_type == QuestionType.FILTERING
        assert "customer_name" in result.columns
        assert len(result.filters) > 0
        assert result.confidence > 0.5
    
    def test_parse_comparison_question(self, nlp_processor, sample_schema):
        """Test parsing comparison questions."""
        question = "Compare sales_amount between product categories"
        result = nlp_processor.parse_question(question, sample_schema)
        
        assert result.question_type == QuestionType.COMPARISON
        assert "sales_amount" in result.columns or "product_category" in result.columns
        assert result.confidence > 0.3
    
    def test_parse_trend_question(self, nlp_processor, sample_schema):
        """Test parsing trend questions."""
        question = "Show me the sales trend over time"
        result = nlp_processor.parse_question(question, sample_schema)
        
        assert result.question_type == QuestionType.TREND
        assert result.confidence > 0.3
    
    def test_parse_ranking_question(self, nlp_processor, sample_schema):
        """Test parsing ranking questions."""
        question = "What are the top 10 customers by sales amount?"
        result = nlp_processor.parse_question(question, sample_schema)
        
        assert result.question_type == QuestionType.RANKING
        assert "sales_amount" in result.columns or "customer_name" in result.columns
        assert 10 in result.entities.get('numbers', [])
    
    def test_extract_column_references(self, nlp_processor, sample_schema):
        """Test column reference extraction."""
        question = "What is the average unit price for premium customers?"
        matches = nlp_processor.extract_column_references(question, sample_schema)
        
        column_names = [match[0] for match in matches]
        assert "unit_price" in column_names
        assert "is_premium" in column_names
    
    def test_classify_question_complexity(self, nlp_processor, sample_schema):
        """Test question complexity classification."""
        # Simple question
        simple_intent = ParsedIntent(
            question_type=QuestionType.AGGREGATION,
            entities={'columns': ['sales_amount']},
            columns=['sales_amount'],
            aggregation=AggregationType.SUM,
            confidence=0.8,
            original_question="What is the total sales?"
        )
        assert nlp_processor.classify_question_complexity(simple_intent) == 'simple'
        
        # Complex question
        complex_intent = ParsedIntent(
            question_type=QuestionType.COMPARISON,
            entities={'columns': ['sales_amount', 'product_category', 'customer_name']},
            columns=['sales_amount', 'product_category', 'customer_name'],
            filters=[{'column': 'order_date', 'operator': '>', 'value': '2023-01-01'}],
            confidence=0.7,
            original_question="Compare sales by category and customer with date filter"
        )
        assert nlp_processor.classify_question_complexity(complex_intent) == 'complex'
    
    def test_suggest_clarifications(self, nlp_processor, sample_schema):
        """Test clarification suggestions."""
        # Low confidence intent
        low_confidence_intent = ParsedIntent(
            question_type=QuestionType.DESCRIPTIVE,
            entities={'columns': []},
            columns=[],
            confidence=0.2,
            original_question="Tell me about the data"
        )
        
        suggestions = nlp_processor.suggest_clarifications(low_confidence_intent, sample_schema)
        assert len(suggestions) > 0
        assert any("rephrase" in s.lower() for s in suggestions)


class TestEntityExtractor:
    """Test cases for entity extractor."""
    
    def test_extract_column_entities(self, entity_extractor, sample_schema):
        """Test column entity extraction."""
        text = "Show me the sales amount and customer name"
        entities = entity_extractor.extract_entities(text, sample_schema)
        
        column_entities = [e for e in entities if e.entity_type == 'column']
        column_names = [e.normalized_value for e in column_entities]
        
        assert "sales_amount" in column_names
        assert "customer_name" in column_names
    
    def test_extract_number_entities(self, entity_extractor, sample_schema):
        """Test number entity extraction."""
        text = "Show me orders with quantity greater than 100 and price $50.99"
        entities = entity_extractor.extract_entities(text, sample_schema)
        
        number_entities = [e for e in entities if e.entity_type == 'number']
        numbers = [e.normalized_value for e in number_entities]
        
        assert 100.0 in numbers
        assert 50.99 in numbers
    
    def test_extract_date_entities(self, entity_extractor, sample_schema):
        """Test date entity extraction."""
        text = "Show orders from 2023-01-01 to 12/31/2023"
        entities = entity_extractor.extract_entities(text, sample_schema)
        
        date_entities = [e for e in entities if e.entity_type == 'date']
        assert len(date_entities) >= 1
    
    def test_extract_operator_entities(self, entity_extractor, sample_schema):
        """Test operator entity extraction."""
        text = "Show records where sales is greater than 1000"
        entities = entity_extractor.extract_entities(text, sample_schema)
        
        operator_entities = [e for e in entities if e.entity_type == 'operator']
        operators = [e.normalized_value for e in operator_entities]
        
        assert 'greater_than' in operators
    
    def test_extract_quoted_values(self, entity_extractor, sample_schema):
        """Test quoted value extraction."""
        text = 'Show customers with name "John Smith" and category "Electronics"'
        entities = entity_extractor.extract_entities(text, sample_schema)
        
        value_entities = [e for e in entities if e.entity_type == 'value']
        values = [e.normalized_value for e in value_entities]
        
        assert "John Smith" in values
        assert "Electronics" in values
    
    def test_find_column_matches_fuzzy(self, entity_extractor, sample_schema):
        """Test fuzzy column matching."""
        matches = entity_extractor.find_column_matches("sales", sample_schema, threshold=0.5)
        
        column_names = [match[0] for match in matches]
        assert "sales_amount" in column_names
    
    def test_extract_filter_conditions(self, entity_extractor, sample_schema):
        """Test filter condition extraction."""
        text = "Show orders where quantity greater than 10"
        conditions = entity_extractor.extract_filter_conditions(text, sample_schema)
        
        assert len(conditions) > 0
        condition = conditions[0]
        assert condition['column'] == 'quantity'
        assert condition['operator'] == 'greater_than'
        assert condition['value'] == 10.0


class TestQuestionClassifier:
    """Test cases for question classifier."""
    
    def test_classify_aggregation_question(self, question_classifier, sample_schema):
        """Test aggregation question classification."""
        question = "What is the total sales amount?"
        result = question_classifier.classify_question(question, sample_schema)
        
        assert result.question_type == QuestionType.AGGREGATION
        assert result.confidence >= 0.5
        assert result.complexity_level in ['simple', 'medium', 'complex']
    
    def test_classify_filtering_question(self, question_classifier, sample_schema):
        """Test filtering question classification."""
        question = "Show me orders where customer name is John"
        result = question_classifier.classify_question(question, sample_schema)
        
        assert result.question_type == QuestionType.FILTERING
        assert result.confidence > 0.3
    
    def test_classify_comparison_question(self, question_classifier, sample_schema):
        """Test comparison question classification."""
        question = "Compare sales between different product categories"
        result = question_classifier.classify_question(question, sample_schema)
        
        assert result.question_type == QuestionType.COMPARISON
        assert result.confidence > 0.3
    
    def test_classify_trend_question(self, question_classifier, sample_schema):
        """Test trend question classification."""
        question = "Show me the sales trend over the last year"
        result = question_classifier.classify_question(question, sample_schema)
        
        assert result.question_type == QuestionType.TREND
        assert result.confidence > 0.3
    
    def test_classify_ranking_question(self, question_classifier, sample_schema):
        """Test ranking question classification."""
        question = "What are the top 5 customers by sales?"
        result = question_classifier.classify_question(question, sample_schema)
        
        assert result.question_type == QuestionType.RANKING
        assert result.confidence > 0.3
    
    def test_classify_distribution_question(self, question_classifier, sample_schema):
        """Test distribution question classification."""
        question = "What is the distribution of sales by category?"
        result = question_classifier.classify_question(question, sample_schema)
        
        assert result.question_type == QuestionType.DISTRIBUTION
        assert result.confidence > 0.3
    
    def test_get_question_features(self, question_classifier):
        """Test question feature extraction."""
        question = "What is the average sales amount for premium customers?"
        features = question_classifier.get_question_features(question)
        
        assert features['word_count'] > 5
        assert 'what' in features['question_words']
        assert 'average' in features['aggregation_words']
    
    def test_suggest_question_improvements(self, question_classifier, sample_schema):
        """Test question improvement suggestions."""
        # Vague question
        question = "Tell me about data"
        result = question_classifier.classify_question(question, sample_schema)
        suggestions = question_classifier.suggest_question_improvements(question, result)
        
        assert len(suggestions) > 0
    
    def test_complexity_determination(self, question_classifier, sample_schema):
        """Test complexity level determination."""
        # Simple question
        simple_q = "What is the total sales?"
        simple_result = question_classifier.classify_question(simple_q, sample_schema)
        assert simple_result.complexity_level == 'simple'
        
        # Complex question
        complex_q = "Compare average sales by category and customer type with correlation analysis"
        complex_result = question_classifier.classify_question(complex_q, sample_schema)
        assert complex_result.complexity_level in ['medium', 'complex']


class TestIntegration:
    """Integration tests for NLP components."""
    
    def test_end_to_end_question_processing(self, nlp_processor, sample_schema):
        """Test complete question processing pipeline."""
        questions = [
            "What is the total sales amount?",
            "Show me customers with sales greater than 1000",
            "Compare sales between Electronics and Clothing categories",
            "What is the trend of sales over time?",
            "Who are the top 10 customers by sales?"
        ]
        
        for question in questions:
            result = nlp_processor.parse_question(question, sample_schema)
            
            # Basic validation
            assert isinstance(result, ParsedIntent)
            assert result.question_type in QuestionType
            assert 0 <= result.confidence <= 1
            assert result.original_question == question
    
    def test_column_reference_accuracy(self, nlp_processor, sample_schema):
        """Test accuracy of column reference extraction."""
        test_cases = [
            ("sales amount", ["sales_amount"]),
            ("customer name", ["customer_name"]),
            ("order date", ["order_date"]),
            ("product category", ["product_category"]),
            ("unit price", ["unit_price"])
        ]
        
        for question_text, expected_columns in test_cases:
            question = f"Show me the {question_text}"
            result = nlp_processor.parse_question(question, sample_schema)
            
            # Check if at least one expected column is found
            found_columns = set(result.columns)
            expected_set = set(expected_columns)
            assert len(found_columns.intersection(expected_set)) > 0
    
    def test_filter_extraction_accuracy(self, entity_extractor, sample_schema):
        """Test accuracy of filter extraction."""
        test_cases = [
            ("quantity greater than 100", "quantity", "greater_than", 100.0),
            ("customer_name equals 'John'", "customer_name", "equals", "John"),
            ("sales_amount > 1000", "sales_amount", "greater_than", 1000.0)
        ]
        
        for text, expected_col, expected_op, expected_val in test_cases:
            conditions = entity_extractor.extract_filter_conditions(text, sample_schema)
            
            if conditions:
                condition = conditions[0]
                assert condition['column'] == expected_col
                assert condition['operator'] == expected_op
                assert condition['value'] == expected_val