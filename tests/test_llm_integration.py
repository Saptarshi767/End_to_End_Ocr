"""
Tests for the LLM Integration Layer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.ai.llm_provider import (
    LLMProviderManager, OpenAIProvider, ClaudeProvider, LocalLLMProvider,
    LLMProvider, LLMResponse
)
from src.ai.prompt_engineer import PromptEngineer, PromptTemplate, PromptContext
from src.ai.response_parser import ResponseParser, ResponseType, ParsedResponse
from src.core.models import DataSchema, ColumnInfo, DataType, Query
from src.ai.nlp_processor import ParsedIntent, QuestionType, AggregationType


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
def sample_intent():
    """Create a sample parsed intent."""
    return ParsedIntent(
        question_type=QuestionType.AGGREGATION,
        entities={'columns': ['sales_amount'], 'values': [], 'numbers': []},
        columns=['sales_amount'],
        aggregation=AggregationType.SUM,
        confidence=0.8,
        original_question="What is the total sales amount?"
    )


@pytest.fixture
def prompt_engineer():
    """Create prompt engineer instance."""
    return PromptEngineer()


@pytest.fixture
def response_parser():
    """Create response parser instance."""
    return ResponseParser()


class TestPromptEngineer:
    """Test cases for prompt engineer."""
    
    def test_generate_query_prompt(self, prompt_engineer, sample_schema, sample_intent):
        """Test query generation prompt."""
        context = PromptContext(schema=sample_schema, intent=sample_intent)
        prompt = prompt_engineer.generate_prompt(PromptTemplate.QUERY_GENERATION, context)
        
        assert "sales_amount" in prompt
        assert "What is the total sales amount?" in prompt
        assert "data_table" in prompt
        assert "SQL" in prompt
    
    def test_generate_analysis_prompt(self, prompt_engineer, sample_schema, sample_intent):
        """Test data analysis prompt."""
        context = PromptContext(schema=sample_schema, intent=sample_intent)
        prompt = prompt_engineer.generate_prompt(PromptTemplate.DATA_ANALYSIS, context)
        
        assert "data analyst" in prompt.lower()
        assert "insights" in prompt.lower()
        assert sample_intent.original_question in prompt
    
    def test_generate_explanation_prompt(self, prompt_engineer, sample_intent):
        """Test explanation prompt."""
        context = PromptContext(intent=sample_intent)
        prompt = prompt_engineer.generate_prompt(PromptTemplate.EXPLANATION, context)
        
        assert "explain" in prompt.lower()
        assert "plain language" in prompt.lower()
        assert sample_intent.original_question in prompt
    
    def test_generate_suggestion_prompt(self, prompt_engineer, sample_schema, sample_intent):
        """Test suggestion prompt."""
        context = PromptContext(schema=sample_schema, intent=sample_intent)
        prompt = prompt_engineer.generate_prompt(PromptTemplate.SUGGESTION, context)
        
        assert "follow-up" in prompt.lower()
        assert "suggest" in prompt.lower()
        assert "business" in prompt.lower()
    
    def test_generate_error_diagnosis_prompt(self, prompt_engineer, sample_schema, sample_intent):
        """Test error diagnosis prompt."""
        context = PromptContext(
            schema=sample_schema, 
            intent=sample_intent,
            error_message="Column 'invalid_column' not found"
        )
        prompt = prompt_engineer.generate_prompt(PromptTemplate.ERROR_DIAGNOSIS, context)
        
        assert "error" in prompt.lower()
        assert "Column 'invalid_column' not found" in prompt
        assert "diagnose" in prompt.lower()
    
    def test_optimize_prompt_for_gpt35(self, prompt_engineer):
        """Test prompt optimization for GPT-3.5."""
        original_prompt = "Generate a SQL query. SQL Query:"
        optimized = prompt_engineer.optimize_prompt_for_model(original_prompt, "gpt-3.5-turbo")
        
        assert "Format:" in optimized
        assert len(optimized) > len(original_prompt)
    
    def test_optimize_prompt_for_claude(self, prompt_engineer):
        """Test prompt optimization for Claude."""
        original_prompt = "You are a data analyst."
        optimized = prompt_engineer.optimize_prompt_for_model(original_prompt, "claude-3-sonnet")
        
        assert "Please help as" in optimized
    
    def test_create_few_shot_examples(self, prompt_engineer, sample_schema):
        """Test few-shot example creation."""
        context = PromptContext(schema=sample_schema)
        examples = prompt_engineer.create_few_shot_examples(PromptTemplate.QUERY_GENERATION, context)
        
        assert len(examples) > 0
        assert all('input' in ex and 'output' in ex for ex in examples)
        assert any('SELECT' in ex['output'] for ex in examples)


class TestResponseParser:
    """Test cases for response parser."""
    
    def test_parse_sql_response_with_code_block(self, response_parser):
        """Test parsing SQL response with code block."""
        response = """
        Here's the SQL query:
        ```sql
        SELECT SUM(sales_amount) as total_sales FROM data_table
        ```
        """
        
        parsed = response_parser.parse_response(response, ResponseType.SQL_QUERY)
        
        assert parsed.response_type == ResponseType.SQL_QUERY
        assert len(parsed.extracted_queries) == 1
        assert "SUM(sales_amount)" in parsed.extracted_queries[0].sql
        assert parsed.structured_data['has_aggregation']
    
    def test_parse_sql_response_without_code_block(self, response_parser):
        """Test parsing SQL response without code block."""
        response = "SELECT * FROM data_table WHERE sales_amount > 1000"
        
        parsed = response_parser.parse_response(response, ResponseType.SQL_QUERY)
        
        assert parsed.response_type == ResponseType.SQL_QUERY
        assert len(parsed.extracted_queries) == 1
        assert "WHERE" in parsed.extracted_queries[0].sql
        assert parsed.structured_data['has_filtering']
    
    def test_parse_analysis_response(self, response_parser):
        """Test parsing analysis response."""
        response = """
        1. Key Insights: Sales have increased by 25% this quarter.
        2. Recommendations: Focus on the Electronics category which shows highest growth.
        3. Next Steps: Investigate the decline in Clothing sales.
        """
        
        parsed = response_parser.parse_response(response, ResponseType.ANALYSIS)
        
        assert parsed.response_type == ResponseType.ANALYSIS
        assert len(parsed.structured_data['sections']) == 3
        assert parsed.structured_data['has_insights']
        assert parsed.structured_data['has_recommendations']
        assert parsed.structured_data['has_numbers']
    
    def test_parse_explanation_response(self, response_parser):
        """Test parsing explanation response."""
        response = """
        This shows that sales are increasing because of improved marketing.
        For example, the Electronics category saw a 30% boost due to targeted campaigns.
        This means we should continue investing in digital marketing.
        """
        
        parsed = response_parser.parse_response(response, ResponseType.EXPLANATION)
        
        assert parsed.response_type == ResponseType.EXPLANATION
        assert parsed.structured_data['has_reasoning']
        assert parsed.structured_data['has_examples']
        assert len(parsed.structured_data['elements']) > 0
    
    def test_parse_suggestion_response(self, response_parser):
        """Test parsing suggestion response."""
        response = """
        1. Analyze monthly sales trends to identify seasonal patterns
        2. Compare performance across different product categories
        3. Investigate customer retention rates by segment
        """
        
        parsed = response_parser.parse_response(response, ResponseType.SUGGESTION)
        
        assert parsed.response_type == ResponseType.SUGGESTION
        assert parsed.structured_data['suggestion_count'] == 3
        assert parsed.structured_data['actionable_count'] > 0
    
    def test_extract_sql_queries_multiple(self, response_parser):
        """Test extracting multiple SQL queries."""
        response = """
        First query:
        ```sql
        SELECT COUNT(*) FROM data_table
        ```
        
        Second query:
        ```sql
        SELECT AVG(sales_amount) FROM data_table WHERE product_category = 'Electronics'
        ```
        """
        
        queries = response_parser.extract_sql_queries(response)
        
        assert len(queries) == 2
        assert "COUNT(*)" in queries[0].sql
        assert "AVG(sales_amount)" in queries[1].sql
    
    def test_validate_sql_queries_with_schema(self, response_parser, sample_schema):
        """Test SQL query validation with schema."""
        queries = [
            Query(sql="SELECT sales_amount, customer_name FROM data_table", columns=['sales_amount', 'customer_name']),
            Query(sql="SELECT invalid_column FROM data_table", columns=['invalid_column'])
        ]
        
        validation = response_parser._validate_sql_queries(queries, sample_schema)
        
        assert validation.is_valid  # Should be valid despite warnings
        assert len(validation.warnings) > 0  # Should have warnings about invalid_column


class TestLLMProviders:
    """Test cases for LLM providers."""
    
    def test_openai_provider_config_validation(self):
        """Test OpenAI provider configuration validation."""
        # Valid config
        valid_config = {
            'api_key': 'test-key',
            'model': 'gpt-3.5-turbo',
            'max_tokens': 1000,
            'temperature': 0.1,
            'timeout_seconds': 30
        }
        
        provider = OpenAIProvider(valid_config)
        validation = provider.validate_config()
        
        assert validation.is_valid
        assert len(validation.errors) == 0
    
    def test_openai_provider_invalid_config(self):
        """Test OpenAI provider with invalid configuration."""
        invalid_config = {
            'api_key': '',  # Empty API key
            'max_tokens': -1,  # Invalid max_tokens
            'temperature': 3.0  # Invalid temperature
        }
        
        provider = OpenAIProvider(invalid_config)
        validation = provider.validate_config()
        
        assert not validation.is_valid
        assert len(validation.errors) > 0
    
    @patch('openai.OpenAI')
    @patch('src.ai.llm_provider.openai')
    def test_openai_provider_generate_response(self, mock_openai_module, mock_openai_class):
        """Test OpenAI response generation."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.model_dump.return_value = {'total_tokens': 100}
        mock_response.model = "gpt-3.5-turbo"
        
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_openai_module.OpenAI = mock_openai_class
        
        config = {'api_key': 'test-key', 'model': 'gpt-3.5-turbo', 'max_tokens': 1000, 'temperature': 0.1}
        provider = OpenAIProvider(config)
        provider._available = True  # Override availability check
        
        response = provider.generate_response("Test prompt")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.provider == LLMProvider.OPENAI
        assert response.confidence > 0
    
    def test_claude_provider_config_validation(self):
        """Test Claude provider configuration validation."""
        valid_config = {
            'api_key': 'test-key',
            'model': 'claude-3-sonnet-20240229',
            'timeout_seconds': 30
        }
        
        provider = ClaudeProvider(valid_config)
        validation = provider.validate_config()
        
        # Should be valid (base validation only checks timeout)
        assert validation.is_valid
    
    def test_local_provider_not_implemented(self):
        """Test local provider not implemented."""
        config = {'model_path': '/path/to/model'}
        provider = LocalLLMProvider(config)
        
        assert not provider.is_available()
        
        with pytest.raises(NotImplementedError):
            provider.generate_response("Test prompt")


class TestLLMProviderManager:
    """Test cases for LLM provider manager."""
    
    @patch('src.ai.llm_provider.config_manager')
    def test_provider_manager_initialization(self, mock_config_manager):
        """Test provider manager initialization."""
        # Mock configuration
        mock_config_manager.get_llm_config.side_effect = lambda provider: {
            'openai': {'api_key': 'test-openai-key', 'model': 'gpt-3.5-turbo'},
            'claude': {'api_key': None},
            'local': {'model_path': None}
        }.get(provider, {})
        
        with patch('src.ai.llm_provider.OpenAIProvider') as mock_openai:
            mock_openai.return_value = Mock()
            
            manager = LLMProviderManager()
            
            # Should initialize OpenAI provider
            assert LLMProvider.OPENAI in manager.providers
            # Should not initialize Claude (no API key)
            assert LLMProvider.CLAUDE not in manager.providers
    
    @patch('src.ai.llm_provider.config_manager')
    def test_provider_manager_fallback(self, mock_config_manager):
        """Test provider manager fallback functionality."""
        mock_config_manager.get_llm_config.return_value = {'api_key': 'test-key'}
        
        manager = LLMProviderManager()
        
        # Mock providers
        mock_provider1 = Mock()
        mock_provider1.is_available.return_value = True
        mock_provider1.generate_response.side_effect = Exception("Provider 1 failed")
        
        mock_provider2 = Mock()
        mock_provider2.is_available.return_value = True
        mock_provider2.generate_response.return_value = LLMResponse(
            content="Success",
            usage={},
            model="test-model",
            provider=LLMProvider.CLAUDE,
            response_time_ms=100
        )
        
        manager.providers = {
            LLMProvider.OPENAI: mock_provider1,
            LLMProvider.CLAUDE: mock_provider2
        }
        
        # Should fallback to second provider
        response = manager.generate_response("Test prompt", preferred_provider=LLMProvider.OPENAI)
        
        assert response.content == "Success"
        assert response.provider == LLMProvider.CLAUDE
    
    def test_provider_manager_no_providers_available(self):
        """Test provider manager with no available providers."""
        manager = LLMProviderManager()
        manager.providers = {}  # No providers
        
        with pytest.raises(RuntimeError, match="No LLM providers available"):
            manager.generate_response("Test prompt")
    
    @patch('src.ai.llm_provider.config_manager')
    def test_get_provider_status(self, mock_config_manager):
        """Test getting provider status."""
        mock_config_manager.get_llm_config.return_value = {'api_key': 'test-key'}
        
        manager = LLMProviderManager()
        
        # Mock provider
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.validate_config.return_value = Mock(is_valid=True)
        
        manager.providers = {LLMProvider.OPENAI: mock_provider}
        
        status = manager.get_provider_status()
        
        assert 'openai' in status
        assert status['openai']['available'] is True
        assert status['openai']['config_valid'] is True


class TestIntegration:
    """Integration tests for LLM components."""
    
    def test_end_to_end_query_generation_flow(self, prompt_engineer, response_parser, sample_schema, sample_intent):
        """Test complete query generation flow."""
        # Generate prompt
        context = PromptContext(schema=sample_schema, intent=sample_intent)
        prompt = prompt_engineer.generate_prompt(PromptTemplate.QUERY_GENERATION, context)
        
        # Simulate LLM response
        mock_llm_response = """
        ```sql
        SELECT SUM(sales_amount) as total_sales FROM data_table
        ```
        """
        
        # Parse response
        parsed = response_parser.parse_response(mock_llm_response, ResponseType.SQL_QUERY, {'schema': sample_schema})
        
        assert len(parsed.extracted_queries) == 1
        assert "SUM(sales_amount)" in parsed.extracted_queries[0].sql
        assert parsed.validation_result.is_valid
        assert parsed.confidence > 0.5
    
    def test_end_to_end_analysis_flow(self, prompt_engineer, response_parser, sample_schema, sample_intent):
        """Test complete analysis flow."""
        # Generate prompt
        context = PromptContext(schema=sample_schema, intent=sample_intent)
        prompt = prompt_engineer.generate_prompt(PromptTemplate.DATA_ANALYSIS, context)
        
        # Simulate LLM response
        mock_llm_response = """
        Based on the sales data analysis:
        
        1. Key Insights: Total sales amount is $150,000 with strong performance in Electronics.
        2. Trends: 25% increase compared to last quarter shows positive growth.
        3. Recommendations: Focus marketing efforts on top-performing categories.
        """
        
        # Parse response
        parsed = response_parser.parse_response(mock_llm_response, ResponseType.ANALYSIS)
        
        assert parsed.structured_data['has_insights']
        assert parsed.structured_data['has_recommendations']
        assert parsed.structured_data['has_numbers']
        assert len(parsed.structured_data['sections']) > 0
        assert parsed.confidence > 0.5
    
    def test_prompt_optimization_and_parsing(self, prompt_engineer, response_parser):
        """Test prompt optimization and response parsing integration."""
        # Test different model optimizations
        base_prompt = "Generate a SQL query. SQL Query:"
        
        gpt35_prompt = prompt_engineer.optimize_prompt_for_model(base_prompt, "gpt-3.5-turbo")
        claude_prompt = prompt_engineer.optimize_prompt_for_model(base_prompt, "claude-3-sonnet")
        
        # Both should be different from base
        assert gpt35_prompt != base_prompt
        assert claude_prompt != base_prompt
        
        # Test parsing responses from different formats
        responses = [
            "SELECT * FROM data_table",  # Plain SQL
            "```sql\nSELECT * FROM data_table\n```",  # Code block
            "Here's the query:\n```\nSELECT * FROM data_table\n```"  # Generic code block
        ]
        
        for response in responses:
            parsed = response_parser.parse_response(response, ResponseType.SQL_QUERY)
            assert len(parsed.extracted_queries) == 1
            assert "SELECT" in parsed.extracted_queries[0].sql