"""
LLM provider abstraction layer for multiple LLM services.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from ..core.interfaces import LLMProviderInterface
from ..core.models import DataSchema, Query, ValidationResult
from ..core.config import config_manager


class LLMProvider(Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    LOCAL = "local"


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    usage: Dict[str, Any]
    model: str
    provider: LLMProvider
    response_time_ms: int
    confidence: float = 0.0


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        pass
    
    def validate_config(self) -> ValidationResult:
        """Validate provider configuration."""
        errors = []
        warnings = []
        
        if 'timeout_seconds' in self.config and self.config['timeout_seconds'] <= 0:
            errors.append("timeout_seconds must be greater than 0")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=1.0 if len(errors) == 0 else 0.0
        )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'gpt-3.5-turbo')
        self.max_tokens = config.get('max_tokens', 2000)
        self.temperature = config.get('temperature', 0.1)
        self.timeout = config.get('timeout_seconds', 30)
        
        # Import OpenAI client
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self._available = True
        except ImportError:
            self.logger.error("OpenAI library not installed. Install with: pip install openai")
            self._available = False
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self._available = False
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self._available:
            raise RuntimeError("OpenAI provider not available")
        
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Add context if provided
            if context:
                system_message = self._build_system_message(context)
                messages.insert(0, {"role": "system", "content": system_message})
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            response_time = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                provider=LLMProvider.OPENAI,
                response_time_ms=response_time,
                confidence=0.9  # High confidence for successful API response
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_query(self, question: str, schema: DataSchema) -> Query:
        """Generate data query from natural language question."""
        context = {
            'schema': schema,
            'task': 'query_generation'
        }
        
        prompt = self._build_query_prompt(question, schema)
        response = self.generate_response(prompt, context)
        
        # Parse SQL from response
        sql = self._extract_sql_from_response(response.content)
        
        return Query(sql=sql, columns=['*'])
    
    def is_available(self) -> bool:
        """Check if OpenAI service is available."""
        if not self._available:
            return False
        
        try:
            # Simple test call
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                timeout=5
            )
            return True
        except Exception:
            return False
    
    def validate_config(self) -> ValidationResult:
        """Validate OpenAI configuration."""
        result = super().validate_config()
        
        if not self.api_key:
            result.errors.append("OpenAI API key is required")
        
        if not self.model:
            result.errors.append("OpenAI model is required")
        
        if self.max_tokens <= 0:
            result.errors.append("max_tokens must be greater than 0")
        
        if not (0 <= self.temperature <= 2):
            result.errors.append("temperature must be between 0 and 2")
        
        result.is_valid = len(result.errors) == 0
        result.confidence = 1.0 if result.is_valid else 0.0
        
        return result
    
    def _build_system_message(self, context: Dict[str, Any]) -> str:
        """Build system message from context."""
        if context.get('task') == 'query_generation':
            return """You are a SQL query generator. Given a natural language question and a database schema, 
            generate a valid SQL query. Return only the SQL query without any explanation or formatting."""
        
        return "You are a helpful assistant for data analysis tasks."
    
    def _build_query_prompt(self, question: str, schema: DataSchema) -> str:
        """Build prompt for query generation."""
        schema_info = []
        for col in schema.columns:
            schema_info.append(f"- {col.name} ({col.data_type.value})")
        
        prompt = f"""
Given the following database schema:
{chr(10).join(schema_info)}

Generate a SQL query to answer this question: "{question}"

Return only the SQL query, no explanation needed.
"""
        return prompt.strip()
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Remove common formatting
        sql = response.strip()
        
        # Remove markdown code blocks
        if sql.startswith('```sql'):
            sql = sql[6:]
        elif sql.startswith('```'):
            sql = sql[3:]
        
        if sql.endswith('```'):
            sql = sql[:-3]
        
        # Remove semicolon at the end
        sql = sql.rstrip(';').strip()
        
        return sql


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.model = config.get('model', 'claude-3-sonnet-20240229')
        self.timeout = config.get('timeout_seconds', 30)
        
        # Import Anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self._available = True
        except ImportError:
            self.logger.error("Anthropic library not installed. Install with: pip install anthropic")
            self._available = False
        except Exception as e:
            self.logger.error(f"Failed to initialize Claude client: {e}")
            self._available = False
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate response using Claude API."""
        if not self._available:
            raise RuntimeError("Claude provider not available")
        
        start_time = time.time()
        
        try:
            # Build system message
            system_message = ""
            if context:
                system_message = self._build_system_message(context)
            
            # Make API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system_message,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout
            )
            
            response_time = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=response.content[0].text,
                usage=response.usage.__dict__ if hasattr(response, 'usage') else {},
                model=response.model,
                provider=LLMProvider.CLAUDE,
                response_time_ms=response_time,
                confidence=0.9
            )
            
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            raise
    
    def generate_query(self, question: str, schema: DataSchema) -> Query:
        """Generate data query from natural language question."""
        context = {
            'schema': schema,
            'task': 'query_generation'
        }
        
        prompt = self._build_query_prompt(question, schema)
        response = self.generate_response(prompt, context)
        
        # Parse SQL from response
        sql = self._extract_sql_from_response(response.content)
        
        return Query(sql=sql, columns=['*'])
    
    def is_available(self) -> bool:
        """Check if Claude service is available."""
        if not self._available:
            return False
        
        try:
            # Simple test call
            test_response = self.client.messages.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hello"}],
                timeout=5
            )
            return True
        except Exception:
            return False
    
    def _build_system_message(self, context: Dict[str, Any]) -> str:
        """Build system message from context."""
        if context.get('task') == 'query_generation':
            return """You are a SQL query generator. Given a natural language question and a database schema, 
            generate a valid SQL query. Return only the SQL query without any explanation or formatting."""
        
        return "You are a helpful assistant for data analysis tasks."
    
    def _build_query_prompt(self, question: str, schema: DataSchema) -> str:
        """Build prompt for query generation."""
        schema_info = []
        for col in schema.columns:
            schema_info.append(f"- {col.name} ({col.data_type.value})")
        
        prompt = f"""
Given the following database schema:
{chr(10).join(schema_info)}

Generate a SQL query to answer this question: "{question}"

Return only the SQL query, no explanation needed.
"""
        return prompt.strip()
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Remove common formatting
        sql = response.strip()
        
        # Remove markdown code blocks
        if sql.startswith('```sql'):
            sql = sql[6:]
        elif sql.startswith('```'):
            sql = sql[3:]
        
        if sql.endswith('```'):
            sql = sql[:-3]
        
        # Remove semicolon at the end
        sql = sql.rstrip(';').strip()
        
        return sql


class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider implementation (placeholder for local models)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get('model_path')
        self._available = False  # Not implemented yet
        
        self.logger.warning("Local LLM provider is not yet implemented")
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> LLMResponse:
        """Generate response using local model."""
        raise NotImplementedError("Local LLM provider not yet implemented")
    
    def generate_query(self, question: str, schema: DataSchema) -> Query:
        """Generate data query from natural language question."""
        raise NotImplementedError("Local LLM provider not yet implemented")
    
    def is_available(self) -> bool:
        """Check if local model is available."""
        return False


class LLMProviderManager:
    """Manager for multiple LLM providers with fallback support."""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        # Initialize OpenAI
        openai_config = config_manager.get_llm_config('openai')
        if openai_config.get('api_key'):
            try:
                self.providers[LLMProvider.OPENAI] = OpenAIProvider(openai_config)
                self.logger.info("OpenAI provider initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        # Initialize Claude
        claude_config = config_manager.get_llm_config('claude')
        if claude_config.get('api_key'):
            try:
                self.providers[LLMProvider.CLAUDE] = ClaudeProvider(claude_config)
                self.logger.info("Claude provider initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Claude provider: {e}")
        
        # Initialize Local (placeholder)
        local_config = config_manager.get_llm_config('local')
        if local_config.get('model_path'):
            try:
                self.providers[LLMProvider.LOCAL] = LocalLLMProvider(local_config)
                self.logger.info("Local LLM provider initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Local LLM provider: {e}")
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers."""
        available = []
        for provider_type, provider in self.providers.items():
            if provider.is_available():
                available.append(provider_type)
        return available
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None, 
                         preferred_provider: Optional[LLMProvider] = None) -> LLMResponse:
        """
        Generate response with fallback support.
        
        Args:
            prompt: Input prompt
            context: Additional context
            preferred_provider: Preferred provider to try first
            
        Returns:
            LLMResponse from successful provider
        """
        providers_to_try = []
        
        # Add preferred provider first
        if preferred_provider and preferred_provider in self.providers:
            providers_to_try.append(preferred_provider)
        
        # Add other available providers
        for provider_type in [LLMProvider.OPENAI, LLMProvider.CLAUDE, LLMProvider.LOCAL]:
            if provider_type not in providers_to_try and provider_type in self.providers:
                providers_to_try.append(provider_type)
        
        last_error = None
        for provider_type in providers_to_try:
            provider = self.providers[provider_type]
            
            try:
                if provider.is_available():
                    return provider.generate_response(prompt, context)
            except Exception as e:
                self.logger.warning(f"Provider {provider_type.value} failed: {e}")
                last_error = e
                continue
        
        if last_error:
            raise last_error
        else:
            raise RuntimeError("No LLM providers available")
    
    def generate_query(self, question: str, schema: DataSchema, 
                      preferred_provider: Optional[LLMProvider] = None) -> Query:
        """
        Generate query with fallback support.
        
        Args:
            question: Natural language question
            schema: Data schema
            preferred_provider: Preferred provider to try first
            
        Returns:
            Generated Query
        """
        providers_to_try = []
        
        # Add preferred provider first
        if preferred_provider and preferred_provider in self.providers:
            providers_to_try.append(preferred_provider)
        
        # Add other available providers
        for provider_type in [LLMProvider.OPENAI, LLMProvider.CLAUDE]:
            if provider_type not in providers_to_try and provider_type in self.providers:
                providers_to_try.append(provider_type)
        
        last_error = None
        for provider_type in providers_to_try:
            provider = self.providers[provider_type]
            
            try:
                if provider.is_available():
                    return provider.generate_query(question, schema)
            except Exception as e:
                self.logger.warning(f"Query generation failed with {provider_type.value}: {e}")
                last_error = e
                continue
        
        if last_error:
            raise last_error
        else:
            raise RuntimeError("No LLM providers available for query generation")
    
    def validate_all_providers(self) -> Dict[LLMProvider, ValidationResult]:
        """Validate all configured providers."""
        results = {}
        
        for provider_type, provider in self.providers.items():
            results[provider_type] = provider.validate_config()
        
        return results
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        status = {}
        
        for provider_type, provider in self.providers.items():
            status[provider_type.value] = {
                'available': provider.is_available(),
                'config_valid': provider.validate_config().is_valid
            }
        
        return status


# Global LLM provider manager
llm_manager = LLMProviderManager()