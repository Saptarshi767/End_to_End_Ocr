"""
Main conversational AI engine that orchestrates all components.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .nlp_processor import NLPProcessor, ParsedIntent
from .query_generator import QueryGenerator
from .query_executor import QueryExecutor, ExecutionContext
from .llm_provider import LLMProviderManager, LLMProvider
from .prompt_engineer import PromptEngineer, PromptTemplate, PromptContext
from .response_parser import ResponseParser, ResponseType
from .response_formatter import ResponseFormatter, FormattingContext
from ..core.models import ConversationResponse, DataSchema, QueryResult
from ..core.interfaces import ConversationalAIInterface


@dataclass
class ConversationContext:
    """Context for conversation management."""
    session_id: str
    user_id: Optional[str] = None
    data_schema: Optional[DataSchema] = None
    conversation_history: List[Dict[str, Any]] = None
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.user_preferences is None:
            self.user_preferences = {}


class ConversationalAIEngine(ConversationalAIInterface):
    """Main conversational AI engine for data analysis."""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.query_generator = QueryGenerator()
        self.query_executor = QueryExecutor()
        self.llm_manager = LLMProviderManager()
        self.prompt_engineer = PromptEngineer()
        self.response_parser = ResponseParser()
        self.response_formatter = ResponseFormatter()
        
        self.logger = logging.getLogger(__name__)
        
        # Conversation contexts by session
        self.contexts: Dict[str, ConversationContext] = {}
    
    def process_question(self, question: str, data_schema: DataSchema, 
                        session_id: str = "default", 
                        execution_context: Optional[ExecutionContext] = None) -> ConversationResponse:
        """
        Process natural language question and generate comprehensive response.
        
        Args:
            question: Natural language question
            data_schema: Data schema for context
            session_id: Conversation session ID
            execution_context: Context for query execution
            
        Returns:
            ConversationResponse with analysis and visualizations
        """
        try:
            # Get or create conversation context
            conv_context = self._get_conversation_context(session_id, data_schema)
            
            # Step 1: Parse the natural language question
            intent = self.nlp_processor.parse_question(question, data_schema)
            self.logger.info(f"Parsed question type: {intent.question_type}, confidence: {intent.confidence}")
            
            # Step 2: Determine processing approach based on intent confidence
            if intent.confidence >= 0.7:
                # High confidence - use rule-based approach
                response = self._process_with_rules(intent, data_schema, execution_context, conv_context)
            else:
                # Low confidence - use LLM-assisted approach
                response = self._process_with_llm(question, intent, data_schema, execution_context, conv_context)
            
            # Step 3: Update conversation history
            self._update_conversation_history(conv_context, question, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return self._create_error_response(str(e), question)
    
    def _process_with_rules(self, intent: ParsedIntent, data_schema: DataSchema,
                           execution_context: Optional[ExecutionContext],
                           conv_context: ConversationContext) -> ConversationResponse:
        """Process question using rule-based approach."""
        try:
            # Generate SQL query
            sql_query = self.query_generator.generate_sql_query(intent, data_schema)
            
            # Execute query if execution context is provided
            query_result = None
            if execution_context:
                try:
                    pandas_query = self.query_generator.generate_pandas_query(intent, data_schema)
                    query_result = self.query_executor.execute_query_with_fallback(
                        sql_query, pandas_query, execution_context
                    )
                except Exception as e:
                    self.logger.warning(f"Query execution failed: {e}")
            
            # Format response
            formatting_context = FormattingContext(
                query_result=query_result,
                schema=data_schema,
                original_question=intent.original_question,
                user_preferences=conv_context.user_preferences
            )
            
            # Create a mock parsed response for formatting
            from .response_parser import ParsedResponse, ValidationResult
            parsed_response = ParsedResponse(
                response_type=ResponseType.SQL_QUERY,
                content=f"Generated query: {sql_query.sql}",
                structured_data={
                    'query_count': 1,
                    'primary_query': sql_query.sql,
                    'has_aggregation': intent.aggregation is not None,
                    'has_filtering': bool(intent.filters)
                },
                confidence=intent.confidence,
                validation_result=ValidationResult(is_valid=True, confidence=1.0),
                extracted_queries=[sql_query]
            )
            
            return self.response_formatter.format_response(parsed_response, formatting_context)
            
        except Exception as e:
            self.logger.error(f"Rule-based processing failed: {e}")
            # Fallback to LLM approach
            return self._process_with_llm(intent.original_question, intent, data_schema, 
                                        execution_context, conv_context)
    
    def _process_with_llm(self, question: str, intent: ParsedIntent, 
                         data_schema: DataSchema, execution_context: Optional[ExecutionContext],
                         conv_context: ConversationContext) -> ConversationResponse:
        """Process question using LLM-assisted approach."""
        try:
            # Determine the type of LLM assistance needed
            if intent.confidence < 0.3:
                # Very low confidence - ask for analysis
                template = PromptTemplate.DATA_ANALYSIS
                expected_response_type = ResponseType.ANALYSIS
            elif intent.filters and not intent.columns:
                # Has filters but unclear columns - ask for query help
                template = PromptTemplate.QUERY_GENERATION
                expected_response_type = ResponseType.SQL_QUERY
            else:
                # Medium confidence - generate query with LLM assistance
                template = PromptTemplate.QUERY_GENERATION
                expected_response_type = ResponseType.SQL_QUERY
            
            # Create prompt context
            prompt_context = PromptContext(
                schema=data_schema,
                intent=intent,
                sample_data=data_schema.sample_data if hasattr(data_schema, 'sample_data') else None,
                previous_queries=self._get_recent_queries(conv_context)
            )
            
            # Generate prompt
            prompt = self.prompt_engineer.generate_prompt(template, prompt_context)
            
            # Get LLM response
            llm_response = self.llm_manager.generate_response(
                prompt, 
                context={'schema': data_schema, 'task': template.value}
            )
            
            # Parse LLM response
            parsed_response = self.response_parser.parse_response(
                llm_response.content, 
                expected_response_type,
                context={'schema': data_schema}
            )
            
            # Execute queries if found and execution context available
            query_result = None
            if (parsed_response.extracted_queries and execution_context and 
                expected_response_type == ResponseType.SQL_QUERY):
                try:
                    query = parsed_response.extracted_queries[0]
                    # Convert to pandas query for execution
                    pandas_query = f"df.query('{query.sql}')" if 'WHERE' in query.sql.upper() else "df"
                    query_result = self.query_executor.execute_pandas_query(pandas_query, execution_context)
                except Exception as e:
                    self.logger.warning(f"LLM query execution failed: {e}")
            
            # Format response
            formatting_context = FormattingContext(
                query_result=query_result,
                schema=data_schema,
                original_question=question,
                user_preferences=conv_context.user_preferences
            )
            
            return self.response_formatter.format_response(parsed_response, formatting_context)
            
        except Exception as e:
            self.logger.error(f"LLM processing failed: {e}")
            return self._create_error_response(f"I encountered an error processing your question: {str(e)}", question)
    
    def generate_query(self, question: str, schema: DataSchema) -> Any:
        """Generate query from natural language question."""
        intent = self.nlp_processor.parse_question(question, schema)
        return self.query_generator.generate_sql_query(intent, schema)
    
    def format_response(self, query_result: QueryResult, question: str) -> ConversationResponse:
        """Format query result into conversational response."""
        # Create a basic formatting context
        formatting_context = FormattingContext(
            query_result=query_result,
            original_question=question
        )
        
        # Create a mock parsed response
        from .response_parser import ParsedResponse, ValidationResult
        parsed_response = ParsedResponse(
            response_type=ResponseType.SQL_QUERY,
            content="Query executed successfully",
            structured_data={'query_count': 1, 'has_data': query_result.row_count > 0},
            confidence=0.9,
            validation_result=ValidationResult(is_valid=True, confidence=1.0)
        )
        
        return self.response_formatter.format_response(parsed_response, formatting_context)
    
    def _get_conversation_context(self, session_id: str, data_schema: DataSchema) -> ConversationContext:
        """Get or create conversation context."""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(
                session_id=session_id,
                data_schema=data_schema
            )
        else:
            # Update schema if it has changed
            self.contexts[session_id].data_schema = data_schema
        
        return self.contexts[session_id]
    
    def _update_conversation_history(self, context: ConversationContext, 
                                   question: str, response: ConversationResponse):
        """Update conversation history."""
        context.conversation_history.append({
            'question': question,
            'response_text': response.text_response,
            'confidence': response.confidence,
            'has_visualizations': len(response.visualizations) > 0,
            'timestamp': None  # Would be set to current time in production
        })
        
        # Keep only last 10 interactions
        if len(context.conversation_history) > 10:
            context.conversation_history = context.conversation_history[-10:]
    
    def _get_recent_queries(self, context: ConversationContext) -> List[str]:
        """Get recent queries from conversation history."""
        queries = []
        for interaction in context.conversation_history[-3:]:  # Last 3 interactions
            # Extract queries from response text (simplified)
            response_text = interaction.get('response_text', '')
            if 'SELECT' in response_text.upper():
                # Try to extract SQL from response
                import re
                sql_matches = re.findall(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL)
                queries.extend(sql_matches)
        
        return queries
    
    def _create_error_response(self, error_message: str, question: str) -> ConversationResponse:
        """Create error response."""
        return ConversationResponse(
            text_response=f"I apologize, but I encountered an issue: {error_message}",
            visualizations=[],
            data_summary={'error': True, 'error_message': error_message},
            suggested_questions=[
                "Try rephrasing your question",
                "Ask for help with data analysis",
                "Check if your data is properly formatted"
            ],
            confidence=0.1
        )
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation session."""
        if session_id not in self.contexts:
            return {'error': 'Session not found'}
        
        context = self.contexts[session_id]
        
        return {
            'session_id': session_id,
            'total_interactions': len(context.conversation_history),
            'recent_questions': [h['question'] for h in context.conversation_history[-5:]],
            'avg_confidence': sum(h['confidence'] for h in context.conversation_history) / len(context.conversation_history) if context.conversation_history else 0,
            'has_data_schema': context.data_schema is not None,
            'user_preferences': context.user_preferences
        }
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for session."""
        if session_id in self.contexts:
            self.contexts[session_id].conversation_history = []
    
    def set_user_preferences(self, session_id: str, preferences: Dict[str, Any]):
        """Set user preferences for session."""
        if session_id in self.contexts:
            self.contexts[session_id].user_preferences.update(preferences)
        else:
            context = ConversationContext(session_id=session_id, user_preferences=preferences)
            self.contexts[session_id] = context
    
    def get_available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        return [provider.value for provider in self.llm_manager.get_available_providers()]
    
    def is_ready(self) -> bool:
        """Check if the conversational AI engine is ready."""
        try:
            # Check if at least one LLM provider is available
            available_providers = self.llm_manager.get_available_providers()
            return len(available_providers) > 0
        except Exception:
            return False