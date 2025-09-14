"""
Tests for Conversational Chat Interface Component

Tests chat UI with message history, context management,
quick question suggestions, and visualization embedding.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import json
import plotly.graph_objects as go

from src.ui.components.chat_interface import ConversationalChatInterface


class TestConversationalChatInterface:
    """Test cases for ConversationalChatInterface"""
    
    @pytest.fixture
    def chat_interface(self):
        """Create ConversationalChatInterface instance"""
        return ConversationalChatInterface(api_base_url="http://test-api:8000")
    
    @pytest.fixture
    def mock_document_context(self):
        """Create mock document context"""
        return {
            'document': {
                'document_id': 'test_doc_123',
                'filename': 'test_document.pdf',
                'status': 'completed'
            },
            'tables': [
                {
                    'table_id': 'table_1',
                    'headers': ['Name', 'Age', 'City'],
                    'row_count': 10
                },
                {
                    'table_id': 'table_2',
                    'headers': ['Product', 'Price', 'Quantity'],
                    'row_count': 15
                }
            ],
            'schema': {
                'tables': [
                    {'table_id': 'table_1', 'columns': ['Name', 'Age', 'City'], 'row_count': 10},
                    {'table_id': 'table_2', 'columns': ['Product', 'Price', 'Quantity'], 'row_count': 15}
                ],
                'columns': ['Name', 'Age', 'City', 'Product', 'Price', 'Quantity']
            }
        }
    
    @pytest.fixture
    def mock_chat_response(self):
        """Create mock chat response"""
        return {
            'response': 'Based on your data, the average age is 32 years.',
            'visualizations': [
                {
                    'chart_type': 'bar',
                    'title': 'Age Distribution',
                    'data': {
                        'x': ['20-30', '30-40', '40-50'],
                        'y': [5, 8, 2]
                    }
                }
            ],
            'data_summary': {
                'metrics': {
                    'average_age': 32,
                    'total_records': 15
                }
            },
            'suggested_questions': [
                'What is the age distribution by city?',
                'Show me the oldest and youngest people'
            ]
        }
    
    @patch('streamlit.session_state', {})
    def test_initialization(self, chat_interface):
        """Test interface initialization"""
        
        assert chat_interface.api_base_url == "http://test-api:8000"
        
        # Check session state initialization
        assert 'chat_history' in st.session_state
        assert 'chat_context' in st.session_state
        assert isinstance(st.session_state.chat_history, list)
        assert isinstance(st.session_state.chat_context, dict)
    
    @patch('requests.get')
    def test_load_document_context_success(self, mock_get, chat_interface):
        """Test successful document context loading"""
        
        # Mock API responses
        doc_response = Mock()
        doc_response.status_code = 200
        doc_response.json.return_value = {'document_id': 'test_doc', 'filename': 'test.pdf'}
        
        tables_response = Mock()
        tables_response.status_code = 200
        tables_response.json.return_value = [
            {'table_id': 'table_1', 'headers': ['Name', 'Age'], 'row_count': 5}
        ]
        
        mock_get.side_effect = [doc_response, tables_response]
        
        with patch('streamlit.session_state', {'auth_token': 'test_token'}):
            context = chat_interface._load_document_context('test_doc')
            
            assert 'document' in context
            assert 'tables' in context
            assert 'schema' in context
            assert context['document']['document_id'] == 'test_doc'
    
    @patch('requests.get')
    def test_load_document_context_failure(self, mock_get, chat_interface):
        """Test failed document context loading"""
        
        # Mock failed API response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        with patch('streamlit.session_state', {'auth_token': 'test_token'}):
            context = chat_interface._load_document_context('test_doc')
            
            assert context == {}
    
    def test_extract_schema_info(self, chat_interface):
        """Test schema information extraction"""
        
        tables = [
            {
                'table_id': 'table_1',
                'table_index': 0,
                'headers': ['Name', 'Age'],
                'row_count': 5
            },
            {
                'table_id': 'table_2',
                'table_index': 1,
                'headers': ['Product', 'Price'],
                'row_count': 10
            }
        ]
        
        schema = chat_interface._extract_schema_info(tables)
        
        assert len(schema['tables']) == 2
        assert len(schema['columns']) == 4
        assert 'Name' in schema['columns']
        assert 'Product' in schema['columns']
    
    def test_generate_question_suggestions_basic(self, chat_interface):
        """Test basic question suggestions generation"""
        
        suggestions = chat_interface._generate_question_suggestions('test_doc')
        
        # Should have basic suggestions
        assert len(suggestions) >= 6
        
        suggestion_texts = [s['text'] for s in suggestions]
        assert any('trends' in text.lower() for text in suggestion_texts)
        assert any('summary' in text.lower() for text in suggestion_texts)
        assert any('highest' in text.lower() for text in suggestion_texts)
    
    @patch('streamlit.session_state', {})
    def test_generate_question_suggestions_contextual(self, chat_interface):
        """Test contextual question suggestions generation"""
        
        # Setup context with schema
        st.session_state.chat_context = {
            'schema': {
                'columns': ['Name', 'Age', 'Salary']
            }
        }
        
        suggestions = chat_interface._generate_question_suggestions('test_doc')
        
        # Should include contextual suggestions based on columns
        suggestion_texts = [s['text'] for s in suggestions]
        contextual_suggestions = [text for text in suggestion_texts if 'average' in text.lower()]
        
        assert len(contextual_suggestions) > 0
    
    @patch('streamlit.session_state', {'chat_history': []})
    def test_handle_question_success(self, chat_interface, mock_chat_response):
        """Test successful question handling"""
        
        with patch.object(chat_interface, '_send_question_to_api', return_value=mock_chat_response):
            with patch('streamlit.spinner'):
                with patch('streamlit.experimental_rerun'):
                    
                    chat_interface._handle_question('test_doc', 'What is the average age?')
                    
                    # Check that messages were added to history
                    assert len(st.session_state.chat_history) == 2
                    
                    # Check user message
                    user_message = st.session_state.chat_history[0]
                    assert user_message['role'] == 'user'
                    assert user_message['content'] == 'What is the average age?'
                    
                    # Check assistant message
                    assistant_message = st.session_state.chat_history[1]
                    assert assistant_message['role'] == 'assistant'
                    assert 'average age is 32' in assistant_message['content']
                    assert 'visualizations' in assistant_message
                    assert 'suggested_questions' in assistant_message
    
    @patch('streamlit.session_state', {'chat_history': []})
    def test_handle_question_failure(self, chat_interface):
        """Test question handling with API failure"""
        
        with patch.object(chat_interface, '_send_question_to_api', side_effect=Exception("API Error")):
            with patch('streamlit.spinner'):
                with patch('streamlit.experimental_rerun'):
                    
                    chat_interface._handle_question('test_doc', 'What is the average age?')
                    
                    # Check that error message was added
                    assert len(st.session_state.chat_history) == 2
                    
                    assistant_message = st.session_state.chat_history[1]
                    assert assistant_message['role'] == 'assistant'
                    assert 'error' in assistant_message['content'].lower()
    
    @patch('requests.post')
    def test_send_question_to_api_success(self, mock_post, chat_interface, mock_chat_response):
        """Test successful API question sending"""
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_chat_response
        mock_post.return_value = mock_response
        
        with patch('streamlit.session_state', {'auth_token': 'test_token', 'chat_context': {}}):
            result = chat_interface._send_question_to_api('test_doc', 'What is the average age?')
            
            assert result == mock_chat_response
            mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_send_question_to_api_failure(self, mock_post, chat_interface):
        """Test failed API question sending"""
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response
        
        with patch('streamlit.session_state', {'auth_token': 'test_token', 'chat_context': {}}):
            with pytest.raises(Exception, match="API request failed"):
                chat_interface._send_question_to_api('test_doc', 'What is the average age?')
    
    def test_create_bar_chart(self, chat_interface):
        """Test bar chart creation"""
        
        data = {
            'x': ['A', 'B', 'C'],
            'y': [10, 20, 15]
        }
        
        fig = chat_interface._create_bar_chart(data, 'Test Chart')
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Test Chart'
        assert len(fig.data) == 1
        assert fig.data[0].type == 'bar'
    
    def test_create_line_chart(self, chat_interface):
        """Test line chart creation"""
        
        data = {
            'x': [1, 2, 3, 4],
            'y': [10, 15, 13, 17]
        }
        
        fig = chat_interface._create_line_chart(data, 'Trend Chart')
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Trend Chart'
        assert len(fig.data) == 1
        assert fig.data[0].type == 'scatter'
        assert fig.data[0].mode == 'lines+markers'
    
    def test_create_pie_chart(self, chat_interface):
        """Test pie chart creation"""
        
        data = {
            'labels': ['Category A', 'Category B', 'Category C'],
            'values': [30, 45, 25]
        }
        
        fig = chat_interface._create_pie_chart(data, 'Distribution Chart')
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Distribution Chart'
        assert len(fig.data) == 1
        assert fig.data[0].type == 'pie'
    
    def test_create_scatter_chart(self, chat_interface):
        """Test scatter chart creation"""
        
        data = {
            'x': [1, 2, 3, 4, 5],
            'y': [2, 5, 3, 8, 7]
        }
        
        fig = chat_interface._create_scatter_chart(data, 'Correlation Chart')
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Correlation Chart'
        assert len(fig.data) == 1
        assert fig.data[0].type == 'scatter'
        assert fig.data[0].mode == 'markers'
    
    def test_render_visualizations(self, chat_interface):
        """Test visualization rendering"""
        
        visualizations = [
            {
                'chart_type': 'bar',
                'title': 'Test Bar Chart',
                'data': {'x': ['A', 'B'], 'y': [1, 2]}
            },
            {
                'chart_type': 'line',
                'title': 'Test Line Chart',
                'data': {'x': [1, 2], 'y': [3, 4]}
            }
        ]
        
        with patch('streamlit.plotly_chart') as mock_plotly:
            chat_interface._render_visualizations(visualizations)
            
            # Should render both charts
            assert mock_plotly.call_count == 2
    
    def test_render_visualizations_error_handling(self, chat_interface):
        """Test visualization rendering with errors"""
        
        # Invalid visualization data
        visualizations = [
            {
                'chart_type': 'invalid_type',
                'title': 'Invalid Chart',
                'data': {}
            }
        ]
        
        with patch('streamlit.plotly_chart', side_effect=Exception("Chart error")):
            with patch('streamlit.error') as mock_error:
                chat_interface._render_visualizations(visualizations)
                
                # Should handle error gracefully
                mock_error.assert_called_once()
    
    def test_render_data_summary(self, chat_interface):
        """Test data summary rendering"""
        
        data_summary = {
            'metrics': {
                'total_records': 100,
                'average_value': 45.6,
                'max_value': 98.7
            },
            'details': {
                'data_quality': 'high',
                'completeness': 0.95
            }
        }
        
        with patch('streamlit.expander') as mock_expander:
            with patch('streamlit.columns') as mock_columns:
                with patch('streamlit.metric') as mock_metric:
                    with patch('streamlit.json') as mock_json:
                        
                        # Mock columns for metrics
                        mock_columns.return_value = [Mock(), Mock(), Mock()]
                        
                        chat_interface._render_data_summary(data_summary)
                        
                        # Should display metrics and details
                        mock_metric.assert_called()
                        mock_json.assert_called_once()
    
    def test_render_suggested_questions(self, chat_interface):
        """Test suggested questions rendering"""
        
        suggested_questions = [
            'What is the distribution by category?',
            'Show me the top 10 values',
            'Are there any outliers?'
        ]
        
        with patch('streamlit.button', return_value=False) as mock_button:
            with patch('streamlit.session_state', {'current_document_id': 'test_doc'}):
                
                chat_interface._render_suggested_questions(suggested_questions)
                
                # Should create buttons for suggestions (limited to 3)
                assert mock_button.call_count == 3
    
    def test_render_suggested_questions_with_click(self, chat_interface):
        """Test suggested questions with button click"""
        
        suggested_questions = ['What is the average?']
        
        with patch('streamlit.button', return_value=True):
            with patch('streamlit.session_state', {'current_document_id': 'test_doc'}):
                with patch.object(chat_interface, '_handle_question') as mock_handle:
                    
                    chat_interface._render_suggested_questions(suggested_questions)
                    
                    # Should handle the question
                    mock_handle.assert_called_once_with('test_doc', 'What is the average?')
    
    @patch('streamlit.session_state', {'chat_history': []})
    def test_export_chat_history_empty(self, chat_interface):
        """Test exporting empty chat history"""
        
        with patch('streamlit.warning') as mock_warning:
            chat_interface._export_chat_history()
            
            mock_warning.assert_called_once_with("No chat history to export")
    
    @patch('streamlit.session_state', {})
    def test_export_chat_history_with_data(self, chat_interface):
        """Test exporting chat history with data"""
        
        # Setup chat history
        st.session_state.chat_history = [
            {
                'role': 'user',
                'content': 'What is the average age?',
                'timestamp': datetime.now()
            },
            {
                'role': 'assistant',
                'content': 'The average age is 32 years.',
                'visualizations': [{'chart_type': 'bar'}],
                'timestamp': datetime.now()
            }
        ]
        
        with patch('streamlit.download_button') as mock_download:
            chat_interface._export_chat_history()
            
            # Should create download button
            mock_download.assert_called_once()
            
            # Check download parameters
            call_args = mock_download.call_args
            assert 'chat_history_' in call_args[1]['file_name']
            assert call_args[1]['mime'] == 'application/json'
    
    @patch('streamlit.session_state', {})
    def test_render_chat_controls(self, chat_interface):
        """Test chat controls rendering"""
        
        with patch('streamlit.sidebar') as mock_sidebar:
            with patch('streamlit.button', return_value=False) as mock_button:
                with patch('streamlit.selectbox') as mock_selectbox:
                    with patch('streamlit.checkbox') as mock_checkbox:
                        
                        # Mock sidebar context
                        mock_sidebar.markdown = Mock()
                        mock_sidebar.button = Mock(return_value=False)
                        mock_sidebar.expander = Mock()
                        
                        chat_interface.render_chat_controls()
                        
                        # Should render sidebar controls
                        assert mock_sidebar.markdown.called
    
    @patch('streamlit.session_state', {'chat_history': [{'role': 'user', 'content': 'test'}]})
    def test_clear_chat_history(self, chat_interface):
        """Test clearing chat history"""
        
        with patch('streamlit.sidebar') as mock_sidebar:
            with patch('streamlit.experimental_rerun') as mock_rerun:
                
                # Mock clear button click
                mock_sidebar.button = Mock(return_value=True)
                
                chat_interface.render_chat_controls()
                
                # Chat history should be cleared
                assert len(st.session_state.chat_history) == 0


@pytest.mark.integration
class TestChatInterfaceIntegration:
    """Integration tests for chat interface workflow"""
    
    @pytest.fixture
    def chat_interface(self):
        """Create ConversationalChatInterface instance"""
        return ConversationalChatInterface()
    
    @patch('requests.get')
    @patch('requests.post')
    @patch('streamlit.session_state', {})
    def test_complete_chat_workflow(self, mock_post, mock_get, chat_interface):
        """Test complete chat workflow from question to response"""
        
        # Mock document context loading
        doc_response = Mock()
        doc_response.status_code = 200
        doc_response.json.return_value = {'document_id': 'test_doc', 'filename': 'test.pdf'}
        
        tables_response = Mock()
        tables_response.status_code = 200
        tables_response.json.return_value = [{'table_id': 'table_1', 'headers': ['Name', 'Age']}]
        
        mock_get.side_effect = [doc_response, tables_response]
        
        # Mock chat API response
        chat_response = Mock()
        chat_response.status_code = 200
        chat_response.json.return_value = {
            'response': 'The average age is 30 years.',
            'visualizations': [],
            'data_summary': {},
            'suggested_questions': []
        }
        mock_post.return_value = chat_response
        
        with patch('streamlit.chat_input', return_value='What is the average age?'):
            with patch('streamlit.spinner'):
                with patch('streamlit.experimental_rerun'):
                    
                    # Render interface (this would trigger question handling)
                    chat_interface.render('test_doc')
                    
                    # Manually trigger question handling for test
                    chat_interface._handle_question('test_doc', 'What is the average age?')
                    
                    # Check that conversation was recorded
                    assert len(st.session_state.chat_history) == 2
                    assert st.session_state.chat_history[0]['role'] == 'user'
                    assert st.session_state.chat_history[1]['role'] == 'assistant'


if __name__ == "__main__":
    pytest.main([__file__])