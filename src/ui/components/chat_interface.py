"""
Conversational Chat Interface Component

Implements chat UI with message history, context management,
quick question suggestions, and visualization embedding.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import requests
import json
from datetime import datetime
import uuid
import plotly.graph_objects as go
import plotly.express as px

try:
    from src.core.logging_system import get_logger
    from src.api.models import ChatRequest, ChatResponse
except ImportError:
    # Mock implementations for standalone usage
    class MockLogger:
        def info(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
    
    def get_logger():
        return MockLogger()
    
    # Mock API models
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional
    
    class ChatRequest(BaseModel):
        document_id: str
        question: str
        context: Optional[Dict[str, Any]] = {}
    
    class ChatResponse(BaseModel):
        response: str
        visualizations: Optional[List[Dict[str, Any]]] = []
        data_summary: Optional[Dict[str, Any]] = {}
        suggested_questions: Optional[List[str]] = []


class ConversationalChatInterface:
    """Conversational chat interface for data analysis"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.logger = get_logger()
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'chat_context' not in st.session_state:
            st.session_state.chat_context = {}
    
    def render(self, document_id: str):
        """Render the conversational chat interface"""
        
        st.title("💬 Data Analysis Chat")
        st.markdown("Ask questions about your data in natural language")
        
        # Load document context
        document_info = self._load_document_context(document_id)
        
        # Chat interface layout
        self._render_chat_header(document_info)
        self._render_quick_suggestions(document_id)
        self._render_chat_history()
        self._render_chat_input(document_id)
    
    def _load_document_context(self, document_id: str) -> Dict[str, Any]:
        """Load document context for chat"""
        
        try:
            # Load document info
            doc_response = requests.get(
                f"{self.api_base_url}/documents/{document_id}/status",
                headers={'Authorization': f"Bearer {st.session_state.get('auth_token', '')}"}
            )
            
            # Load tables info
            tables_response = requests.get(
                f"{self.api_base_url}/documents/{document_id}/tables",
                headers={'Authorization': f"Bearer {st.session_state.get('auth_token', '')}"}
            )
            
            document_info = {}
            if doc_response.status_code == 200:
                document_info = doc_response.json()
            
            tables_info = []
            if tables_response.status_code == 200:
                tables_info = tables_response.json()
            
            return {
                'document': document_info,
                'tables': tables_info,
                'schema': self._extract_schema_info(tables_info)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load document context: {str(e)}")
            return {}
    
    def _extract_schema_info(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract schema information from tables"""
        
        schema = {
            'tables': [],
            'columns': [],
            'data_types': {}
        }
        
        for table in tables:
            table_info = {
                'table_id': table['table_id'],
                'index': table['table_index'],
                'columns': table['headers'],
                'row_count': table['row_count']
            }
            schema['tables'].append(table_info)
            schema['columns'].extend(table['headers'])
        
        return schema
    
    def _render_chat_header(self, document_info: Dict[str, Any]):
        """Render chat header with document context"""
        
        if document_info:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                doc_name = document_info.get('document', {}).get('filename', 'Unknown Document')
                st.markdown(f"**Document:** {doc_name}")
            
            with col2:
                table_count = len(document_info.get('tables', []))
                st.markdown(f"**Tables:** {table_count}")
            
            with col3:
                total_rows = sum(t.get('row_count', 0) for t in document_info.get('tables', []))
                st.markdown(f"**Total Rows:** {total_rows}")
        
        st.divider()
    
    def _render_quick_suggestions(self, document_id: str):
        """Render quick question suggestions"""
        
        st.markdown("**💡 Quick Questions**")
        
        # Generate contextual suggestions
        suggestions = self._generate_question_suggestions(document_id)
        
        # Display suggestions as buttons
        cols = st.columns(min(len(suggestions), 3))
        for i, suggestion in enumerate(suggestions[:3]):
            with cols[i]:
                if st.button(suggestion['text'], key=f"suggestion_{i}"):
                    self._handle_question(document_id, suggestion['text'])
        
        # More suggestions in expander
        if len(suggestions) > 3:
            with st.expander("More Suggestions"):
                for suggestion in suggestions[3:]:
                    if st.button(suggestion['text'], key=f"more_suggestion_{suggestion['text'][:20]}"):
                        self._handle_question(document_id, suggestion['text'])
    
    def _generate_question_suggestions(self, document_id: str) -> List[Dict[str, str]]:
        """Generate contextual question suggestions"""
        
        # Basic suggestions that work for most tables
        basic_suggestions = [
            {"text": "What are the main trends in this data?", "type": "trend"},
            {"text": "Show me a summary of the key metrics", "type": "summary"},
            {"text": "What are the highest and lowest values?", "type": "extremes"},
            {"text": "Create a chart showing the distribution", "type": "distribution"},
            {"text": "Are there any outliers in the data?", "type": "outliers"},
            {"text": "Compare values across different categories", "type": "comparison"}
        ]
        
        # Add contextual suggestions based on document schema
        contextual_suggestions = []
        
        # Get schema info from session state or load it
        schema = st.session_state.chat_context.get('schema', {})
        columns = schema.get('columns', [])
        
        if columns:
            # Suggest questions based on column names
            for col in columns[:3]:  # Limit to first 3 columns
                contextual_suggestions.append({
                    "text": f"What is the average {col}?",
                    "type": "column_specific"
                })
        
        return basic_suggestions + contextual_suggestions
    
    def _render_chat_history(self):
        """Render chat message history"""
        
        st.markdown("**💬 Conversation**")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                self._render_message(message)
        
        # Auto-scroll to bottom
        if st.session_state.chat_history:
            st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)
    
    def _render_message(self, message: Dict[str, Any]):
        """Render a single chat message"""
        
        if message['role'] == 'user':
            # User message
            with st.chat_message("user"):
                st.markdown(message['content'])
                
        else:
            # Assistant message
            with st.chat_message("assistant"):
                st.markdown(message['content'])
                
                # Render visualizations if present
                if 'visualizations' in message and message['visualizations']:
                    self._render_visualizations(message['visualizations'])
                
                # Render data summary if present
                if 'data_summary' in message and message['data_summary']:
                    self._render_data_summary(message['data_summary'])
                
                # Render suggested follow-up questions
                if 'suggested_questions' in message and message['suggested_questions']:
                    self._render_suggested_questions(message['suggested_questions'])
    
    def _render_visualizations(self, visualizations: List[Dict[str, Any]]):
        """Render embedded visualizations"""
        
        st.markdown("**📊 Visualizations**")
        
        for viz in visualizations:
            try:
                chart_type = viz.get('chart_type', 'bar')
                title = viz.get('title', 'Chart')
                data = viz.get('data', {})
                
                # Create plotly chart based on type
                if chart_type == 'bar':
                    fig = self._create_bar_chart(data, title)
                elif chart_type == 'line':
                    fig = self._create_line_chart(data, title)
                elif chart_type == 'pie':
                    fig = self._create_pie_chart(data, title)
                elif chart_type == 'scatter':
                    fig = self._create_scatter_chart(data, title)
                else:
                    fig = self._create_bar_chart(data, title)  # Default to bar
                
                st.plotly_chart(fig, width='stretch')
                
            except Exception as e:
                st.error(f"Failed to render visualization: {str(e)}")
    
    def _create_bar_chart(self, data: Dict[str, Any], title: str) -> go.Figure:
        """Create bar chart"""
        
        x_values = data.get('x', [])
        y_values = data.get('y', [])
        
        fig = go.Figure(data=[go.Bar(x=x_values, y=y_values)])
        fig.update_layout(title=title, xaxis_title="Category", yaxis_title="Value")
        
        return fig
    
    def _create_line_chart(self, data: Dict[str, Any], title: str) -> go.Figure:
        """Create line chart"""
        
        x_values = data.get('x', [])
        y_values = data.get('y', [])
        
        fig = go.Figure(data=[go.Scatter(x=x_values, y=y_values, mode='lines+markers')])
        fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
        
        return fig
    
    def _create_pie_chart(self, data: Dict[str, Any], title: str) -> go.Figure:
        """Create pie chart"""
        
        labels = data.get('labels', [])
        values = data.get('values', [])
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title=title)
        
        return fig
    
    def _create_scatter_chart(self, data: Dict[str, Any], title: str) -> go.Figure:
        """Create scatter chart"""
        
        x_values = data.get('x', [])
        y_values = data.get('y', [])
        
        fig = go.Figure(data=[go.Scatter(x=x_values, y=y_values, mode='markers')])
        fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
        
        return fig
    
    def _render_data_summary(self, data_summary: Dict[str, Any]):
        """Render data summary information"""
        
        with st.expander("📈 Data Summary", expanded=False):
            
            # Display key metrics
            if 'metrics' in data_summary:
                metrics = data_summary['metrics']
                cols = st.columns(len(metrics))
                
                for i, (key, value) in enumerate(metrics.items()):
                    with cols[i]:
                        st.metric(key.title(), value)
            
            # Display additional info
            if 'details' in data_summary:
                st.json(data_summary['details'])
    
    def _render_suggested_questions(self, suggested_questions: List[str]):
        """Render suggested follow-up questions"""
        
        if suggested_questions:
            st.markdown("**🤔 Follow-up Questions**")
            
            for question in suggested_questions[:3]:  # Limit to 3 suggestions
                if st.button(f"💭 {question}", key=f"followup_{hash(question)}"):
                    # Get document_id from current context
                    document_id = st.session_state.get('current_document_id')
                    if document_id:
                        self._handle_question(document_id, question)
    
    def _render_chat_input(self, document_id: str):
        """Render chat input area"""
        
        # Store current document ID for follow-up questions
        st.session_state.current_document_id = document_id
        
        # Chat input
        user_input = st.chat_input("Ask a question about your data...")
        
        if user_input:
            self._handle_question(document_id, user_input)
    
    def _handle_question(self, document_id: str, question: str):
        """Handle user question and get AI response"""
        
        # Add user message to history
        user_message = {
            'role': 'user',
            'content': question,
            'timestamp': datetime.now()
        }
        st.session_state.chat_history.append(user_message)
        
        # Show thinking indicator
        with st.spinner("🤔 Analyzing your question..."):
            try:
                # Send question to API
                response = self._send_question_to_api(document_id, question)
                
                # Add assistant response to history
                assistant_message = {
                    'role': 'assistant',
                    'content': response.get('response', 'I apologize, but I could not process your question.'),
                    'visualizations': response.get('visualizations', []),
                    'data_summary': response.get('data_summary', {}),
                    'suggested_questions': response.get('suggested_questions', []),
                    'timestamp': datetime.now()
                }
                st.session_state.chat_history.append(assistant_message)
                
            except Exception as e:
                self.logger.error(f"Failed to process question: {str(e)}")
                
                # Add error message
                error_message = {
                    'role': 'assistant',
                    'content': f"I'm sorry, I encountered an error while processing your question: {str(e)}",
                    'timestamp': datetime.now()
                }
                st.session_state.chat_history.append(error_message)
        
        # Rerun to update chat display
        st.rerun()
    
    def _send_question_to_api(self, document_id: str, question: str) -> Dict[str, Any]:
        """Send question to conversational AI API"""
        
        # Prepare request
        chat_request = {
            'document_id': document_id,
            'question': question,
            'context': st.session_state.chat_context
        }
        
        # Send to API
        response = requests.post(
            f"{self.api_base_url}/chat/ask",
            json=chat_request,
            headers={'Authorization': f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed: {response.text}")
    
    def render_chat_controls(self):
        """Render chat control buttons"""
        
        st.sidebar.markdown("### 💬 Chat Controls")
        
        # Clear chat history
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Export chat history
        if st.sidebar.button("📤 Export Chat"):
            self._export_chat_history()
        
        # Chat settings
        with st.sidebar.expander("⚙️ Chat Settings"):
            
            # Response length
            response_length = st.selectbox(
                "Response Length",
                ["concise", "detailed", "comprehensive"],
                index=1
            )
            
            # Include visualizations
            include_viz = st.checkbox(
                "Include Visualizations",
                value=True
            )
            
            # Include suggestions
            include_suggestions = st.checkbox(
                "Include Follow-up Suggestions",
                value=True
            )
            
            # Update context
            st.session_state.chat_context.update({
                'response_length': response_length,
                'include_visualizations': include_viz,
                'include_suggestions': include_suggestions
            })
    
    def _export_chat_history(self):
        """Export chat history to file"""
        
        if not st.session_state.chat_history:
            st.warning("No chat history to export")
            return
        
        # Prepare export data
        export_data = []
        for message in st.session_state.chat_history:
            export_item = {
                'role': message['role'],
                'content': message['content'],
                'timestamp': message['timestamp'].isoformat()
            }
            
            if 'visualizations' in message:
                export_item['visualizations'] = len(message['visualizations'])
            
            export_data.append(export_item)
        
        # Create JSON export
        json_data = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Chat History",
            data=json_data,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def render_chat_interface(document_id: str):
    """Render conversational chat interface"""
    
    chat_interface = ConversationalChatInterface()
    
    # Main chat interface
    chat_interface.render(document_id)
    
    # Sidebar controls
    chat_interface.render_chat_controls()


if __name__ == "__main__":
    # Example usage
    if 'selected_document' in st.session_state:
        render_chat_interface(st.session_state.selected_document)
    else:
        st.info("Please select a document to start chatting about your data")