"""
Main Streamlit Application for OCR Table Analytics UI

Integrates document upload, table validation, and conversational chat interfaces.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import requests
from datetime import datetime

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.ui.components.document_upload import render_document_upload
    from src.ui.components.table_validation import render_table_validation
    from src.ui.components.chat_interface import render_chat_interface
    from src.core.logging_system import get_logger
except ImportError:
    # Fallback for when running directly
    from ui.components.document_upload import render_document_upload
    from ui.components.table_validation import render_table_validation
    from ui.components.chat_interface import render_chat_interface
    
    # Mock logger if core module not available
    class MockLogger:
        def info(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
    
    def get_logger():
        return MockLogger()


class OCRTableAnalyticsApp:
    """Main application class for OCR Table Analytics UI"""
    
    def __init__(self):
        self.logger = get_logger()
        self.api_base_url = "http://localhost:8000"
        
        # Configure Streamlit page
        st.set_page_config(
            page_title="OCR Table Analytics",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'upload'
        
        if 'selected_document' not in st.session_state:
            st.session_state.selected_document = None
        
        if 'user_authenticated' not in st.session_state:
            st.session_state.user_authenticated = False
        
        if 'auth_token' not in st.session_state:
            st.session_state.auth_token = None
    
    def run(self):
        """Run the main application"""
        
        # Render header
        self._render_header()
        
        # Check authentication
        if not st.session_state.user_authenticated:
            self._render_login()
            return
        
        # Render sidebar navigation
        self._render_sidebar()
        
        # Render main content based on current page
        self._render_main_content()
        
        # Render footer
        self._render_footer()
    
    def _render_header(self):
        """Render application header"""
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.image("https://via.placeholder.com/100x50/0066CC/FFFFFF?text=OCR", width=100)
        
        with col2:
            st.title("📊 OCR Table Analytics")
            st.markdown("*Transform documents into interactive data insights*")
        
        with col3:
            if st.session_state.user_authenticated:
                if st.button("🚪 Logout"):
                    self._logout()
    
    def _render_login(self):
        """Render login interface"""
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.subheader("🔐 Login")
            
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login", type="primary"):
                    if self._authenticate_user(username, password):
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
            
            # Demo mode
            st.markdown("---")
            if st.button("🎮 Demo Mode", help="Try the app without authentication"):
                st.session_state.user_authenticated = True
                st.session_state.auth_token = "demo_token"
                st.rerun()
    
    def _authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user credentials"""
        
        try:
            # In a real application, this would call the authentication API
            # For demo purposes, accept any non-empty credentials
            if username and password:
                st.session_state.user_authenticated = True
                st.session_state.auth_token = "authenticated_token"
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def _logout(self):
        """Logout user and clear session"""
        
        st.session_state.user_authenticated = False
        st.session_state.auth_token = None
        st.session_state.selected_document = None
        st.session_state.current_page = 'upload'
        
        # Clear other session state
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith(('upload_', 'chat_', 'table_'))]
        for key in keys_to_clear:
            del st.session_state[key]
        
        st.rerun()
    
    def _render_sidebar(self):
        """Render sidebar navigation"""
        
        st.sidebar.title("🧭 Navigation")
        
        # Page navigation
        pages = {
            'upload': '📤 Upload Documents',
            'validation': '🔍 Validate Tables',
            'chat': '💬 Chat Analysis',
            'dashboard': '📊 Dashboard'
        }
        
        for page_key, page_name in pages.items():
            if st.sidebar.button(page_name, key=f"nav_{page_key}"):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Document selector
        self._render_document_selector()
        
        st.sidebar.markdown("---")
        
        # System status
        self._render_system_status()
    
    def _render_document_selector(self):
        """Render document selector in sidebar"""
        
        st.sidebar.subheader("📄 Documents")
        
        # Get user documents
        documents = self._get_user_documents()
        
        if documents:
            doc_options = {doc['document_id']: f"{doc['filename']} ({doc['status']})" 
                          for doc in documents}
            
            selected_doc_id = st.sidebar.selectbox(
                "Select Document",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
                index=0 if not st.session_state.selected_document else 
                      list(doc_options.keys()).index(st.session_state.selected_document) 
                      if st.session_state.selected_document in doc_options else 0
            )
            
            if selected_doc_id != st.session_state.selected_document:
                st.session_state.selected_document = selected_doc_id
                st.rerun()
        
        else:
            st.sidebar.info("No documents uploaded yet")
    
    def _get_user_documents(self) -> list:
        """Get user's documents from API"""
        
        try:
            # In a real application, this would call the API to get user documents
            # For demo purposes, return mock data
            return [
                {
                    'document_id': 'doc_1',
                    'filename': 'sample_table.pdf',
                    'status': 'completed',
                    'uploaded_at': '2024-01-15 10:30:00'
                },
                {
                    'document_id': 'doc_2',
                    'filename': 'financial_report.png',
                    'status': 'processing',
                    'uploaded_at': '2024-01-15 11:15:00'
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get user documents: {str(e)}")
            return []
    
    def _render_system_status(self):
        """Render system status in sidebar"""
        
        st.sidebar.subheader("⚡ System Status")
        
        # Get system health
        health_status = self._get_system_health()
        
        if health_status:
            # Overall status
            overall_status = health_status.get('status', 'unknown')
            status_color = {
                'healthy': '🟢',
                'degraded': '🟡',
                'unhealthy': '🔴'
            }
            
            st.sidebar.markdown(f"{status_color.get(overall_status, '⚪')} **{overall_status.title()}**")
            
            # Service statuses
            services = health_status.get('services', {})
            for service, status in services.items():
                color = '🟢' if status == 'healthy' else '🔴'
                st.sidebar.text(f"{color} {service.title()}")
        
        else:
            st.sidebar.error("Unable to get system status")
    
    def _get_system_health(self) -> Optional[Dict[str, Any]]:
        """Get system health status"""
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {str(e)}")
            return None
    
    def _render_main_content(self):
        """Render main content area based on current page"""
        
        current_page = st.session_state.current_page
        
        if current_page == 'upload':
            self._render_upload_page()
        
        elif current_page == 'validation':
            self._render_validation_page()
        
        elif current_page == 'chat':
            self._render_chat_page()
        
        elif current_page == 'dashboard':
            self._render_dashboard_page()
        
        else:
            st.error(f"Unknown page: {current_page}")
    
    def _render_upload_page(self):
        """Render document upload page"""
        
        render_document_upload()
    
    def _render_validation_page(self):
        """Render table validation page"""
        
        if st.session_state.selected_document:
            render_table_validation(st.session_state.selected_document)
        else:
            st.info("👆 Please select a document from the sidebar to validate tables")
    
    def _render_chat_page(self):
        """Render chat analysis page"""
        
        if st.session_state.selected_document:
            render_chat_interface(st.session_state.selected_document)
        else:
            st.info("👆 Please select a document from the sidebar to start chatting")
    
    def _render_dashboard_page(self):
        """Render dashboard page"""
        
        if st.session_state.selected_document:
            self._render_dashboard_interface()
        else:
            st.info("👆 Please select a document from the sidebar to view dashboard")
    
    def _render_dashboard_interface(self):
        """Render dashboard interface"""
        
        st.title("📊 Interactive Dashboard")
        st.markdown("Auto-generated dashboard from your table data")
        
        document_id = st.session_state.selected_document
        
        # Dashboard generation controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("Dashboard Options")
        
        with col2:
            if st.button("🔄 Regenerate Dashboard"):
                self._generate_dashboard(document_id)
        
        with col3:
            if st.button("📤 Export Dashboard"):
                self._export_dashboard(document_id)
        
        # Dashboard content
        self._render_dashboard_content(document_id)
    
    def _generate_dashboard(self, document_id: str):
        """Generate dashboard for document"""
        
        with st.spinner("Generating dashboard..."):
            try:
                # Call dashboard generation API
                response = requests.post(
                    f"{self.api_base_url}/dashboards/generate",
                    json={'document_id': document_id},
                    headers={'Authorization': f"Bearer {st.session_state.auth_token}"}
                )
                
                if response.status_code == 200:
                    st.success("✅ Dashboard generated successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to generate dashboard: {response.text}")
                    
            except Exception as e:
                st.error(f"Dashboard generation failed: {str(e)}")
    
    def _export_dashboard(self, document_id: str):
        """Export dashboard"""
        
        st.info("Dashboard export functionality would be implemented here")
    
    def _render_dashboard_content(self, document_id: str):
        """Render dashboard content"""
        
        # Mock dashboard content for demo
        st.subheader("📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", "1,234", "12%")
        
        with col2:
            st.metric("Average Value", "45.6", "-2.3%")
        
        with col3:
            st.metric("Max Value", "98.7", "5.1%")
        
        with col4:
            st.metric("Data Quality", "94%", "1.2%")
        
        # Mock charts
        st.subheader("📊 Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample bar chart
            chart_data = pd.DataFrame({
                'Category': ['A', 'B', 'C', 'D'],
                'Value': [23, 45, 56, 78]
            })
            st.bar_chart(chart_data.set_index('Category'))
        
        with col2:
            # Sample line chart
            chart_data = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=10),
                'Value': [20, 25, 30, 28, 35, 40, 38, 42, 45, 48]
            })
            st.line_chart(chart_data.set_index('Date'))
    
    def _render_footer(self):
        """Render application footer"""
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(
                "<div style='text-align: center; color: #666;'>"
                "OCR Table Analytics v1.0.0 | "
                f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                "</div>",
                unsafe_allow_html=True
            )


def main():
    """Main application entry point"""
    
    app = OCRTableAnalyticsApp()
    app.run()


if __name__ == "__main__":
    main()