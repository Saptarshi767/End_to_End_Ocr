"""
Integration tests for service layer demonstrating proper repository usage.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

from src.core.services import (
    DocumentService, DashboardService, ConversationService, SystemMaintenanceService
)
from src.core.models import ProcessingStatus, OCREngine, ChartType


class TestDocumentService:
    """Test document service operations."""
    
    def test_process_document_upload_success(self, repo_manager):
        """Test successful document upload processing."""
        service = DocumentService(repo_manager)
        
        result = service.process_document_upload(
            filename='test_upload.pdf',
            file_path='/path/to/test_upload.pdf',
            file_size=2048,
            mime_type='application/pdf'
        )
        
        assert result['success'] is True
        assert 'document_id' in result
        
        # Verify document was created
        document_id = uuid.UUID(result['document_id'])
        document = repo_manager.documents.get_by_id(document_id)
        assert document is not None
        assert document.filename == 'test_upload.pdf'
        assert document.processing_status == ProcessingStatus.PENDING
        
        # Verify processing log was created
        logs = repo_manager.processing_logs.get_by_document(document_id)
        assert len(logs) == 1
        assert logs[0].stage == 'upload'
        assert logs[0].status == 'completed'
        
        # Verify metrics were recorded
        metrics = repo_manager.system_metrics.get_metrics_by_name('document_uploaded')
        assert len(metrics) > 0
    
    def test_complete_document_processing_success(self, repo_manager):
        """Test successful document processing completion."""
        service = DocumentService(repo_manager)
        
        # Create initial document
        upload_result = service.process_document_upload(
            filename='processing_test.pdf',
            file_path='/path/to/processing_test.pdf',
            file_size=3072,
            mime_type='application/pdf'
        )
        
        document_id = uuid.UUID(upload_result['document_id'])
        
        # Complete processing
        tables_data = [
            {
                'table_index': 0,
                'headers': ['Name', 'Age', 'Salary'],
                'data': [['John', '25', '50000'], ['Jane', '30', '60000']],
                'confidence_score': 0.95
            }
        ]
        
        processing_logs = [
            {
                'stage': 'ocr',
                'status': 'completed',
                'processing_time_ms': 2000,
                'engine_used': OCREngine.TESSERACT,
                'confidence_score': 0.95
            },
            {
                'stage': 'table_extraction',
                'status': 'completed',
                'processing_time_ms': 1500
            }
        ]
        
        result = service.complete_document_processing(
            document_id=document_id,
            tables_data=tables_data,
            processing_logs=processing_logs,
            total_processing_time=3500
        )
        
        assert result['success'] is True
        assert result['document_id'] == str(document_id)
        
        # Verify document status was updated
        document = repo_manager.documents.get_by_id(document_id)
        assert document.processing_status == ProcessingStatus.COMPLETED
        
        # Verify tables were created
        tables = repo_manager.extracted_tables.get_by_document(document_id)
        assert len(tables) == 1
        assert tables[0].headers == ['Name', 'Age', 'Salary']
        
        # Verify all processing logs were created
        all_logs = repo_manager.processing_logs.get_by_document(document_id)
        assert len(all_logs) == 3  # upload + 2 processing logs
    
    def test_get_document_processing_status(self, repo_manager):
        """Test getting document processing status."""
        service = DocumentService(repo_manager)
        
        # Create document
        upload_result = service.process_document_upload(
            filename='status_test.pdf',
            file_path='/path/to/status_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        document_id = uuid.UUID(upload_result['document_id'])
        
        # Get document with tables
        result = service.get_document_with_tables(document_id)
        
        assert result['success'] is True
        assert result['document']['id'] == str(document_id)
        assert result['document']['filename'] == 'status_test.pdf'
        assert 'tables' in result
        assert 'processing_logs' in result


class TestDashboardService:
    """Test dashboard service operations."""
    
    def test_create_dashboard_success(self, repo_manager):
        """Test successful dashboard creation."""
        service = DashboardService(repo_manager)
        
        # Create document first
        document = repo_manager.documents.create(
            filename='dashboard_test.pdf',
            file_path='/path/to/dashboard_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        # Create table
        table = repo_manager.extracted_tables.create(
            document_id=document.id,
            table_index=0,
            headers=['Product', 'Sales', 'Region'],
            data=[['Widget', '100', 'North'], ['Gadget', '150', 'South']],
            confidence_score=0.92
        )
        
        result = service.create_dashboard_from_table(
            table_id=table.id,
            dashboard_title='Sales Dashboard'
        )
        
        assert result['success'] is True
        assert 'dashboard_id' in result
        
        # Verify dashboard was created
        dashboard_id = uuid.UUID(result['dashboard_id'])
        dashboard = repo_manager.dashboards.get_by_id(dashboard_id)
        assert dashboard is not None
        assert dashboard.title == 'Sales Dashboard'
        
        # Verify charts were created
        charts = repo_manager.charts.get_by_dashboard(dashboard_id)
        assert len(charts) > 0


class TestConversationService:
    """Test conversation service operations."""
    
    def test_start_conversation_success(self, repo_manager):
        """Test successful conversation start."""
        service = ConversationService(repo_manager)
        
        # Create document first
        document = repo_manager.documents.create(
            filename='conversation_test.pdf',
            file_path='/path/to/conversation_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        result = service.start_conversation(
            document_id=document.id
        )
        
        assert result['success'] is True
        assert 'session_id' in result
        
        # Verify session was created
        session_id = uuid.UUID(result['session_id'])
        session = repo_manager.conversations.get_by_id(session_id)
        assert session is not None
        assert session.document_id == document.id
        assert session.is_active is True
    
    def test_add_message_success(self, repo_manager):
        """Test successful message addition."""
        service = ConversationService(repo_manager)
        
        # Create document and session
        document = repo_manager.documents.create(
            filename='message_test.pdf',
            file_path='/path/to/message_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        session = repo_manager.conversations.create(
            document_id=document.id,
            session_name='Message Test Session'
        )
        
        result = service.add_user_message(
            session_id=session.id,
            message='What is the total sales?'
        )
        
        assert result['success'] is True
        assert 'message_id' in result
        
        # Verify message was created
        message_id = uuid.UUID(result['message_id'])
        message = repo_manager.conversation_messages.get_by_id(message_id)
        assert message is not None
        assert message.session_id == session.id
        assert message.message_type == 'user'
        assert message.content == 'What is the total sales?'


class TestSystemMaintenanceService:
    """Test system maintenance service operations."""
    
    def test_cleanup_old_data_success(self, repo_manager):
        """Test successful old data cleanup."""
        service = SystemMaintenanceService(repo_manager)
        
        # Create some old data
        document = repo_manager.documents.create(
            filename='cleanup_test.pdf',
            file_path='/path/to/cleanup_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        # Create old processing log
        old_timestamp = datetime.utcnow() - timedelta(days=35)
        repo_manager.processing_logs.create(
            document_id=document.id,
            stage='ocr',
            status='completed',
            timestamp=old_timestamp
        )
        
        # Create old system metric
        repo_manager.system_metrics.create(
            metric_name='old_metric',
            metric_value=100.0,
            timestamp=old_timestamp
        )
        
        result = service.perform_maintenance(days_to_keep=30)
        
        assert result['success'] is True
        assert 'cleanup_counts' in result
        assert result['cleanup_counts']['processing_logs'] >= 0
        assert result['cleanup_counts']['system_metrics'] >= 0
    
    def test_get_system_health_success(self, repo_manager):
        """Test successful system health check."""
        service = SystemMaintenanceService(repo_manager)
        
        # Record some metrics
        repo_manager.system_metrics.record_metric('cpu_usage', 45.0)
        repo_manager.system_metrics.record_metric('memory_usage', 70.5)
        repo_manager.system_metrics.record_metric('response_time', 150.0)
        
        result = service.get_system_health()
        
        assert result['success'] is True
        assert 'system_health' in result
        assert 'performance_metrics' in result['system_health']
        assert 'cpu_usage' in result['system_health']['performance_metrics']
        assert 'memory_usage' in result['system_health']['performance_metrics']
        assert 'response_time' in result['system_health']['performance_metrics']
                