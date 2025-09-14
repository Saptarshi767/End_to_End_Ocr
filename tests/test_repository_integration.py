"""
Integration tests for repository layer with transaction management and error handling.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from unittest.mock import patch

from src.core.repository import (
    RepositoryManager, RepositoryError, RecordNotFoundError, 
    DuplicateRecordError, TransactionError
)
from src.core.models import ProcessingStatus, OCREngine, ChartType
from src.core.database import (
    Document, ExtractedTable, ProcessingLog, ConversationSession,
    ConversationMessage, DataSchema, Dashboard, Chart, User, SystemMetrics
)


class TestRepositoryTransactionManagement:
    """Test transaction management in repositories."""
    
    def test_transaction_context_manager_success(self, repo_manager):
        """Test successful transaction using context manager."""
        document_data = {
            'filename': 'test.pdf',
            'file_path': '/path/to/test.pdf',
            'file_size': 1024,
            'mime_type': 'application/pdf'
        }
        
        with repo_manager.transaction():
            document = repo_manager.documents.create(**document_data)
            assert document.id is not None
            assert document.filename == 'test.pdf'
        
        # Verify document was committed
        retrieved = repo_manager.documents.get_by_id(document.id)
        assert retrieved is not None
        assert retrieved.filename == 'test.pdf'
    
    def test_transaction_context_manager_rollback(self, repo_manager):
        """Test transaction rollback on exception."""
        document_data = {
            'filename': 'test.pdf',
            'file_path': '/path/to/test.pdf',
            'file_size': 1024,
            'mime_type': 'application/pdf'
        }
        
        with pytest.raises(TransactionError):
            with repo_manager.transaction():
                document = repo_manager.documents.create(**document_data)
                document_id = document.id
                # Force an error
                raise ValueError("Test error")
        
        # Verify document was not committed
        retrieved = repo_manager.documents.get_by_id(document_id)
        assert retrieved is None
    
    def test_bulk_process_document_transaction(self, repo_manager):
        """Test bulk document processing in single transaction."""
        document_data = {
            'filename': 'bulk_test.pdf',
            'file_path': '/path/to/bulk_test.pdf',
            'file_size': 2048,
            'mime_type': 'application/pdf'
        }
        
        tables_data = [
            {
                'table_index': 0,
                'headers': ['Name', 'Age'],
                'data': [['John', '25'], ['Jane', '30']],
                'confidence_score': 0.95
            },
            {
                'table_index': 1,
                'headers': ['Product', 'Price'],
                'data': [['Widget', '$10'], ['Gadget', '$20']],
                'confidence_score': 0.88
            }
        ]
        
        processing_logs = [
            {
                'stage': 'ocr',
                'status': 'completed',
                'processing_time_ms': 1500,
                'engine_used': OCREngine.TESSERACT
            },
            {
                'stage': 'table_extraction',
                'status': 'completed',
                'processing_time_ms': 800
            }
        ]
        
        result = repo_manager.bulk_process_document(
            document_data, tables_data, processing_logs
        )
        
        assert result['document'].filename == 'bulk_test.pdf'
        assert len(result['tables']) == 2
        assert len(result['logs']) == 2
        
        # Verify all data was persisted
        document = repo_manager.documents.get_by_id(result['document'].id)
        assert document is not None
        
        tables = repo_manager.extracted_tables.get_by_document(document.id)
        assert len(tables) == 2
        
        logs = repo_manager.processing_logs.get_by_document(document.id)
        assert len(logs) == 2
    
    def test_create_dashboard_with_charts_transaction(self, repo_manager):
        """Test dashboard creation with charts in single transaction."""
        # Create document first
        document = repo_manager.documents.create(
            filename='dashboard_test.pdf',
            file_path='/path/to/dashboard_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        dashboard_data = {
            'document_id': document.id,
            'title': 'Test Dashboard',
            'description': 'Test dashboard description'
        }
        
        charts_data = [
            {
                'chart_type': ChartType.BAR,
                'title': 'Bar Chart',
                'config': {'x_column': 'name', 'y_column': 'value'},
                'data': {'labels': ['A', 'B'], 'values': [10, 20]}
            },
            {
                'chart_type': ChartType.PIE,
                'title': 'Pie Chart',
                'config': {'value_column': 'amount'},
                'data': {'labels': ['X', 'Y'], 'values': [30, 70]}
            }
        ]
        
        result = repo_manager.create_dashboard_with_charts(
            dashboard_data, charts_data
        )
        
        assert result['dashboard'].title == 'Test Dashboard'
        assert len(result['charts']) == 2
        
        # Verify all data was persisted
        dashboard = repo_manager.dashboards.get_by_id(result['dashboard'].id)
        assert dashboard is not None
        
        charts = repo_manager.charts.get_by_dashboard(dashboard.id)
        assert len(charts) == 2


class TestRepositoryErrorHandling:
    """Test error handling in repositories."""
    
    def test_record_not_found_error(self, repo_manager):
        """Test RecordNotFoundError is raised for non-existent records."""
        non_existent_id = uuid.uuid4()
        
        with pytest.raises(RecordNotFoundError):
            repo_manager.documents.get_by_id_or_raise(non_existent_id)
        
        with pytest.raises(RecordNotFoundError):
            repo_manager.documents.update(non_existent_id, filename='new_name.pdf')
        
        with pytest.raises(RecordNotFoundError):
            repo_manager.documents.delete_by_id_or_raise(non_existent_id)
    
    def test_duplicate_record_error(self, repo_manager):
        """Test DuplicateRecordError for constraint violations."""
        # Create user with unique username
        user1 = repo_manager.users.create(
            username='testuser',
            email='test@example.com',
            password_hash='hashed_password'
        )
        
        # Try to create another user with same username
        with pytest.raises(DuplicateRecordError):
            repo_manager.users.create(
                username='testuser',  # Duplicate username
                email='test2@example.com',
                password_hash='hashed_password2'
            )
    
    def test_database_error_handling(self, repo_manager):
        """Test general database error handling."""
        # Mock a database error
        with patch.object(repo_manager.session, 'add', side_effect=Exception("Database connection lost")):
            with pytest.raises(RepositoryError):
                repo_manager.documents.create(
                    filename='test.pdf',
                    file_path='/path/to/test.pdf'
                )


class TestDataSchemaRepository:
    """Test DataSchema repository operations."""
    
    def test_create_and_get_schema(self, repo_manager):
        """Test creating and retrieving data schemas."""
        # Create document and table first
        document = repo_manager.documents.create(
            filename='schema_test.pdf',
            file_path='/path/to/schema_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        table = repo_manager.extracted_tables.create(
            document_id=document.id,
            table_index=0,
            headers=['Name', 'Age', 'Salary'],
            data=[['John', '25', '50000'], ['Jane', '30', '60000']],
            confidence_score=0.95
        )
        
        # Create schema
        columns_info = [
            {'name': 'Name', 'type': 'text', 'nullable': False},
            {'name': 'Age', 'type': 'number', 'nullable': False},
            {'name': 'Salary', 'type': 'currency', 'nullable': True}
        ]
        
        data_types = {
            'Name': 'text',
            'Age': 'number',
            'Salary': 'currency'
        }
        
        sample_data = {
            'Name': ['John', 'Jane'],
            'Age': [25, 30],
            'Salary': [50000, 60000]
        }
        
        schema = repo_manager.data_schemas.create_schema(
            table_id=table.id,
            schema_name='Employee Data',
            columns_info=columns_info,
            data_types=data_types,
            sample_data=sample_data
        )
        
        assert schema.schema_name == 'Employee Data'
        assert schema.is_active is True
        assert schema.version == 1
        assert len(schema.columns_info) == 3
        
        # Test retrieval
        retrieved_schema = repo_manager.data_schemas.get_active_schema(table.id)
        assert retrieved_schema.id == schema.id
        
        all_schemas = repo_manager.data_schemas.get_by_table(table.id)
        assert len(all_schemas) == 1
    
    def test_schema_versioning(self, repo_manager):
        """Test schema versioning functionality."""
        # Create document and table
        document = repo_manager.documents.create(
            filename='versioning_test.pdf',
            file_path='/path/to/versioning_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        table = repo_manager.extracted_tables.create(
            document_id=document.id,
            table_index=0,
            headers=['Name', 'Age'],
            data=[['John', '25']],
            confidence_score=0.95
        )
        
        # Create first schema
        schema_v1 = repo_manager.data_schemas.create_schema(
            table_id=table.id,
            schema_name='Schema V1',
            columns_info=[{'name': 'Name', 'type': 'text'}],
            data_types={'Name': 'text'}
        )
        
        assert schema_v1.version == 1
        assert schema_v1.is_active is True
        
        # Create second schema
        schema_v2 = repo_manager.data_schemas.create_schema(
            table_id=table.id,
            schema_name='Schema V2',
            columns_info=[
                {'name': 'Name', 'type': 'text'},
                {'name': 'Age', 'type': 'number'}
            ],
            data_types={'Name': 'text', 'Age': 'number'}
        )
        
        assert schema_v2.version == 2
        assert schema_v2.is_active is True
        
        # Check that v1 is no longer active
        repo_manager.session.refresh(schema_v1)
        assert schema_v1.is_active is False
        
        # Test activation of older version
        activated = repo_manager.data_schemas.activate_schema(schema_v1.id)
        assert activated.is_active is True
        
        # Check that v2 is no longer active
        repo_manager.session.refresh(schema_v2)
        assert schema_v2.is_active is False


class TestRepositoryBulkOperations:
    """Test bulk operations in repositories."""
    
    def test_bulk_create_documents(self, repo_manager):
        """Test bulk creation of documents."""
        documents_data = [
            {
                'filename': 'bulk1.pdf',
                'file_path': '/path/to/bulk1.pdf',
                'file_size': 1024,
                'mime_type': 'application/pdf'
            },
            {
                'filename': 'bulk2.pdf',
                'file_path': '/path/to/bulk2.pdf',
                'file_size': 2048,
                'mime_type': 'application/pdf'
            },
            {
                'filename': 'bulk3.pdf',
                'file_path': '/path/to/bulk3.pdf',
                'file_size': 3072,
                'mime_type': 'application/pdf'
            }
        ]
        
        created_documents = repo_manager.documents.bulk_create(documents_data)
        
        assert len(created_documents) == 3
        for i, doc in enumerate(created_documents):
            assert doc.filename == f'bulk{i+1}.pdf'
            assert doc.id is not None
        
        # Verify all documents were persisted
        repo_manager.commit()
        for doc in created_documents:
            retrieved = repo_manager.documents.get_by_id(doc.id)
            assert retrieved is not None
    
    def test_bulk_update_documents(self, repo_manager):
        """Test bulk update of documents."""
        # Create documents first
        documents = []
        for i in range(3):
            doc = repo_manager.documents.create(
                filename=f'update{i}.pdf',
                file_path=f'/path/to/update{i}.pdf',
                file_size=1024 * (i + 1),
                mime_type='application/pdf'
            )
            documents.append(doc)
        
        # Prepare bulk updates
        updates = [
            {'id': documents[0].id, 'processing_status': ProcessingStatus.PROCESSING},
            {'id': documents[1].id, 'processing_status': ProcessingStatus.COMPLETED},
            {'id': documents[2].id, 'processing_status': ProcessingStatus.FAILED}
        ]
        
        repo_manager.documents.bulk_update(updates)
        repo_manager.commit()
        
        # Verify updates
        for i, doc in enumerate(documents):
            updated_doc = repo_manager.documents.get_by_id(doc.id)
            expected_status = [ProcessingStatus.PROCESSING, ProcessingStatus.COMPLETED, ProcessingStatus.FAILED][i]
            assert updated_doc.processing_status == expected_status


class TestRepositoryCleanupOperations:
    """Test cleanup operations in repository manager."""
    
    def test_cleanup_old_data(self, repo_manager):
        """Test cleanup of old data."""
        # Create document
        document = repo_manager.documents.create(
            filename='cleanup_test.pdf',
            file_path='/path/to/cleanup_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        # Create old processing logs
        old_timestamp = datetime.utcnow() - timedelta(days=35)
        old_log = repo_manager.processing_logs.create(
            document_id=document.id,
            stage='ocr',
            status='completed',
            timestamp=old_timestamp
        )
        
        # Create recent processing log
        recent_log = repo_manager.processing_logs.create(
            document_id=document.id,
            stage='table_extraction',
            status='completed'
        )
        
        # Create old system metrics
        old_metric = repo_manager.system_metrics.create(
            metric_name='test_metric',
            metric_value=100.0,
            timestamp=old_timestamp
        )
        
        # Create recent system metric
        recent_metric = repo_manager.system_metrics.record_metric(
            'test_metric', 200.0
        )
        
        # Create old inactive conversation session
        old_session = repo_manager.conversations.create(
            document_id=document.id,
            is_active=False,
            last_activity=old_timestamp
        )
        
        # Create recent active session
        recent_session = repo_manager.conversations.create(
            document_id=document.id,
            is_active=True
        )
        
        # Run cleanup
        cleanup_counts = repo_manager.cleanup_old_data(days_to_keep=30)
        
        # Verify cleanup results
        assert cleanup_counts['processing_logs'] == 1
        assert cleanup_counts['system_metrics'] == 1
        assert cleanup_counts['conversation_sessions'] == 1
        
        # Verify old data was deleted
        assert repo_manager.processing_logs.get_by_id(old_log.id) is None
        assert repo_manager.system_metrics.get_by_id(old_metric.id) is None
        assert repo_manager.conversations.get_by_id(old_session.id) is None
        
        # Verify recent data was preserved
        assert repo_manager.processing_logs.get_by_id(recent_log.id) is not None
        assert repo_manager.system_metrics.get_by_id(recent_metric.id) is not None
        assert repo_manager.conversations.get_by_id(recent_session.id) is not None


class TestSystemMetricsRepository:
    """Test system metrics repository enhancements."""
    
    def test_performance_summary(self, repo_manager):
        """Test performance summary calculation."""
        # Record various metrics
        metrics_data = [
            ('response_time', 150.0),
            ('response_time', 200.0),
            ('response_time', 100.0),
            ('memory_usage', 75.5),
            ('memory_usage', 80.2),
            ('cpu_usage', 45.0),
            ('cpu_usage', 55.0),
            ('cpu_usage', 50.0)
        ]
        
        for metric_name, value in metrics_data:
            repo_manager.system_metrics.record_metric(metric_name, value)
        
        # Get performance summary
        summary = repo_manager.system_metrics.get_performance_summary(
            ['response_time', 'memory_usage', 'cpu_usage', 'nonexistent_metric']
        )
        
        # Verify response_time summary
        assert summary['response_time']['count'] == 3
        assert summary['response_time']['avg'] == 150.0
        assert summary['response_time']['min'] == 100.0
        assert summary['response_time']['max'] == 200.0
        
        # Verify memory_usage summary
        assert summary['memory_usage']['count'] == 2
        assert abs(summary['memory_usage']['avg'] - 77.85) < 0.01
        
        # Verify cpu_usage summary
        assert summary['cpu_usage']['count'] == 3
        assert summary['cpu_usage']['avg'] == 50.0
        
        # Verify nonexistent metric
        assert summary['nonexistent_metric']['count'] == 0
        assert summary['nonexistent_metric']['avg'] == 0


class TestRepositoryExists:
    """Test exists functionality in repositories."""
    
    def test_document_exists(self, repo_manager):
        """Test document existence check."""
        # Create document
        document = repo_manager.documents.create(
            filename='exists_test.pdf',
            file_path='/path/to/exists_test.pdf',
            file_size=1024,
            mime_type='application/pdf'
        )
        
        # Test exists
        assert repo_manager.documents.exists(document.id) is True
        
        # Test non-existent
        non_existent_id = uuid.uuid4()
        assert repo_manager.documents.exists(non_existent_id) is False
        
        # Delete and test again
        repo_manager.documents.delete(document.id)
        repo_manager.commit()
        assert repo_manager.documents.exists(document.id) is False