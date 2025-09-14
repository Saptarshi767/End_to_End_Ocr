"""
Tests for database models and operations.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from src.core.database import Base, DatabaseManager, get_database_url
from src.core.database import (
    Document, ExtractedTable, ProcessingLog, ConversationSession,
    ConversationMessage, DataSchema, Dashboard, Chart, User, SystemMetrics
)
from src.core.repository import RepositoryManager
from src.core.models import ProcessingStatus, OCREngine, ChartType
from src.core.config import SystemConfig, DatabaseConfig


@pytest.fixture
def test_db_config():
    """Test database configuration."""
    config = SystemConfig()
    config.database = DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_ocr_analytics",
        username="postgres",
        password="test_password"
    )
    return config


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    from sqlalchemy import event
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Enable foreign key constraints for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal


@pytest.fixture
def db_session(in_memory_db):
    """Create database session for testing."""
    session = in_memory_db()
    yield session
    session.close()


@pytest.fixture
def repo_manager(db_session):
    """Create repository manager for testing."""
    return RepositoryManager(db_session)


class TestDatabaseModels:
    """Test database model creation and relationships."""
    
    def test_document_creation(self, db_session):
        """Test document model creation."""
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_size=1024,
            mime_type="application/pdf",
            processing_status=ProcessingStatus.PENDING
        )
        
        db_session.add(document)
        db_session.commit()
        
        assert document.id is not None
        assert document.filename == "test.pdf"
        assert document.processing_status == ProcessingStatus.PENDING
        assert document.upload_timestamp is not None
    
    def test_extracted_table_creation(self, db_session):
        """Test extracted table model creation."""
        # Create document first
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create extracted table
        table = ExtractedTable(
            document_id=document.id,
            table_index=0,
            headers=["Name", "Age", "City"],
            data=[["John", "25", "NYC"], ["Jane", "30", "LA"]],
            confidence_score=0.95,
            row_count=2,
            column_count=3
        )
        
        db_session.add(table)
        db_session.commit()
        
        assert table.id is not None
        assert table.document_id == document.id
        assert table.headers == ["Name", "Age", "City"]
        assert table.confidence_score == 0.95
        assert table.row_count == 2
    
    def test_processing_log_creation(self, db_session):
        """Test processing log model creation."""
        # Create document first
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create processing log
        log = ProcessingLog(
            document_id=document.id,
            stage="ocr_extraction",
            status="completed",
            processing_time_ms=1500,
            engine_used=OCREngine.TESSERACT,
            confidence_score=0.88
        )
        
        db_session.add(log)
        db_session.commit()
        
        assert log.id is not None
        assert log.document_id == document.id
        assert log.stage == "ocr_extraction"
        assert log.engine_used == OCREngine.TESSERACT
    
    def test_conversation_session_creation(self, db_session):
        """Test conversation session model creation."""
        # Create document first
        document = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf"
        )
        db_session.add(document)
        db_session.commit()
        
        # Create conversation session
        session = ConversationSession(
            document_id=document.id,
            session_name="Analysis Session",
            is_active=True
        )
        
        db_session.add(session)
        db_session.commit()
        
        assert session.id is not None
        assert session.document_id == document.id
        assert session.session_name == "Analysis Session"
        assert session.is_active is True
    
    def test_conversation_message_creation(self, db_session):
        """Test conversation message model creation."""
        # Create document and session first
        document = Document(filename="test.pdf", file_path="/tmp/test.pdf")
        db_session.add(document)
        db_session.commit()
        
        conv_session = ConversationSession(
            document_id=document.id,
            session_name="Test Session"
        )
        db_session.add(conv_session)
        db_session.commit()
        
        # Create message
        message = ConversationMessage(
            session_id=conv_session.id,
            message_type="user",
            content="What is the average age?",
            confidence_score=0.92
        )
        
        db_session.add(message)
        db_session.commit()
        
        assert message.id is not None
        assert message.session_id == conv_session.id
        assert message.message_type == "user"
        assert message.content == "What is the average age?"
    
    def test_dashboard_creation(self, db_session):
        """Test dashboard model creation."""
        # Create document first
        document = Document(filename="test.pdf", file_path="/tmp/test.pdf")
        db_session.add(document)
        db_session.commit()
        
        # Create dashboard
        dashboard = Dashboard(
            document_id=document.id,
            title="Sales Dashboard",
            description="Dashboard for sales data analysis",
            is_public=False,
            share_token="abc123"
        )
        
        db_session.add(dashboard)
        db_session.commit()
        
        assert dashboard.id is not None
        assert dashboard.document_id == document.id
        assert dashboard.title == "Sales Dashboard"
        assert dashboard.share_token == "abc123"
    
    def test_chart_creation(self, db_session):
        """Test chart model creation."""
        # Create document and dashboard first
        document = Document(filename="test.pdf", file_path="/tmp/test.pdf")
        db_session.add(document)
        db_session.commit()
        
        dashboard = Dashboard(
            document_id=document.id,
            title="Test Dashboard"
        )
        db_session.add(dashboard)
        db_session.commit()
        
        # Create chart
        chart = Chart(
            dashboard_id=dashboard.id,
            chart_type=ChartType.BAR,
            title="Age Distribution",
            config={"x_column": "age", "y_column": "count"},
            order_index=1,
            is_visible=True
        )
        
        db_session.add(chart)
        db_session.commit()
        
        assert chart.id is not None
        assert chart.dashboard_id == dashboard.id
        assert chart.chart_type == ChartType.BAR
        assert chart.title == "Age Distribution"
    
    def test_user_creation(self, db_session):
        """Test user model creation."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            is_active=True,
            is_admin=False
        )
        
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
    
    def test_system_metrics_creation(self, db_session):
        """Test system metrics model creation."""
        metric = SystemMetrics(
            metric_name="processing_time",
            metric_value=1250.5,
            metric_unit="ms",
            category="performance",
            tags={"engine": "tesseract", "document_type": "pdf"}
        )
        
        db_session.add(metric)
        db_session.commit()
        
        assert metric.id is not None
        assert metric.metric_name == "processing_time"
        assert metric.metric_value == 1250.5
        assert metric.category == "performance"


class TestDatabaseConstraints:
    """Test database constraints and relationships."""
    
    def test_unique_username_constraint(self, db_session):
        """Test unique username constraint."""
        user1 = User(username="testuser", email="test1@example.com")
        user2 = User(username="testuser", email="test2@example.com")
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_unique_email_constraint(self, db_session):
        """Test unique email constraint."""
        user1 = User(username="user1", email="test@example.com")
        user2 = User(username="user2", email="test@example.com")
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_foreign_key_constraint(self, db_session):
        """Test foreign key constraints."""
        # Try to create extracted table without document
        table = ExtractedTable(
            document_id=uuid.uuid4(),  # Non-existent document ID
            table_index=0,
            headers=["Name", "Age"]
        )
        
        db_session.add(table)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_cascade_delete(self, db_session):
        """Test cascade delete relationships."""
        # Create document with related records
        document = Document(filename="test.pdf", file_path="/tmp/test.pdf")
        db_session.add(document)
        db_session.commit()
        
        # Create related records
        table = ExtractedTable(
            document_id=document.id,
            table_index=0,
            headers=["Name"]
        )
        log = ProcessingLog(
            document_id=document.id,
            stage="test",
            status="completed"
        )
        
        db_session.add_all([table, log])
        db_session.commit()
        
        # Delete document should cascade
        db_session.delete(document)
        db_session.commit()
        
        # Related records should be deleted
        assert db_session.query(ExtractedTable).filter(
            ExtractedTable.document_id == document.id
        ).count() == 0
        assert db_session.query(ProcessingLog).filter(
            ProcessingLog.document_id == document.id
        ).count() == 0


class TestRepositoryOperations:
    """Test repository CRUD operations."""
    
    def test_document_repository_operations(self, repo_manager):
        """Test document repository operations."""
        # Create document
        document = repo_manager.documents.create(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_size=1024,
            mime_type="application/pdf"
        )
        
        assert document.id is not None
        assert document.filename == "test.pdf"
        
        # Get by ID
        retrieved = repo_manager.documents.get_by_id(document.id)
        assert retrieved is not None
        assert retrieved.filename == "test.pdf"
        
        # Get by filename
        by_filename = repo_manager.documents.get_by_filename("test.pdf")
        assert by_filename is not None
        assert by_filename.id == document.id
        
        # Update status
        updated = repo_manager.documents.update_status(
            document.id, ProcessingStatus.COMPLETED
        )
        assert updated.processing_status == ProcessingStatus.COMPLETED
        
        # Get by status
        completed_docs = repo_manager.documents.get_by_status(ProcessingStatus.COMPLETED)
        assert len(completed_docs) == 1
        assert completed_docs[0].id == document.id
    
    def test_extracted_table_repository_operations(self, repo_manager):
        """Test extracted table repository operations."""
        # Create document first
        document = repo_manager.documents.create(
            filename="test.pdf",
            file_path="/tmp/test.pdf"
        )
        
        # Create table
        table = repo_manager.extracted_tables.create(
            document_id=document.id,
            table_index=0,
            headers=["Name", "Age"],
            data=[["John", "25"], ["Jane", "30"]],
            confidence_score=0.95
        )
        
        assert table.id is not None
        
        # Get by document
        tables = repo_manager.extracted_tables.get_by_document(document.id)
        assert len(tables) == 1
        assert tables[0].id == table.id
        
        # Get by confidence threshold
        high_confidence = repo_manager.extracted_tables.get_by_confidence_threshold(0.9)
        assert len(high_confidence) == 1
        
        low_confidence = repo_manager.extracted_tables.get_by_confidence_threshold(0.99)
        assert len(low_confidence) == 0
        
        # Update confidence
        updated = repo_manager.extracted_tables.update_confidence(table.id, 0.88)
        assert updated.confidence_score == 0.88
    
    def test_processing_log_repository_operations(self, repo_manager):
        """Test processing log repository operations."""
        # Create document first
        document = repo_manager.documents.create(
            filename="test.pdf",
            file_path="/tmp/test.pdf"
        )
        
        # Log processing step
        log = repo_manager.processing_logs.log_processing_step(
            document_id=document.id,
            stage="ocr_extraction",
            status="completed",
            processing_time_ms=1500,
            engine_used=OCREngine.TESSERACT,
            confidence_score=0.88
        )
        
        assert log.id is not None
        assert log.stage == "ocr_extraction"
        
        # Get by document
        logs = repo_manager.processing_logs.get_by_document(document.id)
        assert len(logs) == 1
        assert logs[0].id == log.id
        
        # Get by stage
        stage_logs = repo_manager.processing_logs.get_by_stage("ocr_extraction")
        assert len(stage_logs) == 1
        
        # Log error
        error_log = repo_manager.processing_logs.log_processing_step(
            document_id=document.id,
            stage="table_extraction",
            status="error",
            error_message="Failed to extract tables"
        )
        
        # Get errors
        errors = repo_manager.processing_logs.get_errors(document.id)
        assert len(errors) == 1
        assert errors[0].id == error_log.id
    
    def test_conversation_repository_operations(self, repo_manager):
        """Test conversation repository operations."""
        # Create document first
        document = repo_manager.documents.create(
            filename="test.pdf",
            file_path="/tmp/test.pdf"
        )
        
        # Create conversation session
        session = repo_manager.conversations.create(
            document_id=document.id,
            session_name="Test Session",
            is_active=True
        )
        
        assert session.id is not None
        
        # Add messages
        user_msg = repo_manager.conversation_messages.add_message(
            session_id=session.id,
            message_type="user",
            content="What is the average age?"
        )
        
        assistant_msg = repo_manager.conversation_messages.add_message(
            session_id=session.id,
            message_type="assistant",
            content="The average age is 27.5 years.",
            confidence_score=0.92
        )
        
        # Get messages
        messages = repo_manager.conversation_messages.get_by_session(session.id)
        assert len(messages) == 2
        assert messages[0].message_type == "user"
        assert messages[1].message_type == "assistant"
        
        # Update activity (add small delay to ensure timestamp difference)
        import time
        time.sleep(0.01)
        updated_session = repo_manager.conversations.update_activity(session.id)
        assert updated_session.last_activity >= session.last_activity
    
    def test_dashboard_repository_operations(self, repo_manager):
        """Test dashboard repository operations."""
        # Create document first
        document = repo_manager.documents.create(
            filename="test.pdf",
            file_path="/tmp/test.pdf"
        )
        
        # Create dashboard
        dashboard = repo_manager.dashboards.create(
            document_id=document.id,
            title="Sales Dashboard",
            is_public=True,
            share_token="abc123"
        )
        
        # Create charts
        chart1 = repo_manager.charts.create(
            dashboard_id=dashboard.id,
            chart_type=ChartType.BAR,
            title="Age Distribution",
            order_index=1
        )
        
        chart2 = repo_manager.charts.create(
            dashboard_id=dashboard.id,
            chart_type=ChartType.PIE,
            title="City Distribution",
            order_index=2,
            is_visible=False
        )
        
        # Get charts
        all_charts = repo_manager.charts.get_by_dashboard(dashboard.id)
        assert len(all_charts) == 2
        
        visible_charts = repo_manager.charts.get_visible_charts(dashboard.id)
        assert len(visible_charts) == 1
        assert visible_charts[0].id == chart1.id
        
        # Get by share token
        by_token = repo_manager.dashboards.get_by_share_token("abc123")
        assert by_token is not None
        assert by_token.id == dashboard.id
    
    def test_system_metrics_repository_operations(self, repo_manager):
        """Test system metrics repository operations."""
        # Record metrics
        metric1 = repo_manager.system_metrics.record_metric(
            metric_name="processing_time",
            metric_value=1250.5,
            metric_unit="ms",
            category="performance"
        )
        
        metric2 = repo_manager.system_metrics.record_metric(
            metric_name="accuracy_score",
            metric_value=0.95,
            category="quality"
        )
        
        # Get by name
        processing_metrics = repo_manager.system_metrics.get_metrics_by_name("processing_time")
        assert len(processing_metrics) == 1
        assert processing_metrics[0].metric_value == 1250.5
        
        # Get by category
        performance_metrics = repo_manager.system_metrics.get_metrics_by_category("performance")
        assert len(performance_metrics) == 1
        
        # Get recent metrics
        recent_metrics = repo_manager.system_metrics.get_recent_metrics(24)
        assert len(recent_metrics) == 2


class TestDatabaseManager:
    """Test database manager functionality."""
    
    def test_database_url_generation(self, test_db_config):
        """Test database URL generation."""
        url = get_database_url(test_db_config)
        expected = "postgresql://postgres:test_password@localhost:5432/test_ocr_analytics"
        assert url == expected
    
    def test_database_url_without_password(self, test_db_config):
        """Test database URL generation without password."""
        test_db_config.database.password = None
        url = get_database_url(test_db_config)
        expected = "postgresql://postgres@localhost:5432/test_ocr_analytics"
        assert url == expected
    
    def test_database_manager_initialization(self):
        """Test database manager initialization."""
        db_manager = DatabaseManager("sqlite:///:memory:")
        assert db_manager.engine is not None
        assert db_manager.SessionLocal is not None
        
        # Test session creation
        session = db_manager.get_session()
        assert session is not None
        
        db_manager.close_session(session)


if __name__ == "__main__":
    pytest.main([__file__])