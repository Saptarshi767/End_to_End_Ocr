"""
Database models and ORM configuration for the OCR Table Analytics system.
"""

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean, 
    ForeignKey, JSON, BigInteger, Enum as SQLEnum, Index
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import create_engine
from datetime import datetime
import uuid
from typing import Optional, Dict, Any

from .models import ProcessingStatus, OCREngine, DataType, ChartType

Base = declarative_base()


class Document(Base):
    """Document table for storing uploaded document metadata."""
    __tablename__ = 'documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger)
    mime_type = Column(String(100))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.PENDING)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Relationships
    extracted_tables = relationship("ExtractedTable", back_populates="document", cascade="all, delete-orphan")
    processing_logs = relationship("ProcessingLog", back_populates="document", cascade="all, delete-orphan")
    conversation_sessions = relationship("ConversationSession", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_status', 'processing_status'),
        Index('idx_documents_user', 'user_id'),
        Index('idx_documents_upload_time', 'upload_timestamp'),
    )


class ExtractedTable(Base):
    """Extracted tables from documents."""
    __tablename__ = 'extracted_tables'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    table_index = Column(Integer, nullable=False)
    headers = Column(JSON)
    data = Column(JSON)
    confidence_score = Column(Float)
    extraction_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Additional fields for table structure
    row_count = Column(Integer)
    column_count = Column(Integer)
    table_region = Column(JSON)  # Store bounding box and page info
    
    # Relationships
    document = relationship("Document", back_populates="extracted_tables")
    
    # Indexes
    __table_args__ = (
        Index('idx_extracted_tables_document', 'document_id'),
        Index('idx_extracted_tables_confidence', 'confidence_score'),
        Index('idx_extracted_tables_created', 'created_at'),
    )


class ProcessingLog(Base):
    """Processing logs for tracking document processing stages."""
    __tablename__ = 'processing_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    stage = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    error_message = Column(Text)
    processing_time_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Additional fields for detailed logging
    engine_used = Column(SQLEnum(OCREngine))
    confidence_score = Column(Float)
    log_metadata = Column(JSON)
    
    # Relationships
    document = relationship("Document", back_populates="processing_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_processing_logs_document', 'document_id'),
        Index('idx_processing_logs_stage', 'stage'),
        Index('idx_processing_logs_timestamp', 'timestamp'),
        Index('idx_processing_logs_status', 'status'),
    )


class ConversationSession(Base):
    """User conversation sessions for chat interface."""
    __tablename__ = 'conversation_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Session metadata
    session_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    
    # Relationships
    document = relationship("Document", back_populates="conversation_sessions")
    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_conversation_sessions_document', 'document_id'),
        Index('idx_conversation_sessions_user', 'user_id'),
        Index('idx_conversation_sessions_activity', 'last_activity'),
    )


class ConversationMessage(Base):
    """Individual messages in conversation sessions."""
    __tablename__ = 'conversation_messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('conversation_sessions.id'), nullable=False)
    message_type = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    query_executed = Column(Text)
    response_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Additional fields for message metadata
    confidence_score = Column(Float)
    processing_time_ms = Column(Integer)
    
    # Relationships
    session = relationship("ConversationSession", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('idx_conversation_messages_session', 'session_id'),
        Index('idx_conversation_messages_timestamp', 'timestamp'),
        Index('idx_conversation_messages_type', 'message_type'),
    )


class DataSchema(Base):
    """Data schemas for extracted tables."""
    __tablename__ = 'data_schemas'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    table_id = Column(UUID(as_uuid=True), ForeignKey('extracted_tables.id'), nullable=False)
    schema_name = Column(String(255))
    columns_info = Column(JSON)  # Store column information
    data_types = Column(JSON)    # Store data type mappings
    sample_data = Column(JSON)   # Store sample data for each column
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Schema metadata
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_data_schemas_table', 'table_id'),
        Index('idx_data_schemas_active', 'is_active'),
        Index('idx_data_schemas_created', 'created_at'),
    )


class Dashboard(Base):
    """Dashboard configurations."""
    __tablename__ = 'dashboards'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    layout_config = Column(JSON)
    filters_config = Column(JSON)
    kpis_config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Dashboard metadata
    is_public = Column(Boolean, default=False)
    share_token = Column(String(255), unique=True)
    user_id = Column(UUID(as_uuid=True))
    
    # Relationships
    charts = relationship("Chart", back_populates="dashboard", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_dashboards_document', 'document_id'),
        Index('idx_dashboards_user', 'user_id'),
        Index('idx_dashboards_share_token', 'share_token'),
        Index('idx_dashboards_created', 'created_at'),
    )


class Chart(Base):
    """Chart configurations for dashboards."""
    __tablename__ = 'charts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dashboard_id = Column(UUID(as_uuid=True), ForeignKey('dashboards.id'), nullable=False)
    chart_type = Column(SQLEnum(ChartType), nullable=False)
    title = Column(String(255), nullable=False)
    config = Column(JSON)  # Chart configuration (columns, aggregation, etc.)
    data = Column(JSON)    # Chart data
    position = Column(JSON)  # Position in dashboard grid
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Chart metadata
    is_visible = Column(Boolean, default=True)
    order_index = Column(Integer, default=0)
    
    # Relationships
    dashboard = relationship("Dashboard", back_populates="charts")
    
    # Indexes
    __table_args__ = (
        Index('idx_charts_dashboard', 'dashboard_id'),
        Index('idx_charts_type', 'chart_type'),
        Index('idx_charts_order', 'order_index'),
    )


class User(Base):
    """User accounts (optional - for future authentication)."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # User metadata
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    preferences = Column(JSON)
    
    # Indexes
    __table_args__ = (
        Index('idx_users_username', 'username'),
        Index('idx_users_email', 'email'),
        Index('idx_users_active', 'is_active'),
    )


class SystemMetrics(Base):
    """System performance and usage metrics."""
    __tablename__ = 'system_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Metric metadata
    category = Column(String(50))  # 'performance', 'usage', 'error'
    tags = Column(JSON)  # Additional tags for filtering
    
    # Indexes
    __table_args__ = (
        Index('idx_system_metrics_name', 'metric_name'),
        Index('idx_system_metrics_timestamp', 'timestamp'),
        Index('idx_system_metrics_category', 'category'),
    )


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close database session."""
        session.close()


def get_database_url(config) -> str:
    """Construct database URL from configuration."""
    if config.database.password:
        return (f"postgresql://{config.database.username}:{config.database.password}"
                f"@{config.database.host}:{config.database.port}/{config.database.database}")
    else:
        return (f"postgresql://{config.database.username}"
                f"@{config.database.host}:{config.database.port}/{config.database.database}")