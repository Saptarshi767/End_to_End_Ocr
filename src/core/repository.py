"""
Repository pattern implementation for database operations.
"""

from typing import List, Optional, Dict, Any, Callable, TypeVar, Generic
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime, timedelta
import uuid
import time
import logging
from contextlib import contextmanager
from functools import wraps

from .database import (
    Document, ExtractedTable, ProcessingLog, ConversationSession,
    ConversationMessage, DataSchema, Dashboard, Chart, User, SystemMetrics
)
from .models import ProcessingStatus, OCREngine, ChartType

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass


class RecordNotFoundError(RepositoryError):
    """Exception raised when a record is not found."""
    pass


class DuplicateRecordError(RepositoryError):
    """Exception raised when trying to create a duplicate record."""
    pass


class TransactionError(RepositoryError):
    """Exception raised during transaction operations."""
    pass


def handle_db_errors(func: Callable) -> Callable:
    """Decorator to handle database errors consistently."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RecordNotFoundError, DuplicateRecordError, TransactionError):
            # Re-raise our custom exceptions without wrapping
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error in {func.__name__}: {str(e)}")
            raise DuplicateRecordError(f"Record already exists or violates constraints: {str(e)}")
        except SQLAlchemyError as e:
            logger.error(f"Database error in {func.__name__}: {str(e)}")
            raise RepositoryError(f"Database operation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise RepositoryError(f"Unexpected error: {str(e)}")
    return wrapper


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: Session, model_class):
        self.session = session
        self.model_class = model_class
    
    @handle_db_errors
    def create(self, **kwargs) -> T:
        """Create a new record."""
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        self.session.flush()  # Flush to get ID without committing
        self.session.refresh(instance)
        return instance
    
    @handle_db_errors
    def get_by_id(self, record_id: uuid.UUID) -> Optional[T]:
        """Get record by ID."""
        return self.session.query(self.model_class).filter(
            self.model_class.id == record_id
        ).first()
    
    @handle_db_errors
    def get_by_id_or_raise(self, record_id: uuid.UUID) -> T:
        """Get record by ID or raise RecordNotFoundError."""
        instance = self.get_by_id(record_id)
        if not instance:
            raise RecordNotFoundError(f"{self.model_class.__name__} with id {record_id} not found")
        return instance
    
    @handle_db_errors
    def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all records with pagination."""
        return self.session.query(self.model_class).offset(offset).limit(limit).all()
    
    @handle_db_errors
    def update(self, record_id: uuid.UUID, **kwargs) -> T:
        """Update record by ID."""
        instance = self.get_by_id_or_raise(record_id)
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        self.session.flush()
        self.session.refresh(instance)
        return instance
    
    @handle_db_errors
    def delete(self, record_id: uuid.UUID) -> bool:
        """Delete record by ID."""
        instance = self.get_by_id(record_id)
        if instance:
            self.session.delete(instance)
            self.session.flush()
            return True
        return False
    
    @handle_db_errors
    def delete_by_id_or_raise(self, record_id: uuid.UUID) -> None:
        """Delete record by ID or raise RecordNotFoundError."""
        instance = self.get_by_id_or_raise(record_id)
        self.session.delete(instance)
        self.session.flush()
    
    @handle_db_errors
    def count(self) -> int:
        """Count total records."""
        return self.session.query(self.model_class).count()
    
    @handle_db_errors
    def exists(self, record_id: uuid.UUID) -> bool:
        """Check if record exists by ID."""
        return self.session.query(self.model_class).filter(
            self.model_class.id == record_id
        ).first() is not None
    
    @handle_db_errors
    def bulk_create(self, records: List[Dict[str, Any]]) -> List[T]:
        """Create multiple records in bulk."""
        instances = [self.model_class(**record) for record in records]
        self.session.add_all(instances)
        self.session.flush()
        for instance in instances:
            self.session.refresh(instance)
        return instances
    
    @handle_db_errors
    def bulk_update(self, updates: List[Dict[str, Any]]) -> None:
        """Update multiple records in bulk."""
        for update_data in updates:
            record_id = update_data.pop('id')
            self.session.query(self.model_class).filter(
                self.model_class.id == record_id
            ).update(update_data)
        self.session.flush()


class DocumentRepository(BaseRepository):
    """Repository for document operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, Document)
    
    def get_by_filename(self, filename: str) -> Optional[Document]:
        """Get document by filename."""
        return self.session.query(Document).filter(
            Document.filename == filename
        ).first()
    
    def get_by_status(self, status: ProcessingStatus) -> List[Document]:
        """Get documents by processing status."""
        return self.session.query(Document).filter(
            Document.processing_status == status
        ).all()
    
    def get_by_user(self, user_id: uuid.UUID) -> List[Document]:
        """Get documents by user ID."""
        return self.session.query(Document).filter(
            Document.user_id == user_id
        ).order_by(desc(Document.upload_timestamp)).all()
    
    def update_status(self, document_id: uuid.UUID, status: ProcessingStatus) -> Optional[Document]:
        """Update document processing status."""
        return self.update(document_id, processing_status=status)
    
    def get_recent(self, limit: int = 10) -> List[Document]:
        """Get recently uploaded documents."""
        return self.session.query(Document).order_by(
            desc(Document.upload_timestamp)
        ).limit(limit).all()


class ExtractedTableRepository(BaseRepository):
    """Repository for extracted table operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, ExtractedTable)
    
    def get_by_document(self, document_id: uuid.UUID) -> List[ExtractedTable]:
        """Get all tables for a document."""
        return self.session.query(ExtractedTable).filter(
            ExtractedTable.document_id == document_id
        ).order_by(ExtractedTable.table_index).all()
    
    def get_by_confidence_threshold(self, threshold: float) -> List[ExtractedTable]:
        """Get tables above confidence threshold."""
        return self.session.query(ExtractedTable).filter(
            ExtractedTable.confidence_score >= threshold
        ).all()
    
    def get_table_count_by_document(self, document_id: uuid.UUID) -> int:
        """Get count of tables for a document."""
        return self.session.query(ExtractedTable).filter(
            ExtractedTable.document_id == document_id
        ).count()
    
    def update_confidence(self, table_id: uuid.UUID, confidence: float) -> Optional[ExtractedTable]:
        """Update table confidence score."""
        return self.update(table_id, confidence_score=confidence)


class ProcessingLogRepository(BaseRepository):
    """Repository for processing log operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, ProcessingLog)
    
    def get_by_document(self, document_id: uuid.UUID) -> List[ProcessingLog]:
        """Get all logs for a document."""
        return self.session.query(ProcessingLog).filter(
            ProcessingLog.document_id == document_id
        ).order_by(ProcessingLog.timestamp).all()
    
    def get_by_stage(self, stage: str) -> List[ProcessingLog]:
        """Get logs by processing stage."""
        return self.session.query(ProcessingLog).filter(
            ProcessingLog.stage == stage
        ).order_by(desc(ProcessingLog.timestamp)).all()
    
    def get_errors(self, document_id: Optional[uuid.UUID] = None) -> List[ProcessingLog]:
        """Get error logs, optionally filtered by document."""
        query = self.session.query(ProcessingLog).filter(
            ProcessingLog.status == 'error'
        )
        if document_id:
            query = query.filter(ProcessingLog.document_id == document_id)
        return query.order_by(desc(ProcessingLog.timestamp)).all()
    
    def log_processing_step(self, document_id: uuid.UUID, stage: str, status: str,
                           processing_time_ms: Optional[int] = None,
                           engine_used: Optional[OCREngine] = None,
                           confidence_score: Optional[float] = None,
                           error_message: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> ProcessingLog:
        """Log a processing step."""
        return self.create(
            document_id=document_id,
            stage=stage,
            status=status,
            processing_time_ms=processing_time_ms,
            engine_used=engine_used,
            confidence_score=confidence_score,
            error_message=error_message,
            log_metadata=metadata,
            timestamp=datetime.utcnow()
        )


class ConversationRepository(BaseRepository):
    """Repository for conversation operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, ConversationSession)
    
    def get_by_document(self, document_id: uuid.UUID) -> List[ConversationSession]:
        """Get all conversation sessions for a document."""
        return self.session.query(ConversationSession).filter(
            ConversationSession.document_id == document_id
        ).order_by(desc(ConversationSession.last_activity)).all()
    
    def get_active_sessions(self, user_id: Optional[uuid.UUID] = None) -> List[ConversationSession]:
        """Get active conversation sessions."""
        query = self.session.query(ConversationSession).filter(
            ConversationSession.is_active == True
        )
        if user_id:
            query = query.filter(ConversationSession.user_id == user_id)
        return query.order_by(desc(ConversationSession.last_activity)).all()
    
    def update_activity(self, session_id: uuid.UUID) -> Optional[ConversationSession]:
        """Update last activity timestamp."""
        return self.update(session_id, last_activity=datetime.utcnow())
    
    def deactivate_session(self, session_id: uuid.UUID) -> Optional[ConversationSession]:
        """Deactivate a conversation session."""
        return self.update(session_id, is_active=False)


class ConversationMessageRepository(BaseRepository):
    """Repository for conversation message operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, ConversationMessage)
    
    def get_by_session(self, session_id: uuid.UUID, limit: int = 50) -> List[ConversationMessage]:
        """Get messages for a conversation session."""
        return self.session.query(ConversationMessage).filter(
            ConversationMessage.session_id == session_id
        ).order_by(ConversationMessage.timestamp).limit(limit).all()
    
    def add_message(self, session_id: uuid.UUID, message_type: str, content: str,
                   query_executed: Optional[str] = None,
                   response_data: Optional[Dict[str, Any]] = None,
                   confidence_score: Optional[float] = None,
                   processing_time_ms: Optional[int] = None) -> ConversationMessage:
        """Add a new message to conversation."""
        return self.create(
            session_id=session_id,
            message_type=message_type,
            content=content,
            query_executed=query_executed,
            response_data=response_data,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow()
        )
    
    def get_recent_messages(self, session_id: uuid.UUID, count: int = 10) -> List[ConversationMessage]:
        """Get recent messages from a session."""
        return self.session.query(ConversationMessage).filter(
            ConversationMessage.session_id == session_id
        ).order_by(desc(ConversationMessage.timestamp)).limit(count).all()


class DashboardRepository(BaseRepository):
    """Repository for dashboard operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, Dashboard)
    
    def get_by_document(self, document_id: uuid.UUID) -> List[Dashboard]:
        """Get dashboards for a document."""
        return self.session.query(Dashboard).filter(
            Dashboard.document_id == document_id
        ).order_by(desc(Dashboard.created_at)).all()
    
    def get_by_share_token(self, share_token: str) -> Optional[Dashboard]:
        """Get dashboard by share token."""
        return self.session.query(Dashboard).filter(
            Dashboard.share_token == share_token
        ).first()
    
    def get_public_dashboards(self) -> List[Dashboard]:
        """Get public dashboards."""
        return self.session.query(Dashboard).filter(
            Dashboard.is_public == True
        ).order_by(desc(Dashboard.created_at)).all()
    
    def update_share_token(self, dashboard_id: uuid.UUID, share_token: str) -> Optional[Dashboard]:
        """Update dashboard share token."""
        return self.update(dashboard_id, share_token=share_token)


class ChartRepository(BaseRepository):
    """Repository for chart operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, Chart)
    
    def get_by_dashboard(self, dashboard_id: uuid.UUID) -> List[Chart]:
        """Get charts for a dashboard."""
        return self.session.query(Chart).filter(
            Chart.dashboard_id == dashboard_id
        ).order_by(Chart.order_index).all()
    
    def get_visible_charts(self, dashboard_id: uuid.UUID) -> List[Chart]:
        """Get visible charts for a dashboard."""
        return self.session.query(Chart).filter(
            and_(Chart.dashboard_id == dashboard_id, Chart.is_visible == True)
        ).order_by(Chart.order_index).all()
    
    def update_order(self, chart_id: uuid.UUID, order_index: int) -> Optional[Chart]:
        """Update chart order index."""
        return self.update(chart_id, order_index=order_index)
    
    def toggle_visibility(self, chart_id: uuid.UUID) -> Optional[Chart]:
        """Toggle chart visibility."""
        chart = self.get_by_id(chart_id)
        if chart:
            return self.update(chart_id, is_visible=not chart.is_visible)
        return None


class UserRepository(BaseRepository):
    """Repository for user operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, User)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.session.query(User).filter(
            User.username == username
        ).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.session.query(User).filter(
            User.email == email
        ).first()
    
    def get_active_users(self) -> List[User]:
        """Get active users."""
        return self.session.query(User).filter(
            User.is_active == True
        ).all()
    
    def update_last_login(self, user_id: uuid.UUID) -> Optional[User]:
        """Update user last login timestamp."""
        return self.update(user_id, last_login=datetime.utcnow())


class DataSchemaRepository(BaseRepository):
    """Repository for data schema operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, DataSchema)
    
    @handle_db_errors
    def get_by_table(self, table_id: uuid.UUID) -> List[DataSchema]:
        """Get all schemas for a table."""
        return self.session.query(DataSchema).filter(
            DataSchema.table_id == table_id
        ).order_by(desc(DataSchema.created_at)).all()
    
    @handle_db_errors
    def get_active_schema(self, table_id: uuid.UUID) -> Optional[DataSchema]:
        """Get active schema for a table."""
        return self.session.query(DataSchema).filter(
            and_(DataSchema.table_id == table_id, DataSchema.is_active == True)
        ).first()
    
    @handle_db_errors
    def create_schema(self, table_id: uuid.UUID, schema_name: str,
                     columns_info: List[Dict[str, Any]],
                     data_types: Dict[str, str],
                     sample_data: Optional[Dict[str, List[Any]]] = None) -> DataSchema:
        """Create a new data schema."""
        # Deactivate existing active schemas
        self.session.query(DataSchema).filter(
            and_(DataSchema.table_id == table_id, DataSchema.is_active == True)
        ).update({'is_active': False})
        
        return self.create(
            table_id=table_id,
            schema_name=schema_name,
            columns_info=columns_info,
            data_types=data_types,
            sample_data=sample_data or {},
            is_active=True,
            version=self._get_next_version(table_id)
        )
    
    @handle_db_errors
    def activate_schema(self, schema_id: uuid.UUID) -> DataSchema:
        """Activate a specific schema version."""
        schema = self.get_by_id_or_raise(schema_id)
        
        # Deactivate other schemas for the same table
        self.session.query(DataSchema).filter(
            and_(DataSchema.table_id == schema.table_id, DataSchema.is_active == True)
        ).update({'is_active': False})
        
        return self.update(schema_id, is_active=True)
    
    def _get_next_version(self, table_id: uuid.UUID) -> int:
        """Get next version number for a table."""
        max_version = self.session.query(DataSchema).filter(
            DataSchema.table_id == table_id
        ).order_by(desc(DataSchema.version)).first()
        
        return (max_version.version + 1) if max_version else 1


class SystemMetricsRepository(BaseRepository):
    """Repository for system metrics operations."""
    
    def __init__(self, session: Session):
        super().__init__(session, SystemMetrics)
    
    @handle_db_errors
    def record_metric(self, metric_name: str, metric_value: float,
                     metric_unit: Optional[str] = None,
                     category: Optional[str] = None,
                     tags: Optional[Dict[str, Any]] = None) -> SystemMetrics:
        """Record a system metric."""
        return self.create(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            category=category,
            tags=tags,
            timestamp=datetime.utcnow()
        )
    
    @handle_db_errors
    def get_metrics_by_name(self, metric_name: str, limit: int = 100) -> List[SystemMetrics]:
        """Get metrics by name."""
        return self.session.query(SystemMetrics).filter(
            SystemMetrics.metric_name == metric_name
        ).order_by(desc(SystemMetrics.timestamp)).limit(limit).all()
    
    @handle_db_errors
    def get_metrics_by_category(self, category: str, limit: int = 100) -> List[SystemMetrics]:
        """Get metrics by category."""
        return self.session.query(SystemMetrics).filter(
            SystemMetrics.category == category
        ).order_by(desc(SystemMetrics.timestamp)).limit(limit).all()
    
    @handle_db_errors
    def get_recent_metrics(self, hours: int = 24) -> List[SystemMetrics]:
        """Get recent metrics within specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return self.session.query(SystemMetrics).filter(
            SystemMetrics.timestamp >= cutoff_time
        ).order_by(desc(SystemMetrics.timestamp)).all()
    
    @handle_db_errors
    def get_performance_summary(self, metric_names: List[str], hours: int = 24) -> Dict[str, Dict[str, float]]:
        """Get performance summary for specified metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        summary = {}
        
        for metric_name in metric_names:
            metrics = self.session.query(SystemMetrics).filter(
                and_(
                    SystemMetrics.metric_name == metric_name,
                    SystemMetrics.timestamp >= cutoff_time
                )
            ).all()
            
            if metrics:
                values = [m.metric_value for m in metrics]
                summary[metric_name] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else 0
                }
            else:
                summary[metric_name] = {
                    'count': 0, 'avg': 0, 'min': 0, 'max': 0, 'latest': 0
                }
        
        return summary


class RepositoryManager:
    """Manager class for all repositories with transaction management."""
    
    def __init__(self, session: Session):
        self.session = session
        self.documents = DocumentRepository(session)
        self.extracted_tables = ExtractedTableRepository(session)
        self.processing_logs = ProcessingLogRepository(session)
        self.conversations = ConversationRepository(session)
        self.conversation_messages = ConversationMessageRepository(session)
        self.dashboards = DashboardRepository(session)
        self.charts = ChartRepository(session)
        self.users = UserRepository(session)
        self.system_metrics = SystemMetricsRepository(session)
        self.data_schemas = DataSchemaRepository(session)
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self
            self.session.commit()
            logger.debug("Transaction committed successfully")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Transaction rolled back due to error: {str(e)}")
            raise TransactionError(f"Transaction failed: {str(e)}")
    
    def commit(self):
        """Commit current transaction."""
        try:
            self.session.commit()
            logger.debug("Manual commit successful")
        except SQLAlchemyError as e:
            logger.error(f"Commit failed: {str(e)}")
            raise TransactionError(f"Commit failed: {str(e)}")
    
    def rollback(self):
        """Rollback current transaction."""
        try:
            self.session.rollback()
            logger.debug("Manual rollback successful")
        except SQLAlchemyError as e:
            logger.error(f"Rollback failed: {str(e)}")
            raise TransactionError(f"Rollback failed: {str(e)}")
    
    def flush(self):
        """Flush pending changes without committing."""
        try:
            self.session.flush()
            logger.debug("Session flushed successfully")
        except SQLAlchemyError as e:
            logger.error(f"Flush failed: {str(e)}")
            raise TransactionError(f"Flush failed: {str(e)}")
    
    def close(self):
        """Close session."""
        try:
            self.session.close()
            logger.debug("Session closed successfully")
        except SQLAlchemyError as e:
            logger.error(f"Session close failed: {str(e)}")
    
    def refresh(self, instance):
        """Refresh instance from database."""
        try:
            self.session.refresh(instance)
        except SQLAlchemyError as e:
            logger.error(f"Refresh failed: {str(e)}")
            raise TransactionError(f"Refresh failed: {str(e)}")
    
    @handle_db_errors
    def bulk_process_document(self, document_data: Dict[str, Any], 
                            tables_data: List[Dict[str, Any]],
                            processing_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process document with tables and logs in a single transaction."""
        with self.transaction():
            # Create document
            document = self.documents.create(**document_data)
            
            # Create tables
            created_tables = []
            for table_data in tables_data:
                table_data['document_id'] = document.id
                table = self.extracted_tables.create(**table_data)
                created_tables.append(table)
            
            # Create processing logs
            created_logs = []
            for log_data in processing_logs:
                log_data['document_id'] = document.id
                log = self.processing_logs.create(**log_data)
                created_logs.append(log)
            
            return {
                'document': document,
                'tables': created_tables,
                'logs': created_logs
            }
    
    @handle_db_errors
    def create_dashboard_with_charts(self, dashboard_data: Dict[str, Any],
                                   charts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create dashboard with charts in a single transaction."""
        with self.transaction():
            # Create dashboard
            dashboard = self.dashboards.create(**dashboard_data)
            
            # Create charts
            created_charts = []
            for chart_data in charts_data:
                chart_data['dashboard_id'] = dashboard.id
                chart = self.charts.create(**chart_data)
                created_charts.append(chart)
            
            return {
                'dashboard': dashboard,
                'charts': created_charts
            }
    
    @handle_db_errors
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data beyond retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cleanup_counts = {}
        
        with self.transaction():
            # Clean up old processing logs
            old_logs = self.session.query(ProcessingLog).filter(
                ProcessingLog.timestamp < cutoff_date
            ).count()
            
            self.session.query(ProcessingLog).filter(
                ProcessingLog.timestamp < cutoff_date
            ).delete()
            cleanup_counts['processing_logs'] = old_logs
            
            # Clean up old system metrics
            old_metrics = self.session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_date
            ).count()
            
            self.session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_date
            ).delete()
            cleanup_counts['system_metrics'] = old_metrics
            
            # Clean up inactive conversation sessions
            old_sessions = self.session.query(ConversationSession).filter(
                and_(
                    ConversationSession.last_activity < cutoff_date,
                    ConversationSession.is_active == False
                )
            ).count()
            
            self.session.query(ConversationSession).filter(
                and_(
                    ConversationSession.last_activity < cutoff_date,
                    ConversationSession.is_active == False
                )
            ).delete()
            cleanup_counts['conversation_sessions'] = old_sessions
            
        return cleanup_counts