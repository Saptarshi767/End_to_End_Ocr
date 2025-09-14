"""
Service layer demonstrating proper use of repository pattern with transaction management.
"""

from typing import List, Dict, Any, Optional
import uuid
import logging
from datetime import datetime

from .repository import RepositoryManager, RepositoryError, RecordNotFoundError
from .models import ProcessingStatus, OCREngine, ChartType
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document operations with business logic."""
    
    def __init__(self, repo_manager: RepositoryManager):
        self.repo_manager = repo_manager
    
    def process_document_upload(self, filename: str, file_path: str, 
                              file_size: int, mime_type: str,
                              user_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """Process document upload with validation and logging."""
        try:
            with self.repo_manager.transaction():
                # Create document record
                document = self.repo_manager.documents.create(
                    filename=filename,
                    file_path=file_path,
                    file_size=file_size,
                    mime_type=mime_type,
                    user_id=user_id,
                    processing_status=ProcessingStatus.PENDING
                )
                
                # Log upload
                self.repo_manager.processing_logs.log_processing_step(
                    document_id=document.id,
                    stage='upload',
                    status='completed',
                    processing_time_ms=0
                )
                
                # Record metrics
                self.repo_manager.system_metrics.record_metric(
                    'document_uploaded',
                    1.0,
                    category='usage'
                )
                
                logger.info(f"Document uploaded successfully: {document.id}")
                
                return {
                    'success': True,
                    'document_id': str(document.id),
                    'message': 'Document uploaded successfully'
                }
                
        except Exception as e:
            logger.error(f"Document upload failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Document upload failed'
            }
    
    def complete_document_processing(self, document_id: uuid.UUID,
                                   tables_data: List[Dict[str, Any]],
                                   processing_logs: List[Dict[str, Any]],
                                   total_processing_time: int) -> Dict[str, Any]:
        """Complete document processing with tables and logs."""
        try:
            with self.repo_manager.transaction():
                # Update document status
                document = self.repo_manager.documents.update(
                    document_id,
                    processing_status=ProcessingStatus.COMPLETED
                )
                
                # Create extracted tables
                created_tables = []
                for table_data in tables_data:
                    table_data['document_id'] = document_id
                    table = self.repo_manager.extracted_tables.create(**table_data)
                    created_tables.append(table)
                
                # Create processing logs
                for log_data in processing_logs:
                    log_data['document_id'] = document_id
                    self.repo_manager.processing_logs.create(**log_data)
                
                # Record completion metrics
                self.repo_manager.system_metrics.record_metric(
                    'document_processed',
                    1.0,
                    category='usage'
                )
                
                self.repo_manager.system_metrics.record_metric(
                    'processing_time_ms',
                    float(total_processing_time),
                    metric_unit='milliseconds',
                    category='performance'
                )
                
                logger.info(f"Document processing completed: {document_id}")
                
                return {
                    'success': True,
                    'document_id': str(document_id),
                    'tables_count': len(created_tables),
                    'message': 'Document processing completed successfully'
                }
                
        except RecordNotFoundError:
            logger.error(f"Document not found: {document_id}")
            return {
                'success': False,
                'error': 'Document not found',
                'message': 'Document processing failed'
            }
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Document processing failed'
            }
    
    def get_document_with_tables(self, document_id: uuid.UUID) -> Dict[str, Any]:
        """Get document with all associated tables and metadata."""
        try:
            document = self.repo_manager.documents.get_by_id_or_raise(document_id)
            tables = self.repo_manager.extracted_tables.get_by_document(document_id)
            logs = self.repo_manager.processing_logs.get_by_document(document_id)
            
            return {
                'success': True,
                'document': {
                    'id': str(document.id),
                    'filename': document.filename,
                    'file_size': document.file_size,
                    'mime_type': document.mime_type,
                    'upload_timestamp': document.upload_timestamp.isoformat(),
                    'processing_status': document.processing_status.value
                },
                'tables': [
                    {
                        'id': str(table.id),
                        'table_index': table.table_index,
                        'headers': table.headers,
                        'row_count': table.row_count,
                        'confidence_score': table.confidence_score
                    }
                    for table in tables
                ],
                'processing_logs': [
                    {
                        'stage': log.stage,
                        'status': log.status,
                        'processing_time_ms': log.processing_time_ms,
                        'timestamp': log.timestamp.isoformat()
                    }
                    for log in logs
                ]
            }
            
        except RecordNotFoundError:
            return {
                'success': False,
                'error': 'Document not found',
                'message': 'Document not found'
            }
        except Exception as e:
            logger.error(f"Failed to get document with tables: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve document data'
            }


class DashboardService:
    """Service for dashboard operations with business logic."""
    
    def __init__(self, repo_manager: RepositoryManager):
        self.repo_manager = repo_manager
    
    def create_dashboard_from_table(self, table_id: uuid.UUID, 
                                  dashboard_title: str) -> Dict[str, Any]:
        """Create dashboard with auto-generated charts from table data."""
        try:
            with self.repo_manager.transaction():
                # Get table data
                table = self.repo_manager.extracted_tables.get_by_id_or_raise(table_id)
                
                # Create dashboard
                dashboard = self.repo_manager.dashboards.create(
                    document_id=table.document_id,
                    title=dashboard_title,
                    description=f"Auto-generated dashboard for table {table.table_index}"
                )
                
                # Auto-generate charts based on table structure
                charts_data = self._generate_charts_from_table(table)
                
                created_charts = []
                for chart_data in charts_data:
                    chart_data['dashboard_id'] = dashboard.id
                    chart = self.repo_manager.charts.create(**chart_data)
                    created_charts.append(chart)
                
                # Create data schema for the table
                schema_data = self._infer_schema_from_table(table)
                schema = self.repo_manager.data_schemas.create_schema(
                    table_id=table.id,
                    schema_name=f"Schema for {dashboard_title}",
                    columns_info=schema_data['columns_info'],
                    data_types=schema_data['data_types'],
                    sample_data=schema_data['sample_data']
                )
                
                # Record metrics
                self.repo_manager.system_metrics.record_metric(
                    'dashboard_created',
                    1.0,
                    category='usage'
                )
                
                logger.info(f"Dashboard created successfully: {dashboard.id}")
                
                return {
                    'success': True,
                    'dashboard_id': str(dashboard.id),
                    'charts_count': len(created_charts),
                    'schema_id': str(schema.id),
                    'message': 'Dashboard created successfully'
                }
                
        except RecordNotFoundError:
            return {
                'success': False,
                'error': 'Table not found',
                'message': 'Dashboard creation failed'
            }
        except Exception as e:
            logger.error(f"Dashboard creation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Dashboard creation failed'
            }
    
    def _generate_charts_from_table(self, table) -> List[Dict[str, Any]]:
        """Generate chart configurations based on table data."""
        charts = []
        
        if not table.headers or not table.data:
            return charts
        
        headers = table.headers
        data_rows = table.data
        
        # Generate summary table chart
        charts.append({
            'chart_type': ChartType.TABLE,
            'title': f'Table {table.table_index} Data',
            'config': {
                'columns': headers,
                'show_pagination': True,
                'page_size': 10
            },
            'data': {
                'headers': headers,
                'rows': data_rows[:10]  # Show first 10 rows
            },
            'order_index': 0
        })
        
        # If we have numeric columns, create a bar chart
        numeric_columns = self._identify_numeric_columns(headers, data_rows)
        if len(numeric_columns) > 0 and len(headers) > 1:
            charts.append({
                'chart_type': ChartType.BAR,
                'title': f'Bar Chart - {headers[0]} vs {numeric_columns[0]}',
                'config': {
                    'x_column': headers[0],
                    'y_column': numeric_columns[0],
                    'aggregation': 'sum'
                },
                'data': self._prepare_chart_data(headers, data_rows, headers[0], numeric_columns[0]),
                'order_index': 1
            })
        
        return charts
    
    def _identify_numeric_columns(self, headers: List[str], data_rows: List[List[str]]) -> List[str]:
        """Identify numeric columns in table data."""
        numeric_columns = []
        
        for col_idx, header in enumerate(headers):
            numeric_count = 0
            total_count = 0
            
            for row in data_rows[:10]:  # Sample first 10 rows
                if col_idx < len(row):
                    value = row[col_idx].strip()
                    if value:
                        total_count += 1
                        try:
                            # Try to convert to float
                            float(value.replace(',', '').replace('$', ''))
                            numeric_count += 1
                        except ValueError:
                            pass
            
            # If more than 70% of values are numeric, consider it numeric
            if total_count > 0 and (numeric_count / total_count) > 0.7:
                numeric_columns.append(header)
        
        return numeric_columns
    
    def _prepare_chart_data(self, headers: List[str], data_rows: List[List[str]], 
                          x_column: str, y_column: str) -> Dict[str, Any]:
        """Prepare data for chart visualization."""
        x_idx = headers.index(x_column)
        y_idx = headers.index(y_column)
        
        chart_data = {'labels': [], 'values': []}
        
        for row in data_rows[:20]:  # Limit to first 20 rows
            if x_idx < len(row) and y_idx < len(row):
                x_val = row[x_idx].strip()
                y_val = row[y_idx].strip()
                
                if x_val and y_val:
                    try:
                        y_numeric = float(y_val.replace(',', '').replace('$', ''))
                        chart_data['labels'].append(x_val)
                        chart_data['values'].append(y_numeric)
                    except ValueError:
                        continue
        
        return chart_data
    
    def _infer_schema_from_table(self, table) -> Dict[str, Any]:
        """Infer data schema from table structure."""
        if not table.headers or not table.data:
            return {
                'columns_info': [],
                'data_types': {},
                'sample_data': {}
            }
        
        headers = table.headers
        data_rows = table.data
        
        columns_info = []
        data_types = {}
        sample_data = {}
        
        for col_idx, header in enumerate(headers):
            # Collect sample values
            sample_values = []
            for row in data_rows[:5]:  # Sample first 5 rows
                if col_idx < len(row) and row[col_idx].strip():
                    sample_values.append(row[col_idx].strip())
            
            # Infer data type
            inferred_type = self._infer_column_type(sample_values)
            
            columns_info.append({
                'name': header,
                'type': inferred_type,
                'nullable': True,
                'sample_count': len(sample_values)
            })
            
            data_types[header] = inferred_type
            sample_data[header] = sample_values
        
        return {
            'columns_info': columns_info,
            'data_types': data_types,
            'sample_data': sample_data
        }
    
    def _infer_column_type(self, sample_values: List[str]) -> str:
        """Infer column data type from sample values."""
        if not sample_values:
            return 'text'
        
        numeric_count = 0
        date_count = 0
        
        for value in sample_values:
            # Check if numeric
            try:
                float(value.replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
                continue
            except ValueError:
                pass
            
            # Check if date
            try:
                datetime.strptime(value, '%Y-%m-%d')
                date_count += 1
                continue
            except ValueError:
                try:
                    datetime.strptime(value, '%m/%d/%Y')
                    date_count += 1
                    continue
                except ValueError:
                    pass
        
        total_count = len(sample_values)
        
        # Determine type based on majority
        if numeric_count / total_count > 0.7:
            # Check for currency or percentage
            if any('$' in v for v in sample_values):
                return 'currency'
            elif any('%' in v for v in sample_values):
                return 'percentage'
            else:
                return 'number'
        elif date_count / total_count > 0.7:
            return 'date'
        else:
            return 'text'


class ConversationService:
    """Service for conversation operations with business logic."""
    
    def __init__(self, repo_manager: RepositoryManager):
        self.repo_manager = repo_manager
    
    def start_conversation(self, document_id: uuid.UUID, 
                         user_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """Start a new conversation session."""
        try:
            with self.repo_manager.transaction():
                # Verify document exists
                document = self.repo_manager.documents.get_by_id_or_raise(document_id)
                
                # Create conversation session
                session = self.repo_manager.conversations.create(
                    document_id=document_id,
                    user_id=user_id,
                    session_name=f"Chat about {document.filename}",
                    is_active=True
                )
                
                # Add welcome message
                welcome_message = self.repo_manager.conversation_messages.add_message(
                    session_id=session.id,
                    message_type='assistant',
                    content=f"Hello! I'm ready to help you analyze the data from {document.filename}. What would you like to know?"
                )
                
                # Record metrics
                self.repo_manager.system_metrics.record_metric(
                    'conversation_started',
                    1.0,
                    category='usage'
                )
                
                logger.info(f"Conversation started: {session.id}")
                
                return {
                    'success': True,
                    'session_id': str(session.id),
                    'welcome_message': welcome_message.content,
                    'message': 'Conversation started successfully'
                }
                
        except RecordNotFoundError:
            return {
                'success': False,
                'error': 'Document not found',
                'message': 'Failed to start conversation'
            }
        except Exception as e:
            logger.error(f"Failed to start conversation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to start conversation'
            }
    
    def add_user_message(self, session_id: uuid.UUID, message: str) -> Dict[str, Any]:
        """Add user message to conversation."""
        try:
            with self.repo_manager.transaction():
                # Verify session exists and is active
                session = self.repo_manager.conversations.get_by_id_or_raise(session_id)
                if not session.is_active:
                    return {
                        'success': False,
                        'error': 'Session is not active',
                        'message': 'Cannot add message to inactive session'
                    }
                
                # Add user message
                user_message = self.repo_manager.conversation_messages.add_message(
                    session_id=session_id,
                    message_type='user',
                    content=message
                )
                
                # Update session activity
                self.repo_manager.conversations.update_activity(session_id)
                
                # Record metrics
                self.repo_manager.system_metrics.record_metric(
                    'user_message_sent',
                    1.0,
                    category='usage'
                )
                
                return {
                    'success': True,
                    'message_id': str(user_message.id),
                    'message': 'Message added successfully'
                }
                
        except RecordNotFoundError:
            return {
                'success': False,
                'error': 'Session not found',
                'message': 'Failed to add message'
            }
        except Exception as e:
            logger.error(f"Failed to add user message: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to add message'
            }


class SystemMaintenanceService:
    """Service for system maintenance operations."""
    
    def __init__(self, repo_manager: RepositoryManager):
        self.repo_manager = repo_manager
    
    def perform_maintenance(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Perform system maintenance including data cleanup."""
        try:
            start_time = datetime.utcnow()
            
            # Clean up old data
            cleanup_counts = self.repo_manager.cleanup_old_data(days_to_keep)
            
            # Record maintenance metrics
            self.repo_manager.system_metrics.record_metric(
                'maintenance_performed',
                1.0,
                category='system'
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.repo_manager.system_metrics.record_metric(
                'maintenance_time_ms',
                processing_time,
                metric_unit='milliseconds',
                category='performance'
            )
            
            logger.info(f"System maintenance completed in {processing_time:.2f}ms")
            
            return {
                'success': True,
                'cleanup_counts': cleanup_counts,
                'processing_time_ms': processing_time,
                'message': 'System maintenance completed successfully'
            }
            
        except Exception as e:
            logger.error(f"System maintenance failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'System maintenance failed'
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics and status."""
        try:
            # Get recent metrics
            performance_metrics = self.repo_manager.system_metrics.get_performance_summary(
                ['response_time', 'memory_usage', 'cpu_usage', 'processing_time_ms'],
                hours=24
            )
            
            # Get document processing stats
            total_documents = self.repo_manager.documents.count()
            pending_documents = len(self.repo_manager.documents.get_by_status(ProcessingStatus.PENDING))
            failed_documents = len(self.repo_manager.documents.get_by_status(ProcessingStatus.FAILED))
            
            # Get recent errors
            recent_errors = self.repo_manager.processing_logs.get_errors()[:10]
            
            return {
                'success': True,
                'system_health': {
                    'total_documents': total_documents,
                    'pending_documents': pending_documents,
                    'failed_documents': failed_documents,
                    'error_rate': (failed_documents / max(total_documents, 1)) * 100,
                    'performance_metrics': performance_metrics,
                    'recent_errors_count': len(recent_errors)
                },
                'message': 'System health retrieved successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve system health'
            }