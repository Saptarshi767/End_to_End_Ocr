"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types
    processing_status_enum = postgresql.ENUM(
        'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED',
        name='processingstatus'
    )
    processing_status_enum.create(op.get_bind())
    
    ocr_engine_enum = postgresql.ENUM(
        'TESSERACT', 'EASYOCR', 'CLOUD_VISION', 'LAYOUTLM', 'AUTO', 'MOCK',
        name='ocrengine'
    )
    ocr_engine_enum.create(op.get_bind())
    
    chart_type_enum = postgresql.ENUM(
        'BAR', 'LINE', 'PIE', 'SCATTER', 'HISTOGRAM', 'HEATMAP', 'TABLE',
        name='charttype'
    )
    chart_type_enum.create(op.get_bind())
    
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('username', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_admin', sa.Boolean(), nullable=True),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index('idx_users_active', 'users', ['is_active'], unique=False)
    op.create_index('idx_users_email', 'users', ['email'], unique=False)
    op.create_index('idx_users_username', 'users', ['username'], unique=False)
    
    # Create documents table
    op.create_table('documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('mime_type', sa.String(length=100), nullable=True),
        sa.Column('upload_timestamp', sa.DateTime(), nullable=True),
        sa.Column('processing_status', processing_status_enum, nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_documents_status', 'documents', ['processing_status'], unique=False)
    op.create_index('idx_documents_upload_time', 'documents', ['upload_timestamp'], unique=False)
    op.create_index('idx_documents_user', 'documents', ['user_id'], unique=False)
    
    # Create extracted_tables table
    op.create_table('extracted_tables',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('table_index', sa.Integer(), nullable=False),
        sa.Column('headers', sa.JSON(), nullable=True),
        sa.Column('data', sa.JSON(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('extraction_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('row_count', sa.Integer(), nullable=True),
        sa.Column('column_count', sa.Integer(), nullable=True),
        sa.Column('table_region', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_extracted_tables_confidence', 'extracted_tables', ['confidence_score'], unique=False)
    op.create_index('idx_extracted_tables_created', 'extracted_tables', ['created_at'], unique=False)
    op.create_index('idx_extracted_tables_document', 'extracted_tables', ['document_id'], unique=False)
    
    # Create processing_logs table
    op.create_table('processing_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('stage', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('engine_used', ocr_engine_enum, nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('log_metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_processing_logs_document', 'processing_logs', ['document_id'], unique=False)
    op.create_index('idx_processing_logs_stage', 'processing_logs', ['stage'], unique=False)
    op.create_index('idx_processing_logs_status', 'processing_logs', ['status'], unique=False)
    op.create_index('idx_processing_logs_timestamp', 'processing_logs', ['timestamp'], unique=False)
    
    # Create conversation_sessions table
    op.create_table('conversation_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_activity', sa.DateTime(), nullable=True),
        sa.Column('session_name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_conversation_sessions_activity', 'conversation_sessions', ['last_activity'], unique=False)
    op.create_index('idx_conversation_sessions_document', 'conversation_sessions', ['document_id'], unique=False)
    op.create_index('idx_conversation_sessions_user', 'conversation_sessions', ['user_id'], unique=False)
    
    # Create conversation_messages table
    op.create_table('conversation_messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('message_type', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('query_executed', sa.Text(), nullable=True),
        sa.Column('response_data', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['conversation_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_conversation_messages_session', 'conversation_messages', ['session_id'], unique=False)
    op.create_index('idx_conversation_messages_timestamp', 'conversation_messages', ['timestamp'], unique=False)
    op.create_index('idx_conversation_messages_type', 'conversation_messages', ['message_type'], unique=False)
    
    # Create data_schemas table
    op.create_table('data_schemas',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('table_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('schema_name', sa.String(length=255), nullable=True),
        sa.Column('columns_info', sa.JSON(), nullable=True),
        sa.Column('data_types', sa.JSON(), nullable=True),
        sa.Column('sample_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['table_id'], ['extracted_tables.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_data_schemas_active', 'data_schemas', ['is_active'], unique=False)
    op.create_index('idx_data_schemas_created', 'data_schemas', ['created_at'], unique=False)
    op.create_index('idx_data_schemas_table', 'data_schemas', ['table_id'], unique=False)
    
    # Create dashboards table
    op.create_table('dashboards',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('layout_config', sa.JSON(), nullable=True),
        sa.Column('filters_config', sa.JSON(), nullable=True),
        sa.Column('kpis_config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_public', sa.Boolean(), nullable=True),
        sa.Column('share_token', sa.String(length=255), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('share_token')
    )
    op.create_index('idx_dashboards_created', 'dashboards', ['created_at'], unique=False)
    op.create_index('idx_dashboards_document', 'dashboards', ['document_id'], unique=False)
    op.create_index('idx_dashboards_share_token', 'dashboards', ['share_token'], unique=False)
    op.create_index('idx_dashboards_user', 'dashboards', ['user_id'], unique=False)
    
    # Create charts table
    op.create_table('charts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('dashboard_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chart_type', chart_type_enum, nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('data', sa.JSON(), nullable=True),
        sa.Column('position', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_visible', sa.Boolean(), nullable=True),
        sa.Column('order_index', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['dashboard_id'], ['dashboards.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_charts_dashboard', 'charts', ['dashboard_id'], unique=False)
    op.create_index('idx_charts_order', 'charts', ['order_index'], unique=False)
    op.create_index('idx_charts_type', 'charts', ['chart_type'], unique=False)
    
    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_unit', sa.String(length=50), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_system_metrics_category', 'system_metrics', ['category'], unique=False)
    op.create_index('idx_system_metrics_name', 'system_metrics', ['metric_name'], unique=False)
    op.create_index('idx_system_metrics_timestamp', 'system_metrics', ['timestamp'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('system_metrics')
    op.drop_table('charts')
    op.drop_table('dashboards')
    op.drop_table('data_schemas')
    op.drop_table('conversation_messages')
    op.drop_table('conversation_sessions')
    op.drop_table('processing_logs')
    op.drop_table('extracted_tables')
    op.drop_table('documents')
    op.drop_table('users')
    
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS charttype')
    op.execute('DROP TYPE IF EXISTS ocrengine')
    op.execute('DROP TYPE IF EXISTS processingstatus')