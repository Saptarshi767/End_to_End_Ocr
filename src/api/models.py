"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ProcessingOptionsRequest(BaseModel):
    """Request model for document processing options"""
    ocr_engine: Optional[str] = Field(default="auto", description="OCR engine to use")
    preprocessing: Optional[bool] = Field(default=True, description="Apply image preprocessing")
    table_detection: Optional[bool] = Field(default=True, description="Enable table detection")
    confidence_threshold: Optional[float] = Field(default=0.8, description="Minimum confidence threshold")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    filename: str
    status: str
    message: str


class DocumentStatusResponse(BaseModel):
    """Response model for document processing status"""
    document_id: str
    filename: str
    status: str
    progress: int = Field(ge=0, le=100)
    created_at: datetime
    completed_at: Optional[datetime] = None


class TableResponse(BaseModel):
    """Response model for extracted table"""
    table_id: str
    document_id: str
    table_index: int
    headers: List[str]
    row_count: int
    confidence_score: float
    created_at: datetime


class ChartConfig(BaseModel):
    """Chart configuration model"""
    chart_type: str
    title: str
    data: Dict[str, Any]
    options: Dict[str, Any] = {}


class KPIModel(BaseModel):
    """KPI model"""
    name: str
    value: Union[int, float, str]
    unit: str = ""
    trend: Optional[str] = None
    change: Optional[float] = None


class FilterModel(BaseModel):
    """Filter model"""
    column: str
    filter_type: str
    values: List[Any]
    operator: str = "in"


class LayoutModel(BaseModel):
    """Dashboard layout model"""
    grid_columns: int = 12
    chart_positions: Dict[str, Dict[str, int]] = {}


class DashboardGenerationRequest(BaseModel):
    """Request model for dashboard generation"""
    document_id: str
    table_ids: Optional[List[str]] = None
    options: Optional[Dict[str, Any]] = {}


class DashboardResponse(BaseModel):
    """Response model for dashboard"""
    dashboard_id: str
    document_id: str
    charts: List[ChartConfig]
    kpis: List[KPIModel]
    filters: List[FilterModel]
    layout: LayoutModel
    created_at: datetime


class ExportFormat(str, Enum):
    """Supported export formats"""
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    JSON = "json"


class ExportRequest(BaseModel):
    """Request model for data export"""
    format: ExportFormat
    output_path: Optional[str] = None
    options: Optional[Dict[str, Any]] = {}


class DashboardExportRequest(BaseModel):
    """Request model for dashboard export"""
    format: ExportFormat
    include_data: bool = True
    output_path: Optional[str] = None
    options: Optional[Dict[str, Any]] = {}


class BatchExportRequest(BaseModel):
    """Request model for batch export"""
    table_ids: List[str]
    format: ExportFormat
    output_dir: Optional[str] = None


class BatchExportResponse(BaseModel):
    """Response model for batch export"""
    exported_files: List[str]
    total_count: int
    format: str


class ChatRequest(BaseModel):
    """Request model for conversational AI"""
    document_id: str
    question: str
    context: Optional[Dict[str, Any]] = {}


class ChatResponse(BaseModel):
    """Response model for conversational AI"""
    response: str
    visualizations: Optional[List[ChartConfig]] = []
    data_summary: Optional[Dict[str, Any]] = {}
    suggested_questions: Optional[List[str]] = []


class ShareRequest(BaseModel):
    """Request model for dashboard sharing"""
    dashboard_id: str
    share_type: str = Field(description="Type of sharing: 'link', 'embed', 'team'")
    permissions: List[str] = Field(default=["view"], description="Permissions: view, edit, export")
    expiry_date: Optional[datetime] = None
    password_protected: bool = False
    password: Optional[str] = None


class ShareResponse(BaseModel):
    """Response model for dashboard sharing"""
    share_id: str
    share_url: str
    embed_code: Optional[str] = None
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None


class TeamAccessRequest(BaseModel):
    """Request model for team access"""
    dashboard_id: str
    user_emails: List[str]
    permissions: List[str] = Field(default=["view"])
    message: Optional[str] = None


class TeamAccessResponse(BaseModel):
    """Response model for team access"""
    dashboard_id: str
    invited_users: List[str]
    failed_invitations: List[str]
    access_url: str


class EmbedWidgetRequest(BaseModel):
    """Request model for embed widget"""
    dashboard_id: str
    widget_type: str = Field(description="Type of widget: 'chart', 'kpi', 'table'")
    widget_id: str
    customization: Optional[Dict[str, Any]] = {}


class EmbedWidgetResponse(BaseModel):
    """Response model for embed widget"""
    widget_id: str
    embed_code: str
    preview_url: str
    customization_options: Dict[str, Any]


class APIKeyRequest(BaseModel):
    """Request model for API key generation"""
    name: str
    permissions: List[str]
    rate_limit: Optional[int] = 1000  # requests per hour
    expires_at: Optional[datetime] = None


class APIKeyResponse(BaseModel):
    """Response model for API key"""
    key_id: str
    api_key: str
    name: str
    permissions: List[str]
    rate_limit: int
    created_at: datetime
    expires_at: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    services: Dict[str, str]
    timestamp: datetime
    version: str = "1.0.0"


class RateLimitInfo(BaseModel):
    """Rate limit information model"""
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None