"""
Core data models and type definitions for the OCR Table Analytics system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import uuid
from datetime import datetime
import numpy as np


class ProcessingStatus(Enum):
    """Document processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OCREngine(Enum):
    """Available OCR engines."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    CLOUD_VISION = "cloud_vision"
    LAYOUTLM = "layoutlm"
    AUTO = "auto"
    MOCK = "mock"


class DataType(Enum):
    """Data type enumeration for columns."""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"


class ChartType(Enum):
    """Chart type enumeration for visualizations."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    TABLE = "table"


@dataclass
class BoundingBox:
    """Bounding box coordinates for text regions."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0


@dataclass
class WordData:
    """Individual word data from OCR."""
    text: str
    confidence: float
    bounding_box: BoundingBox


@dataclass
class TableRegion:
    """Table region information."""
    bounding_box: BoundingBox
    confidence: float
    page_number: int = 1


@dataclass
class ProcessingOptions:
    """Configuration options for document processing."""
    ocr_engine: OCREngine = OCREngine.AUTO
    preprocessing: bool = True
    table_detection: bool = True
    confidence_threshold: float = 0.8
    language: str = "eng"
    dpi: int = 300


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    bounding_boxes: List[BoundingBox]
    word_level_data: List[WordData]
    processing_time_ms: int = 0
    engine_used: OCREngine = OCREngine.AUTO


@dataclass
class ColumnInfo:
    """Information about a data column."""
    name: str
    data_type: DataType
    nullable: bool = True
    unique_values: int = 0
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class Table:
    """Extracted table structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    confidence: float = 0.0
    region: Optional[TableRegion] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DataSchema:
    """Schema information for extracted data."""
    columns: List[ColumnInfo]
    row_count: int
    data_types: Dict[str, DataType]
    sample_data: Dict[str, List[Any]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Filter:
    """Dashboard filter configuration."""
    column: str
    filter_type: str  # 'select', 'range', 'search'
    values: List[Any] = field(default_factory=list)
    default_value: Any = None


@dataclass
class KPI:
    """Key Performance Indicator configuration."""
    name: str
    value: Union[int, float, str]
    format_type: str = "number"  # 'number', 'currency', 'percentage'
    description: str = ""
    trend: Optional[float] = None


@dataclass
class ChartConfig:
    """Chart configuration for visualizations."""
    chart_type: ChartType
    title: str
    x_column: str
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    aggregation: str = "count"  # 'count', 'sum', 'avg', 'min', 'max'
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chart:
    """Chart data and configuration."""
    config: ChartConfig
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""
    grid_columns: int = 12
    chart_positions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    responsive: bool = True


@dataclass
class Dashboard:
    """Complete dashboard configuration."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "Data Dashboard"
    charts: List[Chart] = field(default_factory=list)
    filters: List[Filter] = field(default_factory=list)
    kpis: List[KPI] = field(default_factory=list)
    layout: DashboardLayout = field(default_factory=DashboardLayout)
    export_options: List[str] = field(default_factory=lambda: ["pdf", "excel", "csv"])
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Query:
    """Data query representation."""
    sql: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)
    limit: Optional[int] = None


@dataclass
class QueryResult:
    """Result from data query execution."""
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time_ms: int = 0
    query: Optional[Query] = None


@dataclass
class ConversationResponse:
    """Response from conversational AI."""
    text_response: str
    visualizations: List[Chart] = field(default_factory=list)
    data_summary: Dict[str, Any] = field(default_factory=dict)
    suggested_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ProcessingResult:
    """Result from document processing."""
    success: bool
    message: str = ""
    metadata: Optional[Any] = None
    document_id: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    tables: List[Table] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time_ms: int = 0


@dataclass
class ValidationResult:
    """Result from data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Document:
    """Document metadata and information."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    file_path: str = ""
    file_size: int = 0
    mime_type: str = ""
    upload_timestamp: datetime = field(default_factory=datetime.now)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    user_id: Optional[str] = None