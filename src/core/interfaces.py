"""
Base interfaces and abstract classes for the OCR Table Analytics system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from .models import (
    Document, ProcessingOptions, ProcessingResult, OCRResult, Table,
    TableRegion, DataSchema, Dashboard, ConversationResponse, Query,
    QueryResult, ValidationResult, Chart, ChartConfig
)


class DocumentProcessorInterface(ABC):
    """Interface for document processing components."""
    
    @abstractmethod
    def process_document(self, file_path: str, options: ProcessingOptions) -> ProcessingResult:
        """Process uploaded document and extract metadata."""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image preprocessing for better OCR accuracy."""
        pass
    
    @abstractmethod
    def validate_format(self, file_path: str) -> bool:
        """Validate supported document formats."""
        pass


class OCREngineInterface(ABC):
    """Interface for OCR engine implementations."""
    
    @abstractmethod
    def extract_text(self, image: np.ndarray, **kwargs) -> OCRResult:
        """Extract text from image using OCR."""
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for this engine."""
        pass
    
    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if engine supports specified language."""
        pass


class OCREngineManagerInterface(ABC):
    """Interface for OCR engine management."""
    
    @abstractmethod
    def extract_text(self, image: np.ndarray, engine: str = 'auto') -> OCRResult:
        """Extract text using specified or auto-selected OCR engine."""
        pass
    
    @abstractmethod
    def detect_tables(self, image: np.ndarray) -> List[TableRegion]:
        """Detect table regions in document."""
        pass
    
    @abstractmethod
    def register_engine(self, name: str, engine: OCREngineInterface) -> None:
        """Register a new OCR engine."""
        pass


class TableExtractionInterface(ABC):
    """Interface for table extraction services."""
    
    @abstractmethod
    def extract_table_structure(self, ocr_result: OCRResult, table_regions: List[TableRegion]) -> List[Table]:
        """Extract structured table data from OCR results."""
        pass
    
    @abstractmethod
    def merge_table_fragments(self, tables: List[Table]) -> List[Table]:
        """Merge table fragments across pages."""
        pass
    
    @abstractmethod
    def validate_table_structure(self, table: Table) -> ValidationResult:
        """Validate extracted table structure and data quality."""
        pass


class DataCleaningInterface(ABC):
    """Interface for data cleaning and standardization."""
    
    @abstractmethod
    def standardize_data_types(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and convert data types."""
        pass
    
    @abstractmethod
    def handle_missing_values(self, dataframe: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Handle missing or corrupted values."""
        pass
    
    @abstractmethod
    def remove_duplicates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows and consolidate headers."""
        pass
    
    @abstractmethod
    def detect_schema(self, dataframe: pd.DataFrame) -> DataSchema:
        """Detect and return data schema information."""
        pass


class ChartEngineInterface(ABC):
    """Interface for chart generation engines."""
    
    @abstractmethod
    def create_chart(self, config: ChartConfig, data: pd.DataFrame) -> Chart:
        """Create chart from configuration and data."""
        pass
    
    @abstractmethod
    def get_supported_chart_types(self) -> List[str]:
        """Get list of supported chart types."""
        pass
    
    @abstractmethod
    def auto_select_chart_type(self, data: pd.DataFrame, columns: List[str]) -> str:
        """Automatically select appropriate chart type for data."""
        pass


class DashboardGeneratorInterface(ABC):
    """Interface for dashboard generation."""
    
    @abstractmethod
    def generate_dashboard(self, dataframe: pd.DataFrame) -> Dashboard:
        """Generate interactive dashboard from structured data."""
        pass
    
    @abstractmethod
    def auto_select_visualizations(self, dataframe: pd.DataFrame) -> List[ChartConfig]:
        """Automatically select appropriate chart types."""
        pass
    
    @abstractmethod
    def create_filters(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create interactive filters based on data columns."""
        pass


class LLMProviderInterface(ABC):
    """Interface for LLM service providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    def generate_query(self, question: str, schema: DataSchema) -> Query:
        """Generate data query from natural language question."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        pass


class QueryGeneratorInterface(ABC):
    """Interface for query generation from natural language."""
    
    @abstractmethod
    def parse_question(self, question: str, schema: DataSchema) -> Dict[str, Any]:
        """Parse natural language question and extract intent."""
        pass
    
    @abstractmethod
    def generate_sql(self, intent: Dict[str, Any], schema: DataSchema) -> Query:
        """Generate SQL query from parsed intent."""
        pass
    
    @abstractmethod
    def validate_query(self, query: Query, schema: DataSchema) -> ValidationResult:
        """Validate generated query against schema."""
        pass


class QueryExecutorInterface(ABC):
    """Interface for query execution."""
    
    @abstractmethod
    def execute_query(self, query: Query, dataframe: pd.DataFrame) -> QueryResult:
        """Execute query against dataframe."""
        pass
    
    @abstractmethod
    def execute_sql(self, sql: str, dataframe: pd.DataFrame) -> QueryResult:
        """Execute SQL query against dataframe."""
        pass


class ConversationalAIInterface(ABC):
    """Interface for conversational AI engine."""
    
    @abstractmethod
    def process_question(self, question: str, data_schema: DataSchema) -> ConversationResponse:
        """Process natural language question and generate response."""
        pass
    
    @abstractmethod
    def generate_query(self, question: str, schema: DataSchema) -> Query:
        """Convert natural language to data query."""
        pass
    
    @abstractmethod
    def format_response(self, query_result: QueryResult, question: str) -> ConversationResponse:
        """Format response with text and visualizations."""
        pass


class DataStorageInterface(ABC):
    """Interface for data storage and retrieval."""
    
    @abstractmethod
    def store_document(self, document: Document) -> str:
        """Store document metadata and return document ID."""
        pass
    
    @abstractmethod
    def store_table(self, table: Table, document_id: str) -> str:
        """Store extracted table data."""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        pass
    
    @abstractmethod
    def get_tables(self, document_id: str) -> List[Table]:
        """Retrieve all tables for a document."""
        pass
    
    @abstractmethod
    def store_dashboard(self, dashboard: Dashboard, document_id: str) -> str:
        """Store dashboard configuration."""
        pass


class ExportService(ABC):
    """Interface for data export services."""
    
    @abstractmethod
    def export_table_data(self, table: Table, format: str, output_path: Optional[str] = None) -> str:
        """Export table data to specified format."""
        pass
    
    @abstractmethod
    def export_dashboard(self, dashboard: Dashboard, format: str = 'pdf', 
                        include_data: bool = True, output_path: Optional[str] = None) -> str:
        """Export dashboard with visualizations."""
        pass
    
    @abstractmethod
    def batch_export_tables(self, tables: List[Table], format: str, 
                           output_dir: Optional[str] = None) -> List[str]:
        """Export multiple tables in batch."""
        pass
    
    @abstractmethod
    def get_export_formats(self) -> List[str]:
        """Get list of supported export formats."""
        pass
    
    @abstractmethod
    def validate_export_data(self, data: Any, format: str) -> bool:
        """Validate data before export."""
        pass


class ErrorHandlerInterface(ABC):
    """Interface for error handling and recovery."""
    
    @abstractmethod
    def handle_ocr_error(self, error: Exception, document: Document) -> Dict[str, Any]:
        """Handle OCR processing errors."""
        pass
    
    @abstractmethod
    def handle_table_extraction_error(self, error: Exception, table_region: TableRegion) -> Dict[str, Any]:
        """Handle table extraction errors."""
        pass
    
    @abstractmethod
    def handle_query_error(self, error: Exception, query: Query) -> Dict[str, Any]:
        """Handle query execution errors."""
        pass
    
    @abstractmethod
    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with context information."""
        pass


class ValidationInterface(ABC):
    """Interface for data validation services."""
    
    @abstractmethod
    def validate_table(self, table: Table) -> ValidationResult:
        """Validate table structure and data quality."""
        pass


class ConfigurationManagerInterface(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def get_ocr_config(self, engine: str) -> Dict[str, Any]:
        """Get configuration for OCR engine."""
        pass
    
    @abstractmethod
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for LLM provider."""
        pass
    
    @abstractmethod
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update configuration value."""
        pass
    
    @abstractmethod
    def validate_config(self) -> ValidationResult:
        """Validate current configuration."""
        pass