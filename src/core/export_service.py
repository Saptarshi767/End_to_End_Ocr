"""
Data Export Service

Handles export of data and visualizations to multiple formats including
Excel, CSV, PDF, and JSON with batch export capabilities.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
import base64
from datetime import datetime

from .interfaces import ExportService as IExportService
from .models import Table, Dashboard, Chart
from .exceptions import ExportError


class ExportService(IExportService):
    """Service for exporting data and visualizations to various formats"""
    
    def __init__(self):
        self.supported_formats = ['excel', 'csv', 'pdf', 'json']
        self.export_history = []
    
    def export_table_data(
        self, 
        table: Table, 
        format: str, 
        output_path: Optional[str] = None
    ) -> Union[str, bytes]:
        """
        Export table data to specified format
        
        Args:
            table: Table object containing data to export
            format: Export format ('excel', 'csv', 'pdf', 'json')
            output_path: Optional file path for saving
            
        Returns:
            File path or binary data depending on format
        """
        if format not in self.supported_formats:
            raise ExportError(f"Unsupported format: {format}")
        
        try:
            # Convert table to DataFrame
            df = self._table_to_dataframe(table)
            
            if format == 'csv':
                return self._export_csv(df, output_path)
            elif format == 'excel':
                return self._export_excel(df, output_path, table)
            elif format == 'json':
                return self._export_json(table, output_path)
            elif format == 'pdf':
                return self._export_pdf(df, output_path, table)
                
        except Exception as e:
            raise ExportError(f"Export failed: {str(e)}")
    
    def export_dashboard(
        self, 
        dashboard: Dashboard, 
        format: str = 'pdf',
        include_data: bool = True,
        output_path: Optional[str] = None
    ) -> Union[str, bytes]:
        """
        Export dashboard with visualizations
        
        Args:
            dashboard: Dashboard object to export
            format: Export format
            include_data: Whether to include raw data
            output_path: Optional file path for saving
            
        Returns:
            File path or binary data
        """
        try:
            if format == 'pdf':
                return self._export_dashboard_pdf(dashboard, include_data, output_path)
            elif format == 'json':
                return self._export_dashboard_json(dashboard, include_data, output_path)
            else:
                raise ExportError(f"Dashboard export not supported for format: {format}")
                
        except Exception as e:
            raise ExportError(f"Dashboard export failed: {str(e)}")
    
    def batch_export_tables(
        self, 
        tables: List[Table], 
        format: str,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Export multiple tables in batch
        
        Args:
            tables: List of tables to export
            format: Export format
            output_dir: Directory for output files
            
        Returns:
            List of exported file paths
        """
        if not output_dir:
            output_dir = f"exports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        exported_files = []
        
        for i, table in enumerate(tables):
            filename = f"table_{i+1}_{table.metadata.get('name', 'unnamed')}.{format}"
            output_path = Path(output_dir) / filename
            
            try:
                result = self.export_table_data(table, format, str(output_path))
                exported_files.append(str(output_path))
                
            except Exception as e:
                print(f"Failed to export table {i+1}: {str(e)}")
                continue
        
        return exported_files
    
    def get_export_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return self.supported_formats.copy()
    
    def validate_export_data(self, data: Any, format: str) -> bool:
        """
        Validate data before export
        
        Args:
            data: Data to validate
            format: Target export format
            
        Returns:
            True if data is valid for export
        """
        try:
            if format == 'csv':
                return self._validate_csv_data(data)
            elif format == 'excel':
                return self._validate_excel_data(data)
            elif format == 'json':
                return self._validate_json_data(data)
            elif format == 'pdf':
                return self._validate_pdf_data(data)
            return False
            
        except Exception:
            return False
    
    def _table_to_dataframe(self, table: Table) -> pd.DataFrame:
        """Convert Table object to pandas DataFrame"""
        data = {}
        for i, header in enumerate(table.headers):
            column_data = []
            for row in table.rows:
                if i < len(row):
                    column_data.append(row[i])
                else:
                    column_data.append(None)
            data[header] = column_data
        
        return pd.DataFrame(data)
    
    def _export_csv(self, df: pd.DataFrame, output_path: Optional[str]) -> str:
        """Export DataFrame to CSV"""
        if output_path:
            df.to_csv(output_path, index=False)
            return output_path
        else:
            return df.to_csv(index=False)
    
    def _export_excel(self, df: pd.DataFrame, output_path: Optional[str], table: Table) -> str:
        """Export DataFrame to Excel with formatting"""
        if not output_path:
            output_path = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Add metadata sheet
            metadata_df = pd.DataFrame([
                ['Table Name', table.metadata.get('name', 'Unnamed')],
                ['Extraction Date', table.metadata.get('extraction_date', 'Unknown')],
                ['Confidence Score', table.confidence],
                ['Row Count', len(table.rows)],
                ['Column Count', len(table.headers)]
            ], columns=['Property', 'Value'])
            
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        return output_path
    
    def _export_json(self, table: Table, output_path: Optional[str]) -> str:
        """Export table to JSON format"""
        export_data = {
            'metadata': table.metadata,
            'confidence': table.confidence,
            'headers': table.headers,
            'rows': table.rows,
            'export_timestamp': datetime.now().isoformat()
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return output_path
        else:
            return json_str
    
    def _export_pdf(self, df: pd.DataFrame, output_path: Optional[str], table: Table) -> str:
        """Export DataFrame to PDF (requires reportlab)"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table as PDFTable, TableStyle, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            
        except ImportError:
            raise ExportError("PDF export requires reportlab package: pip install reportlab")
        
        if not output_path:
            output_path = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Add title
        title = Paragraph(f"Table Export - {table.metadata.get('name', 'Unnamed')}", styles['Title'])
        elements.append(title)
        
        # Add metadata
        metadata_text = f"Confidence: {table.confidence:.2f} | Rows: {len(table.rows)} | Columns: {len(table.headers)}"
        metadata = Paragraph(metadata_text, styles['Normal'])
        elements.append(metadata)
        
        # Create table data
        table_data = [table.headers] + table.rows
        
        # Create PDF table
        pdf_table = PDFTable(table_data)
        pdf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(pdf_table)
        doc.build(elements)
        
        return output_path
    
    def _export_dashboard_pdf(self, dashboard: Dashboard, include_data: bool, output_path: Optional[str]) -> str:
        """Export dashboard to PDF with visualizations"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            
        except ImportError:
            raise ExportError("PDF export requires reportlab package: pip install reportlab")
        
        if not output_path:
            output_path = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Add title
        title = Paragraph("Dashboard Export", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Add KPIs
        if dashboard.kpis:
            kpi_title = Paragraph("Key Performance Indicators", styles['Heading2'])
            elements.append(kpi_title)
            
            for kpi in dashboard.kpis:
                kpi_text = f"{kpi.name}: {kpi.value} {kpi.unit}"
                kpi_para = Paragraph(kpi_text, styles['Normal'])
                elements.append(kpi_para)
            
            elements.append(Spacer(1, 12))
        
        # Add charts (as placeholders - would need actual chart rendering)
        if dashboard.charts:
            charts_title = Paragraph("Visualizations", styles['Heading2'])
            elements.append(charts_title)
            
            for chart in dashboard.charts:
                chart_desc = f"Chart: {chart.title} (Type: {chart.chart_type})"
                chart_para = Paragraph(chart_desc, styles['Normal'])
                elements.append(chart_para)
                elements.append(Spacer(1, 6))
        
        doc.build(elements)
        return output_path
    
    def _export_dashboard_json(self, dashboard: Dashboard, include_data: bool, output_path: Optional[str]) -> str:
        """Export dashboard configuration to JSON"""
        export_data = {
            'dashboard_config': {
                'layout': dashboard.layout.__dict__ if dashboard.layout else None,
                'export_options': dashboard.export_options
            },
            'kpis': [kpi.__dict__ for kpi in dashboard.kpis],
            'charts': [chart.__dict__ for chart in dashboard.charts],
            'filters': [filter.__dict__ for filter in dashboard.filters],
            'export_timestamp': datetime.now().isoformat()
        }
        
        if include_data:
            # Add raw data if available
            export_data['raw_data'] = getattr(dashboard, 'raw_data', None)
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return output_path
        else:
            return json_str
    
    def _validate_csv_data(self, data: Any) -> bool:
        """Validate data for CSV export"""
        return hasattr(data, 'headers') and hasattr(data, 'rows')
    
    def _validate_excel_data(self, data: Any) -> bool:
        """Validate data for Excel export"""
        return hasattr(data, 'headers') and hasattr(data, 'rows')
    
    def _validate_json_data(self, data: Any) -> bool:
        """Validate data for JSON export"""
        try:
            json.dumps(data.__dict__ if hasattr(data, '__dict__') else data)
            return True
        except (TypeError, ValueError):
            return False
    
    def _validate_pdf_data(self, data: Any) -> bool:
        """Validate data for PDF export"""
        return hasattr(data, 'headers') and hasattr(data, 'rows')