"""
Tests for export service functionality
"""

import pytest
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.export_service import ExportService
from src.core.models import Table, Dashboard, Chart, KPI
from src.core.exceptions import ExportError


class TestExportService:
    """Test cases for ExportService"""
    
    @pytest.fixture
    def export_service(self):
        """Create ExportService instance"""
        return ExportService()
    
    @pytest.fixture
    def sample_table(self):
        """Create sample table for testing"""
        return Table(
            headers=["Name", "Age", "City"],
            rows=[
                ["Alice", "25", "New York"],
                ["Bob", "30", "Los Angeles"],
                ["Charlie", "35", "Chicago"]
            ],
            confidence=0.95,
            region=None,
            metadata={
                "name": "test_table",
                "extraction_date": "2024-01-01"
            }
        )
    
    @pytest.fixture
    def sample_dashboard(self):
        """Create sample dashboard for testing"""
        charts = [
            Chart(
                id="chart1",
                title="Age Distribution",
                chart_type="bar",
                data={"labels": ["20-30", "30-40"], "values": [2, 1]},
                options={}
            )
        ]
        
        kpis = [
            KPI(
                id="kpi1",
                name="Average Age",
                value=30,
                unit="years",
                trend="up",
                change=5.0
            )
        ]
        
        return Dashboard(
            id="dash1",
            charts=charts,
            kpis=kpis,
            filters=[],
            layout=None,
            export_options=["pdf", "json"]
        )
    
    def test_export_table_csv(self, export_service, sample_table):
        """Test CSV export functionality"""
        
        # Test string output
        result = export_service.export_table_data(sample_table, "csv")
        
        assert isinstance(result, str)
        assert "Name,Age,City" in result
        assert "Alice,25,New York" in result
        
        # Test file output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            result = export_service.export_table_data(sample_table, "csv", output_path)
            assert result == output_path
            assert os.path.exists(output_path)
            
            # Verify file content
            df = pd.read_csv(output_path)
            assert len(df) == 3
            assert list(df.columns) == ["Name", "Age", "City"]
            
        finally:
            os.unlink(output_path)
    
    def test_export_table_json(self, export_service, sample_table):
        """Test JSON export functionality"""
        
        result = export_service.export_table_data(sample_table, "json")
        
        # Parse JSON to verify structure
        data = json.loads(result)
        
        assert "headers" in data
        assert "rows" in data
        assert "metadata" in data
        assert "confidence" in data
        assert "export_timestamp" in data
        
        assert data["headers"] == ["Name", "Age", "City"]
        assert len(data["rows"]) == 3
        assert data["confidence"] == 0.95
    
    @patch('src.core.export_service.pd.ExcelWriter')
    def test_export_table_excel(self, mock_excel_writer, export_service, sample_table):
        """Test Excel export functionality"""
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            output_path = f.name
        
        try:
            result = export_service.export_table_data(sample_table, "excel", output_path)
            assert result == output_path
            
            # Verify ExcelWriter was called
            mock_excel_writer.assert_called_once()
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_export_unsupported_format(self, export_service, sample_table):
        """Test error handling for unsupported formats"""
        
        with pytest.raises(ExportError) as exc_info:
            export_service.export_table_data(sample_table, "unsupported")
        
        assert "Unsupported format" in str(exc_info.value)
    
    def test_batch_export_tables(self, export_service, sample_table):
        """Test batch export functionality"""
        
        # Create multiple tables
        tables = [sample_table, sample_table, sample_table]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = export_service.batch_export_tables(
                tables, "csv", temp_dir
            )
            
            assert len(exported_files) == 3
            
            for file_path in exported_files:
                assert os.path.exists(file_path)
                assert file_path.endswith('.csv')
                
                # Verify file content
                df = pd.read_csv(file_path)
                assert len(df) == 3
    
    def test_export_dashboard_json(self, export_service, sample_dashboard):
        """Test dashboard JSON export"""
        
        result = export_service.export_dashboard(
            sample_dashboard, "json", include_data=True
        )
        
        data = json.loads(result)
        
        assert "dashboard_config" in data
        assert "kpis" in data
        assert "charts" in data
        assert "export_timestamp" in data
        
        assert len(data["kpis"]) == 1
        assert len(data["charts"]) == 1
    
    def test_validate_export_data(self, export_service, sample_table):
        """Test export data validation"""
        
        # Valid data
        assert export_service.validate_export_data(sample_table, "csv") == True
        assert export_service.validate_export_data(sample_table, "excel") == True
        assert export_service.validate_export_data(sample_table, "json") == True
        
        # Invalid data
        invalid_data = {"no_headers": True}
        assert export_service.validate_export_data(invalid_data, "csv") == False
    
    def test_get_export_formats(self, export_service):
        """Test getting supported export formats"""
        
        formats = export_service.get_export_formats()
        
        assert isinstance(formats, list)
        assert "csv" in formats
        assert "excel" in formats
        assert "json" in formats
        assert "pdf" in formats
    
    def test_table_to_dataframe_conversion(self, export_service, sample_table):
        """Test internal table to DataFrame conversion"""
        
        df = export_service._table_to_dataframe(sample_table)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["Name", "Age", "City"]
        assert df.iloc[0]["Name"] == "Alice"
    
    def test_export_with_missing_data(self, export_service):
        """Test export with missing/incomplete data"""
        
        # Table with missing cells
        incomplete_table = Table(
            headers=["A", "B", "C"],
            rows=[
                ["1", "2"],  # Missing third column
                ["3", "4", "5"],
                ["6"]  # Missing two columns
            ],
            confidence=0.8,
            region=None,
            metadata={}
        )
        
        result = export_service.export_table_data(incomplete_table, "csv")
        
        # Should handle missing data gracefully
        assert isinstance(result, str)
        assert "A,B,C" in result
    
    def test_export_error_handling(self, export_service, sample_table):
        """Test error handling during export"""
        
        # Mock pandas to raise an exception
        with patch('pandas.DataFrame.to_csv', side_effect=Exception("Export failed")):
            with pytest.raises(ExportError) as exc_info:
                export_service.export_table_data(sample_table, "csv")
            
            assert "Export failed" in str(exc_info.value)
    
    def test_export_large_table(self, export_service):
        """Test export of large table"""
        
        # Create large table
        headers = [f"Col_{i}" for i in range(50)]
        rows = [[f"val_{i}_{j}" for j in range(50)] for i in range(1000)]
        
        large_table = Table(
            headers=headers,
            rows=rows,
            confidence=0.9,
            region=None,
            metadata={"size": "large"}
        )
        
        # Should handle large tables without issues
        result = export_service.export_table_data(large_table, "json")
        
        data = json.loads(result)
        assert len(data["headers"]) == 50
        assert len(data["rows"]) == 1000
    
    def test_dashboard_export_without_data(self, export_service, sample_dashboard):
        """Test dashboard export without including raw data"""
        
        result = export_service.export_dashboard(
            sample_dashboard, "json", include_data=False
        )
        
        data = json.loads(result)
        
        assert "raw_data" not in data or data["raw_data"] is None
        assert "dashboard_config" in data
        assert "kpis" in data
        assert "charts" in data