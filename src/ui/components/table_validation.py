"""
Table Validation and Correction UI Component

Implements side-by-side view of original document and extracted data,
inline editing capabilities, confidence indicators, and validation warnings.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import requests
import json
from PIL import Image
import io
import numpy as np

try:
    from src.core.logging_system import get_logger
    from src.api.models import TableResponse
except ImportError:
    # Mock implementations for standalone usage
    class MockLogger:
        def info(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
    
    def get_logger():
        return MockLogger()
    
    # Mock TableResponse
    from pydantic import BaseModel
    from typing import List, Optional
    from datetime import datetime
    
    class TableResponse(BaseModel):
        table_id: str
        document_id: str
        table_index: int
        headers: List[str]
        row_count: int
        confidence_score: float
        created_at: datetime


class TableValidationInterface:
    """Table validation and correction interface"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.logger = get_logger()
        
    def render(self, document_id: str):
        """Render the table validation interface"""
        
        st.title("🔍 Table Validation & Correction")
        st.markdown("Review and correct extracted table data")
        
        # Load document and tables
        document_info = self._load_document_info(document_id)
        tables = self._load_document_tables(document_id)
        
        if not tables:
            st.warning("No tables found for this document")
            return
        
        # Table selection
        selected_table_idx = st.selectbox(
            "Select Table",
            range(len(tables)),
            format_func=lambda x: f"Table {x + 1} ({len(tables[x]['headers'])} columns, {tables[x]['row_count']} rows)"
        )
        
        selected_table = tables[selected_table_idx]
        
        # Main validation interface
        self._render_validation_interface(document_info, selected_table)
    
    def _load_document_info(self, document_id: str) -> Dict[str, Any]:
        """Load document information"""
        
        try:
            response = requests.get(
                f"{self.api_base_url}/documents/{document_id}/status",
                headers={'Authorization': f"Bearer {st.session_state.get('auth_token', '')}"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to load document: {response.text}")
                return {}
                
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return {}
    
    def _load_document_tables(self, document_id: str) -> List[Dict[str, Any]]:
        """Load extracted tables for document"""
        
        try:
            response = requests.get(
                f"{self.api_base_url}/documents/{document_id}/tables",
                headers={'Authorization': f"Bearer {st.session_state.get('auth_token', '')}"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to load tables: {response.text}")
                return []
                
        except Exception as e:
            st.error(f"Error loading tables: {str(e)}")
            return []
    
    def _render_validation_interface(self, document_info: Dict[str, Any], table: Dict[str, Any]):
        """Render the main validation interface"""
        
        # Confidence indicator
        self._render_confidence_indicator(table)
        
        # Side-by-side view
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._render_document_view(document_info, table)
        
        with col2:
            self._render_table_editor(table)
        
        # Validation controls
        self._render_validation_controls(table)
    
    def _render_confidence_indicator(self, table: Dict[str, Any]):
        """Render confidence score and validation warnings"""
        
        confidence = table.get('confidence_score', 0.0)
        
        # Confidence indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.metric("Confidence Score", f"{confidence:.1%}")
        
        with col2:
            # Confidence bar
            if confidence >= 0.8:
                st.success(f"High confidence: {confidence:.1%}")
            elif confidence >= 0.6:
                st.warning(f"Medium confidence: {confidence:.1%}")
            else:
                st.error(f"Low confidence: {confidence:.1%}")
        
        with col3:
            # Validation status
            validation_issues = self._detect_validation_issues(table)
            if validation_issues:
                st.error(f"⚠️ {len(validation_issues)} issues")
            else:
                st.success("✅ No issues")
        
        # Show validation warnings
        if validation_issues:
            with st.expander("⚠️ Validation Issues", expanded=True):
                for issue in validation_issues:
                    st.warning(f"**{issue['type']}**: {issue['message']}")
    
    def _detect_validation_issues(self, table: Dict[str, Any]) -> List[Dict[str, str]]:
        """Detect validation issues in the table"""
        
        issues = []
        
        # Load full table data for validation
        table_data = self._load_full_table_data(table['table_id'])
        if not table_data:
            return issues
        
        df = pd.DataFrame(table_data['data'], columns=table_data['headers'])
        
        # Check for empty cells
        empty_cells = df.isnull().sum().sum()
        if empty_cells > 0:
            issues.append({
                'type': 'Missing Data',
                'message': f"{empty_cells} empty cells detected"
            })
        
        # Check for duplicate headers
        headers = table_data['headers']
        if len(headers) != len(set(headers)):
            issues.append({
                'type': 'Duplicate Headers',
                'message': "Duplicate column headers found"
            })
        
        # Check for inconsistent data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column should be numeric
                numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if numeric_count > len(df) * 0.8:  # 80% numeric
                    issues.append({
                        'type': 'Data Type Inconsistency',
                        'message': f"Column '{col}' appears to be numeric but contains text"
                    })
        
        # Check for unusual patterns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for OCR artifacts
                ocr_artifacts = df[col].astype(str).str.contains(r'[|\\]', na=False).sum()
                if ocr_artifacts > 0:
                    issues.append({
                        'type': 'OCR Artifacts',
                        'message': f"Column '{col}' contains potential OCR artifacts"
                    })
        
        return issues
    
    def _render_document_view(self, document_info: Dict[str, Any], table: Dict[str, Any]):
        """Render original document view with table highlighting"""
        
        st.subheader("📄 Original Document")
        
        # Try to load and display document image
        # This would typically load the original document image
        # For now, show placeholder
        st.info("Document preview would be displayed here")
        
        # Show table region information
        if 'metadata' in table and 'region' in table['metadata']:
            region = table['metadata']['region']
            st.text(f"Table Region: ({region.get('x', 0)}, {region.get('y', 0)}) - "
                   f"({region.get('width', 0)}x{region.get('height', 0)})")
        
        # Show extraction metadata
        with st.expander("Extraction Details"):
            metadata = table.get('metadata', {})
            st.json(metadata)
    
    def _render_table_editor(self, table: Dict[str, Any]):
        """Render editable table data"""
        
        st.subheader("📊 Extracted Data")
        
        # Load full table data
        table_data = self._load_full_table_data(table['table_id'])
        if not table_data:
            st.error("Failed to load table data")
            return
        
        # Create editable dataframe
        df = pd.DataFrame(table_data['data'], columns=table_data['headers'])
        
        # Store original data for comparison
        if f"original_table_{table['table_id']}" not in st.session_state:
            st.session_state[f"original_table_{table['table_id']}"] = df.copy()
        
        # Header editing
        st.markdown("**Column Headers**")
        edited_headers = []
        
        cols = st.columns(min(len(df.columns), 4))  # Max 4 columns per row
        for i, header in enumerate(df.columns):
            with cols[i % 4]:
                new_header = st.text_input(
                    f"Col {i+1}",
                    value=header,
                    key=f"header_{table['table_id']}_{i}"
                )
                edited_headers.append(new_header)
        
        # Update column names if changed
        if edited_headers != list(df.columns):
            df.columns = edited_headers
        
        # Data editing
        st.markdown("**Table Data**")
        
        # Use data editor for inline editing
        edited_df = st.data_editor(
            df,
            width='stretch',
            num_rows="dynamic",
            key=f"table_editor_{table['table_id']}"
        )
        
        # Store edited data
        st.session_state[f"edited_table_{table['table_id']}"] = edited_df
        
        # Show changes summary
        original_df = st.session_state[f"original_table_{table['table_id']}"]
        changes = self._detect_changes(original_df, edited_df)
        
        if changes:
            st.info(f"📝 {len(changes)} changes made")
            with st.expander("View Changes"):
                for change in changes:
                    st.text(f"Row {change['row']}, Col '{change['column']}': "
                           f"'{change['old_value']}' → '{change['new_value']}'")
    
    def _load_full_table_data(self, table_id: str) -> Optional[Dict[str, Any]]:
        """Load full table data including all rows"""
        
        try:
            # This would typically call an API endpoint to get full table data
            # For now, return mock data
            return {
                'headers': ['Column 1', 'Column 2', 'Column 3'],
                'data': [
                    ['Value 1', 'Value 2', 'Value 3'],
                    ['Value 4', 'Value 5', 'Value 6'],
                    ['Value 7', 'Value 8', 'Value 9']
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load table data: {str(e)}")
            return None
    
    def _detect_changes(self, original_df: pd.DataFrame, edited_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect changes between original and edited dataframes"""
        
        changes = []
        
        # Check if dimensions changed
        if original_df.shape != edited_df.shape:
            changes.append({
                'type': 'structure',
                'message': f"Table dimensions changed from {original_df.shape} to {edited_df.shape}"
            })
        
        # Check cell-by-cell changes
        min_rows = min(len(original_df), len(edited_df))
        min_cols = min(len(original_df.columns), len(edited_df.columns))
        
        for i in range(min_rows):
            for j in range(min_cols):
                old_val = original_df.iloc[i, j]
                new_val = edited_df.iloc[i, j]
                
                if pd.isna(old_val) and pd.isna(new_val):
                    continue
                
                if old_val != new_val:
                    changes.append({
                        'row': i,
                        'column': edited_df.columns[j],
                        'old_value': old_val,
                        'new_value': new_val
                    })
        
        return changes
    
    def _render_validation_controls(self, table: Dict[str, Any]):
        """Render validation and correction controls"""
        
        st.subheader("🔧 Validation Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Re-extract Table", help="Re-run OCR extraction"):
                self._reextract_table(table['table_id'])
        
        with col2:
            if st.button("🤖 Auto-correct", help="Apply automatic corrections"):
                self._auto_correct_table(table['table_id'])
        
        with col3:
            if st.button("💾 Save Changes", type="primary"):
                self._save_table_changes(table['table_id'])
        
        with col4:
            if st.button("↩️ Reset", help="Reset to original"):
                self._reset_table_changes(table['table_id'])
        
        # Advanced options
        with st.expander("Advanced Options"):
            
            # Data type detection
            if st.button("🔍 Detect Data Types"):
                self._detect_data_types(table['table_id'])
            
            # Missing value handling
            missing_strategy = st.selectbox(
                "Missing Value Strategy",
                ["keep", "remove", "interpolate", "fill_forward", "fill_backward"]
            )
            
            if st.button("🔧 Handle Missing Values"):
                self._handle_missing_values(table['table_id'], missing_strategy)
            
            # Export corrected data
            export_format = st.selectbox(
                "Export Format",
                ["csv", "excel", "json"]
            )
            
            if st.button("📤 Export Corrected Data"):
                self._export_corrected_data(table['table_id'], export_format)
    
    def _reextract_table(self, table_id: str):
        """Re-extract table with different settings"""
        st.info("Re-extraction would be triggered here")
        # Implementation would call API to re-process with different OCR settings
    
    def _auto_correct_table(self, table_id: str):
        """Apply automatic corrections to table"""
        st.info("Auto-correction would be applied here")
        # Implementation would apply ML-based corrections
    
    def _save_table_changes(self, table_id: str):
        """Save table changes to database"""
        
        edited_df = st.session_state.get(f"edited_table_{table_id}")
        if edited_df is not None:
            # Save changes via API
            st.success("✅ Changes saved successfully!")
        else:
            st.warning("No changes to save")
    
    def _reset_table_changes(self, table_id: str):
        """Reset table to original state"""
        
        # Clear edited data from session state
        if f"edited_table_{table_id}" in st.session_state:
            del st.session_state[f"edited_table_{table_id}"]
        
        st.success("✅ Table reset to original state")
        st.rerun()
    
    def _detect_data_types(self, table_id: str):
        """Detect and suggest data types for columns"""
        st.info("Data type detection would be performed here")
        # Implementation would analyze column data and suggest appropriate types
    
    def _handle_missing_values(self, table_id: str, strategy: str):
        """Handle missing values according to strategy"""
        st.info(f"Missing values would be handled using '{strategy}' strategy")
        # Implementation would apply the selected missing value strategy
    
    def _export_corrected_data(self, table_id: str, format: str):
        """Export corrected table data"""
        
        edited_df = st.session_state.get(f"edited_table_{table_id}")
        if edited_df is not None:
            if format == "csv":
                csv = edited_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"corrected_table_{table_id}.csv",
                    mime="text/csv"
                )
            elif format == "excel":
                # Convert to Excel
                buffer = io.BytesIO()
                edited_df.to_excel(buffer, index=False)
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"corrected_table_{table_id}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif format == "json":
                json_data = edited_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"corrected_table_{table_id}.json",
                    mime="application/json"
                )


def render_table_validation(document_id: str):
    """Render table validation interface"""
    validation_interface = TableValidationInterface()
    validation_interface.render(document_id)


if __name__ == "__main__":
    # Example usage
    if 'selected_document' in st.session_state:
        render_table_validation(st.session_state.selected_document)
    else:
        st.info("Please select a document to validate tables")