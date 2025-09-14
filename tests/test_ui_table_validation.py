"""
Tests for Table Validation UI Component

Tests side-by-side view, inline editing capabilities,
confidence indicators, and validation warnings.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import requests

from src.ui.components.table_validation import TableValidationInterface


class TestTableValidationInterface:
    """Test cases for TableValidationInterface"""
    
    @pytest.fixture
    def validation_interface(self):
        """Create TableValidationInterface instance"""
        return TableValidationInterface(api_base_url="http://test-api:8000")
    
    @pytest.fixture
    def mock_document_info(self):
        """Create mock document information"""
        return {
            'document_id': 'test_doc_123',
            'filename': 'test_document.pdf',
            'status': 'completed',
            'progress': 100
        }
    
    @pytest.fixture
    def mock_table_data(self):
        """Create mock table data"""
        return {
            'table_id': 'table_123',
            'document_id': 'test_doc_123',
            'table_index': 0,
            'headers': ['Name', 'Age', 'City'],
            'row_count': 3,
            'confidence_score': 0.85,
            'metadata': {
                'region': {'x': 100, 'y': 200, 'width': 300, 'height': 150}
            }
        }
    
    @pytest.fixture
    def mock_full_table_data(self):
        """Create mock full table data"""
        return {
            'headers': ['Name', 'Age', 'City'],
            'data': [
                ['John Doe', '30', 'New York'],
                ['Jane Smith', '25', 'Los Angeles'],
                ['Bob Johnson', '35', 'Chicago']
            ]
        }
    
    def test_initialization(self, validation_interface):
        """Test interface initialization"""
        
        assert validation_interface.api_base_url == "http://test-api:8000"
    
    @patch('requests.get')
    def test_load_document_info_success(self, mock_get, validation_interface, mock_document_info):
        """Test successful document info loading"""
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_document_info
        mock_get.return_value = mock_response
        
        with patch('streamlit.session_state', {'auth_token': 'test_token'}):
            result = validation_interface._load_document_info('test_doc_123')
            
            assert result == mock_document_info
            mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_load_document_info_failure(self, mock_get, validation_interface):
        """Test failed document info loading"""
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Document not found"
        mock_get.return_value = mock_response
        
        with patch('streamlit.error') as mock_error:
            with patch('streamlit.session_state', {'auth_token': 'test_token'}):
                result = validation_interface._load_document_info('test_doc_123')
                
                assert result == {}
                mock_error.assert_called_once()
    
    @patch('requests.get')
    def test_load_document_tables_success(self, mock_get, validation_interface, mock_table_data):
        """Test successful table loading"""
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [mock_table_data]
        mock_get.return_value = mock_response
        
        with patch('streamlit.session_state', {'auth_token': 'test_token'}):
            result = validation_interface._load_document_tables('test_doc_123')
            
            assert len(result) == 1
            assert result[0] == mock_table_data
    
    @patch('requests.get')
    def test_load_document_tables_failure(self, mock_get, validation_interface):
        """Test failed table loading"""
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_get.return_value = mock_response
        
        with patch('streamlit.error') as mock_error:
            with patch('streamlit.session_state', {'auth_token': 'test_token'}):
                result = validation_interface._load_document_tables('test_doc_123')
                
                assert result == []
                mock_error.assert_called_once()
    
    def test_detect_validation_issues_no_issues(self, validation_interface, mock_full_table_data):
        """Test validation issue detection with clean data"""
        
        with patch.object(validation_interface, '_load_full_table_data', return_value=mock_full_table_data):
            issues = validation_interface._detect_validation_issues({'table_id': 'test_table'})
            
            # Clean data should have no issues
            assert len(issues) == 0
    
    def test_detect_validation_issues_missing_data(self, validation_interface):
        """Test validation issue detection with missing data"""
        
        table_data_with_nulls = {
            'headers': ['Name', 'Age', 'City'],
            'data': [
                ['John Doe', '30', 'New York'],
                ['Jane Smith', None, 'Los Angeles'],  # Missing age
                ['Bob Johnson', '35', None]  # Missing city
            ]
        }
        
        with patch.object(validation_interface, '_load_full_table_data', return_value=table_data_with_nulls):
            issues = validation_interface._detect_validation_issues({'table_id': 'test_table'})
            
            # Should detect missing data issue
            missing_data_issues = [issue for issue in issues if issue['type'] == 'Missing Data']
            assert len(missing_data_issues) > 0
    
    def test_detect_validation_issues_duplicate_headers(self, validation_interface):
        """Test validation issue detection with duplicate headers"""
        
        table_data_with_duplicates = {
            'headers': ['Name', 'Age', 'Age'],  # Duplicate header
            'data': [
                ['John Doe', '30', '31'],
                ['Jane Smith', '25', '26']
            ]
        }
        
        with patch.object(validation_interface, '_load_full_table_data', return_value=table_data_with_duplicates):
            issues = validation_interface._detect_validation_issues({'table_id': 'test_table'})
            
            # Should detect duplicate headers issue
            duplicate_issues = [issue for issue in issues if issue['type'] == 'Duplicate Headers']
            assert len(duplicate_issues) > 0
    
    def test_detect_validation_issues_ocr_artifacts(self, validation_interface):
        """Test validation issue detection with OCR artifacts"""
        
        table_data_with_artifacts = {
            'headers': ['Name', 'Value', 'Notes'],
            'data': [
                ['John Doe', '30|31', 'Good'],  # OCR artifact |
                ['Jane Smith', '25\\26', 'Fair'],  # OCR artifact \
                ['Bob Johnson', '35', 'Excellent']
            ]
        }
        
        with patch.object(validation_interface, '_load_full_table_data', return_value=table_data_with_artifacts):
            issues = validation_interface._detect_validation_issues({'table_id': 'test_table'})
            
            # Should detect OCR artifacts
            ocr_issues = [issue for issue in issues if issue['type'] == 'OCR Artifacts']
            assert len(ocr_issues) > 0
    
    def test_detect_changes_no_changes(self, validation_interface):
        """Test change detection with identical dataframes"""
        
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        df2 = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        
        changes = validation_interface._detect_changes(df1, df2)
        
        assert len(changes) == 0
    
    def test_detect_changes_with_modifications(self, validation_interface):
        """Test change detection with modified data"""
        
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        df2 = pd.DataFrame({'A': [1, 5, 3], 'B': ['x', 'modified', 'z']})
        
        changes = validation_interface._detect_changes(df1, df2)
        
        # Should detect 2 changes
        assert len(changes) == 2
        
        # Check specific changes
        change_values = [(c['old_value'], c['new_value']) for c in changes]
        assert (2, 5) in change_values
        assert ('y', 'modified') in change_values
    
    def test_detect_changes_dimension_change(self, validation_interface):
        """Test change detection with dimension changes"""
        
        df1 = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
        df2 = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        
        changes = validation_interface._detect_changes(df1, df2)
        
        # Should detect structure change
        structure_changes = [c for c in changes if c.get('type') == 'structure']
        assert len(structure_changes) > 0
    
    @patch('streamlit.session_state', {})
    def test_save_table_changes_with_edits(self, validation_interface):
        """Test saving table changes"""
        
        # Setup edited table data
        edited_df = pd.DataFrame({'Name': ['John', 'Jane'], 'Age': [30, 25]})
        st.session_state['edited_table_test_table'] = edited_df
        
        with patch('streamlit.success') as mock_success:
            validation_interface._save_table_changes('test_table')
            
            mock_success.assert_called_once_with("✅ Changes saved successfully!")
    
    @patch('streamlit.session_state', {})
    def test_save_table_changes_no_edits(self, validation_interface):
        """Test saving table changes with no edits"""
        
        with patch('streamlit.warning') as mock_warning:
            validation_interface._save_table_changes('test_table')
            
            mock_warning.assert_called_once_with("No changes to save")
    
    @patch('streamlit.session_state', {})
    def test_reset_table_changes(self, validation_interface):
        """Test resetting table changes"""
        
        # Setup edited table data
        st.session_state['edited_table_test_table'] = pd.DataFrame({'A': [1, 2]})
        
        with patch('streamlit.success') as mock_success:
            with patch('streamlit.experimental_rerun') as mock_rerun:
                validation_interface._reset_table_changes('test_table')
                
                # Check that edited data was cleared
                assert 'edited_table_test_table' not in st.session_state
                
                mock_success.assert_called_once()
                mock_rerun.assert_called_once()
    
    def test_render_confidence_indicator_high_confidence(self, validation_interface, mock_table_data):
        """Test rendering confidence indicator for high confidence"""
        
        mock_table_data['confidence_score'] = 0.9
        
        with patch('streamlit.columns') as mock_columns:
            with patch('streamlit.metric') as mock_metric:
                with patch('streamlit.success') as mock_success:
                    with patch.object(validation_interface, '_detect_validation_issues', return_value=[]):
                        
                        mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
                        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
                        
                        validation_interface._render_confidence_indicator(mock_table_data)
                        
                        # Should show success for high confidence
                        mock_success.assert_called()
    
    def test_render_confidence_indicator_low_confidence(self, validation_interface, mock_table_data):
        """Test rendering confidence indicator for low confidence"""
        
        mock_table_data['confidence_score'] = 0.4
        
        with patch('streamlit.columns') as mock_columns:
            with patch('streamlit.metric') as mock_metric:
                with patch('streamlit.error') as mock_error:
                    with patch.object(validation_interface, '_detect_validation_issues', return_value=[]):
                        
                        mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
                        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
                        
                        validation_interface._render_confidence_indicator(mock_table_data)
                        
                        # Should show error for low confidence
                        mock_error.assert_called()
    
    def test_render_confidence_indicator_with_issues(self, validation_interface, mock_table_data):
        """Test rendering confidence indicator with validation issues"""
        
        mock_issues = [
            {'type': 'Missing Data', 'message': 'Some cells are empty'},
            {'type': 'OCR Artifacts', 'message': 'Contains OCR artifacts'}
        ]
        
        with patch('streamlit.columns') as mock_columns:
            with patch('streamlit.error') as mock_error:
                with patch('streamlit.expander') as mock_expander:
                    with patch.object(validation_interface, '_detect_validation_issues', return_value=mock_issues):
                        
                        mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
                        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
                        
                        validation_interface._render_confidence_indicator(mock_table_data)
                        
                        # Should show error for issues
                        mock_error.assert_called()
    
    @patch('streamlit.session_state', {})
    def test_render_table_editor(self, validation_interface, mock_table_data, mock_full_table_data):
        """Test rendering table editor"""
        
        with patch.object(validation_interface, '_load_full_table_data', return_value=mock_full_table_data):
            with patch('streamlit.text_input') as mock_text_input:
                with patch('streamlit.data_editor') as mock_data_editor:
                    with patch('streamlit.columns') as mock_columns:
                        
                        # Mock column inputs for headers
                        mock_text_input.side_effect = ['Name', 'Age', 'City']
                        
                        # Mock data editor
                        edited_df = pd.DataFrame(mock_full_table_data['data'], columns=mock_full_table_data['headers'])
                        mock_data_editor.return_value = edited_df
                        
                        # Mock columns
                        mock_columns.return_value = [Mock(), Mock(), Mock()]
                        
                        validation_interface._render_table_editor(mock_table_data)
                        
                        # Check that original data was stored
                        assert f"original_table_{mock_table_data['table_id']}" in st.session_state
                        
                        # Check that edited data was stored
                        assert f"edited_table_{mock_table_data['table_id']}" in st.session_state
    
    def test_export_corrected_data_csv(self, validation_interface):
        """Test exporting corrected data as CSV"""
        
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        
        with patch('streamlit.session_state', {'edited_table_test_table': test_df}):
            with patch('streamlit.download_button') as mock_download:
                
                validation_interface._export_corrected_data('test_table', 'csv')
                
                # Check that download button was created
                mock_download.assert_called_once()
                
                # Check download parameters
                call_args = mock_download.call_args
                assert call_args[1]['file_name'] == 'corrected_table_test_table.csv'
                assert call_args[1]['mime'] == 'text/csv'
    
    def test_export_corrected_data_excel(self, validation_interface):
        """Test exporting corrected data as Excel"""
        
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        
        with patch('streamlit.session_state', {'edited_table_test_table': test_df}):
            with patch('streamlit.download_button') as mock_download:
                
                validation_interface._export_corrected_data('test_table', 'excel')
                
                # Check that download button was created
                mock_download.assert_called_once()
                
                # Check download parameters
                call_args = mock_download.call_args
                assert call_args[1]['file_name'] == 'corrected_table_test_table.xlsx'
                assert 'spreadsheetml' in call_args[1]['mime']
    
    def test_export_corrected_data_json(self, validation_interface):
        """Test exporting corrected data as JSON"""
        
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        
        with patch('streamlit.session_state', {'edited_table_test_table': test_df}):
            with patch('streamlit.download_button') as mock_download:
                
                validation_interface._export_corrected_data('test_table', 'json')
                
                # Check that download button was created
                mock_download.assert_called_once()
                
                # Check download parameters
                call_args = mock_download.call_args
                assert call_args[1]['file_name'] == 'corrected_table_test_table.json'
                assert call_args[1]['mime'] == 'application/json'


@pytest.mark.integration
class TestTableValidationIntegration:
    """Integration tests for table validation workflow"""
    
    @pytest.fixture
    def validation_interface(self):
        """Create TableValidationInterface instance"""
        return TableValidationInterface()
    
    @patch('requests.get')
    @patch('streamlit.session_state', {})
    def test_complete_validation_workflow(self, mock_get, validation_interface):
        """Test complete validation workflow"""
        
        # Mock API responses
        document_response = Mock()
        document_response.status_code = 200
        document_response.json.return_value = {
            'document_id': 'test_doc',
            'filename': 'test.pdf',
            'status': 'completed'
        }
        
        tables_response = Mock()
        tables_response.status_code = 200
        tables_response.json.return_value = [{
            'table_id': 'table_1',
            'headers': ['Name', 'Value'],
            'row_count': 2,
            'confidence_score': 0.8
        }]
        
        mock_get.side_effect = [document_response, tables_response]
        
        with patch('streamlit.selectbox', return_value=0):
            with patch.object(validation_interface, '_render_validation_interface') as mock_render:
                
                validation_interface.render('test_doc')
                
                # Check that validation interface was rendered
                mock_render.assert_called_once()
    
    @patch('requests.get')
    def test_validation_workflow_no_tables(self, mock_get, validation_interface):
        """Test validation workflow when no tables are found"""
        
        # Mock API responses
        document_response = Mock()
        document_response.status_code = 200
        document_response.json.return_value = {'document_id': 'test_doc'}
        
        tables_response = Mock()
        tables_response.status_code = 200
        tables_response.json.return_value = []  # No tables
        
        mock_get.side_effect = [document_response, tables_response]
        
        with patch('streamlit.warning') as mock_warning:
            validation_interface.render('test_doc')
            
            mock_warning.assert_called_once_with("No tables found for this document")


if __name__ == "__main__":
    pytest.main([__file__])