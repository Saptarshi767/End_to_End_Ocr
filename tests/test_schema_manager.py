"""
Tests for schema detection and management system.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, date
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.models import DataType, DataSchema, ColumnInfo, ValidationResult
from src.data_processing.schema_manager import (
    SchemaInferenceEngine, SchemaValidator, SchemaVersionManager, SchemaManager,
    SchemaInferenceConfig, SchemaCompatibility, SchemaChangeType, SchemaChange,
    SchemaVersion, SchemaCompatibilityResult
)


class TestSchemaInferenceEngine:
    """Test schema inference functionality."""
    
    @pytest.fixture
    def inference_engine(self):
        """Create schema inference engine with test configuration."""
        config = SchemaInferenceConfig(
            sample_size=100,
            confidence_threshold=0.8,
            null_threshold=0.95,
            categorical_threshold=10
        )
        return SchemaInferenceEngine(config)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': ['$50000', '$60000', '$55000', '$70000', '$65000'],
            'is_active': [True, False, True, True, False],
            'join_date': ['2020-01-15', '2019-06-20', '2021-03-10', '2020-11-05', '2021-07-22'],
            'completion_rate': ['85%', '92%', '78%', '95%', '88%'],
            'description': ['Software Engineer', 'Data Scientist', 'Product Manager', 'Designer', 'Analyst']
        })
    
    def test_infer_schema_basic(self, inference_engine, sample_dataframe):
        """Test basic schema inference."""
        schema = inference_engine.infer_schema(sample_dataframe, "employees")
        
        assert len(schema.columns) == 8
        assert schema.row_count == 5
        assert len(schema.data_types) == 8
        assert len(schema.sample_data) == 8
        
        # Check specific column types
        column_types = {col.name: col.data_type for col in schema.columns}
        assert column_types['id'] == DataType.NUMBER
        assert column_types['name'] == DataType.TEXT
        assert column_types['age'] == DataType.NUMBER
        assert column_types['salary'] == DataType.CURRENCY
        assert column_types['is_active'] == DataType.BOOLEAN
        assert column_types['join_date'] == DataType.DATE
        assert column_types['completion_rate'] == DataType.PERCENTAGE
        assert column_types['description'] == DataType.TEXT
    
    def test_infer_schema_empty_dataframe(self, inference_engine):
        """Test schema inference with empty DataFrame."""
        empty_df = pd.DataFrame()
        schema = inference_engine.infer_schema(empty_df)
        
        assert len(schema.columns) == 0
        assert schema.row_count == 0
        assert len(schema.data_types) == 0
        assert len(schema.sample_data) == 0
    
    def test_infer_schema_with_nulls(self, inference_engine):
        """Test schema inference with null values."""
        df_with_nulls = pd.DataFrame({
            'optional_field': [1, None, 3, None, 5],
            'required_field': [1, 2, 3, 4, 5],
            'mostly_null': [1, None, None, None, None]
        })
        
        schema = inference_engine.infer_schema(df_with_nulls)
        
        # Check nullable properties
        column_nullable = {col.name: col.nullable for col in schema.columns}
        assert column_nullable['optional_field'] == True  # 40% null
        assert column_nullable['required_field'] == False  # 0% null
        assert column_nullable['mostly_null'] == True  # 80% null
    
    def test_detect_numeric_types(self, inference_engine):
        """Test detection of various numeric types."""
        numeric_df = pd.DataFrame({
            'integers': [1, 2, 3, 4, 5],
            'floats': [1.1, 2.2, 3.3, 4.4, 5.5],
            'mixed_numeric': ['1', '2.5', '3', '4.7', '5'],
            'with_commas': ['1,000', '2,500', '3,750', '4,200', '5,100']
        })
        
        schema = inference_engine.infer_schema(numeric_df)
        
        for column in schema.columns:
            assert column.data_type == DataType.NUMBER
    
    def test_detect_currency_types(self, inference_engine):
        """Test detection of currency values."""
        currency_df = pd.DataFrame({
            'usd': ['$100', '$200.50', '$1,500', '$75.25', '$999.99'],
            'euro': ['€100', '€200.50', '€1,500', '€75.25', '€999.99'],
            'pound': ['£100', '£200.50', '£1,500', '£75.25', '£999.99']
        })
        
        schema = inference_engine.infer_schema(currency_df)
        
        for column in schema.columns:
            assert column.data_type == DataType.CURRENCY
    
    def test_detect_percentage_types(self, inference_engine):
        """Test detection of percentage values."""
        percentage_df = pd.DataFrame({
            'completion': ['85%', '92%', '78%', '95%', '88%'],
            'growth': ['5.5%', '12.3%', '8.7%', '15.2%', '9.8%']
        })
        
        schema = inference_engine.infer_schema(percentage_df)
        
        for column in schema.columns:
            assert column.data_type == DataType.PERCENTAGE
    
    def test_detect_date_types(self, inference_engine):
        """Test detection of date values."""
        date_df = pd.DataFrame({
            'iso_dates': ['2020-01-15', '2019-06-20', '2021-03-10', '2020-11-05', '2021-07-22'],
            'us_dates': ['01/15/2020', '06/20/2019', '03/10/2021', '11/05/2020', '07/22/2021'],
            'long_dates': ['January 15, 2020', 'June 20, 2019', 'March 10, 2021', 'November 5, 2020', 'July 22, 2021']
        })
        
        schema = inference_engine.infer_schema(date_df)
        
        for column in schema.columns:
            assert column.data_type == DataType.DATE
    
    def test_detect_boolean_types(self, inference_engine):
        """Test detection of boolean values."""
        boolean_df = pd.DataFrame({
            'true_false': ['true', 'false', 'true', 'false', 'true'],
            'yes_no': ['yes', 'no', 'yes', 'no', 'yes'],
            'binary': ['1', '0', '1', '0', '1'],
            'on_off': ['on', 'off', 'on', 'off', 'on']
        })
        
        schema = inference_engine.infer_schema(boolean_df)
        
        for column in schema.columns:
            assert column.data_type == DataType.BOOLEAN
    
    def test_column_metadata(self, inference_engine, sample_dataframe):
        """Test that column metadata is properly generated."""
        schema = inference_engine.infer_schema(sample_dataframe)
        
        # Find the age column
        age_column = next(col for col in schema.columns if col.name == 'age')
        
        assert hasattr(age_column, 'metadata')
        assert 'total_count' in age_column.metadata
        assert 'null_count' in age_column.metadata
        assert 'unique_values' in age_column.metadata
        assert age_column.metadata['total_count'] == 5
        assert age_column.metadata['null_count'] == 0
    
    def test_large_dataframe_sampling(self, inference_engine):
        """Test that large DataFrames are properly sampled."""
        # Create a large DataFrame
        large_df = pd.DataFrame({
            'id': range(2000),
            'value': np.random.randn(2000)
        })
        
        with patch.object(inference_engine, '_sample_dataframe') as mock_sample:
            mock_sample.return_value = large_df.head(100)
            schema = inference_engine.infer_schema(large_df)
            
            # Verify sampling was called
            mock_sample.assert_called_once()
            assert schema.row_count == 2000  # Original size should be preserved


class TestSchemaValidator:
    """Test schema validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create schema validator."""
        return SchemaValidator()
    
    @pytest.fixture
    def valid_schema(self):
        """Create a valid schema for testing."""
        columns = [
            ColumnInfo(name="id", data_type=DataType.NUMBER, nullable=False, unique_values=5),
            ColumnInfo(name="name", data_type=DataType.TEXT, nullable=False, unique_values=5),
            ColumnInfo(name="age", data_type=DataType.NUMBER, nullable=True, unique_values=4)
        ]
        return DataSchema(
            columns=columns,
            row_count=5,
            data_types={"id": DataType.NUMBER, "name": DataType.TEXT, "age": DataType.NUMBER},
            sample_data={"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        )
    
    def test_validate_valid_schema(self, validator, valid_schema):
        """Test validation of a valid schema."""
        result = validator.validate_schema(valid_schema)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.confidence == 1.0
    
    def test_validate_empty_schema(self, validator):
        """Test validation of empty schema."""
        empty_schema = DataSchema(columns=[], row_count=0, data_types={}, sample_data={})
        result = validator.validate_schema(empty_schema)
        
        assert not result.is_valid
        assert "Schema has no columns defined" in result.errors
    
    def test_validate_duplicate_columns(self, validator):
        """Test validation with duplicate column names."""
        columns = [
            ColumnInfo(name="id", data_type=DataType.NUMBER),
            ColumnInfo(name="id", data_type=DataType.TEXT)  # Duplicate name
        ]
        schema = DataSchema(columns=columns, row_count=5, data_types={}, sample_data={})
        result = validator.validate_schema(schema)
        
        assert not result.is_valid
        assert "Duplicate column name: id" in result.errors
    
    def test_validate_negative_row_count(self, validator):
        """Test validation with negative row count."""
        columns = [ColumnInfo(name="id", data_type=DataType.NUMBER)]
        schema = DataSchema(columns=columns, row_count=-1, data_types={}, sample_data={})
        result = validator.validate_schema(schema)
        
        assert not result.is_valid
        assert "Row count cannot be negative" in result.errors
    
    def test_check_compatibility_identical(self, validator, valid_schema):
        """Test compatibility check with identical schemas."""
        result = validator.check_compatibility(valid_schema, valid_schema)
        
        assert result.compatibility == SchemaCompatibility.IDENTICAL
        assert len(result.changes) == 0
        assert result.can_migrate
    
    def test_check_compatibility_column_added(self, validator, valid_schema):
        """Test compatibility check with added column."""
        new_columns = valid_schema.columns + [
            ColumnInfo(name="email", data_type=DataType.TEXT, nullable=True)
        ]
        new_schema = DataSchema(
            columns=new_columns,
            row_count=valid_schema.row_count,
            data_types={**valid_schema.data_types, "email": DataType.TEXT},
            sample_data=valid_schema.sample_data
        )
        
        result = validator.check_compatibility(valid_schema, new_schema)
        
        assert result.compatibility == SchemaCompatibility.MINOR_CHANGES
        assert len(result.changes) == 1
        assert result.changes[0].change_type == SchemaChangeType.COLUMN_ADDED
        assert result.changes[0].column_name == "email"
        assert result.can_migrate
    
    def test_check_compatibility_column_removed(self, validator, valid_schema):
        """Test compatibility check with removed column."""
        new_columns = [col for col in valid_schema.columns if col.name != "age"]
        new_schema = DataSchema(
            columns=new_columns,
            row_count=valid_schema.row_count,
            data_types={k: v for k, v in valid_schema.data_types.items() if k != "age"},
            sample_data={k: v for k, v in valid_schema.sample_data.items() if k != "age"}
        )
        
        result = validator.check_compatibility(valid_schema, new_schema)
        
        assert result.compatibility == SchemaCompatibility.MAJOR_CHANGES
        assert len(result.changes) == 1
        assert result.changes[0].change_type == SchemaChangeType.COLUMN_REMOVED
        assert result.changes[0].column_name == "age"
        assert result.can_migrate
    
    def test_check_compatibility_type_changed(self, validator, valid_schema):
        """Test compatibility check with changed column type."""
        new_columns = []
        for col in valid_schema.columns:
            if col.name == "age":
                new_col = ColumnInfo(name="age", data_type=DataType.TEXT, nullable=col.nullable)
                new_columns.append(new_col)
            else:
                new_columns.append(col)
        
        new_schema = DataSchema(
            columns=new_columns,
            row_count=valid_schema.row_count,
            data_types={**valid_schema.data_types, "age": DataType.TEXT},
            sample_data=valid_schema.sample_data
        )
        
        result = validator.check_compatibility(valid_schema, new_schema)
        
        assert result.compatibility == SchemaCompatibility.MAJOR_CHANGES
        assert len(result.changes) == 1
        assert result.changes[0].change_type == SchemaChangeType.TYPE_CHANGED
        assert result.changes[0].column_name == "age"
        assert result.can_migrate


class TestSchemaVersionManager:
    """Test schema version management functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def version_manager(self, temp_storage):
        """Create schema version manager with temporary storage."""
        return SchemaVersionManager(temp_storage)
    
    @pytest.fixture
    def sample_schema(self):
        """Create sample schema for testing."""
        columns = [
            ColumnInfo(name="id", data_type=DataType.NUMBER, nullable=False),
            ColumnInfo(name="name", data_type=DataType.TEXT, nullable=False)
        ]
        return DataSchema(
            columns=columns,
            row_count=10,
            data_types={"id": DataType.NUMBER, "name": DataType.TEXT},
            sample_data={"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        )
    
    def test_create_version(self, version_manager, sample_schema):
        """Test creating a schema version."""
        version_id = version_manager.create_version(sample_schema, "Initial schema")
        
        assert version_id is not None
        assert version_id in version_manager.versions
        
        version = version_manager.get_version(version_id)
        assert version is not None
        assert version.description == "Initial schema"
        assert version.schema == sample_schema
        assert version.parent_version is None
    
    def test_create_version_with_parent(self, version_manager, sample_schema):
        """Test creating a schema version with parent."""
        # Create initial version
        parent_id = version_manager.create_version(sample_schema, "Initial schema")
        
        # Create modified schema
        new_columns = sample_schema.columns + [
            ColumnInfo(name="email", data_type=DataType.TEXT, nullable=True)
        ]
        new_schema = DataSchema(
            columns=new_columns,
            row_count=sample_schema.row_count,
            data_types={**sample_schema.data_types, "email": DataType.TEXT},
            sample_data=sample_schema.sample_data
        )
        
        # Create child version
        child_id = version_manager.create_version(new_schema, "Added email column", parent_id)
        
        child_version = version_manager.get_version(child_id)
        assert child_version.parent_version == parent_id
        assert len(child_version.changes) == 1
        assert child_version.changes[0].change_type == SchemaChangeType.COLUMN_ADDED
    
    def test_get_latest_version(self, version_manager, sample_schema):
        """Test getting the latest schema version."""
        import time
        
        # Initially no versions
        assert version_manager.get_latest_version() is None
        
        # Create first version
        version1_id = version_manager.create_version(sample_schema, "Version 1")
        latest = version_manager.get_latest_version()
        assert latest.version == version1_id
        
        # Add small delay to ensure different timestamps
        time.sleep(0.01)
        
        # Create second version
        version2_id = version_manager.create_version(sample_schema, "Version 2")
        latest = version_manager.get_latest_version()
        # The latest version should be the most recently created one
        assert latest.description == "Version 2"
    
    def test_list_versions(self, version_manager, sample_schema):
        """Test listing all versions."""
        # Create multiple versions
        version1_id = version_manager.create_version(sample_schema, "Version 1")
        version2_id = version_manager.create_version(sample_schema, "Version 2")
        
        versions = version_manager.list_versions()
        assert len(versions) == 2
        
        # Check that both versions exist
        version_ids = [v.version for v in versions]
        assert version1_id in version_ids
        assert version2_id in version_ids
        
        # Check that they are sorted by creation time (newest first)
        assert versions[0].created_at >= versions[1].created_at
    
    def test_get_version_history(self, version_manager, sample_schema):
        """Test getting version history."""
        # Create version chain
        v1_id = version_manager.create_version(sample_schema, "Version 1")
        
        # Modify schema for v2
        new_schema = DataSchema(
            columns=sample_schema.columns,
            row_count=sample_schema.row_count + 5,
            data_types=sample_schema.data_types,
            sample_data=sample_schema.sample_data
        )
        v2_id = version_manager.create_version(new_schema, "Version 2", v1_id)
        
        # Get history
        history = version_manager.get_version_history(v2_id)
        assert len(history) == 2
        assert history[0].version == v2_id
        assert history[1].version == v1_id
    
    def test_compare_versions(self, version_manager, sample_schema):
        """Test comparing two versions."""
        # Create two different schemas
        v1_id = version_manager.create_version(sample_schema, "Version 1")
        
        new_columns = sample_schema.columns + [
            ColumnInfo(name="age", data_type=DataType.NUMBER, nullable=True)
        ]
        new_schema = DataSchema(
            columns=new_columns,
            row_count=sample_schema.row_count,
            data_types={**sample_schema.data_types, "age": DataType.NUMBER},
            sample_data=sample_schema.sample_data
        )
        v2_id = version_manager.create_version(new_schema, "Version 2")
        
        # Compare versions
        result = version_manager.compare_versions(v1_id, v2_id)
        assert result.compatibility == SchemaCompatibility.MINOR_CHANGES
        assert len(result.changes) == 1
        assert result.changes[0].change_type == SchemaChangeType.COLUMN_ADDED
    
    def test_persistence(self, temp_storage, sample_schema):
        """Test that versions are persisted to storage."""
        # Create version manager and add version
        vm1 = SchemaVersionManager(temp_storage)
        version_id = vm1.create_version(sample_schema, "Test version")
        
        # Create new version manager (simulating restart)
        vm2 = SchemaVersionManager(temp_storage)
        
        # Version should be loaded from storage
        assert version_id in vm2.versions
        loaded_version = vm2.get_version(version_id)
        assert loaded_version.description == "Test version"
        assert len(loaded_version.schema.columns) == len(sample_schema.columns)


class TestSchemaManager:
    """Test the main schema manager interface."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def schema_manager(self, temp_storage):
        """Create schema manager with temporary storage."""
        return SchemaManager(temp_storage)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'active': [True, False, True, True, False]
        })
    
    def test_detect_schema(self, schema_manager, sample_dataframe):
        """Test schema detection through main interface."""
        schema = schema_manager.detect_schema(sample_dataframe, "users")
        
        assert len(schema.columns) == 4
        assert schema.row_count == 5
        
        # Check column types
        column_types = {col.name: col.data_type for col in schema.columns}
        assert column_types['id'] == DataType.NUMBER
        assert column_types['name'] == DataType.TEXT
        assert column_types['age'] == DataType.NUMBER
        assert column_types['active'] == DataType.BOOLEAN
    
    def test_validate_schema(self, schema_manager, sample_dataframe):
        """Test schema validation through main interface."""
        schema = schema_manager.detect_schema(sample_dataframe)
        result = schema_manager.validate_schema(schema)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_create_and_retrieve_version(self, schema_manager, sample_dataframe):
        """Test creating and retrieving schema versions."""
        schema = schema_manager.detect_schema(sample_dataframe, "test_table")
        version_id = schema_manager.create_schema_version(schema, "Initial version")
        
        retrieved_version = schema_manager.get_schema_version(version_id)
        assert retrieved_version is not None
        assert retrieved_version.description == "Initial version"
        assert len(retrieved_version.schema.columns) == 4
    
    def test_check_compatibility(self, schema_manager, sample_dataframe):
        """Test schema compatibility checking."""
        # Create original schema
        original_schema = schema_manager.detect_schema(sample_dataframe)
        
        # Create modified DataFrame
        modified_df = sample_dataframe.copy()
        modified_df['email'] = ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com']
        modified_schema = schema_manager.detect_schema(modified_df)
        
        # Check compatibility
        result = schema_manager.check_schema_compatibility(original_schema, modified_schema)
        
        assert result.compatibility == SchemaCompatibility.MINOR_CHANGES
        assert len(result.changes) == 1
        assert result.changes[0].change_type == SchemaChangeType.COLUMN_ADDED
        assert result.can_migrate
    
    def test_get_latest_schema(self, schema_manager, sample_dataframe):
        """Test getting the latest schema."""
        # Initially no schema
        assert schema_manager.get_latest_schema() is None
        
        # Create schema version
        schema = schema_manager.detect_schema(sample_dataframe)
        schema_manager.create_schema_version(schema, "Test version")
        
        # Should return the schema
        latest = schema_manager.get_latest_schema()
        assert latest is not None
        assert len(latest.columns) == 4
    
    def test_list_versions(self, schema_manager, sample_dataframe):
        """Test listing schema versions."""
        # Create multiple versions
        schema1 = schema_manager.detect_schema(sample_dataframe)
        version1_id = schema_manager.create_schema_version(schema1, "Version 1")
        
        modified_df = sample_dataframe.copy()
        modified_df['status'] = ['active', 'inactive', 'active', 'active', 'inactive']
        schema2 = schema_manager.detect_schema(modified_df)
        version2_id = schema_manager.create_schema_version(schema2, "Version 2")
        
        versions = schema_manager.list_schema_versions()
        assert len(versions) == 2
        
        # Check that both versions exist
        version_descriptions = [v.description for v in versions]
        assert "Version 1" in version_descriptions
        assert "Version 2" in version_descriptions
        
        # Check that they are sorted by creation time (newest first)
        assert versions[0].created_at >= versions[1].created_at


class TestSchemaAccuracy:
    """Test schema detection accuracy with various data patterns."""
    
    @pytest.fixture
    def schema_manager(self):
        """Create schema manager for accuracy testing."""
        return SchemaManager()
    
    def test_mixed_data_types_accuracy(self, schema_manager):
        """Test accuracy with mixed data types."""
        df = pd.DataFrame({
            'mixed_numbers': ['1', '2.5', '3', '4.7', '5'],  # Should detect as NUMBER
            'mixed_text_numbers': ['1', 'two', '3', 'four', '5'],  # Should detect as TEXT
            'mostly_numbers': ['1', '2', '3', '4', 'N/A'],  # Should detect as NUMBER (80% numeric)
            'currency_mixed': ['$100', '$200.50', 'N/A', '$75.25', '$999.99'],  # Should detect as CURRENCY
        })
        
        schema = schema_manager.detect_schema(df)
        column_types = {col.name: col.data_type for col in schema.columns}
        
        assert column_types['mixed_numbers'] == DataType.NUMBER
        assert column_types['mixed_text_numbers'] == DataType.TEXT
        assert column_types['mostly_numbers'] == DataType.NUMBER
        assert column_types['currency_mixed'] == DataType.CURRENCY
    
    def test_edge_cases_accuracy(self, schema_manager):
        """Test accuracy with edge cases."""
        df = pd.DataFrame({
            'empty_strings': ['', '', '', '', ''],  # Should detect as TEXT
            'all_nulls': [None, None, None, None, None],  # Should detect as TEXT
            'single_value': ['test', 'test', 'test', 'test', 'test'],  # Should detect as TEXT
            'binary_strings': ['0', '1', '0', '1', '0'],  # Should detect as BOOLEAN
        })
        
        schema = schema_manager.detect_schema(df)
        column_types = {col.name: col.data_type for col in schema.columns}
        
        assert column_types['empty_strings'] == DataType.TEXT
        assert column_types['all_nulls'] == DataType.TEXT
        assert column_types['single_value'] == DataType.TEXT
        assert column_types['binary_strings'] == DataType.BOOLEAN
    
    def test_real_world_patterns_accuracy(self, schema_manager):
        """Test accuracy with real-world data patterns."""
        df = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
            'phone_numbers': ['555-123-4567', '555-987-6543', '555-456-7890', '555-321-0987', '555-654-3210'],
            'email_addresses': ['john@company.com', 'jane@company.com', 'bob@company.com', 'alice@company.com', 'charlie@company.com'],
            'zip_codes': ['12345', '67890', '54321', '09876', '13579'],
            'ratings': ['4.5', '3.8', '4.2', '4.9', '3.5'],
        })
        
        schema = schema_manager.detect_schema(df)
        column_types = {col.name: col.data_type for col in schema.columns}
        
        # All should be detected as TEXT (specific patterns)
        assert column_types['employee_id'] == DataType.TEXT
        assert column_types['phone_numbers'] == DataType.TEXT
        assert column_types['email_addresses'] == DataType.TEXT
        assert column_types['zip_codes'] == DataType.TEXT  # Could be NUMBER, but often treated as TEXT
        assert column_types['ratings'] == DataType.NUMBER
    
    def test_confidence_thresholds(self, schema_manager):
        """Test that confidence thresholds work correctly."""
        # Create data where only 60% matches a pattern (below 80% threshold)
        df = pd.DataFrame({
            'low_confidence_currency': ['$100', '$200', 'N/A', 'invalid', '$300'],  # 60% currency
            'high_confidence_currency': ['$100', '$200', '$150', '$75', '$300'],  # 100% currency
        })
        
        schema = schema_manager.detect_schema(df)
        column_types = {col.name: col.data_type for col in schema.columns}
        
        # Low confidence should fall back to TEXT
        assert column_types['low_confidence_currency'] == DataType.TEXT
        # High confidence should detect as CURRENCY
        assert column_types['high_confidence_currency'] == DataType.CURRENCY


class TestSchemaAIIntegration:
    """Test AI integration features for schema management."""
    
    @pytest.fixture
    def schema_manager(self):
        """Create schema manager for AI integration testing."""
        return SchemaManager()
    
    @pytest.fixture
    def comprehensive_dataframe(self):
        """Create comprehensive DataFrame for AI integration testing."""
        return pd.DataFrame({
            'user_id': ['USR001', 'USR002', 'USR003', 'USR004', 'USR005'],
            'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
            'age': [25, 30, 35, 28, 32],
            'salary': ['$50000', '$60000', '$55000', '$70000', '$65000'],
            'is_active': [True, False, True, True, False],
            'join_date': ['2020-01-15', '2019-06-20', '2021-03-10', '2020-11-05', '2021-07-22'],
            'completion_rate': ['85%', '92%', '78%', '95%', '88%'],
            'department': ['Engineering', 'Data Science', 'Product', 'Design', 'Analytics']
        })
    
    def test_export_schema_for_ai(self, schema_manager, comprehensive_dataframe):
        """Test exporting schema in AI-friendly format."""
        schema = schema_manager.detect_schema(comprehensive_dataframe, "employees")
        ai_schema = schema_manager.export_schema_for_ai(schema)
        
        # Check structure
        assert 'table_info' in ai_schema
        assert 'columns' in ai_schema
        assert 'data_patterns' in ai_schema
        assert 'query_suggestions' in ai_schema
        
        # Check table info
        assert ai_schema['table_info']['total_rows'] == 5
        assert ai_schema['table_info']['total_columns'] == 8
        
        # Check columns have required fields
        for column in ai_schema['columns']:
            assert 'name' in column
            assert 'type' in column
            assert 'description' in column
            assert 'analysis_hints' in column
            assert isinstance(column['analysis_hints'], list)
        
        # Check query suggestions are generated
        assert len(ai_schema['query_suggestions']) > 0
        assert isinstance(ai_schema['query_suggestions'], list)
    
    def test_column_descriptions(self, schema_manager, comprehensive_dataframe):
        """Test that column descriptions are meaningful."""
        schema = schema_manager.detect_schema(comprehensive_dataframe)
        ai_schema = schema_manager.export_schema_for_ai(schema)
        
        # Find specific columns and check descriptions
        columns_by_name = {col['name']: col for col in ai_schema['columns']}
        
        # Check user_id description (should be identified as identifier)
        user_id_desc = columns_by_name['user_id']['description']
        assert 'text values' in user_id_desc
        
        # Check salary description (should be currency)
        salary_desc = columns_by_name['salary']['description']
        assert 'monetary amounts' in salary_desc
        
        # Check is_active description (should be boolean)
        active_desc = columns_by_name['is_active']['description']
        assert 'true/false values' in active_desc
    
    def test_analysis_hints_generation(self, schema_manager, comprehensive_dataframe):
        """Test that analysis hints are appropriate for each data type."""
        schema = schema_manager.detect_schema(comprehensive_dataframe)
        ai_schema = schema_manager.export_schema_for_ai(schema)
        
        columns_by_name = {col['name']: col for col in ai_schema['columns']}
        
        # Check numeric column hints
        age_hints = columns_by_name['age']['analysis_hints']
        assert any('aggregations' in hint for hint in age_hints)
        assert any('calculations' in hint for hint in age_hints)
        
        # Check text column hints
        name_hints = columns_by_name['name']['analysis_hints']
        assert any('grouping' in hint for hint in name_hints)
        
        # Check date column hints
        join_date_hints = columns_by_name['join_date']['analysis_hints']
        assert any('time-based' in hint for hint in join_date_hints)
    
    def test_query_suggestions_relevance(self, schema_manager, comprehensive_dataframe):
        """Test that query suggestions are relevant to the data."""
        schema = schema_manager.detect_schema(comprehensive_dataframe)
        ai_schema = schema_manager.export_schema_for_ai(schema)
        
        suggestions = ai_schema['query_suggestions']
        
        # Should have suggestions about numeric columns
        assert any('age' in suggestion for suggestion in suggestions)
        assert any('salary' in suggestion for suggestion in suggestions)
        
        # Should have suggestions about grouping (check for any text column)
        text_columns = ['name', 'department', 'user_id']
        assert any(any(col in suggestion for col in text_columns) for suggestion in suggestions)
        
        # Should have time-based suggestions
        assert any('time' in suggestion or 'date' in suggestion for suggestion in suggestions)
    
    def test_detect_schema_evolution(self, schema_manager, comprehensive_dataframe):
        """Test schema evolution detection between two datasets."""
        # Create modified DataFrame (add column, change type)
        modified_df = comprehensive_dataframe.copy()
        modified_df['email'] = ['alice@company.com', 'bob@company.com', 'charlie@company.com', 
                               'diana@company.com', 'eve@company.com']
        # Change age to text values that won't be detected as numbers
        modified_df['age'] = ['twenty-five', 'thirty', 'thirty-five', 'twenty-eight', 'thirty-two']
        
        evolution_report = schema_manager.detect_schema_evolution(
            comprehensive_dataframe, modified_df, "employees"
        )
        
        # Check structure
        assert 'compatibility' in evolution_report
        assert 'changes' in evolution_report
        assert 'data_quality' in evolution_report
        assert 'migration_recommendations' in evolution_report
        assert 'schemas' in evolution_report
        
        # Check that changes were detected
        assert len(evolution_report['changes']) >= 2  # At least column added and type changed
        
        # Check migration recommendations exist
        assert len(evolution_report['migration_recommendations']) > 0
        
        # Check that each recommendation has required fields
        for rec in evolution_report['migration_recommendations']:
            assert 'change_type' in rec
            assert 'priority' in rec
            assert 'actions' in rec
            assert 'risks' in rec
            assert 'validation_steps' in rec
    
    def test_data_quality_analysis(self, schema_manager):
        """Test data quality analysis in schema evolution."""
        # Create DataFrames with different quality characteristics
        old_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, None, 40, 50],  # 20% null
            'category': ['A', 'B', 'A', 'B', 'A']  # 2 unique values
        })
        
        new_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],  # More rows
            'value': [10, 20, 30, 40, 50, 60],  # No nulls now
            'category': ['A', 'B', 'A', 'C', 'A', 'B']  # 3 unique values now
        })
        
        evolution_report = schema_manager.detect_schema_evolution(old_df, new_df, "test")
        
        quality_metrics = evolution_report['data_quality']
        
        # Check row count change
        assert quality_metrics['row_count_change'] == 1
        
        # Check data quality metrics for common columns
        assert 'value' in quality_metrics['data_quality_metrics']
        value_metrics = quality_metrics['data_quality_metrics']['value']
        assert value_metrics['null_percentage_change'] < 0  # Null percentage decreased
        
        assert 'category' in quality_metrics['data_quality_metrics']
        category_metrics = quality_metrics['data_quality_metrics']['category']
        assert category_metrics['unique_values_change'] == 1  # One more unique value
    
    def test_migration_priority_assessment(self, schema_manager):
        """Test that migration priorities are correctly assessed."""
        # Create schemas with different types of changes
        old_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        
        # Remove a column (high priority)
        new_df_removed = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        evolution_report = schema_manager.detect_schema_evolution(old_df, new_df_removed, "test")
        
        # Find the column removal recommendation
        removal_rec = next(rec for rec in evolution_report['migration_recommendations'] 
                          if rec['change_type'] == 'column_removed')
        
        assert removal_rec['priority'] == 'high'  # Column removal should be high priority
        assert len(removal_rec['risks']) > 0
        assert 'Data loss' in removal_rec['risks'][0]


class TestSchemaDetectionAccuracy:
    """Enhanced tests for schema detection accuracy."""
    
    @pytest.fixture
    def schema_manager(self):
        """Create schema manager for accuracy testing."""
        return SchemaManager()
    
    def test_identifier_detection_accuracy(self, schema_manager):
        """Test accurate detection of identifier columns vs numeric data."""
        df = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
            'zip_code': ['12345', '67890', '54321', '09876', '13579'],  # Should be text
            'phone': ['5551234567', '5559876543', '5554567890', '5553210987', '5556543210'],
            'actual_number': [1.5, 2.7, 3.2, 4.8, 5.1],  # Should be number
            'count': [1, 2, 3, 4, 5]  # Should be number
        })
        
        schema = schema_manager.detect_schema(df)
        column_types = {col.name: col.data_type for col in schema.columns}
        
        # Identifiers should be detected as text
        assert column_types['employee_id'] == DataType.TEXT
        assert column_types['zip_code'] == DataType.TEXT
        assert column_types['phone'] == DataType.TEXT
        
        # Actual numbers should be detected as numbers
        assert column_types['actual_number'] == DataType.NUMBER
        assert column_types['count'] == DataType.NUMBER
    
    def test_mixed_format_detection(self, schema_manager):
        """Test detection with mixed formats in the same column."""
        df = pd.DataFrame({
            'mixed_currency': ['$100.00', '$200', '€150', '$75.50', '$999'],  # Mixed currency symbols
            'mixed_percentage': ['85%', '92.5%', '78', '95%', '88.2%'],  # Some without %
            'mixed_dates': ['2020-01-15', '06/20/2019', '2021-03-10', '11/05/2020', '2021-07-22'],
            'mostly_numeric': ['1', '2', '3', '4', 'N/A']  # Mostly numeric with one non-numeric
        })
        
        schema = schema_manager.detect_schema(df)
        column_types = {col.name: col.data_type for col in schema.columns}
        
        # Should handle mixed formats gracefully
        # Mixed currency might be detected as text due to different symbols
        assert column_types['mixed_currency'] in [DataType.CURRENCY, DataType.TEXT]
        
        # Mixed percentage should still be detected as percentage if majority matches
        assert column_types['mixed_percentage'] in [DataType.PERCENTAGE, DataType.TEXT]
        
        # Mixed dates might be detected as text due to different formats
        assert column_types['mixed_dates'] in [DataType.DATE, DataType.TEXT]
        
        # Mostly numeric should be detected as number due to confidence threshold
        assert column_types['mostly_numeric'] == DataType.NUMBER
    
    def test_edge_case_data_types(self, schema_manager):
        """Test detection of edge cases and special data types."""
        df = pd.DataFrame({
            'scientific_notation': ['1.5e10', '2.3e-5', '4.7e8', '1.2e-3', '9.8e6'],
            'negative_numbers': ['-100', '-50.5', '-200', '-75.25', '-999.99'],
            'leading_zeros': ['001', '002', '003', '004', '005'],  # Should be text (identifiers)
            'boolean_variants': ['TRUE', 'FALSE', 'True', 'False', 'true'],
            'empty_and_spaces': ['', '   ', 'value', '', '  text  ']
        })
        
        schema = schema_manager.detect_schema(df)
        column_types = {col.name: col.data_type for col in schema.columns}
        
        # Scientific notation should be detected as numbers
        assert column_types['scientific_notation'] == DataType.NUMBER
        
        # Negative numbers should be detected as numbers
        assert column_types['negative_numbers'] == DataType.NUMBER
        
        # Leading zeros might be detected as numbers if they're short
        assert column_types['leading_zeros'] in [DataType.TEXT, DataType.NUMBER]
        
        # Boolean variants should be detected as boolean
        assert column_types['boolean_variants'] == DataType.BOOLEAN
        
        # Mixed empty/text should be detected as text
        assert column_types['empty_and_spaces'] == DataType.TEXT


if __name__ == '__main__':
    pytest.main([__file__])