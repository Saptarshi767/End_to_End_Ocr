"""
Tests for privacy manager
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, mock_open
import json

from src.security.privacy_manager import (
    PrivacyManager, DataRecord, DataClassification, 
    RetentionPolicy, PIIPattern
)


class TestPrivacyManager:
    """Test privacy manager"""
    
    @pytest.fixture
    def privacy_manager(self):
        """Create privacy manager for testing"""
        with patch('src.security.privacy_manager.get_settings') as mock_settings:
            mock_settings.return_value.DATA_DIR = "/tmp/test_data"
            return PrivacyManager()
    
    def test_pii_detection_email(self, privacy_manager):
        """Test PII detection for email addresses"""
        text = "Contact us at john.doe@example.com or support@company.org"
        
        pii_detected = privacy_manager.detect_pii(text)
        
        email_pii = [p for p in pii_detected if p["type"] == "email"]
        assert len(email_pii) == 2
        assert "john.doe@example.com" in [p["value"] for p in email_pii]
        assert "support@company.org" in [p["value"] for p in email_pii]
    
    def test_pii_detection_phone(self, privacy_manager):
        """Test PII detection for phone numbers"""
        text = "Call us at 555-123-4567 or 555.987.6543"
        
        pii_detected = privacy_manager.detect_pii(text)
        
        phone_pii = [p for p in pii_detected if p["type"] == "phone"]
        assert len(phone_pii) == 2
        assert "555-123-4567" in [p["value"] for p in phone_pii]
        assert "555.987.6543" in [p["value"] for p in phone_pii]
    
    def test_pii_detection_ssn(self, privacy_manager):
        """Test PII detection for SSN"""
        text = "SSN: 123-45-6789"
        
        pii_detected = privacy_manager.detect_pii(text)
        
        ssn_pii = [p for p in pii_detected if p["type"] == "ssn"]
        assert len(ssn_pii) == 1
        assert ssn_pii[0]["value"] == "123-45-6789"
        assert ssn_pii[0]["confidence"] == 0.95
    
    def test_pii_detection_credit_card(self, privacy_manager):
        """Test PII detection for credit card numbers"""
        text = "Card number: 4532-1234-5678-9012"
        
        pii_detected = privacy_manager.detect_pii(text)
        
        card_pii = [p for p in pii_detected if p["type"] == "credit_card"]
        assert len(card_pii) == 1
        assert card_pii[0]["value"] == "4532-1234-5678-9012"
    
    def test_data_classification_with_pii(self, privacy_manager):
        """Test data classification with PII"""
        # High confidence PII should be RESTRICTED
        data_with_ssn = "Employee SSN: 123-45-6789"
        classification = privacy_manager.classify_data(data_with_ssn, "document")
        assert classification == DataClassification.RESTRICTED
        
        # Lower confidence PII should be CONFIDENTIAL
        data_with_name = "Employee: John Smith"
        classification = privacy_manager.classify_data(data_with_name, "document")
        assert classification == DataClassification.CONFIDENTIAL
    
    def test_data_classification_without_pii(self, privacy_manager):
        """Test data classification without PII"""
        # Regular document data should be INTERNAL
        data = "This is a regular business document"
        classification = privacy_manager.classify_data(data, "document")
        assert classification == DataClassification.INTERNAL
        
        # Other data types should be PUBLIC
        data = "Public information"
        classification = privacy_manager.classify_data(data, "public_info")
        assert classification == DataClassification.PUBLIC
    
    def test_data_anonymization_string(self, privacy_manager):
        """Test data anonymization for strings"""
        text = "Contact John Doe at john.doe@example.com or 555-123-4567"
        
        anonymized = privacy_manager.anonymize_data(text)
        
        assert "[NAME_REDACTED]" in anonymized
        assert "[EMAIL_REDACTED]" in anonymized
        assert "[PHONE_REDACTED]" in anonymized
        assert "john.doe@example.com" not in anonymized
        assert "555-123-4567" not in anonymized
    
    def test_data_anonymization_dict(self, privacy_manager):
        """Test data anonymization for dictionaries"""
        data = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
            "department": "Engineering"
        }
        
        anonymized = privacy_manager.anonymize_data(data)
        
        # Check that PII is anonymized but other data remains
        assert "[NAME_REDACTED]" in str(anonymized)
        assert "[EMAIL_REDACTED]" in str(anonymized)
        assert "[PHONE_REDACTED]" in str(anonymized)
        assert anonymized["department"] == "Engineering"
    
    def test_data_registration(self, privacy_manager):
        """Test data registration with privacy metadata"""
        data = {
            "employee_id": "EMP001",
            "name": "John Doe",
            "email": "john.doe@example.com"
        }
        
        record = privacy_manager.register_data(
            record_id="test_record_1",
            user_id="user_123",
            data=data,
            data_type="employee_data",
            retention_policy=RetentionPolicy.LONG_TERM,
            processing_purposes={"hr_management", "payroll"},
            consent_given=True
        )
        
        assert record.record_id == "test_record_1"
        assert record.user_id == "user_123"
        assert record.data_type == "employee_data"
        assert record.classification == DataClassification.RESTRICTED  # Due to PII
        assert record.retention_policy == RetentionPolicy.LONG_TERM
        assert record.pii_detected is True
        assert record.consent_given is True
        assert "hr_management" in record.processing_purposes
    
    def test_data_access_control(self, privacy_manager):
        """Test data access control"""
        # Register data
        record = privacy_manager.register_data(
            record_id="access_test",
            user_id="user_123",
            data="Sensitive information",
            data_type="document",
            processing_purposes={"analytics", "reporting"}
        )
        
        # Valid access
        access_granted = privacy_manager.access_data(
            "access_test",
            "user_123",
            "analytics"
        )
        assert access_granted is True
        
        # Invalid purpose
        access_denied = privacy_manager.access_data(
            "access_test",
            "user_123",
            "marketing"
        )
        assert access_denied is False
    
    def test_data_access_without_consent(self, privacy_manager):
        """Test data access without consent for PII"""
        # Register PII data without consent
        record = privacy_manager.register_data(
            record_id="pii_test",
            user_id="user_123",
            data="SSN: 123-45-6789",
            data_type="document",
            consent_given=False
        )
        
        # Access should be denied
        access_granted = privacy_manager.access_data(
            "pii_test",
            "user_123",
            "processing"
        )
        assert access_granted is False
    
    def test_expired_data_access(self, privacy_manager):
        """Test access to expired data"""
        # Register data with short retention
        record = privacy_manager.register_data(
            record_id="expired_test",
            user_id="user_123",
            data="Test data",
            data_type="document",
            retention_policy=RetentionPolicy.SHORT_TERM
        )
        
        # Manually set expiry to past
        record.expires_at = datetime.now() - timedelta(days=1)
        
        # Access should be denied
        access_granted = privacy_manager.access_data(
            "expired_test",
            "user_123",
            "processing"
        )
        assert access_granted is False
    
    def test_data_deletion(self, privacy_manager):
        """Test data deletion"""
        # Register data
        privacy_manager.register_data(
            record_id="delete_test",
            user_id="user_123",
            data="Data to delete",
            data_type="document"
        )
        
        # Delete data
        deleted = privacy_manager.delete_data(
            "delete_test",
            "user_123",
            "user_request"
        )
        
        assert deleted is True
        assert "delete_test" not in privacy_manager.data_records
    
    def test_expired_data_cleanup(self, privacy_manager):
        """Test automatic cleanup of expired data"""
        # Register multiple records with different expiry dates
        privacy_manager.register_data(
            record_id="expired_1",
            user_id="user_123",
            data="Expired data 1",
            data_type="document",
            retention_policy=RetentionPolicy.SHORT_TERM
        )
        
        privacy_manager.register_data(
            record_id="expired_2",
            user_id="user_123",
            data="Expired data 2",
            data_type="document",
            retention_policy=RetentionPolicy.SHORT_TERM
        )
        
        privacy_manager.register_data(
            record_id="valid",
            user_id="user_123",
            data="Valid data",
            data_type="document",
            retention_policy=RetentionPolicy.LONG_TERM
        )
        
        # Manually expire some records
        privacy_manager.data_records["expired_1"].expires_at = datetime.now() - timedelta(days=1)
        privacy_manager.data_records["expired_2"].expires_at = datetime.now() - timedelta(days=1)
        
        # Run cleanup
        cleanup_result = privacy_manager.cleanup_expired_data()
        
        assert cleanup_result["expired_records_found"] == 2
        assert cleanup_result["records_deleted"] == 2
        assert "expired_1" not in privacy_manager.data_records
        assert "expired_2" not in privacy_manager.data_records
        assert "valid" in privacy_manager.data_records
    
    def test_privacy_report_generation(self, privacy_manager):
        """Test privacy compliance report generation"""
        # Register various types of data
        privacy_manager.register_data(
            record_id="pii_data",
            user_id="user_123",
            data="SSN: 123-45-6789",
            data_type="document",
            retention_policy=RetentionPolicy.LONG_TERM
        )
        
        privacy_manager.register_data(
            record_id="regular_data",
            user_id="user_123",
            data="Regular business data",
            data_type="document",
            retention_policy=RetentionPolicy.MEDIUM_TERM
        )
        
        # Generate report
        report = privacy_manager.generate_privacy_report()
        
        assert report["summary"]["total_records"] == 2
        assert report["summary"]["pii_records"] == 1
        assert report["summary"]["encrypted_records"] == 1  # PII should be encrypted
        assert "classification_breakdown" in report
        assert "retention_breakdown" in report
        assert "compliance_status" in report
    
    def test_user_data_export(self, privacy_manager):
        """Test user data export for GDPR compliance"""
        # Register data for user
        privacy_manager.register_data(
            record_id="user_data_1",
            user_id="user_123",
            data="User's document 1",
            data_type="document"
        )
        
        privacy_manager.register_data(
            record_id="user_data_2",
            user_id="user_123",
            data="User's document 2",
            data_type="table"
        )
        
        # Register data for different user
        privacy_manager.register_data(
            record_id="other_user_data",
            user_id="user_456",
            data="Other user's data",
            data_type="document"
        )
        
        # Export data for user_123
        exported_data = privacy_manager.export_user_data("user_123")
        
        assert exported_data["user_id"] == "user_123"
        assert len(exported_data["records"]) == 2
        
        record_ids = [r["record_id"] for r in exported_data["records"]]
        assert "user_data_1" in record_ids
        assert "user_data_2" in record_ids
        assert "other_user_data" not in record_ids
    
    def test_pii_pattern_confidence_levels(self, privacy_manager):
        """Test PII pattern confidence levels"""
        # High confidence pattern (SSN)
        ssn_text = "SSN: 123-45-6789"
        ssn_pii = privacy_manager.detect_pii(ssn_text)
        ssn_pattern = next(p for p in ssn_pii if p["type"] == "ssn")
        assert ssn_pattern["confidence"] == 0.95
        
        # Medium confidence pattern (email)
        email_text = "Email: user@example.com"
        email_pii = privacy_manager.detect_pii(email_text)
        email_pattern = next(p for p in email_pii if p["type"] == "email")
        assert email_pattern["confidence"] == 0.9
        
        # Lower confidence pattern (name)
        name_text = "Employee: John Smith"
        name_pii = privacy_manager.detect_pii(name_text)
        name_pattern = next(p for p in name_pii if p["type"] == "name")
        assert name_pattern["confidence"] == 0.6
    
    @patch('builtins.open', mock_open(read_data='{}'))
    @patch('os.path.exists', return_value=True)
    def test_load_data_records(self, mock_exists, privacy_manager):
        """Test loading data records from storage"""
        # This tests the _load_data_records method indirectly
        # The mock_open returns empty JSON, so no records should be loaded
        assert len(privacy_manager.data_records) == 0
    
    def test_retention_policy_expiry_calculation(self, privacy_manager):
        """Test retention policy expiry calculation"""
        # Short term retention
        record_short = privacy_manager.register_data(
            record_id="short_term",
            user_id="user_123",
            data="Short term data",
            data_type="document",
            retention_policy=RetentionPolicy.SHORT_TERM
        )
        
        # Should expire in ~30 days
        days_until_expiry = (record_short.expires_at - datetime.now()).days
        assert 29 <= days_until_expiry <= 30
        
        # Permanent retention
        record_permanent = privacy_manager.register_data(
            record_id="permanent",
            user_id="user_123",
            data="Permanent data",
            data_type="document",
            retention_policy=RetentionPolicy.PERMANENT
        )
        
        # Should not have expiry date
        assert record_permanent.expires_at is None