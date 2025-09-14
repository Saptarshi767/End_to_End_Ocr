"""
Integration tests for security features
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.security.auth_manager import AuthManager, UserRole, Permission
from src.security.encryption_service import EncryptionService
from src.security.privacy_manager import PrivacyManager, DataClassification, RetentionPolicy
from src.security.audit_logger import AuditLogger
from src.security.session_manager import SessionManager
from src.core.exceptions import AuthenticationError, AuthorizationError, EncryptionError


class TestSecurityIntegration:
    """Integration tests for security components"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup handled by system
    
    @pytest.fixture
    def auth_manager(self):
        """Create auth manager for testing"""
        return AuthManager()
    
    @pytest.fixture
    def encryption_service(self, temp_dir):
        """Create encryption service for testing"""
        with patch('src.security.encryption_service.get_settings') as mock_settings:
            mock_settings.return_value.DATA_DIR = temp_dir
            return EncryptionService()
    
    @pytest.fixture
    def privacy_manager(self, temp_dir):
        """Create privacy manager for testing"""
        with patch('src.security.privacy_manager.get_settings') as mock_settings:
            mock_settings.return_value.DATA_DIR = temp_dir
            return PrivacyManager()
    
    @pytest.fixture
    def audit_logger(self, temp_dir):
        """Create audit logger for testing"""
        with patch('src.security.audit_logger.get_settings') as mock_settings:
            mock_settings.return_value.DATA_DIR = temp_dir
            return AuditLogger()
    
    @pytest.fixture
    def session_manager(self):
        """Create session manager for testing"""
        return SessionManager()
    
    def test_complete_user_lifecycle(self, auth_manager, audit_logger):
        """Test complete user lifecycle with security features"""
        
        # Create user
        user = auth_manager.create_user(
            email="test@example.com",
            password="SecurePass123!",
            name="Test User",
            role=UserRole.ANALYST
        )
        
        assert user.email == "test@example.com"
        assert user.role == UserRole.ANALYST
        assert Permission.DOCUMENT_READ in user.permissions
        
        # Authenticate user
        authenticated_user = auth_manager.authenticate_user(
            "test@example.com",
            "SecurePass123!",
            "192.168.1.1"
        )
        
        assert authenticated_user is not None
        assert authenticated_user.user_id == user.user_id
        
        # Create session
        session = auth_manager.create_session(
            authenticated_user,
            "192.168.1.1",
            "Mozilla/5.0"
        )
        
        assert session.user_id == user.user_id
        assert session.is_active
        
        # Generate tokens
        tokens = auth_manager.generate_token(authenticated_user, session.session_id)
        
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        
        # Verify token
        payload = auth_manager.verify_token(tokens["access_token"])
        assert payload["user_id"] == user.user_id
        
        # Create API key
        api_key_info = auth_manager.create_api_key(
            user.user_id,
            "Test API Key",
            {Permission.DOCUMENT_READ, Permission.DATA_EXPORT}
        )
        
        assert "api_key" in api_key_info
        
        # Verify API key
        api_key_obj = auth_manager.verify_api_key(api_key_info["api_key"])
        assert api_key_obj is not None
        assert api_key_obj.user_id == user.user_id
    
    def test_data_encryption_with_privacy(self, encryption_service, privacy_manager):
        """Test data encryption integrated with privacy management"""
        
        # Sensitive data with PII
        sensitive_data = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "email": "john.doe@example.com",
            "salary": 75000
        }
        
        # Register data with privacy manager
        record = privacy_manager.register_data(
            record_id="sensitive_employee_data",
            user_id="user_123",
            data=sensitive_data,
            data_type="employee_record",
            retention_policy=RetentionPolicy.LONG_TERM,
            consent_given=True
        )
        
        # Should be classified as restricted due to PII
        assert record.classification == DataClassification.RESTRICTED
        assert record.pii_detected is True
        assert record.is_encrypted is True
        
        # Encrypt the actual data
        encrypted_package = encryption_service.encrypt_data(sensitive_data)
        
        # Verify encryption
        assert "encrypted_data" in encrypted_package
        assert "metadata" in encrypted_package
        
        # Decrypt and verify
        decrypted_data = encryption_service.decrypt_data(encrypted_package)
        assert decrypted_data == sensitive_data
        
        # Test data anonymization
        anonymized_data = privacy_manager.anonymize_data(sensitive_data)
        anonymized_str = str(anonymized_data)
        
        # PII should be anonymized
        assert "123-45-6789" not in anonymized_str
        assert "john.doe@example.com" not in anonymized_str
        assert "[SSN_REDACTED]" in anonymized_str
        assert "[EMAIL_REDACTED]" in anonymized_str
        
        # Non-PII data should remain
        assert anonymized_data["salary"] == 75000
    
    def test_access_control_with_audit_logging(self, auth_manager, privacy_manager, audit_logger):
        """Test access control with comprehensive audit logging"""
        
        # Create users with different roles
        admin_user = auth_manager.create_user(
            "admin@example.com",
            "AdminPass123!",
            "Admin User",
            UserRole.ADMIN
        )
        
        viewer_user = auth_manager.create_user(
            "viewer@example.com",
            "ViewerPass123!",
            "Viewer User",
            UserRole.VIEWER
        )
        
        # Register sensitive data
        record = privacy_manager.register_data(
            record_id="confidential_doc",
            user_id=admin_user.user_id,
            data="Confidential business information",
            data_type="document",
            processing_purposes={"business_analysis", "reporting"}
        )
        
        # Admin should have access
        admin_access = privacy_manager.access_data(
            "confidential_doc",
            admin_user.user_id,
            "business_analysis"
        )
        assert admin_access is True
        
        # Viewer should not have access to admin's data
        viewer_access = privacy_manager.access_data(
            "confidential_doc",
            viewer_user.user_id,
            "business_analysis"
        )
        assert viewer_access is False
        
        # Check audit logs
        audit_events = audit_logger.get_audit_events(limit=10)
        
        # Should have events for user creation, data registration, and access attempts
        event_types = [event.event_type for event in audit_events]
        assert "user_created" in event_types
        assert "data_registered" in event_types
        assert "data_accessed" in event_types
        assert "data_access_denied" in event_types
    
    def test_session_security_with_monitoring(self, session_manager, audit_logger):
        """Test session security with monitoring"""
        
        # Create session
        session_id = session_manager.create_session(
            user_id="user_123",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0",
            permissions={"read", "write"}
        )
        
        # Validate session
        session = session_manager.validate_session(session_id, "192.168.1.100")
        assert session is not None
        assert session.user_id == "user_123"
        
        # Test IP address mismatch detection
        session_different_ip = session_manager.validate_session(session_id, "192.168.1.200")
        # Should still work but log warning
        assert session_different_ip is not None
        
        # Test session revocation
        revoked = session_manager.revoke_session(session_id, "user_logout")
        assert revoked is True
        
        # Session should no longer be valid
        invalid_session = session_manager.validate_session(session_id)
        assert invalid_session is None
        
        # Test failed attempt tracking
        for _ in range(3):
            session_manager.record_failed_attempt("192.168.1.50")
        
        # IP should not be locked yet (threshold is 10)
        assert not session_manager._is_ip_locked("192.168.1.50")
        
        # Exceed threshold
        for _ in range(8):
            session_manager.record_failed_attempt("192.168.1.50")
        
        # IP should now be locked
        assert session_manager._is_ip_locked("192.168.1.50")
    
    def test_api_key_security_lifecycle(self, auth_manager, audit_logger):
        """Test API key security lifecycle"""
        
        # Create user
        user = auth_manager.create_user(
            "api_user@example.com",
            "ApiPass123!",
            "API User",
            UserRole.ANALYST
        )
        
        # Create API key with limited permissions
        api_key_info = auth_manager.create_api_key(
            user.user_id,
            "Limited API Key",
            {Permission.DOCUMENT_READ},
            rate_limit=100,
            expires_at=datetime.now() + timedelta(days=30)
        )
        
        # Verify API key works
        api_key_obj = auth_manager.verify_api_key(api_key_info["api_key"])
        assert api_key_obj is not None
        assert api_key_obj.permissions == {Permission.DOCUMENT_READ}
        assert api_key_obj.rate_limit == 100
        
        # Test permission checking
        assert auth_manager.check_permission(api_key_obj.permissions, Permission.DOCUMENT_READ)
        assert not auth_manager.check_permission(api_key_obj.permissions, Permission.USER_MANAGE)
        
        # Test usage tracking
        initial_usage = api_key_obj.usage_count
        
        # Use API key multiple times
        for _ in range(5):
            auth_manager.verify_api_key(api_key_info["api_key"])
        
        # Check usage increased
        final_api_key_obj = auth_manager.verify_api_key(api_key_info["api_key"])
        assert final_api_key_obj.usage_count > initial_usage
        
        # Revoke API key
        revoked = auth_manager.revoke_api_key(api_key_info["key_id"], user.user_id)
        assert revoked is True
        
        # API key should no longer work
        invalid_key = auth_manager.verify_api_key(api_key_info["api_key"])
        assert invalid_key is None
    
    def test_data_retention_and_cleanup(self, privacy_manager):
        """Test data retention policies and cleanup"""
        
        # Register data with short retention
        short_term_record = privacy_manager.register_data(
            record_id="short_term_data",
            user_id="user_123",
            data="Temporary data",
            data_type="temp_document",
            retention_policy=RetentionPolicy.SHORT_TERM
        )
        
        # Register data with long retention
        long_term_record = privacy_manager.register_data(
            record_id="long_term_data",
            user_id="user_123",
            data="Important data",
            data_type="important_document",
            retention_policy=RetentionPolicy.LONG_TERM
        )
        
        # Manually expire short-term data
        short_term_record.expires_at = datetime.now() - timedelta(days=1)
        
        # Run cleanup
        cleanup_result = privacy_manager.cleanup_expired_data()
        
        assert cleanup_result["expired_records_found"] == 1
        assert cleanup_result["records_deleted"] == 1
        
        # Short-term data should be deleted
        assert "short_term_data" not in privacy_manager.data_records
        
        # Long-term data should remain
        assert "long_term_data" in privacy_manager.data_records
    
    def test_audit_log_integrity(self, audit_logger):
        """Test audit log integrity and verification"""
        
        # Log various events
        events = [
            ("user_login", "user_123", {"ip": "192.168.1.1"}),
            ("data_access", "user_123", {"resource": "document_1"}),
            ("permission_change", "admin", {"target_user": "user_123"}),
            ("data_export", "user_123", {"format": "csv", "records": 100})
        ]
        
        event_ids = []
        for event_type, user_id, details in events:
            event_id = audit_logger.log_event(
                event_type=event_type,
                user_id=user_id,
                details=details
            )
            event_ids.append(event_id)
        
        # Verify all events were logged
        assert len(event_ids) == 4
        
        # Retrieve events
        logged_events = audit_logger.get_audit_events(limit=10)
        assert len(logged_events) >= 4
        
        # Verify integrity
        integrity_result = audit_logger.verify_audit_integrity()
        assert integrity_result["status"] == "verified"
        assert integrity_result["corrupted_events"] == 0
        assert integrity_result["integrity_rate"] == 100.0
        
        # Generate audit report
        report = audit_logger.generate_audit_report()
        assert report["summary"]["total_events"] >= 4
        assert "event_types" in report
        assert "security_alerts" in report
    
    def test_encryption_key_rotation(self, encryption_service):
        """Test encryption key rotation"""
        
        # Encrypt data with initial key
        original_data = "Sensitive information"
        encrypted_v1 = encryption_service.encrypt_data(original_data)
        
        # Rotate key
        rotation_result = encryption_service.rotate_key()
        assert "Key rotated" in rotation_result
        
        # Encrypt new data with rotated key
        new_data = "New sensitive information"
        encrypted_v2 = encryption_service.encrypt_data(new_data)
        
        # Both should decrypt correctly
        decrypted_v1 = encryption_service.decrypt_data(encrypted_v1)
        decrypted_v2 = encryption_service.decrypt_data(encrypted_v2)
        
        assert decrypted_v1 == original_data
        assert decrypted_v2 == new_data
        
        # Key versions should be different
        assert encrypted_v1["metadata"]["key_id"] != encrypted_v2["metadata"]["key_id"]
    
    def test_privacy_compliance_report(self, privacy_manager):
        """Test privacy compliance reporting"""
        
        # Register various types of data
        test_data = [
            ("pii_data", "SSN: 123-45-6789", RetentionPolicy.LONG_TERM),
            ("business_data", "Revenue: $1,000,000", RetentionPolicy.MEDIUM_TERM),
            ("temp_data", "Temporary information", RetentionPolicy.SHORT_TERM),
            ("public_data", "Public announcement", RetentionPolicy.PERMANENT)
        ]
        
        for record_id, data, retention in test_data:
            privacy_manager.register_data(
                record_id=record_id,
                user_id="user_123",
                data=data,
                data_type="document",
                retention_policy=retention,
                consent_given=True
            )
        
        # Generate privacy report
        report = privacy_manager.generate_privacy_report()
        
        assert report["summary"]["total_records"] == 4
        assert report["summary"]["pii_records"] == 1  # Only SSN data
        assert report["summary"]["encrypted_records"] == 1  # PII should be encrypted
        
        # Check classification breakdown
        assert "restricted" in report["classification_breakdown"]
        assert "internal" in report["classification_breakdown"]
        
        # Check retention breakdown
        assert "long_term" in report["retention_breakdown"]
        assert "medium_term" in report["retention_breakdown"]
        assert "short_term" in report["retention_breakdown"]
        assert "permanent" in report["retention_breakdown"]
        
        # Check compliance status
        assert report["compliance_status"]["pii_encryption_rate"] == 100.0
    
    def test_gdpr_data_export(self, privacy_manager):
        """Test GDPR-compliant data export"""
        
        # Register data for multiple users
        users_data = [
            ("user_123", "User 123's document"),
            ("user_123", "User 123's table data"),
            ("user_456", "User 456's document"),
            ("user_789", "User 789's data")
        ]
        
        for i, (user_id, data) in enumerate(users_data):
            privacy_manager.register_data(
                record_id=f"record_{i}",
                user_id=user_id,
                data=data,
                data_type="document"
            )
        
        # Export data for user_123
        exported_data = privacy_manager.export_user_data("user_123")
        
        assert exported_data["user_id"] == "user_123"
        assert len(exported_data["records"]) == 2
        
        # Verify only user_123's data is included
        for record in exported_data["records"]:
            assert record["record_id"] in ["record_0", "record_1"]
        
        # Export should not include other users' data
        record_ids = [r["record_id"] for r in exported_data["records"]]
        assert "record_2" not in record_ids
        assert "record_3" not in record_ids
    
    def test_security_error_handling(self, auth_manager, encryption_service, privacy_manager):
        """Test security error handling"""
        
        # Test authentication errors
        with pytest.raises(AuthenticationError, match="User already exists"):
            auth_manager.create_user("test@example.com", "Pass123!", "User 1")
            auth_manager.create_user("test@example.com", "Pass123!", "User 2")
        
        # Test authorization errors
        user = auth_manager.create_user("limited@example.com", "Pass123!", "Limited User", UserRole.VIEWER)
        
        with pytest.raises(AuthorizationError, match="permissions user doesn't have"):
            auth_manager.create_api_key(
                user.user_id,
                "Invalid Key",
                {Permission.USER_MANAGE}  # Viewer doesn't have this permission
            )
        
        # Test encryption errors
        with pytest.raises(EncryptionError, match="Failed to decrypt data"):
            invalid_package = {
                "encrypted_data": "invalid_data",
                "metadata": {
                    "algorithm": "Fernet",
                    "key_id": "v1",
                    "encrypted_at": "2023-01-01T00:00:00",
                    "checksum": "invalid"
                }
            }
            encryption_service.decrypt_data(invalid_package)
        
        # Test privacy manager error handling
        # Access to non-existent record should return False
        access_result = privacy_manager.access_data("non_existent", "user_123", "purpose")
        assert access_result is False