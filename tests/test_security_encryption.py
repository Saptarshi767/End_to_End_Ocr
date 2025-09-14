"""
Tests for encryption service
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, mock_open

from src.security.encryption_service import EncryptionService
from src.core.exceptions import EncryptionError


class TestEncryptionService:
    """Test encryption service"""
    
    @pytest.fixture
    def encryption_service(self):
        """Create encryption service for testing"""
        with patch('src.security.encryption_service.get_settings') as mock_settings:
            mock_settings.return_value.DATA_DIR = tempfile.mkdtemp()
            return EncryptionService()
    
    def test_encrypt_decrypt_string(self, encryption_service):
        """Test string encryption and decryption"""
        original_data = "This is sensitive information"
        
        # Encrypt data
        encrypted_package = encryption_service.encrypt_data(original_data)
        
        assert "encrypted_data" in encrypted_package
        assert "metadata" in encrypted_package
        assert encrypted_package["metadata"]["algorithm"] == "Fernet"
        
        # Decrypt data
        decrypted_data = encryption_service.decrypt_data(encrypted_package)
        
        assert decrypted_data == original_data
    
    def test_encrypt_decrypt_dict(self, encryption_service):
        """Test dictionary encryption and decryption"""
        original_data = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "account": "ACC123456"
        }
        
        # Encrypt data
        encrypted_package = encryption_service.encrypt_data(original_data)
        
        # Decrypt data
        decrypted_data = encryption_service.decrypt_data(encrypted_package)
        
        assert decrypted_data == original_data
    
    def test_encrypt_decrypt_bytes(self, encryption_service):
        """Test bytes encryption and decryption"""
        original_data = b"Binary data content"
        
        # Encrypt data
        encrypted_package = encryption_service.encrypt_data(original_data)
        
        # Decrypt data
        decrypted_data = encryption_service.decrypt_data(encrypted_package)
        
        assert decrypted_data == original_data
    
    def test_encryption_metadata(self, encryption_service):
        """Test encryption metadata"""
        data = "Test data"
        encrypted_package = encryption_service.encrypt_data(data, key_id="test_key")
        
        metadata = encrypted_package["metadata"]
        assert metadata["algorithm"] == "Fernet"
        assert metadata["key_id"] == "test_key"
        assert "encrypted_at" in metadata
        assert "checksum" in metadata
    
    def test_data_integrity_verification(self, encryption_service):
        """Test data integrity verification"""
        data = "Important data"
        encrypted_package = encryption_service.encrypt_data(data)
        
        # Tamper with encrypted data
        tampered_package = encrypted_package.copy()
        tampered_package["metadata"]["checksum"] = "invalid_checksum"
        
        # Decryption should fail
        with pytest.raises(EncryptionError, match="integrity check failed"):
            encryption_service.decrypt_data(tampered_package)
    
    def test_file_encryption_decryption(self, encryption_service):
        """Test file encryption and decryption"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("This is file content")
            temp_file_path = temp_file.name
        
        try:
            # Encrypt file
            encrypted_file_path = encryption_service.encrypt_file(temp_file_path)
            assert os.path.exists(encrypted_file_path)
            
            # Verify encrypted file contains JSON
            with open(encrypted_file_path, 'r') as f:
                encrypted_data = json.load(f)
            assert "encrypted_data" in encrypted_data
            assert "metadata" in encrypted_data
            
            # Decrypt file
            decrypted_file_path = encryption_service.decrypt_file(encrypted_file_path)
            assert os.path.exists(decrypted_file_path)
            
            # Verify decrypted content
            with open(decrypted_file_path, 'r') as f:
                decrypted_content = f.read()
            assert decrypted_content == "This is file content"
            
        finally:
            # Clean up
            for path in [temp_file_path, encrypted_file_path, decrypted_file_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_file_not_found_error(self, encryption_service):
        """Test encryption of non-existent file"""
        with pytest.raises(EncryptionError, match="File not found"):
            encryption_service.encrypt_file("nonexistent_file.txt")
    
    def test_key_pair_generation(self, encryption_service):
        """Test RSA key pair generation"""
        key_pair = encryption_service.generate_key_pair()
        
        assert "private_key" in key_pair
        assert "public_key" in key_pair
        assert key_pair["private_key"].startswith("-----BEGIN PRIVATE KEY-----")
        assert key_pair["public_key"].startswith("-----BEGIN PUBLIC KEY-----")
    
    def test_asymmetric_encryption_decryption(self, encryption_service):
        """Test asymmetric encryption and decryption"""
        # Generate key pair
        key_pair = encryption_service.generate_key_pair()
        
        original_data = "Secret message"
        
        # Encrypt with public key
        encrypted_data = encryption_service.encrypt_with_public_key(
            original_data,
            key_pair["public_key"]
        )
        
        # Decrypt with private key
        decrypted_data = encryption_service.decrypt_with_private_key(
            encrypted_data,
            key_pair["private_key"]
        )
        
        assert decrypted_data.decode('utf-8') == original_data
    
    def test_key_rotation(self, encryption_service):
        """Test encryption key rotation"""
        # Encrypt data with current key
        original_data = "Data before rotation"
        encrypted_package_v1 = encryption_service.encrypt_data(original_data)
        
        # Rotate key
        rotation_result = encryption_service.rotate_key()
        assert "Key rotated from v1 to" in rotation_result
        
        # Encrypt new data with new key
        new_data = "Data after rotation"
        encrypted_package_v2 = encryption_service.encrypt_data(new_data)
        
        # Both packages should decrypt correctly
        decrypted_v1 = encryption_service.decrypt_data(encrypted_package_v1)
        decrypted_v2 = encryption_service.decrypt_data(encrypted_package_v2)
        
        assert decrypted_v1 == original_data
        assert decrypted_v2 == new_data
        
        # Key versions should be different
        assert encrypted_package_v1["metadata"]["key_id"] != encrypted_package_v2["metadata"]["key_id"]
    
    def test_secure_file_deletion(self, encryption_service):
        """Test secure file deletion"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("Sensitive content to be deleted")
            temp_file_path = temp_file.name
        
        # Verify file exists
        assert os.path.exists(temp_file_path)
        
        # Securely delete file
        result = encryption_service.secure_delete(temp_file_path)
        
        assert result is True
        assert not os.path.exists(temp_file_path)
    
    def test_secure_delete_nonexistent_file(self, encryption_service):
        """Test secure deletion of non-existent file"""
        result = encryption_service.secure_delete("nonexistent_file.txt")
        assert result is True  # Should return True for non-existent files
    
    def test_encryption_error_handling(self, encryption_service):
        """Test encryption error handling"""
        # Test with invalid key for asymmetric encryption
        with pytest.raises(EncryptionError, match="Failed to encrypt with public key"):
            encryption_service.encrypt_with_public_key("data", "invalid_key")
        
        # Test with invalid encrypted data for decryption
        invalid_package = {
            "encrypted_data": "invalid_base64",
            "metadata": {
                "algorithm": "Fernet",
                "key_id": "v1",
                "encrypted_at": "2023-01-01T00:00:00",
                "checksum": "invalid"
            }
        }
        
        with pytest.raises(EncryptionError, match="Failed to decrypt data"):
            encryption_service.decrypt_data(invalid_package)
    
    def test_unknown_key_version_error(self, encryption_service):
        """Test decryption with unknown key version"""
        data = "Test data"
        encrypted_package = encryption_service.encrypt_data(data)
        
        # Change key version to unknown
        encrypted_package["metadata"]["key_id"] = "unknown_version"
        
        with pytest.raises(EncryptionError, match="Unknown key version"):
            encryption_service.decrypt_data(encrypted_package)
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_file_permission_error(self, mock_file, encryption_service):
        """Test file encryption with permission error"""
        with pytest.raises(EncryptionError, match="Failed to encrypt file"):
            encryption_service.encrypt_file("test_file.txt")
    
    def test_large_data_encryption(self, encryption_service):
        """Test encryption of large data"""
        # Create large string (1MB)
        large_data = "A" * (1024 * 1024)
        
        # Encrypt and decrypt
        encrypted_package = encryption_service.encrypt_data(large_data)
        decrypted_data = encryption_service.decrypt_data(encrypted_package)
        
        assert decrypted_data == large_data
    
    def test_empty_data_encryption(self, encryption_service):
        """Test encryption of empty data"""
        empty_data = ""
        
        encrypted_package = encryption_service.encrypt_data(empty_data)
        decrypted_data = encryption_service.decrypt_data(encrypted_package)
        
        assert decrypted_data == empty_data