"""
Data encryption service for sensitive documents and data
"""

import os
import base64
import hashlib
from typing import Optional, Dict, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from dataclasses import dataclass
from datetime import datetime
import json

from ..core.config import get_settings
from ..core.exceptions import EncryptionError


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted data"""
    algorithm: str
    key_id: str
    encrypted_at: datetime
    checksum: str


class EncryptionService:
    """Service for encrypting and decrypting sensitive data"""
    
    def __init__(self):
        self.settings = get_settings()
        self.master_key = self._get_or_create_master_key()
        self.fernet = Fernet(self.master_key)
        
        # Key rotation tracking
        self.key_versions: Dict[str, bytes] = {}
        self.current_key_version = "v1"
        self.key_versions[self.current_key_version] = self.master_key
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        
        key_file = os.path.join(self.settings.DATA_DIR, ".encryption_key")
        
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            
            # Store securely (in production, use HSM or key management service)
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, "wb") as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            return key
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]], key_id: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt data with metadata"""
        
        try:
            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Calculate checksum
            checksum = hashlib.sha256(data_bytes).hexdigest()
            
            # Encrypt data
            encrypted_data = self.fernet.encrypt(data_bytes)
            
            # Create metadata
            metadata = EncryptionMetadata(
                algorithm="Fernet",
                key_id=key_id or self.current_key_version,
                encrypted_at=datetime.now(),
                checksum=checksum
            )
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "metadata": {
                    "algorithm": metadata.algorithm,
                    "key_id": metadata.key_id,
                    "encrypted_at": metadata.encrypted_at.isoformat(),
                    "checksum": metadata.checksum
                }
            }
            
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt data: {str(e)}")
    
    def decrypt_data(self, encrypted_package: Dict[str, Any]) -> Union[str, bytes, Dict[str, Any]]:
        """Decrypt data with verification"""
        
        try:
            encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
            metadata = encrypted_package["metadata"]
            
            # Get appropriate key version
            key_id = metadata.get("key_id", "v1")
            if key_id not in self.key_versions:
                raise EncryptionError(f"Unknown key version: {key_id}")
            
            # Decrypt data
            fernet = Fernet(self.key_versions[key_id])
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Verify checksum
            checksum = hashlib.sha256(decrypted_data).hexdigest()
            if checksum != metadata.get("checksum"):
                raise EncryptionError("Data integrity check failed")
            
            # Try to parse as JSON, fallback to string/bytes
            try:
                return json.loads(decrypted_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    return decrypted_data.decode('utf-8')
                except UnicodeDecodeError:
                    return decrypted_data
            
        except Exception as e:
            raise EncryptionError(f"Failed to decrypt data: {str(e)}")
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt file and return encrypted file path"""
        
        try:
            if not os.path.exists(file_path):
                raise EncryptionError(f"File not found: {file_path}")
            
            # Read file
            with open(file_path, "rb") as f:
                file_data = f.read()
            
            # Encrypt data
            encrypted_package = self.encrypt_data(file_data)
            
            # Determine output path
            if not output_path:
                output_path = f"{file_path}.encrypted"
            
            # Write encrypted file
            with open(output_path, "w") as f:
                json.dump(encrypted_package, f)
            
            return output_path
            
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt file: {str(e)}")
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt file and return decrypted file path"""
        
        try:
            if not os.path.exists(encrypted_file_path):
                raise EncryptionError(f"Encrypted file not found: {encrypted_file_path}")
            
            # Read encrypted file
            with open(encrypted_file_path, "r") as f:
                encrypted_package = json.load(f)
            
            # Decrypt data
            decrypted_data = self.decrypt_data(encrypted_package)
            
            # Determine output path
            if not output_path:
                if encrypted_file_path.endswith(".encrypted"):
                    output_path = encrypted_file_path[:-10]  # Remove .encrypted
                else:
                    output_path = f"{encrypted_file_path}.decrypted"
            
            # Write decrypted file
            if isinstance(decrypted_data, bytes):
                with open(output_path, "wb") as f:
                    f.write(decrypted_data)
            else:
                with open(output_path, "w") as f:
                    if isinstance(decrypted_data, dict):
                        json.dump(decrypted_data, f)
                    else:
                        f.write(str(decrypted_data))
            
            return output_path
            
        except Exception as e:
            raise EncryptionError(f"Failed to decrypt file: {str(e)}")
    
    def generate_key_pair(self) -> Dict[str, str]:
        """Generate RSA key pair for asymmetric encryption"""
        
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Get public key
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return {
                "private_key": private_pem.decode('utf-8'),
                "public_key": public_pem.decode('utf-8')
            }
            
        except Exception as e:
            raise EncryptionError(f"Failed to generate key pair: {str(e)}")
    
    def encrypt_with_public_key(self, data: Union[str, bytes], public_key_pem: str) -> str:
        """Encrypt data with RSA public key"""
        
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(public_key_pem.encode('utf-8'))
            
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Encrypt data
            encrypted_data = public_key.encrypt(
                data_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt with public key: {str(e)}")
    
    def decrypt_with_private_key(self, encrypted_data: str, private_key_pem: str) -> bytes:
        """Decrypt data with RSA private key"""
        
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                private_key_pem.encode('utf-8'),
                password=None
            )
            
            # Decrypt data
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted_data = private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted_data
            
        except Exception as e:
            raise EncryptionError(f"Failed to decrypt with private key: {str(e)}")
    
    def rotate_key(self) -> str:
        """Rotate encryption key"""
        
        try:
            # Generate new key
            new_key = Fernet.generate_key()
            
            # Create new version
            import time
            new_version = f"v{int(time.time())}"
            self.key_versions[new_version] = new_key
            
            # Update current version
            old_version = self.current_key_version
            self.current_key_version = new_version
            self.master_key = new_key
            self.fernet = Fernet(new_key)
            
            # Store new key
            key_file = os.path.join(self.settings.DATA_DIR, ".encryption_key")
            with open(key_file, "wb") as f:
                f.write(new_key)
            
            return f"Key rotated from {old_version} to {new_version}"
            
        except Exception as e:
            raise EncryptionError(f"Failed to rotate key: {str(e)}")
    
    def secure_delete(self, file_path: str) -> bool:
        """Securely delete file by overwriting"""
        
        try:
            if not os.path.exists(file_path):
                return True
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Overwrite with random data multiple times
            with open(file_path, "r+b") as f:
                for _ in range(3):
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Remove file
            os.remove(file_path)
            
            return True
            
        except Exception as e:
            raise EncryptionError(f"Failed to securely delete file: {str(e)}")