"""
Data privacy and compliance manager
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re

from ..core.config import get_settings
from ..core.exceptions import PrivacyError
from .encryption_service import EncryptionService
from .audit_logger import AuditLogger


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class RetentionPolicy(Enum):
    """Data retention policies"""
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 1 year
    LONG_TERM = "long_term"  # 7 years
    PERMANENT = "permanent"


@dataclass
class DataRecord:
    """Data record with privacy metadata"""
    record_id: str
    user_id: str
    data_type: str
    classification: DataClassification
    retention_policy: RetentionPolicy
    created_at: datetime
    last_accessed: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_encrypted: bool = False
    pii_detected: bool = False
    consent_given: bool = False
    processing_purposes: Set[str] = field(default_factory=set)


@dataclass
class PIIPattern:
    """Pattern for detecting PII"""
    name: str
    pattern: str
    confidence: float
    category: str


class PrivacyManager:
    """Manager for data privacy and compliance"""
    
    def __init__(self):
        self.settings = get_settings()
        self.encryption_service = EncryptionService()
        self.audit_logger = AuditLogger()
        
        # Retention periods
        self.retention_periods = {
            RetentionPolicy.SHORT_TERM: timedelta(days=30),
            RetentionPolicy.MEDIUM_TERM: timedelta(days=365),
            RetentionPolicy.LONG_TERM: timedelta(days=2555),  # 7 years
            RetentionPolicy.PERMANENT: None
        }
        
        # PII detection patterns
        self.pii_patterns = [
            PIIPattern("email", r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.9, "contact"),
            PIIPattern("phone", r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 0.8, "contact"),
            PIIPattern("ssn", r'\b\d{3}-\d{2}-\d{4}\b', 0.95, "identity"),
            PIIPattern("credit_card", r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 0.9, "financial"),
            PIIPattern("ip_address", r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 0.7, "technical"),
            PIIPattern("name", r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 0.6, "identity"),
        ]
        
        # Data records storage
        self.data_records: Dict[str, DataRecord] = {}
        
        # Load existing records
        self._load_data_records()
    
    def classify_data(self, data: Any, data_type: str) -> DataClassification:
        """Automatically classify data based on content"""
        
        # Convert data to string for analysis
        if isinstance(data, dict):
            data_str = json.dumps(data)
        elif isinstance(data, (list, tuple)):
            data_str = str(data)
        else:
            data_str = str(data)
        
        # Check for PII
        pii_detected = self.detect_pii(data_str)
        
        if pii_detected:
            # High-confidence PII patterns suggest restricted classification
            high_confidence_pii = [p for p in pii_detected if p["confidence"] > 0.8]
            if high_confidence_pii:
                return DataClassification.RESTRICTED
            else:
                return DataClassification.CONFIDENTIAL
        
        # Default classification based on data type
        if data_type in ["document", "table", "extracted_data"]:
            return DataClassification.INTERNAL
        else:
            return DataClassification.PUBLIC
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect personally identifiable information in text"""
        
        detected_pii = []
        
        for pattern in self.pii_patterns:
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
            for match in matches:
                detected_pii.append({
                    "type": pattern.name,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": pattern.confidence,
                    "category": pattern.category
                })
        
        return detected_pii
    
    def anonymize_data(self, data: Any, pii_patterns: List[Dict[str, Any]] = None) -> Any:
        """Anonymize data by removing or masking PII"""
        
        if isinstance(data, str):
            return self._anonymize_text(data, pii_patterns)
        elif isinstance(data, dict):
            return {k: self.anonymize_data(v, pii_patterns) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.anonymize_data(item, pii_patterns) for item in data]
        else:
            return data
    
    def _anonymize_text(self, text: str, pii_patterns: List[Dict[str, Any]] = None) -> str:
        """Anonymize text by masking PII"""
        
        if not pii_patterns:
            pii_patterns = self.detect_pii(text)
        
        # Sort patterns by start position (reverse order for replacement)
        pii_patterns.sort(key=lambda x: x["start"], reverse=True)
        
        anonymized_text = text
        for pattern in pii_patterns:
            start, end = pattern["start"], pattern["end"]
            pii_type = pattern["type"]
            
            # Create mask based on PII type
            if pii_type == "email":
                mask = "[EMAIL_REDACTED]"
            elif pii_type == "phone":
                mask = "[PHONE_REDACTED]"
            elif pii_type == "ssn":
                mask = "[SSN_REDACTED]"
            elif pii_type == "credit_card":
                mask = "[CARD_REDACTED]"
            elif pii_type == "name":
                mask = "[NAME_REDACTED]"
            else:
                mask = "[PII_REDACTED]"
            
            anonymized_text = anonymized_text[:start] + mask + anonymized_text[end:]
        
        return anonymized_text
    
    def register_data(
        self,
        record_id: str,
        user_id: str,
        data: Any,
        data_type: str,
        retention_policy: RetentionPolicy = RetentionPolicy.MEDIUM_TERM,
        processing_purposes: Set[str] = None,
        consent_given: bool = False
    ) -> DataRecord:
        """Register data with privacy metadata"""
        
        # Classify data
        classification = self.classify_data(data, data_type)
        
        # Detect PII
        pii_detected = bool(self.detect_pii(str(data)))
        
        # Calculate expiry date
        expires_at = None
        if retention_policy != RetentionPolicy.PERMANENT:
            retention_period = self.retention_periods[retention_policy]
            expires_at = datetime.now() + retention_period
        
        # Create data record
        record = DataRecord(
            record_id=record_id,
            user_id=user_id,
            data_type=data_type,
            classification=classification,
            retention_policy=retention_policy,
            created_at=datetime.now(),
            expires_at=expires_at,
            pii_detected=pii_detected,
            consent_given=consent_given,
            processing_purposes=processing_purposes or set()
        )
        
        # Encrypt sensitive data
        if classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
            record.is_encrypted = True
        
        self.data_records[record_id] = record
        self._save_data_records()
        
        # Log data registration
        self.audit_logger.log_event(
            event_type="data_registered",
            user_id=user_id,
            details={
                "record_id": record_id,
                "data_type": data_type,
                "classification": classification.value,
                "pii_detected": pii_detected,
                "retention_policy": retention_policy.value
            }
        )
        
        return record
    
    def access_data(self, record_id: str, user_id: str, purpose: str) -> bool:
        """Log data access and check permissions"""
        
        record = self.data_records.get(record_id)
        if not record:
            return False
        
        # Check if data has expired
        if record.expires_at and datetime.now() > record.expires_at:
            self.audit_logger.log_event(
                event_type="data_access_denied",
                user_id=user_id,
                details={
                    "record_id": record_id,
                    "reason": "data_expired",
                    "purpose": purpose
                }
            )
            return False
        
        # Check consent for PII data
        if record.pii_detected and not record.consent_given:
            self.audit_logger.log_event(
                event_type="data_access_denied",
                user_id=user_id,
                details={
                    "record_id": record_id,
                    "reason": "no_consent",
                    "purpose": purpose
                }
            )
            return False
        
        # Check processing purpose
        if record.processing_purposes and purpose not in record.processing_purposes:
            self.audit_logger.log_event(
                event_type="data_access_denied",
                user_id=user_id,
                details={
                    "record_id": record_id,
                    "reason": "invalid_purpose",
                    "purpose": purpose,
                    "allowed_purposes": list(record.processing_purposes)
                }
            )
            return False
        
        # Update last accessed
        record.last_accessed = datetime.now()
        self._save_data_records()
        
        # Log access
        self.audit_logger.log_event(
            event_type="data_accessed",
            user_id=user_id,
            details={
                "record_id": record_id,
                "purpose": purpose,
                "classification": record.classification.value
            }
        )
        
        return True
    
    def delete_data(self, record_id: str, user_id: str, reason: str = "user_request") -> bool:
        """Delete data and associated records"""
        
        record = self.data_records.get(record_id)
        if not record:
            return False
        
        # Remove from records
        del self.data_records[record_id]
        self._save_data_records()
        
        # Log deletion
        self.audit_logger.log_event(
            event_type="data_deleted",
            user_id=user_id,
            details={
                "record_id": record_id,
                "reason": reason,
                "classification": record.classification.value,
                "was_encrypted": record.is_encrypted
            }
        )
        
        return True
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data records"""
        
        current_time = datetime.now()
        expired_records = []
        
        for record_id, record in self.data_records.items():
            if record.expires_at and current_time > record.expires_at:
                expired_records.append(record_id)
        
        # Delete expired records
        deleted_count = 0
        for record_id in expired_records:
            if self.delete_data(record_id, "system", "automatic_cleanup"):
                deleted_count += 1
        
        # Log cleanup
        self.audit_logger.log_event(
            event_type="data_cleanup",
            user_id="system",
            details={
                "expired_records_found": len(expired_records),
                "records_deleted": deleted_count
            }
        )
        
        return {
            "expired_records_found": len(expired_records),
            "records_deleted": deleted_count
        }
    
    def generate_privacy_report(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        
        # Filter records by user if specified
        records = self.data_records.values()
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        
        # Calculate statistics
        total_records = len(records)
        pii_records = len([r for r in records if r.pii_detected])
        encrypted_records = len([r for r in records if r.is_encrypted])
        expired_records = len([r for r in records if r.expires_at and datetime.now() > r.expires_at])
        
        # Classification breakdown
        classification_counts = {}
        for classification in DataClassification:
            classification_counts[classification.value] = len([r for r in records if r.classification == classification])
        
        # Retention policy breakdown
        retention_counts = {}
        for policy in RetentionPolicy:
            retention_counts[policy.value] = len([r for r in records if r.retention_policy == policy])
        
        return {
            "generated_at": datetime.now().isoformat(),
            "user_id": user_id,
            "summary": {
                "total_records": total_records,
                "pii_records": pii_records,
                "encrypted_records": encrypted_records,
                "expired_records": expired_records
            },
            "classification_breakdown": classification_counts,
            "retention_breakdown": retention_counts,
            "compliance_status": {
                "pii_encryption_rate": (encrypted_records / pii_records * 100) if pii_records > 0 else 100,
                "expired_data_present": expired_records > 0
            }
        }
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user (GDPR compliance)"""
        
        user_records = [r for r in self.data_records.values() if r.user_id == user_id]
        
        exported_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "records": []
        }
        
        for record in user_records:
            exported_data["records"].append({
                "record_id": record.record_id,
                "data_type": record.data_type,
                "classification": record.classification.value,
                "created_at": record.created_at.isoformat(),
                "last_accessed": record.last_accessed.isoformat() if record.last_accessed else None,
                "expires_at": record.expires_at.isoformat() if record.expires_at else None,
                "pii_detected": record.pii_detected,
                "processing_purposes": list(record.processing_purposes)
            })
        
        # Log export
        self.audit_logger.log_event(
            event_type="user_data_exported",
            user_id=user_id,
            details={
                "records_exported": len(user_records)
            }
        )
        
        return exported_data
    
    def _load_data_records(self):
        """Load data records from storage"""
        
        records_file = os.path.join(self.settings.DATA_DIR, "privacy_records.json")
        
        if os.path.exists(records_file):
            try:
                with open(records_file, "r") as f:
                    records_data = json.load(f)
                
                for record_id, record_dict in records_data.items():
                    # Convert datetime strings back to datetime objects
                    record_dict["created_at"] = datetime.fromisoformat(record_dict["created_at"])
                    if record_dict.get("last_accessed"):
                        record_dict["last_accessed"] = datetime.fromisoformat(record_dict["last_accessed"])
                    if record_dict.get("expires_at"):
                        record_dict["expires_at"] = datetime.fromisoformat(record_dict["expires_at"])
                    
                    # Convert enums
                    record_dict["classification"] = DataClassification(record_dict["classification"])
                    record_dict["retention_policy"] = RetentionPolicy(record_dict["retention_policy"])
                    record_dict["processing_purposes"] = set(record_dict["processing_purposes"])
                    
                    self.data_records[record_id] = DataRecord(**record_dict)
                    
            except Exception as e:
                print(f"Error loading privacy records: {e}")
    
    def _save_data_records(self):
        """Save data records to storage"""
        
        records_file = os.path.join(self.settings.DATA_DIR, "privacy_records.json")
        os.makedirs(os.path.dirname(records_file), exist_ok=True)
        
        # Convert records to serializable format
        records_data = {}
        for record_id, record in self.data_records.items():
            records_data[record_id] = {
                "record_id": record.record_id,
                "user_id": record.user_id,
                "data_type": record.data_type,
                "classification": record.classification.value,
                "retention_policy": record.retention_policy.value,
                "created_at": record.created_at.isoformat(),
                "last_accessed": record.last_accessed.isoformat() if record.last_accessed else None,
                "expires_at": record.expires_at.isoformat() if record.expires_at else None,
                "is_encrypted": record.is_encrypted,
                "pii_detected": record.pii_detected,
                "consent_given": record.consent_given,
                "processing_purposes": list(record.processing_purposes)
            }
        
        with open(records_file, "w") as f:
            json.dump(records_data, f, indent=2)