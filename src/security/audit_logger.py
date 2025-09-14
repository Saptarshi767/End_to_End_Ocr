"""
Audit logging system for compliance and security monitoring
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import threading

from ..core.config import get_settings


class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGED = "password_changed"
    ACCOUNT_LOCKED = "account_locked"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"
    
    # Data events
    DATA_CREATED = "data_created"
    DATA_ACCESSED = "data_accessed"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    DATA_SHARED = "data_shared"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGED = "config_changed"
    ERROR_OCCURRED = "error_occurred"
    
    # Privacy events
    DATA_REGISTERED = "data_registered"
    DATA_CLEANUP = "data_cleanup"
    CONSENT_GIVEN = "consent_given"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    
    # API events
    API_KEY_CREATED = "api_key_created"
    API_KEY_USED = "api_key_used"
    API_KEY_REVOKED = "api_key_revoked"
    
    # User management events
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    
    # Session events
    SESSION_CREATED = "session_created"
    SESSION_REVOKED = "session_revoked"


@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str
    event_type: str
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    result: str  # success, failure, error
    details: Dict[str, Any]
    risk_level: str  # low, medium, high, critical
    checksum: str


class AuditLogger:
    """Audit logging system with integrity protection"""
    
    def __init__(self):
        self.settings = get_settings()
        self.audit_file = os.path.join(self.settings.DATA_DIR, "logs", "audit.jsonl")
        self.lock = threading.Lock()
        
        # Ensure audit directory exists
        os.makedirs(os.path.dirname(self.audit_file), exist_ok=True)
        
        # Risk level mapping
        self.risk_levels = {
            # High risk events
            AuditEventType.LOGIN_FAILED: "medium",
            AuditEventType.ACCOUNT_LOCKED: "high",
            AuditEventType.ACCESS_DENIED: "medium",
            AuditEventType.DATA_DELETED: "high",
            AuditEventType.DATA_EXPORTED: "medium",
            AuditEventType.PERMISSION_CHANGED: "high",
            AuditEventType.CONFIG_CHANGED: "high",
            AuditEventType.ERROR_OCCURRED: "medium",
            AuditEventType.API_KEY_CREATED: "medium",
            AuditEventType.USER_CREATED: "medium",
            AuditEventType.USER_DELETED: "high",
            
            # Medium risk events
            AuditEventType.LOGIN_SUCCESS: "low",
            AuditEventType.DATA_ACCESSED: "low",
            AuditEventType.DATA_CREATED: "low",
            AuditEventType.DATA_MODIFIED: "medium",
            AuditEventType.SESSION_CREATED: "low",
            
            # Low risk events (default)
        }
    
    def log_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an audit event"""
        
        # Generate event ID
        event_id = self._generate_event_id()
        
        # Determine risk level
        try:
            event_enum = AuditEventType(event_type)
            risk_level = self.risk_levels.get(event_enum, "low")
        except ValueError:
            risk_level = "low"
        
        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            risk_level=risk_level,
            checksum=""  # Will be calculated
        )
        
        # Calculate checksum for integrity
        event.checksum = self._calculate_checksum(event)
        
        # Write to audit log
        self._write_audit_event(event)
        
        return event_id
    
    def get_audit_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Retrieve audit events with filtering"""
        
        events = []
        
        try:
            with open(self.audit_file, "r") as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        
                        # Parse timestamp
                        event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                        
                        event = AuditEvent(**event_data)
                        
                        # Apply filters
                        if start_date and event.timestamp < start_date:
                            continue
                        if end_date and event.timestamp > end_date:
                            continue
                        if user_id and event.user_id != user_id:
                            continue
                        if event_type and event.event_type != event_type:
                            continue
                        if risk_level and event.risk_level != risk_level:
                            continue
                        
                        events.append(event)
                        
                        if len(events) >= limit:
                            break
                            
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        # Log corrupted audit entry
                        print(f"Corrupted audit entry: {e}")
                        continue
                        
        except FileNotFoundError:
            pass
        
        return events
    
    def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify audit log integrity"""
        
        total_events = 0
        corrupted_events = 0
        integrity_errors = []
        
        try:
            with open(self.audit_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        event_data = json.loads(line.strip())
                        event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                        
                        event = AuditEvent(**event_data)
                        total_events += 1
                        
                        # Verify checksum
                        stored_checksum = event.checksum
                        event.checksum = ""  # Reset for calculation
                        calculated_checksum = self._calculate_checksum(event)
                        
                        if stored_checksum != calculated_checksum:
                            corrupted_events += 1
                            integrity_errors.append({
                                "line": line_num,
                                "event_id": event.event_id,
                                "error": "checksum_mismatch"
                            })
                        
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        corrupted_events += 1
                        integrity_errors.append({
                            "line": line_num,
                            "error": str(e)
                        })
                        
        except FileNotFoundError:
            return {
                "status": "no_audit_file",
                "total_events": 0,
                "corrupted_events": 0,
                "integrity_errors": []
            }
        
        return {
            "status": "verified" if corrupted_events == 0 else "corrupted",
            "total_events": total_events,
            "corrupted_events": corrupted_events,
            "integrity_rate": ((total_events - corrupted_events) / total_events * 100) if total_events > 0 else 100,
            "integrity_errors": integrity_errors
        }
    
    def generate_audit_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate audit report for compliance"""
        
        events = self.get_audit_events(start_date=start_date, end_date=end_date)
        
        # Event type statistics
        event_type_counts = {}
        risk_level_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        result_counts = {"success": 0, "failure": 0, "error": 0}
        user_activity = {}
        
        for event in events:
            # Event types
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
            
            # Risk levels
            risk_level_counts[event.risk_level] += 1
            
            # Results
            result_counts[event.result] = result_counts.get(event.result, 0) + 1
            
            # User activity
            if event.user_id:
                if event.user_id not in user_activity:
                    user_activity[event.user_id] = {"events": 0, "last_activity": None}
                user_activity[event.user_id]["events"] += 1
                if not user_activity[event.user_id]["last_activity"] or event.timestamp > user_activity[event.user_id]["last_activity"]:
                    user_activity[event.user_id]["last_activity"] = event.timestamp
        
        # Security alerts (high-risk events)
        security_alerts = [
            event for event in events 
            if event.risk_level in ["high", "critical"] or event.result in ["failure", "error"]
        ]
        
        return {
            "report_generated": datetime.now().isoformat(),
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "summary": {
                "total_events": len(events),
                "unique_users": len(user_activity),
                "security_alerts": len(security_alerts)
            },
            "event_types": event_type_counts,
            "risk_levels": risk_level_counts,
            "results": result_counts,
            "top_users": sorted(
                user_activity.items(),
                key=lambda x: x[1]["events"],
                reverse=True
            )[:10],
            "security_alerts": [
                {
                    "event_id": alert.event_id,
                    "event_type": alert.event_type,
                    "timestamp": alert.timestamp.isoformat(),
                    "user_id": alert.user_id,
                    "risk_level": alert.risk_level,
                    "result": alert.result,
                    "details": alert.details
                }
                for alert in security_alerts[:50]  # Limit to 50 most recent alerts
            ]
        }
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate checksum for event integrity"""
        
        # Create deterministic string representation
        event_dict = asdict(event)
        event_dict.pop("checksum", None)  # Remove checksum field
        
        # Sort keys for consistency
        event_str = json.dumps(event_dict, sort_keys=True, default=str)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(event_str.encode()).hexdigest()
    
    def _write_audit_event(self, event: AuditEvent):
        """Write audit event to log file"""
        
        with self.lock:
            try:
                # Convert to dictionary for JSON serialization
                event_dict = asdict(event)
                event_dict["timestamp"] = event.timestamp.isoformat()
                
                # Write to file
                with open(self.audit_file, "a") as f:
                    f.write(json.dumps(event_dict) + "\n")
                    
            except Exception as e:
                # Fallback logging to stderr
                print(f"Failed to write audit event: {e}", file=__import__('sys').stderr)