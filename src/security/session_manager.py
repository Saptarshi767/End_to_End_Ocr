"""
Session management with security features
"""

import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Set
from dataclasses import dataclass, asdict
import redis
import threading

from ..core.config import get_settings
from ..core.exceptions import SessionError
from .audit_logger import AuditLogger


@dataclass
class SessionData:
    """Session data structure"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    permissions: Set[str]
    is_active: bool = True
    device_fingerprint: Optional[str] = None
    location: Optional[str] = None


class SessionManager:
    """Secure session management system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.settings = get_settings()
        self.redis_client = redis_client
        self.audit_logger = AuditLogger()
        
        # Session configuration
        self.session_timeout = timedelta(hours=24)
        self.max_sessions_per_user = 5
        self.session_renewal_threshold = timedelta(hours=1)
        
        # In-memory fallback if Redis not available
        self.memory_sessions: Dict[str, SessionData] = {}
        self.lock = threading.Lock()
        
        # Suspicious activity tracking
        self.failed_session_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 10
        self.lockout_duration = timedelta(minutes=15)
    
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        permissions: Set[str],
        device_fingerprint: Optional[str] = None,
        location: Optional[str] = None
    ) -> str:
        """Create new user session"""
        
        # Check for too many failed attempts from this IP
        if self._is_ip_locked(ip_address):
            self.audit_logger.log_event(
                event_type="session_creation_blocked",
                user_id=user_id,
                ip_address=ip_address,
                result="failure",
                details={"reason": "ip_locked"}
            )
            raise SessionError("Too many failed attempts. IP temporarily blocked.")
        
        # Generate session ID
        session_id = self._generate_session_id()
        
        # Check existing sessions for user
        existing_sessions = self.get_user_sessions(user_id)
        if len(existing_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest_session = min(existing_sessions, key=lambda s: s.last_activity)
            self.revoke_session(oldest_session.session_id, "max_sessions_exceeded")
        
        # Create session data
        now = datetime.now()
        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            expires_at=now + self.session_timeout,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions,
            device_fingerprint=device_fingerprint,
            location=location
        )
        
        # Store session
        self._store_session(session_data)
        
        # Log session creation
        self.audit_logger.log_event(
            event_type="session_created",
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "device_fingerprint": device_fingerprint,
                "location": location,
                "permissions": list(permissions)
            }
        )
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str = None) -> Optional[SessionData]:
        """Validate and refresh session"""
        
        session = self._get_session(session_id)
        if not session:
            return None
        
        # Check if session is active
        if not session.is_active:
            return None
        
        # Check expiration
        if datetime.now() > session.expires_at:
            self.revoke_session(session_id, "expired")
            return None
        
        # Check IP address consistency (optional security check)
        if ip_address and session.ip_address != ip_address:
            self.audit_logger.log_event(
                event_type="session_ip_mismatch",
                user_id=session.user_id,
                session_id=session_id,
                ip_address=ip_address,
                result="warning",
                details={
                    "original_ip": session.ip_address,
                    "current_ip": ip_address
                }
            )
            # Could revoke session here for strict security
        
        # Update last activity
        session.last_activity = datetime.now()
        
        # Renew session if close to expiry
        if session.expires_at - datetime.now() < self.session_renewal_threshold:
            session.expires_at = datetime.now() + self.session_timeout
        
        # Store updated session
        self._store_session(session)
        
        return session
    
    def revoke_session(self, session_id: str, reason: str = "user_logout") -> bool:
        """Revoke a session"""
        
        session = self._get_session(session_id)
        if not session:
            return False
        
        # Mark as inactive
        session.is_active = False
        self._store_session(session)
        
        # Remove from storage
        self._remove_session(session_id)
        
        # Log session revocation
        self.audit_logger.log_event(
            event_type="session_revoked",
            user_id=session.user_id,
            session_id=session_id,
            details={"reason": reason}
        )
        
        return True
    
    def revoke_user_sessions(self, user_id: str, except_session: str = None) -> int:
        """Revoke all sessions for a user"""
        
        sessions = self.get_user_sessions(user_id)
        revoked_count = 0
        
        for session in sessions:
            if session.session_id != except_session:
                if self.revoke_session(session.session_id, "user_sessions_revoked"):
                    revoked_count += 1
        
        return revoked_count
    
    def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all active sessions for a user"""
        
        sessions = []
        
        if self.redis_client:
            # Get from Redis
            try:
                pattern = f"session:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    session_data = self.redis_client.get(key)
                    if session_data:
                        session = self._deserialize_session(session_data)
                        if session and session.user_id == user_id and session.is_active:
                            sessions.append(session)
            except Exception as e:
                print(f"Error getting user sessions from Redis: {e}")
        else:
            # Get from memory
            with self.lock:
                for session in self.memory_sessions.values():
                    if session.user_id == user_id and session.is_active:
                        sessions.append(session)
        
        return sessions
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        
        expired_count = 0
        current_time = datetime.now()
        
        if self.redis_client:
            # Redis handles TTL automatically, but we can clean up manually
            try:
                pattern = f"session:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    session_data = self.redis_client.get(key)
                    if session_data:
                        session = self._deserialize_session(session_data)
                        if session and current_time > session.expires_at:
                            self.redis_client.delete(key)
                            expired_count += 1
            except Exception as e:
                print(f"Error cleaning up Redis sessions: {e}")
        else:
            # Clean up memory sessions
            with self.lock:
                expired_sessions = []
                for session_id, session in self.memory_sessions.items():
                    if current_time > session.expires_at:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.memory_sessions[session_id]
                    expired_count += 1
        
        if expired_count > 0:
            self.audit_logger.log_event(
                event_type="sessions_cleaned",
                user_id="system",
                details={"expired_sessions": expired_count}
            )
        
        return expired_count
    
    def get_session_statistics(self) -> Dict[str, int]:
        """Get session statistics"""
        
        stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "unique_users": set(),
            "unique_ips": set()
        }
        
        if self.redis_client:
            try:
                pattern = f"session:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    session_data = self.redis_client.get(key)
                    if session_data:
                        session = self._deserialize_session(session_data)
                        if session:
                            stats["total_sessions"] += 1
                            if session.is_active and datetime.now() <= session.expires_at:
                                stats["active_sessions"] += 1
                                stats["unique_users"].add(session.user_id)
                                stats["unique_ips"].add(session.ip_address)
            except Exception as e:
                print(f"Error getting session statistics from Redis: {e}")
        else:
            with self.lock:
                for session in self.memory_sessions.values():
                    stats["total_sessions"] += 1
                    if session.is_active and datetime.now() <= session.expires_at:
                        stats["active_sessions"] += 1
                        stats["unique_users"].add(session.user_id)
                        stats["unique_ips"].add(session.ip_address)
        
        # Convert sets to counts
        stats["unique_users"] = len(stats["unique_users"])
        stats["unique_ips"] = len(stats["unique_ips"])
        
        return stats
    
    def record_failed_attempt(self, ip_address: str):
        """Record failed session attempt"""
        
        now = datetime.now()
        
        if ip_address not in self.failed_session_attempts:
            self.failed_session_attempts[ip_address] = []
        
        # Add current attempt
        self.failed_session_attempts[ip_address].append(now)
        
        # Clean up old attempts
        cutoff = now - self.lockout_duration
        self.failed_session_attempts[ip_address] = [
            attempt for attempt in self.failed_session_attempts[ip_address]
            if attempt > cutoff
        ]
    
    def _is_ip_locked(self, ip_address: str) -> bool:
        """Check if IP is locked due to failed attempts"""
        
        if ip_address not in self.failed_session_attempts:
            return False
        
        recent_attempts = self.failed_session_attempts[ip_address]
        return len(recent_attempts) >= self.max_failed_attempts
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)
    
    def _store_session(self, session: SessionData):
        """Store session data"""
        
        if self.redis_client:
            try:
                # Store in Redis with TTL
                session_key = f"session:{session.session_id}"
                session_json = self._serialize_session(session)
                ttl = int((session.expires_at - datetime.now()).total_seconds())
                self.redis_client.setex(session_key, ttl, session_json)
            except Exception as e:
                print(f"Error storing session in Redis: {e}")
                # Fallback to memory
                with self.lock:
                    self.memory_sessions[session.session_id] = session
        else:
            # Store in memory
            with self.lock:
                self.memory_sessions[session.session_id] = session
    
    def _get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data"""
        
        if self.redis_client:
            try:
                session_key = f"session:{session_id}"
                session_data = self.redis_client.get(session_key)
                if session_data:
                    return self._deserialize_session(session_data)
            except Exception as e:
                print(f"Error getting session from Redis: {e}")
        
        # Try memory storage
        with self.lock:
            return self.memory_sessions.get(session_id)
    
    def _remove_session(self, session_id: str):
        """Remove session from storage"""
        
        if self.redis_client:
            try:
                session_key = f"session:{session_id}"
                self.redis_client.delete(session_key)
            except Exception as e:
                print(f"Error removing session from Redis: {e}")
        
        # Remove from memory
        with self.lock:
            self.memory_sessions.pop(session_id, None)
    
    def _serialize_session(self, session: SessionData) -> str:
        """Serialize session data to JSON"""
        
        session_dict = asdict(session)
        session_dict["created_at"] = session.created_at.isoformat()
        session_dict["last_activity"] = session.last_activity.isoformat()
        session_dict["expires_at"] = session.expires_at.isoformat()
        session_dict["permissions"] = list(session.permissions)
        
        return json.dumps(session_dict)
    
    def _deserialize_session(self, session_data: str) -> Optional[SessionData]:
        """Deserialize session data from JSON"""
        
        try:
            session_dict = json.loads(session_data)
            
            # Convert datetime strings
            session_dict["created_at"] = datetime.fromisoformat(session_dict["created_at"])
            session_dict["last_activity"] = datetime.fromisoformat(session_dict["last_activity"])
            session_dict["expires_at"] = datetime.fromisoformat(session_dict["expires_at"])
            session_dict["permissions"] = set(session_dict["permissions"])
            
            return SessionData(**session_dict)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error deserializing session: {e}")
            return None