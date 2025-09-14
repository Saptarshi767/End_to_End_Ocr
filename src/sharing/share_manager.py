"""
Dashboard sharing manager

Handles creation and management of shareable dashboard links with
security controls, expiration, and access tracking.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..core.exceptions import SharingError
from ..core.models import Dashboard


class ShareType(Enum):
    """Types of dashboard sharing"""
    PUBLIC_LINK = "public_link"
    SECURE_LINK = "secure_link"
    EMBED_WIDGET = "embed_widget"
    TEAM_ACCESS = "team_access"


class Permission(Enum):
    """Sharing permissions"""
    VIEW = "view"
    INTERACT = "interact"
    EXPORT = "export"
    EDIT = "edit"


@dataclass
class ShareLink:
    """Shareable link model"""
    share_id: str
    dashboard_id: str
    owner_id: str
    share_type: ShareType
    permissions: List[Permission]
    share_url: str
    access_count: int = 0
    max_access: Optional[int] = None
    password_hash: Optional[str] = None
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    is_active: bool = True


@dataclass
class AccessLog:
    """Access log entry"""
    share_id: str
    ip_address: str
    user_agent: str
    accessed_at: datetime
    user_id: Optional[str] = None


class ShareManager:
    """Manager for dashboard sharing functionality"""
    
    def __init__(self, base_url: str = "https://app.example.com"):
        self.base_url = base_url.rstrip('/')
        
        # In-memory storage for demo (use database in production)
        self.share_links: Dict[str, ShareLink] = {}
        self.access_logs: List[AccessLog] = []
    
    def create_share_link(
        self,
        dashboard_id: str,
        owner_id: str,
        share_type: ShareType = ShareType.SECURE_LINK,
        permissions: List[Permission] = None,
        expires_in_hours: Optional[int] = None,
        max_access: Optional[int] = None,
        password: Optional[str] = None
    ) -> ShareLink:
        """
        Create a shareable link for dashboard
        
        Args:
            dashboard_id: Dashboard to share
            owner_id: Owner user ID
            share_type: Type of sharing
            permissions: List of permissions to grant
            expires_in_hours: Hours until expiration
            max_access: Maximum number of accesses
            password: Optional password protection
            
        Returns:
            ShareLink object
        """
        
        if permissions is None:
            permissions = [Permission.VIEW]
        
        # Generate secure share ID
        share_id = self._generate_share_id()
        
        # Calculate expiration
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        # Hash password if provided
        password_hash = None
        if password:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Generate share URL
        share_url = f"{self.base_url}/shared/{share_id}"
        
        # Create share link
        share_link = ShareLink(
            share_id=share_id,
            dashboard_id=dashboard_id,
            owner_id=owner_id,
            share_type=share_type,
            permissions=permissions,
            share_url=share_url,
            max_access=max_access,
            password_hash=password_hash,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        self.share_links[share_id] = share_link
        
        return share_link
    
    def get_share_link(self, share_id: str) -> Optional[ShareLink]:
        """Get share link by ID"""
        return self.share_links.get(share_id)
    
    def validate_access(
        self,
        share_id: str,
        password: Optional[str] = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> bool:
        """
        Validate access to shared dashboard
        
        Args:
            share_id: Share link ID
            password: Password if required
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            True if access is granted
        """
        
        share_link = self.share_links.get(share_id)
        
        if not share_link or not share_link.is_active:
            return False
        
        # Check expiration
        if share_link.expires_at and datetime.now() > share_link.expires_at:
            return False
        
        # Check access count
        if share_link.max_access and share_link.access_count >= share_link.max_access:
            return False
        
        # Check password
        if share_link.password_hash:
            if not password:
                return False
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if password_hash != share_link.password_hash:
                return False
        
        # Log access
        self._log_access(share_id, ip_address, user_agent)
        
        # Update access count and last accessed
        share_link.access_count += 1
        share_link.last_accessed = datetime.now()
        
        return True
    
    def revoke_share_link(self, share_id: str, owner_id: str) -> bool:
        """
        Revoke a share link
        
        Args:
            share_id: Share link ID
            owner_id: Owner user ID (for authorization)
            
        Returns:
            True if revoked successfully
        """
        
        share_link = self.share_links.get(share_id)
        
        if not share_link or share_link.owner_id != owner_id:
            return False
        
        share_link.is_active = False
        return True
    
    def list_dashboard_shares(self, dashboard_id: str, owner_id: str) -> List[ShareLink]:
        """
        List all share links for a dashboard
        
        Args:
            dashboard_id: Dashboard ID
            owner_id: Owner user ID
            
        Returns:
            List of share links
        """
        
        shares = []
        for share_link in self.share_links.values():
            if (share_link.dashboard_id == dashboard_id and 
                share_link.owner_id == owner_id):
                shares.append(share_link)
        
        return shares
    
    def get_share_analytics(self, share_id: str, owner_id: str) -> Dict[str, Any]:
        """
        Get analytics for a share link
        
        Args:
            share_id: Share link ID
            owner_id: Owner user ID
            
        Returns:
            Analytics data
        """
        
        share_link = self.share_links.get(share_id)
        
        if not share_link or share_link.owner_id != owner_id:
            raise SharingError("Share link not found or access denied")
        
        # Get access logs for this share
        share_logs = [log for log in self.access_logs if log.share_id == share_id]
        
        # Calculate analytics
        total_accesses = len(share_logs)
        unique_ips = len(set(log.ip_address for log in share_logs if log.ip_address))
        
        # Access by day (last 30 days)
        access_by_day = {}
        for log in share_logs:
            day = log.accessed_at.date()
            access_by_day[day] = access_by_day.get(day, 0) + 1
        
        return {
            "share_id": share_id,
            "total_accesses": total_accesses,
            "unique_ips": unique_ips,
            "created_at": share_link.created_at,
            "last_accessed": share_link.last_accessed,
            "expires_at": share_link.expires_at,
            "is_active": share_link.is_active,
            "access_by_day": access_by_day,
            "recent_accesses": [
                {
                    "ip_address": log.ip_address,
                    "accessed_at": log.accessed_at,
                    "user_agent": log.user_agent
                }
                for log in sorted(share_logs, key=lambda x: x.accessed_at, reverse=True)[:10]
            ]
        }
    
    def update_share_settings(
        self,
        share_id: str,
        owner_id: str,
        permissions: Optional[List[Permission]] = None,
        expires_in_hours: Optional[int] = None,
        max_access: Optional[int] = None,
        password: Optional[str] = None
    ) -> bool:
        """
        Update share link settings
        
        Args:
            share_id: Share link ID
            owner_id: Owner user ID
            permissions: New permissions
            expires_in_hours: New expiration time
            max_access: New max access count
            password: New password
            
        Returns:
            True if updated successfully
        """
        
        share_link = self.share_links.get(share_id)
        
        if not share_link or share_link.owner_id != owner_id:
            return False
        
        # Update permissions
        if permissions is not None:
            share_link.permissions = permissions
        
        # Update expiration
        if expires_in_hours is not None:
            if expires_in_hours > 0:
                share_link.expires_at = datetime.now() + timedelta(hours=expires_in_hours)
            else:
                share_link.expires_at = None
        
        # Update max access
        if max_access is not None:
            share_link.max_access = max_access
        
        # Update password
        if password is not None:
            if password:
                share_link.password_hash = hashlib.sha256(password.encode()).hexdigest()
            else:
                share_link.password_hash = None
        
        return True
    
    def cleanup_expired_shares(self):
        """Remove expired share links"""
        
        current_time = datetime.now()
        expired_shares = []
        
        for share_id, share_link in self.share_links.items():
            if (share_link.expires_at and current_time > share_link.expires_at):
                expired_shares.append(share_id)
        
        for share_id in expired_shares:
            del self.share_links[share_id]
        
        return len(expired_shares)
    
    def _generate_share_id(self) -> str:
        """Generate secure share ID"""
        return secrets.token_urlsafe(32)
    
    def _log_access(
        self,
        share_id: str,
        ip_address: Optional[str],
        user_agent: Optional[str],
        user_id: Optional[str] = None
    ):
        """Log access to shared dashboard"""
        
        access_log = AccessLog(
            share_id=share_id,
            ip_address=ip_address or "unknown",
            user_agent=user_agent or "unknown",
            accessed_at=datetime.now(),
            user_id=user_id
        )
        
        self.access_logs.append(access_log)
        
        # Keep only last 10000 logs to prevent memory issues
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-10000:]