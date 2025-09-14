"""
FastAPI dependencies for authentication, rate limiting, and common functionality
"""

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional
import re

from .auth import AuthManager
from .rate_limiter import RateLimiter
from ..core.exceptions import AuthenticationError, AuthorizationError


# Global instances (in production, use dependency injection)
auth_manager = AuthManager()
rate_limiter = RateLimiter()
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token or API key
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        User information dictionary
        
    Raises:
        HTTPException: If authentication fails
    """
    
    token = credentials.credentials
    
    try:
        # Try JWT token first
        if token.startswith("eyJ"):  # JWT tokens start with eyJ
            payload = auth_manager.verify_token(token)
            if payload:
                return {
                    "user_id": payload["user_id"],
                    "email": payload["email"],
                    "permissions": payload["permissions"],
                    "auth_type": "jwt"
                }
        
        # Try API key
        elif token.startswith("ocr_"):
            api_key_obj = auth_manager.verify_api_key(token)
            if api_key_obj:
                user = auth_manager.users.get(api_key_obj.user_id)
                if user:
                    return {
                        "user_id": user.user_id,
                        "email": user.email,
                        "permissions": api_key_obj.permissions,
                        "auth_type": "api_key",
                        "api_key_id": api_key_obj.key_id
                    }
        
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")


async def get_optional_user(
    request: Request
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, None otherwise
    Used for endpoints that work with or without authentication
    
    Args:
        request: FastAPI request object
        
    Returns:
        User information dictionary or None
    """
    
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        return await get_current_user(credentials)
        
    except HTTPException:
        return None


def require_permission(permission: str):
    """
    Dependency factory for requiring specific permissions
    
    Args:
        permission: Required permission string
        
    Returns:
        Dependency function
    """
    
    async def check_permission(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Check if user has required permission"""
        
        user_permissions = current_user.get("permissions", [])
        
        if not auth_manager.check_permission(user_permissions, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        
        return current_user
    
    return check_permission


def require_permissions(permissions: list):
    """
    Dependency factory for requiring multiple permissions (all required)
    
    Args:
        permissions: List of required permission strings
        
    Returns:
        Dependency function
    """
    
    async def check_permissions(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Check if user has all required permissions"""
        
        user_permissions = current_user.get("permissions", [])
        
        for permission in permissions:
            if not auth_manager.check_permission(user_permissions, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {', '.join(permissions)}"
                )
        
        return current_user
    
    return check_permissions


def require_any_permission(permissions: list):
    """
    Dependency factory for requiring any of the specified permissions
    
    Args:
        permissions: List of permission strings (any one required)
        
    Returns:
        Dependency function
    """
    
    async def check_any_permission(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Check if user has any of the required permissions"""
        
        user_permissions = current_user.get("permissions", [])
        
        for permission in permissions:
            if auth_manager.check_permission(user_permissions, permission):
                return current_user
        
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Required any of: {', '.join(permissions)}"
        )
    
    return check_any_permission


async def get_rate_limiter() -> RateLimiter:
    """
    Get rate limiter instance
    
    Returns:
        RateLimiter instance
    """
    return rate_limiter


def rate_limit(endpoint_type: str, cost: int = 1):
    """
    Dependency factory for rate limiting specific endpoints
    
    Args:
        endpoint_type: Type of endpoint for rate limiting
        cost: Cost of the request (default 1)
        
    Returns:
        Dependency function
    """
    
    async def check_rate_limit(
        current_user: Dict[str, Any] = Depends(get_current_user),
        limiter: RateLimiter = Depends(get_rate_limiter)
    ) -> Dict[str, Any]:
        """Check rate limits for user and endpoint"""
        
        user_id = current_user["user_id"]
        
        if not limiter.check_limit(user_id, endpoint_type, cost):
            limit_info = limiter.get_limit_info(user_id, endpoint_type)
            
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(limit_info["limit"]),
                    "X-RateLimit-Remaining": str(limit_info["remaining"]),
                    "X-RateLimit-Reset": str(int(limit_info["reset_time"])),
                    "Retry-After": str(limit_info.get("retry_after", 60))
                }
            )
        
        return current_user
    
    return check_rate_limit


async def validate_document_access(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate that user has access to specified document
    
    Args:
        document_id: Document ID to check access for
        current_user: Current authenticated user
        
    Returns:
        User information if access is granted
        
    Raises:
        HTTPException: If access is denied
    """
    
    # Import here to avoid circular imports
    from ..core.repository import DocumentRepository
    
    doc_repo = DocumentRepository()
    document = doc_repo.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check ownership or shared access
    if document.user_id != current_user["user_id"]:
        # Check if document is shared with user (implement sharing logic)
        if not _check_document_sharing(document_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to document")
    
    return current_user


async def validate_dashboard_access(
    dashboard_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Validate that user has access to specified dashboard
    
    Args:
        dashboard_id: Dashboard ID to check access for
        current_user: Current authenticated user
        
    Returns:
        User information if access is granted
        
    Raises:
        HTTPException: If access is denied
    """
    
    # Import here to avoid circular imports
    from ..core.services import DashboardService
    from ..core.repository import DocumentRepository
    
    dashboard_service = DashboardService()
    doc_repo = DocumentRepository()
    
    dashboard = dashboard_service.get_dashboard(dashboard_id)
    
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    # Check access through document ownership
    document = doc_repo.get_document(dashboard.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Associated document not found")
    
    if document.user_id != current_user["user_id"]:
        # Check if dashboard is shared with user
        if not _check_dashboard_sharing(dashboard_id, current_user["user_id"]):
            raise HTTPException(status_code=403, detail="Access denied to dashboard")
    
    return current_user


def _check_document_sharing(document_id: str, user_id: str) -> bool:
    """
    Check if document is shared with user
    
    Args:
        document_id: Document ID
        user_id: User ID to check access for
        
    Returns:
        True if user has access, False otherwise
    """
    
    # Implement document sharing logic
    # This would check a sharing table in the database
    # For now, return False (no sharing implemented)
    return False


def _check_dashboard_sharing(dashboard_id: str, user_id: str) -> bool:
    """
    Check if dashboard is shared with user
    
    Args:
        dashboard_id: Dashboard ID
        user_id: User ID to check access for
        
    Returns:
        True if user has access, False otherwise
    """
    
    # Implement dashboard sharing logic
    # This would check a sharing table in the database
    # For now, return False (no sharing implemented)
    return False


async def get_pagination_params(
    page: int = 1,
    size: int = 20,
    max_size: int = 100
) -> Dict[str, int]:
    """
    Get pagination parameters with validation
    
    Args:
        page: Page number (1-based)
        size: Page size
        max_size: Maximum allowed page size
        
    Returns:
        Dictionary with pagination parameters
        
    Raises:
        HTTPException: If parameters are invalid
    """
    
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be >= 1")
    
    if size < 1:
        raise HTTPException(status_code=400, detail="Size must be >= 1")
    
    if size > max_size:
        raise HTTPException(status_code=400, detail=f"Size must be <= {max_size}")
    
    return {
        "page": page,
        "size": size,
        "offset": (page - 1) * size,
        "limit": size
    }