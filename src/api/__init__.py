"""
API module for external integrations

Provides RESTful API endpoints for all core functionalities including
document processing, data extraction, dashboard generation, and export.
"""

from .app import create_app
from .auth import AuthManager
from .rate_limiter import RateLimiter
from .documentation import generate_openapi_spec

__all__ = ['create_app', 'AuthManager', 'RateLimiter', 'generate_openapi_spec']