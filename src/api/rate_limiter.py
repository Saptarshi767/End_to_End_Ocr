"""
Rate limiting system for API endpoints
"""

import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int  # Number of requests allowed
    window: int    # Time window in seconds
    burst: int = 0 # Burst allowance


@dataclass
class UserRateState:
    """Rate limiting state for a user"""
    requests: deque = field(default_factory=deque)
    last_request: float = 0
    blocked_until: float = 0


class RateLimiter:
    """Token bucket rate limiter with sliding window"""
    
    def __init__(self):
        self.limits = {
            # Default limits per endpoint type
            "upload": RateLimit(requests=10, window=3600, burst=2),      # 10 uploads per hour, 2 burst
            "export": RateLimit(requests=50, window=3600, burst=5),      # 50 exports per hour, 5 burst
            "chat": RateLimit(requests=100, window=3600, burst=10),      # 100 questions per hour, 10 burst
            "dashboard": RateLimit(requests=20, window=3600, burst=3),   # 20 dashboards per hour, 3 burst
            "api": RateLimit(requests=1000, window=3600, burst=50),      # 1000 API calls per hour, 50 burst
        }
        
        # User state tracking
        self.user_states: Dict[str, Dict[str, UserRateState]] = defaultdict(
            lambda: defaultdict(UserRateState)
        )
        
        # Thread lock for concurrent access
        self.lock = threading.Lock()
    
    def check_limit(
        self, 
        user_id: str, 
        endpoint_type: str,
        cost: int = 1
    ) -> bool:
        """
        Check if request is within rate limits
        
        Args:
            user_id: User identifier
            endpoint_type: Type of endpoint (upload, export, etc.)
            cost: Request cost (default 1)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        
        with self.lock:
            current_time = time.time()
            
            # Get rate limit for endpoint type
            limit = self.limits.get(endpoint_type, self.limits["api"])
            
            # Get user state for this endpoint
            user_state = self.user_states[user_id][endpoint_type]
            
            # Check if user is currently blocked
            if current_time < user_state.blocked_until:
                return False
            
            # Clean old requests outside the window
            cutoff_time = current_time - limit.window
            while user_state.requests and user_state.requests[0] <= cutoff_time:
                user_state.requests.popleft()
            
            # Check if adding this request would exceed the limit
            if len(user_state.requests) + cost > limit.requests:
                # Check burst allowance
                time_since_last = current_time - user_state.last_request
                if time_since_last < 1.0 and len(user_state.requests) + cost > limit.requests + limit.burst:
                    # Block user for a short period
                    user_state.blocked_until = current_time + 60  # 1 minute block
                    return False
                elif len(user_state.requests) + cost > limit.requests + limit.burst:
                    return False
            
            # Allow the request
            for _ in range(cost):
                user_state.requests.append(current_time)
            user_state.last_request = current_time
            
            return True
    
    def get_limit_info(self, user_id: str, endpoint_type: str) -> Dict[str, any]:
        """
        Get rate limit information for user and endpoint
        
        Args:
            user_id: User identifier
            endpoint_type: Type of endpoint
            
        Returns:
            Dictionary with limit information
        """
        
        with self.lock:
            current_time = time.time()
            
            # Get rate limit for endpoint type
            limit = self.limits.get(endpoint_type, self.limits["api"])
            
            # Get user state for this endpoint
            user_state = self.user_states[user_id][endpoint_type]
            
            # Clean old requests
            cutoff_time = current_time - limit.window
            while user_state.requests and user_state.requests[0] <= cutoff_time:
                user_state.requests.popleft()
            
            # Calculate remaining requests
            remaining = max(0, limit.requests - len(user_state.requests))
            
            # Calculate reset time
            if user_state.requests:
                reset_time = user_state.requests[0] + limit.window
            else:
                reset_time = current_time + limit.window
            
            # Calculate retry after if blocked
            retry_after = None
            if current_time < user_state.blocked_until:
                retry_after = int(user_state.blocked_until - current_time)
            
            return {
                "limit": limit.requests,
                "remaining": remaining,
                "reset_time": reset_time,
                "window": limit.window,
                "burst": limit.burst,
                "retry_after": retry_after,
                "blocked_until": user_state.blocked_until if current_time < user_state.blocked_until else None
            }
    
    def set_custom_limit(
        self, 
        user_id: str, 
        endpoint_type: str, 
        requests: int, 
        window: int,
        burst: int = 0
    ):
        """
        Set custom rate limit for specific user and endpoint
        
        Args:
            user_id: User identifier
            endpoint_type: Type of endpoint
            requests: Number of requests allowed
            window: Time window in seconds
            burst: Burst allowance
        """
        
        # Store custom limit (in production, persist to database)
        custom_key = f"{user_id}:{endpoint_type}"
        self.limits[custom_key] = RateLimit(requests=requests, window=window, burst=burst)
    
    def reset_user_limits(self, user_id: str, endpoint_type: Optional[str] = None):
        """
        Reset rate limits for user
        
        Args:
            user_id: User identifier
            endpoint_type: Specific endpoint type, or None for all
        """
        
        with self.lock:
            if endpoint_type:
                if user_id in self.user_states and endpoint_type in self.user_states[user_id]:
                    del self.user_states[user_id][endpoint_type]
            else:
                if user_id in self.user_states:
                    del self.user_states[user_id]
    
    def get_user_stats(self, user_id: str) -> Dict[str, Dict[str, any]]:
        """
        Get rate limiting statistics for user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with stats for each endpoint type
        """
        
        stats = {}
        
        for endpoint_type in self.limits.keys():
            if ":" not in endpoint_type:  # Skip custom user limits
                stats[endpoint_type] = self.get_limit_info(user_id, endpoint_type)
        
        return stats
    
    def cleanup_old_states(self, max_age: int = 86400):
        """
        Clean up old user states to prevent memory leaks
        
        Args:
            max_age: Maximum age in seconds for keeping states
        """
        
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - max_age
            
            users_to_remove = []
            
            for user_id, endpoints in self.user_states.items():
                endpoints_to_remove = []
                
                for endpoint_type, state in endpoints.items():
                    # Remove if no recent activity
                    if state.last_request < cutoff_time and not state.requests:
                        endpoints_to_remove.append(endpoint_type)
                
                # Remove empty endpoint states
                for endpoint_type in endpoints_to_remove:
                    del endpoints[endpoint_type]
                
                # Mark user for removal if no endpoints left
                if not endpoints:
                    users_to_remove.append(user_id)
            
            # Remove empty user states
            for user_id in users_to_remove:
                del self.user_states[user_id]


class RateLimitMiddleware:
    """Middleware for applying rate limits to FastAPI"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request, call_next):
        """Process request with rate limiting"""
        
        # Extract user ID from request (implement based on auth system)
        user_id = self._extract_user_id(request)
        
        if user_id:
            # Determine endpoint type from path
            endpoint_type = self._get_endpoint_type(request.url.path)
            
            # Check rate limit
            if not self.rate_limiter.check_limit(user_id, endpoint_type):
                from fastapi import HTTPException
                
                limit_info = self.rate_limiter.get_limit_info(user_id, endpoint_type)
                
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
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        if user_id:
            limit_info = self.rate_limiter.get_limit_info(user_id, endpoint_type)
            response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(int(limit_info["reset_time"]))
        
        return response
    
    def _extract_user_id(self, request) -> Optional[str]:
        """Extract user ID from request"""
        # Implement based on your authentication system
        # This could check JWT token, API key, etc.
        return getattr(request.state, "user_id", None)
    
    def _get_endpoint_type(self, path: str) -> str:
        """Determine endpoint type from request path"""
        
        if "/upload" in path:
            return "upload"
        elif "/export" in path:
            return "export"
        elif "/chat" in path:
            return "chat"
        elif "/dashboard" in path:
            return "dashboard"
        else:
            return "api"