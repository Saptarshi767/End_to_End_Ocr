"""
API documentation generator for OpenAPI specifications
"""

from typing import Dict, Any
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI


def generate_openapi_spec(app: FastAPI) -> Dict[str, Any]:
    """
    Generate OpenAPI specification for the API
    
    Args:
        app: FastAPI application instance
        
    Returns:
        OpenAPI specification dictionary
    """
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="OCR Table Analytics API",
        version="1.0.0",
        description="""
        # OCR Table Analytics API
        
        A comprehensive API for OCR-based table recognition and conversational data analysis.
        
        ## Features
        
        - **Document Processing**: Upload and process documents with OCR
        - **Table Extraction**: Automatically extract and structure table data
        - **Dashboard Generation**: Create interactive dashboards from extracted data
        - **Conversational AI**: Ask questions about your data in natural language
        - **Data Export**: Export data and dashboards in multiple formats
        - **Sharing & Collaboration**: Share dashboards securely with teams
        - **Embed Widgets**: Create embeddable widgets for external sites
        
        ## Authentication
        
        The API supports two authentication methods:
        
        1. **JWT Tokens**: For user authentication
        2. **API Keys**: For programmatic access
        
        Include authentication in the `Authorization` header:
        ```
        Authorization: Bearer <token_or_api_key>
        ```
        
        ## Rate Limiting
        
        API endpoints are rate limited based on user and endpoint type:
        
        - **Upload**: 10 requests per hour
        - **Export**: 50 requests per hour  
        - **Chat**: 100 requests per hour
        - **Dashboard**: 20 requests per hour
        - **General API**: 1000 requests per hour
        
        Rate limit information is included in response headers:
        - `X-RateLimit-Limit`: Request limit
        - `X-RateLimit-Remaining`: Remaining requests
        - `X-RateLimit-Reset`: Reset timestamp
        
        ## Error Handling
        
        The API uses standard HTTP status codes:
        
        - `200`: Success
        - `400`: Bad Request - Invalid input
        - `401`: Unauthorized - Authentication required
        - `403`: Forbidden - Insufficient permissions
        - `404`: Not Found - Resource not found
        - `422`: Validation Error - Invalid request data
        - `429`: Too Many Requests - Rate limit exceeded
        - `500`: Internal Server Error
        
        Error responses include detailed information:
        ```json
        {
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": {
                "field": "document_id",
                "issue": "required field missing"
            }
        }
        ```
        
        ## Pagination
        
        List endpoints support pagination with query parameters:
        - `page`: Page number (default: 1)
        - `size`: Page size (default: 20, max: 100)
        
        ## Data Formats
        
        ### Supported Document Formats
        - PDF
        - PNG, JPEG, TIFF (images)
        - Multi-page documents
        
        ### Export Formats
        - **CSV**: Comma-separated values
        - **Excel**: Microsoft Excel (.xlsx)
        - **JSON**: JavaScript Object Notation
        - **PDF**: Portable Document Format (for dashboards)
        
        ## Webhooks
        
        Configure webhooks to receive notifications about processing events:
        - Document processing completed
        - Dashboard generated
        - Export completed
        
        ## SDKs and Libraries
        
        Official SDKs available for:
        - Python
        - JavaScript/Node.js
        - Java
        - C#
        
        ## Support
        
        For API support and questions:
        - Documentation: https://docs.example.com
        - Support: support@example.com
        - Status Page: https://status.example.com
        """,
        routes=app.routes,
        servers=[
            {"url": "https://api.example.com", "description": "Production server"},
            {"url": "https://staging-api.example.com", "description": "Staging server"},
            {"url": "http://localhost:8000", "description": "Development server"}
        ]
    )
    
    # Add custom schema components
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token or API key"
        }
    }
    
    # Add security requirement globally
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    # Add custom response schemas
    openapi_schema["components"]["schemas"].update({
        "ErrorResponse": {
            "type": "object",
            "properties": {
                "error": {"type": "string", "description": "Error type"},
                "message": {"type": "string", "description": "Error message"},
                "details": {"type": "object", "description": "Additional error details"},
                "timestamp": {"type": "string", "format": "date-time"}
            },
            "required": ["error", "message", "timestamp"]
        },
        "PaginatedResponse": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {}},
                "total": {"type": "integer", "description": "Total number of items"},
                "page": {"type": "integer", "description": "Current page number"},
                "size": {"type": "integer", "description": "Page size"},
                "pages": {"type": "integer", "description": "Total number of pages"}
            },
            "required": ["items", "total", "page", "size", "pages"]
        }
    })
    
    # Add examples for common operations
    openapi_schema["components"]["examples"] = {
        "DocumentUpload": {
            "summary": "Upload a PDF document",
            "description": "Example of uploading a PDF document for processing",
            "value": {
                "processing_options": {
                    "ocr_engine": "tesseract",
                    "preprocessing": True,
                    "table_detection": True,
                    "confidence_threshold": 0.8
                }
            }
        },
        "DashboardGeneration": {
            "summary": "Generate dashboard from document",
            "description": "Example of generating a dashboard from extracted tables",
            "value": {
                "document_id": "doc_123456789",
                "table_ids": ["table_1", "table_2"],
                "options": {
                    "auto_charts": True,
                    "include_kpis": True,
                    "theme": "light"
                }
            }
        },
        "ChatQuestion": {
            "summary": "Ask a question about data",
            "description": "Example of asking a natural language question",
            "value": {
                "document_id": "doc_123456789",
                "question": "What is the average age of customers by city?",
                "context": {
                    "previous_questions": [],
                    "focus_tables": ["customers"]
                }
            }
        },
        "ShareDashboard": {
            "summary": "Create shareable dashboard link",
            "description": "Example of creating a secure shareable link",
            "value": {
                "dashboard_id": "dash_123456789",
                "share_type": "secure_link",
                "permissions": ["view", "interact"],
                "expiry_date": "2024-12-31T23:59:59Z",
                "password_protected": True,
                "password": "secure123"
            }
        }
    }
    
    # Add tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "Health",
            "description": "Health check and system status endpoints"
        },
        {
            "name": "Documents",
            "description": "Document upload and processing operations"
        },
        {
            "name": "Dashboards", 
            "description": "Dashboard generation and management"
        },
        {
            "name": "Export",
            "description": "Data and dashboard export operations"
        },
        {
            "name": "Chat",
            "description": "Conversational AI and natural language queries"
        },
        {
            "name": "Sharing",
            "description": "Dashboard sharing and collaboration features"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def get_api_info() -> Dict[str, Any]:
    """
    Get basic API information
    
    Returns:
        API information dictionary
    """
    
    return {
        "name": "OCR Table Analytics API",
        "version": "1.0.0",
        "description": "API for OCR-based table recognition and conversational data analysis",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "authentication": ["JWT", "API Key"],
        "rate_limits": {
            "upload": "10/hour",
            "export": "50/hour", 
            "chat": "100/hour",
            "dashboard": "20/hour",
            "api": "1000/hour"
        },
        "supported_formats": {
            "input": ["PDF", "PNG", "JPEG", "TIFF"],
            "export": ["CSV", "Excel", "JSON", "PDF"]
        }
    }


def generate_postman_collection(app: FastAPI) -> Dict[str, Any]:
    """
    Generate Postman collection for API testing
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Postman collection dictionary
    """
    
    collection = {
        "info": {
            "name": "OCR Table Analytics API",
            "description": "Postman collection for OCR Table Analytics API",
            "version": "1.0.0",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "auth": {
            "type": "bearer",
            "bearer": [
                {
                    "key": "token",
                    "value": "{{auth_token}}",
                    "type": "string"
                }
            ]
        },
        "variable": [
            {
                "key": "base_url",
                "value": "https://api.example.com",
                "type": "string"
            },
            {
                "key": "auth_token",
                "value": "your_jwt_token_or_api_key",
                "type": "string"
            }
        ],
        "item": [
            {
                "name": "Health Check",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/health",
                        "host": ["{{base_url}}"],
                        "path": ["health"]
                    }
                }
            },
            {
                "name": "Upload Document",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Authorization",
                            "value": "Bearer {{auth_token}}"
                        }
                    ],
                    "body": {
                        "mode": "formdata",
                        "formdata": [
                            {
                                "key": "file",
                                "type": "file",
                                "src": []
                            },
                            {
                                "key": "processing_options",
                                "value": "{\"ocr_engine\": \"auto\", \"preprocessing\": true}",
                                "type": "text"
                            }
                        ]
                    },
                    "url": {
                        "raw": "{{base_url}}/documents/upload",
                        "host": ["{{base_url}}"],
                        "path": ["documents", "upload"]
                    }
                }
            },
            {
                "name": "Generate Dashboard",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Authorization",
                            "value": "Bearer {{auth_token}}"
                        },
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": "{\n  \"document_id\": \"doc_123\",\n  \"options\": {\n    \"auto_charts\": true\n  }\n}"
                    },
                    "url": {
                        "raw": "{{base_url}}/dashboards/generate",
                        "host": ["{{base_url}}"],
                        "path": ["dashboards", "generate"]
                    }
                }
            },
            {
                "name": "Ask Question",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Authorization",
                            "value": "Bearer {{auth_token}}"
                        },
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": "{\n  \"document_id\": \"doc_123\",\n  \"question\": \"What is the average age?\",\n  \"context\": {}\n}"
                    },
                    "url": {
                        "raw": "{{base_url}}/chat/ask",
                        "host": ["{{base_url}}"],
                        "path": ["chat", "ask"]
                    }
                }
            },
            {
                "name": "Export Table",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Authorization",
                            "value": "Bearer {{auth_token}}"
                        },
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": "{\n  \"format\": \"csv\",\n  \"output_path\": null\n}"
                    },
                    "url": {
                        "raw": "{{base_url}}/export/table/table_123",
                        "host": ["{{base_url}}"],
                        "path": ["export", "table", "table_123"]
                    }
                }
            },
            {
                "name": "Share Dashboard",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Authorization",
                            "value": "Bearer {{auth_token}}"
                        },
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": "{\n  \"dashboard_id\": \"dash_123\",\n  \"share_type\": \"secure_link\",\n  \"permissions\": [\"view\"],\n  \"password_protected\": false\n}"
                    },
                    "url": {
                        "raw": "{{base_url}}/share/dashboard",
                        "host": ["{{base_url}}"],
                        "path": ["share", "dashboard"]
                    }
                }
            }
        ]
    }
    
    return collection