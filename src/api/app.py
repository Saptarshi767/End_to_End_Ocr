"""
FastAPI application for OCR Table Analytics API

Provides RESTful endpoints for document processing, data extraction,
dashboard generation, and export functionality.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import asyncio
import os

from .auth import AuthManager
from .rate_limiter import RateLimiter
from .models import *
from .dependencies import get_current_user, get_rate_limiter
from ..core.services import DocumentProcessingService, DashboardService
from ..core.export_service import ExportService
from ..core.repository import DocumentRepository, TableRepository
from ..core.exceptions import ProcessingError, ExportError
from ..core.error_handler import handle_error, create_error_context, ErrorResponse
from ..core.logging_system import get_logger, get_metrics, get_audit_logger, get_monitor
from ..core.monitoring import get_health_status, system_monitor


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="OCR Table Analytics API",
        description="API for OCR-based table recognition and conversational data analysis",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize services
    auth_manager = AuthManager()
    rate_limiter = RateLimiter()
    document_service = DocumentProcessingService()
    dashboard_service = DashboardService()
    export_service = ExportService()
    document_repo = DocumentRepository()
    table_repo = TableRepository()
    
    # Initialize logging and monitoring
    logger = get_logger()
    metrics = get_metrics()
    audit_logger = get_audit_logger()
    monitor = get_monitor()
    
    # Start system monitoring
    system_monitor.start_monitoring()
    
    # Security scheme
    security = HTTPBearer()
    
    # Add error handling middleware
    @app.middleware("http")
    async def error_handling_middleware(request: Request, call_next):
        """Global error handling middleware"""
        
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            # Log request start
            logger.info(
                f"Request started: {request.method} {request.url.path}",
                component="api",
                operation="request_handling",
                request_id=request_id,
                metadata={
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else None
                }
            )
            
            # Record request metric
            metrics.increment_counter("api.requests.total", 
                                    tags={"method": request.method, "path": request.url.path})
            
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log successful request
            logger.info(
                f"Request completed: {response.status_code}",
                component="api",
                operation="request_handling",
                request_id=request_id,
                duration_ms=duration_ms,
                status="success",
                metadata={"status_code": response.status_code}
            )
            
            # Record success metrics
            metrics.record_timer("api.request.duration", duration_ms,
                               tags={"method": request.method, "status": str(response.status_code)})
            metrics.increment_counter("api.requests.success",
                                    tags={"method": request.method, "status": str(response.status_code)})
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create error context
            error_context = create_error_context(
                operation="request_handling",
                component="api",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else None
            )
            
            # Handle error
            error_response = handle_error(e, error_context, include_technical_details=False)
            
            # Record error metrics
            metrics.increment_counter("api.requests.errors",
                                    tags={"method": request.method, "error_type": type(e).__name__})
            metrics.record_timer("api.request.duration", duration_ms,
                               tags={"method": request.method, "status": "error"})
            
            # Return error response
            status_code = 500
            if isinstance(e, HTTPException):
                status_code = e.status_code
            elif "authentication" in str(e).lower():
                status_code = 401
            elif "authorization" in str(e).lower() or "permission" in str(e).lower():
                status_code = 403
            elif "not found" in str(e).lower():
                status_code = 404
            elif "validation" in str(e).lower():
                status_code = 422
            elif "rate limit" in str(e).lower():
                status_code = 429
            
            return JSONResponse(
                status_code=status_code,
                content=error_response.to_dict()
            )
    
    @app.get("/", tags=["Health"])
    async def root():
        """Health check endpoint"""
        return {
            "message": "OCR Table Analytics API",
            "version": "1.0.0",
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Detailed health check with comprehensive system status"""
        try:
            health_status = get_health_status()
            
            # Log health check request
            logger.info("Health check requested", 
                       component="api", operation="health_check")
            
            # Record health check metric
            metrics.increment_counter("api.health_check.requests")
            
            return health_status
            
        except Exception as e:
            error_context = create_error_context(
                operation="health_check",
                component="api"
            )
            error_response = handle_error(e, error_context)
            
            return JSONResponse(
                status_code=500,
                content=error_response.to_dict()
            )
    
    # Document Processing Endpoints
    
    @app.post("/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
    async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        processing_options: Optional[ProcessingOptionsRequest] = None,
        current_user: dict = Depends(get_current_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter)
    ):
        """Upload and process a document"""
        
        # Check rate limits
        if not rate_limiter.check_limit(current_user["user_id"], "upload"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")
            
            # Save uploaded file
            file_id = str(uuid.uuid4())
            file_path = f"uploads/{file_id}_{file.filename}"
            os.makedirs("uploads", exist_ok=True)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Create document record
            document = document_repo.create_document(
                filename=file.filename,
                file_path=file_path,
                file_size=len(content),
                mime_type=file.content_type,
                user_id=current_user["user_id"]
            )
            
            # Start background processing
            background_tasks.add_task(
                process_document_async,
                document.id,
                file_path,
                processing_options
            )
            
            return DocumentUploadResponse(
                document_id=document.id,
                filename=file.filename,
                status="processing",
                message="Document uploaded successfully and processing started"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    @app.get("/documents/{document_id}/status", response_model=DocumentStatusResponse, tags=["Documents"])
    async def get_document_status(
        document_id: str,
        current_user: dict = Depends(get_current_user)
    ):
        """Get document processing status"""
        
        try:
            document = document_repo.get_document(document_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Check ownership
            if document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return DocumentStatusResponse(
                document_id=document.id,
                filename=document.filename,
                status=document.processing_status,
                progress=document.processing_progress or 0,
                created_at=document.upload_timestamp,
                completed_at=document.completion_timestamp
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
    
    @app.get("/documents/{document_id}/tables", response_model=List[TableResponse], tags=["Documents"])
    async def get_document_tables(
        document_id: str,
        current_user: dict = Depends(get_current_user)
    ):
        """Get extracted tables from document"""
        
        try:
            document = document_repo.get_document(document_id)
            if not document or document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=404, detail="Document not found")
            
            tables = table_repo.get_tables_by_document(document_id)
            
            return [
                TableResponse(
                    table_id=table.id,
                    document_id=table.document_id,
                    table_index=table.table_index,
                    headers=table.headers,
                    row_count=len(table.data) if table.data else 0,
                    confidence_score=table.confidence_score,
                    created_at=table.created_at
                )
                for table in tables
            ]
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get tables: {str(e)}")
    
    # Dashboard Endpoints
    
    @app.post("/dashboards/generate", response_model=DashboardResponse, tags=["Dashboards"])
    async def generate_dashboard(
        request: DashboardGenerationRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """Generate dashboard from extracted table data"""
        
        try:
            # Verify document ownership
            document = document_repo.get_document(request.document_id)
            if not document or document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Get tables
            tables = table_repo.get_tables_by_document(request.document_id)
            if not tables:
                raise HTTPException(status_code=400, detail="No tables found for document")
            
            # Generate dashboard
            dashboard = dashboard_service.generate_dashboard(
                tables,
                options=request.options
            )
            
            return DashboardResponse(
                dashboard_id=dashboard.id,
                document_id=request.document_id,
                charts=dashboard.charts,
                kpis=dashboard.kpis,
                filters=dashboard.filters,
                layout=dashboard.layout,
                created_at=datetime.now()
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")
    
    @app.get("/dashboards/{dashboard_id}", response_model=DashboardResponse, tags=["Dashboards"])
    async def get_dashboard(
        dashboard_id: str,
        current_user: dict = Depends(get_current_user)
    ):
        """Get dashboard by ID"""
        
        try:
            dashboard = dashboard_service.get_dashboard(dashboard_id)
            if not dashboard:
                raise HTTPException(status_code=404, detail="Dashboard not found")
            
            # Check ownership through document
            document = document_repo.get_document(dashboard.document_id)
            if not document or document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return DashboardResponse(
                dashboard_id=dashboard.id,
                document_id=dashboard.document_id,
                charts=dashboard.charts,
                kpis=dashboard.kpis,
                filters=dashboard.filters,
                layout=dashboard.layout,
                created_at=dashboard.created_at
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")
    
    # Export Endpoints
    
    @app.post("/export/table/{table_id}", tags=["Export"])
    async def export_table(
        table_id: str,
        request: ExportRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """Export table data to specified format"""
        
        try:
            table = table_repo.get_table(table_id)
            if not table:
                raise HTTPException(status_code=404, detail="Table not found")
            
            # Check ownership
            document = document_repo.get_document(table.document_id)
            if not document or document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Export table
            output_path = export_service.export_table_data(
                table,
                request.format,
                request.output_path
            )
            
            if request.format in ['pdf', 'excel']:
                return FileResponse(
                    path=output_path,
                    filename=f"table_{table_id}.{request.format}",
                    media_type='application/octet-stream'
                )
            else:
                return JSONResponse(content={"data": output_path})
            
        except HTTPException:
            raise
        except ExportError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    
    @app.post("/export/dashboard/{dashboard_id}", tags=["Export"])
    async def export_dashboard(
        dashboard_id: str,
        request: DashboardExportRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """Export dashboard with visualizations"""
        
        try:
            dashboard = dashboard_service.get_dashboard(dashboard_id)
            if not dashboard:
                raise HTTPException(status_code=404, detail="Dashboard not found")
            
            # Check ownership
            document = document_repo.get_document(dashboard.document_id)
            if not document or document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Export dashboard
            output_path = export_service.export_dashboard(
                dashboard,
                request.format,
                request.include_data,
                request.output_path
            )
            
            if request.format == 'pdf':
                return FileResponse(
                    path=output_path,
                    filename=f"dashboard_{dashboard_id}.pdf",
                    media_type='application/pdf'
                )
            else:
                return JSONResponse(content={"data": output_path})
            
        except HTTPException:
            raise
        except ExportError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Dashboard export failed: {str(e)}")
    
    @app.post("/export/batch", tags=["Export"])
    async def batch_export_tables(
        request: BatchExportRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """Export multiple tables in batch"""
        
        try:
            tables = []
            for table_id in request.table_ids:
                table = table_repo.get_table(table_id)
                if table:
                    # Check ownership
                    document = document_repo.get_document(table.document_id)
                    if document and document.user_id == current_user["user_id"]:
                        tables.append(table)
            
            if not tables:
                raise HTTPException(status_code=400, detail="No accessible tables found")
            
            # Batch export
            exported_files = export_service.batch_export_tables(
                tables,
                request.format,
                request.output_dir
            )
            
            return BatchExportResponse(
                exported_files=exported_files,
                total_count=len(exported_files),
                format=request.format
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch export failed: {str(e)}")
    
    # Conversational AI Endpoints
    
    @app.post("/chat/ask", response_model=ChatResponse, tags=["Chat"])
    async def ask_question(
        request: ChatRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """Ask natural language question about data"""
        
        try:
            # Verify document access
            document = document_repo.get_document(request.document_id)
            if not document or document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Process question using conversational AI
            from ..ai.conversational_engine import ConversationalEngine
            
            ai_engine = ConversationalEngine()
            response = ai_engine.process_question(
                request.question,
                request.document_id,
                request.context
            )
            
            return ChatResponse(
                response=response.text_response,
                visualizations=response.visualizations,
                data_summary=response.data_summary,
                suggested_questions=response.suggested_questions
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")
    
    # Utility functions
    
    async def process_document_async(
        document_id: str,
        file_path: str,
        options: Optional[ProcessingOptionsRequest]
    ):
        """Background task for document processing"""
        try:
            # Update status to processing
            document_repo.update_document_status(document_id, "processing", 10)
            
            # Process document
            result = document_service.process_document(file_path, options)
            
            # Save extracted tables
            for i, table in enumerate(result.tables):
                table_repo.create_table(
                    document_id=document_id,
                    table_index=i,
                    headers=table.headers,
                    data=table.rows,
                    confidence_score=table.confidence,
                    extraction_metadata=table.metadata
                )
            
            # Update status to completed
            document_repo.update_document_status(document_id, "completed", 100)
            
        except Exception as e:
            # Update status to failed
            document_repo.update_document_status(document_id, "failed", 0, str(e))
    
    return app    

    # Sharing Endpoints
    
    @app.post("/share/dashboard", response_model=ShareResponse, tags=["Sharing"])
    async def create_dashboard_share(
        request: ShareRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """Create shareable link for dashboard"""
        
        try:
            # Validate dashboard access
            dashboard = dashboard_service.get_dashboard(request.dashboard_id)
            if not dashboard:
                raise HTTPException(status_code=404, detail="Dashboard not found")
            
            document = document_repo.get_document(dashboard.document_id)
            if not document or document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Create share link
            from ..sharing.share_manager import ShareManager, ShareType, Permission
            
            share_manager = ShareManager()
            
            # Map request permissions to enum
            permissions = [Permission(p) for p in request.permissions]
            share_type = ShareType(request.share_type)
            
            share_link = share_manager.create_share_link(
                dashboard_id=request.dashboard_id,
                owner_id=current_user["user_id"],
                share_type=share_type,
                permissions=permissions,
                expires_in_hours=int((request.expiry_date - datetime.now()).total_seconds() / 3600) if request.expiry_date else None,
                password=request.password if request.password_protected else None
            )
            
            # Generate embed code if needed
            embed_code = None
            if share_type == ShareType.EMBED_WIDGET:
                from ..sharing.embed_service import EmbedService
                embed_service = EmbedService()
                widget = embed_service.create_dashboard_embed(request.dashboard_id)
                embed_code = embed_service.generate_embed_code(widget.widget_id)
            
            return ShareResponse(
                share_id=share_link.share_id,
                share_url=share_link.share_url,
                embed_code=embed_code,
                permissions=request.permissions,
                created_at=share_link.created_at,
                expires_at=share_link.expires_at
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Share creation failed: {str(e)}")
    
    @app.get("/share/{share_id}", tags=["Sharing"])
    async def access_shared_dashboard(
        share_id: str,
        password: Optional[str] = None,
        request: Request = None
    ):
        """Access shared dashboard"""
        
        try:
            from ..sharing.share_manager import ShareManager
            
            share_manager = ShareManager()
            
            # Validate access
            ip_address = request.client.host if request else None
            user_agent = request.headers.get("user-agent") if request else None
            
            if not share_manager.validate_access(share_id, password, ip_address, user_agent):
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Get share link
            share_link = share_manager.get_share_link(share_id)
            if not share_link:
                raise HTTPException(status_code=404, detail="Share link not found")
            
            # Get dashboard
            dashboard = dashboard_service.get_dashboard(share_link.dashboard_id)
            if not dashboard:
                raise HTTPException(status_code=404, detail="Dashboard not found")
            
            # Return dashboard data based on permissions
            response_data = {
                "dashboard_id": dashboard.id,
                "title": getattr(dashboard, 'title', 'Shared Dashboard'),
                "permissions": [p.value for p in share_link.permissions]
            }
            
            # Add data based on permissions
            from ..sharing.share_manager import Permission
            
            if Permission.VIEW in share_link.permissions:
                response_data.update({
                    "charts": dashboard.charts,
                    "kpis": dashboard.kpis,
                    "layout": dashboard.layout
                })
            
            if Permission.INTERACT in share_link.permissions:
                response_data["filters"] = dashboard.filters
            
            return JSONResponse(content=response_data)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Access failed: {str(e)}")
    
    @app.post("/embed/widget", response_model=EmbedWidgetResponse, tags=["Sharing"])
    async def create_embed_widget(
        request: EmbedWidgetRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """Create embeddable widget"""
        
        try:
            # Validate dashboard access
            dashboard = dashboard_service.get_dashboard(request.dashboard_id)
            if not dashboard:
                raise HTTPException(status_code=404, detail="Dashboard not found")
            
            document = document_repo.get_document(dashboard.document_id)
            if not document or document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Create embed widget
            from ..sharing.embed_service import EmbedService, WidgetType
            
            embed_service = EmbedService()
            widget_type = WidgetType(request.widget_type)
            
            if widget_type == WidgetType.CHART:
                widget = embed_service.create_chart_embed(
                    dashboard_id=request.dashboard_id,
                    chart_id=request.widget_id,
                    customization=request.customization
                )
            elif widget_type == WidgetType.KPI:
                widget = embed_service.create_kpi_embed(
                    dashboard_id=request.dashboard_id,
                    kpi_id=request.widget_id,
                    customization=request.customization
                )
            else:
                widget = embed_service.create_dashboard_embed(
                    dashboard_id=request.dashboard_id,
                    customization=request.customization
                )
            
            # Generate embed code
            embed_code = embed_service.generate_embed_code(widget.widget_id)
            preview_url = f"{request.url.scheme}://{request.url.netloc}/embed/{widget.widget_id}"
            
            return EmbedWidgetResponse(
                widget_id=widget.widget_id,
                embed_code=embed_code,
                preview_url=preview_url,
                customization_options={
                    "theme": ["light", "dark", "auto"],
                    "auto_refresh": [True, False],
                    "show_title": [True, False],
                    "show_legend": [True, False]
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Widget creation failed: {str(e)}")
    
    @app.get("/embed/{widget_id}", tags=["Sharing"])
    async def get_embed_widget_data(widget_id: str):
        """Get embed widget data for rendering"""
        
        try:
            from ..sharing.embed_service import EmbedService
            
            embed_service = EmbedService()
            widget_data = embed_service.get_widget_data(widget_id)
            
            return JSONResponse(content=widget_data)
            
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Widget data retrieval failed: {str(e)}")
    
    @app.post("/team/invite", response_model=TeamAccessResponse, tags=["Sharing"])
    async def invite_team_members(
        request: TeamAccessRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """Invite team members to access dashboard"""
        
        try:
            # Validate dashboard access
            dashboard = dashboard_service.get_dashboard(request.dashboard_id)
            if not dashboard:
                raise HTTPException(status_code=404, detail="Dashboard not found")
            
            document = document_repo.get_document(dashboard.document_id)
            if not document or document.user_id != current_user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Process team invitations (implement email sending logic)
            invited_users = []
            failed_invitations = []
            
            for email in request.user_emails:
                try:
                    # Create team access record (implement team access storage)
                    # Send invitation email (implement email service)
                    invited_users.append(email)
                except Exception:
                    failed_invitations.append(email)
            
            # Generate team access URL
            access_url = f"{request.url.scheme}://{request.url.netloc}/team/{request.dashboard_id}"
            
            return TeamAccessResponse(
                dashboard_id=request.dashboard_id,
                invited_users=invited_users,
                failed_invitations=failed_invitations,
                access_url=access_url
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Team invitation failed: {str(e)}")