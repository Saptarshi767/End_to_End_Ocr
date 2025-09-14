# Export System, API Integration, and Dashboard Sharing Features

This document describes the newly implemented features for data export, API integration, and dashboard sharing in the OCR Table Analytics system.

## 🚀 Features Implemented

### 1. Data Export System (`src/core/export_service.py`)

A comprehensive export system supporting multiple formats with batch processing capabilities.

#### Supported Export Formats
- **CSV**: Comma-separated values for data analysis
- **Excel**: Microsoft Excel format with metadata sheets
- **JSON**: Structured data with full metadata
- **PDF**: Formatted reports with tables and visualizations

#### Key Features
- **Single Table Export**: Export individual tables to any supported format
- **Dashboard Export**: Export complete dashboards with visualizations
- **Batch Export**: Process multiple tables simultaneously
- **Format Validation**: Automatic data validation before export
- **Metadata Preservation**: Maintain extraction confidence and source information

#### Usage Examples

```python
from src.core.export_service import ExportService

export_service = ExportService()

# Export single table to CSV
csv_data = export_service.export_table_data(table, "csv")

# Export to Excel file
excel_path = export_service.export_table_data(table, "excel", "output.xlsx")

# Batch export multiple tables
exported_files = export_service.batch_export_tables(tables, "csv", "exports/")

# Export dashboard with visualizations
pdf_path = export_service.export_dashboard(dashboard, "pdf", include_data=True)
```

### 2. RESTful API System (`src/api/`)

A complete FastAPI-based REST API with authentication, rate limiting, and comprehensive documentation.

#### API Endpoints

##### Document Processing
- `POST /documents/upload` - Upload and process documents
- `GET /documents/{id}/status` - Check processing status
- `GET /documents/{id}/tables` - Get extracted tables

##### Dashboard Management
- `POST /dashboards/generate` - Generate dashboard from data
- `GET /dashboards/{id}` - Retrieve dashboard configuration

##### Data Export
- `POST /export/table/{id}` - Export table data
- `POST /export/dashboard/{id}` - Export dashboard
- `POST /export/batch` - Batch export multiple tables

##### Conversational AI
- `POST /chat/ask` - Ask natural language questions

##### Sharing & Collaboration
- `POST /share/dashboard` - Create shareable links
- `GET /share/{share_id}` - Access shared dashboard
- `POST /embed/widget` - Create embeddable widgets
- `POST /team/invite` - Invite team members

#### Authentication & Security
- **JWT Token Authentication**: Secure user sessions
- **API Key Support**: Programmatic access
- **Rate Limiting**: Configurable limits per endpoint type
- **Permission-based Access**: Granular access control

#### Rate Limits (per hour)
- Upload: 10 requests
- Export: 50 requests
- Chat: 100 requests
- Dashboard: 20 requests
- General API: 1000 requests

#### Usage Examples

```bash
# Upload document
curl -X POST "https://api.example.com/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"

# Generate dashboard
curl -X POST "https://api.example.com/dashboards/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc_123", "options": {}}'

# Export table to CSV
curl -X POST "https://api.example.com/export/table/table_123" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"format": "csv"}'

# Ask question about data
curl -X POST "https://api.example.com/chat/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc_123", "question": "What is the average age?"}'
```

### 3. Dashboard Sharing System (`src/sharing/`)

Secure sharing and collaboration features for dashboards with embedded widgets.

#### Share Types
- **Public Links**: Open access without authentication
- **Secure Links**: Password-protected with expiration
- **Embed Widgets**: Embeddable components for external sites
- **Team Access**: Collaborative access with permissions

#### Sharing Features
- **Secure Link Generation**: Cryptographically secure share IDs
- **Access Control**: View, interact, export, and edit permissions
- **Expiration Management**: Time-based link expiration
- **Access Tracking**: Detailed analytics and logging
- **Password Protection**: Optional password security

#### Embed Widgets
- **Chart Widgets**: Individual chart embedding
- **KPI Widgets**: Key performance indicator displays
- **Dashboard Widgets**: Full dashboard embedding
- **Customizable Themes**: Light, dark, and custom styling

#### Usage Examples

```python
from src.sharing.share_manager import ShareManager, ShareType, Permission

share_manager = ShareManager()

# Create secure share link
share_link = share_manager.create_share_link(
    dashboard_id="dash_123",
    owner_id="user_456",
    share_type=ShareType.SECURE_LINK,
    permissions=[Permission.VIEW, Permission.INTERACT],
    expires_in_hours=24,
    password="secure123"
)

# Create embeddable widget
from src.sharing.embed_service import EmbedService

embed_service = EmbedService()
widget = embed_service.create_chart_embed(
    dashboard_id="dash_123",
    chart_id="chart_456",
    theme="dark",
    width=800,
    height=400
)

# Generate embed code
embed_code = embed_service.generate_embed_code(widget.widget_id)
```

## 🧪 Testing

Comprehensive test suites ensure reliability and accuracy:

### Test Coverage
- **Export Service Tests** (`tests/test_export_service.py`)
  - Format accuracy validation
  - Batch processing verification
  - Error handling scenarios
  - Large dataset performance

- **API Endpoint Tests** (`tests/test_api_endpoints.py`)
  - Authentication and authorization
  - Rate limiting functionality
  - Request/response validation
  - Error handling

- **Sharing System Tests** (`tests/test_sharing_system.py`)
  - Access control security
  - Link expiration handling
  - Embed widget generation
  - Analytics tracking

### Running Tests

```bash
# Run all export tests
pytest tests/test_export_service.py -v

# Run API tests
pytest tests/test_api_endpoints.py -v

# Run sharing tests
pytest tests/test_sharing_system.py -v

# Run all new feature tests
pytest tests/test_export_service.py tests/test_api_endpoints.py tests/test_sharing_system.py -v
```

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: Available at `/docs`
- **ReDoc**: Available at `/redoc`
- **OpenAPI Spec**: Available at `/openapi.json`

### Postman Collection
Generate a Postman collection for API testing:

```python
from src.api.documentation import generate_postman_collection
from src.api.app import create_app

app = create_app()
collection = generate_postman_collection(app)
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_BASE_URL=https://api.example.com

# Authentication
JWT_SECRET_KEY=your-secret-key
API_KEY_PREFIX=ocr_

# Rate Limiting
RATE_LIMIT_UPLOAD=10
RATE_LIMIT_EXPORT=50
RATE_LIMIT_CHAT=100

# Export Settings
EXPORT_MAX_FILE_SIZE=100MB
EXPORT_TEMP_DIR=/tmp/exports

# Sharing Settings
SHARE_BASE_URL=https://app.example.com
SHARE_DEFAULT_EXPIRY=24h
```

### Database Setup

The system requires database tables for sharing and API functionality:

```sql
-- Share links table
CREATE TABLE share_links (
    share_id VARCHAR(64) PRIMARY KEY,
    dashboard_id UUID NOT NULL,
    owner_id UUID NOT NULL,
    share_type VARCHAR(20) NOT NULL,
    permissions JSONB NOT NULL,
    password_hash VARCHAR(64),
    max_access INTEGER,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Access logs table
CREATE TABLE access_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    share_id VARCHAR(64) REFERENCES share_links(share_id),
    ip_address INET,
    user_agent TEXT,
    user_id UUID,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API keys table
CREATE TABLE api_keys (
    key_id VARCHAR(32) PRIMARY KEY,
    user_id UUID NOT NULL,
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(64) NOT NULL,
    permissions JSONB NOT NULL,
    rate_limit INTEGER DEFAULT 1000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_used TIMESTAMP
);
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
```

### Production Considerations

1. **Security**
   - Use HTTPS in production
   - Implement proper CORS policies
   - Secure API keys and JWT secrets
   - Regular security audits

2. **Performance**
   - Configure appropriate rate limits
   - Use Redis for caching
   - Implement database connection pooling
   - Monitor API performance

3. **Scalability**
   - Use load balancers for multiple instances
   - Implement horizontal scaling
   - Consider microservices architecture
   - Use CDN for static assets

## 📈 Monitoring & Analytics

### Health Checks
- `/health` endpoint for system status
- Database connectivity checks
- Service availability monitoring

### Metrics Collection
- Request/response times
- Error rates by endpoint
- Rate limit violations
- Export success rates

### Logging
- Structured logging with JSON format
- Request/response logging
- Error tracking with stack traces
- Performance metrics

## 🔄 Integration Examples

### Python SDK Example

```python
import requests

class OCRAnalyticsClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def upload_document(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/documents/upload",
                files=files,
                headers=self.headers
            )
        return response.json()
    
    def generate_dashboard(self, document_id):
        data = {"document_id": document_id}
        response = requests.post(
            f"{self.base_url}/dashboards/generate",
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def export_table(self, table_id, format="csv"):
        data = {"format": format}
        response = requests.post(
            f"{self.base_url}/export/table/{table_id}",
            json=data,
            headers=self.headers
        )
        return response.content if format in ['pdf', 'excel'] else response.json()

# Usage
client = OCRAnalyticsClient("https://api.example.com", "your-api-key")
result = client.upload_document("document.pdf")
dashboard = client.generate_dashboard(result["document_id"])
```

### JavaScript/Node.js Example

```javascript
class OCRAnalyticsClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async askQuestion(documentId, question) {
        const response = await fetch(`${this.baseUrl}/chat/ask`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                document_id: documentId,
                question: question
            })
        });
        return response.json();
    }
    
    async createShareLink(dashboardId, options = {}) {
        const response = await fetch(`${this.baseUrl}/share/dashboard`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                dashboard_id: dashboardId,
                share_type: 'secure_link',
                permissions: ['view'],
                ...options
            })
        });
        return response.json();
    }
}

// Usage
const client = new OCRAnalyticsClient('https://api.example.com', 'your-api-key');
const answer = await client.askQuestion('doc_123', 'What is the total revenue?');
const shareLink = await client.createShareLink('dash_456');
```

## 🎯 Next Steps

The implemented features provide a solid foundation for:

1. **Enterprise Integration**: RESTful API enables seamless integration with existing systems
2. **Data Accessibility**: Multiple export formats ensure data can be used in various tools
3. **Collaboration**: Sharing features enable team-based data analysis
4. **Embedding**: Widgets allow integration into external websites and applications

These features fulfill requirements 7.1, 7.2, 7.3, 7.4, and 3.7, providing comprehensive export, API, and sharing capabilities for the OCR Table Analytics system.