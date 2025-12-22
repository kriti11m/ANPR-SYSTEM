# Backend Service

FastAPI-based backend service for the ANPR system providing REST API endpoints and real-time processing capabilities.

## 🏗️ Structure

```
backend/
├── app/
│   ├── api/           # API routes and endpoints
│   ├── core/          # Core configurations and settings
│   ├── models/        # Database models (SQLAlchemy)
│   ├── services/      # Business logic services
│   └── main.py        # FastAPI application entry point
├── tests/             # Unit and integration tests
└── requirements.txt   # Python dependencies
```

## 🚀 Features

- **Async Processing**: High-performance async/await patterns
- **WebSocket Support**: Real-time communication for live feeds
- **Database Integration**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: JWT-based user authentication
- **API Documentation**: Auto-generated OpenAPI/Swagger docs
- **Background Tasks**: Celery for heavy processing tasks

## 🛠️ Setup

### Local Development
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost/anpr_db"
export SECRET_KEY="your-secret-key"

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker
```bash
docker build -t anpr-backend .
docker run -p 8000:8000 anpr-backend
```

## 📊 API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `POST /auth/refresh` - Token refresh

### ANPR Operations
- `POST /anpr/process` - Process single image/frame
- `GET /anpr/detections` - Get detection history
- `GET /anpr/detections/{id}` - Get specific detection
- `DELETE /anpr/detections/{id}` - Delete detection

### Real-time Feeds
- `WS /ws/feed/{feed_id}` - WebSocket for live video feed
- `GET /feeds/` - List available feeds
- `POST /feeds/` - Add new feed source
- `PUT /feeds/{id}` - Update feed configuration

### System Management
- `GET /health` - Health check endpoint
- `GET /metrics` - System metrics
- `GET /status` - Service status

## 🏛️ Database Models

### Detection
```python
class Detection(Base):
    id: UUID
    timestamp: datetime
    license_plate: str
    confidence: float
    bounding_box: dict
    image_path: str
    feed_id: UUID
```

### Feed
```python
class Feed(Base):
    id: UUID
    name: str
    source_type: str  # webcam, rtsp, file
    source_url: str
    is_active: bool
    created_at: datetime
```

## 🔧 Configuration

Environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `SECRET_KEY` - JWT secret key
- `REDIS_URL` - Redis connection for Celery
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_api.py
```

## 🚀 Deployment

### Docker Compose
```yaml
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/anpr_db
    depends_on:
      - db
      - redis
```

### Production Considerations
- Use gunicorn with uvicorn workers
- Set up reverse proxy (nginx)
- Configure SSL/TLS certificates
- Implement proper logging and monitoring
- Set up health checks and auto-scaling

## 📖 API Documentation

Access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## 🔍 Monitoring

The backend includes built-in monitoring endpoints:
- `/health` - Basic health check
- `/metrics` - Prometheus-compatible metrics
- `/status` - Detailed system status

## 🛠️ Development Tools

- **Code Formatting**: Black, isort
- **Linting**: flake8, mypy
- **Testing**: pytest, pytest-asyncio
- **Database**: Alembic for migrations
- **Task Queue**: Celery with Redis broker
