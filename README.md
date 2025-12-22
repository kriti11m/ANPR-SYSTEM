# Real-Time ANPR (Automatic Number Plate Recognition) System

A comprehensive monorepo solution for real-time license plate detection and recognition using YOLOv11, FastAPI, PostgreSQL, and React.

## 🏗️ Architecture Overview

```
anpr-system/
├── backend/           # FastAPI backend service
├── frontend/          # React.js web dashboard
├── vision/            # Computer vision processing with YOLOv11
├── database/          # PostgreSQL schemas and migrations
├── scripts/           # Deployment and utility scripts
├── config/            # Configuration files
├── docker/            # Docker configurations
└── docs/              # Documentation
```

## 🚀 Features

- **Real-time Processing**: Supports webcam, RTSP streams, and video files
- **YOLOv11 Integration**: State-of-the-art object detection for license plates
- **REST API**: FastAPI backend with async processing
- **Real-time Dashboard**: React frontend with live updates
- **Database Storage**: PostgreSQL for storing detection results
- **Scalable Architecture**: Microservices-ready design

## 📋 Prerequisites

- Python 3.9+
- Node.js 16+
- PostgreSQL 13+
- Docker & Docker Compose (optional)
- OpenCV compatible camera or RTSP stream

## 🛠️ Quick Start

### Using Docker Compose (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd anpr-system

# Start all services
docker-compose up -d

# Access the dashboard
open http://localhost:3000
```

### Manual Setup
```bash
# Backend setup
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend setup
cd frontend
npm install
npm start

# Vision service setup
cd vision
pip install -r requirements.txt
python main.py
```

## 📁 Project Structure

### Backend (`/backend`)
- FastAPI application with async processing
- RESTful API endpoints for ANPR operations
- Database models and migrations
- Authentication and authorization
- Real-time WebSocket connections

### Frontend (`/frontend`)
- React.js dashboard for monitoring
- Real-time video feeds display
- Detection results visualization
- System configuration interface
- Historical data analysis

### Vision (`/vision`)
- YOLOv11 model integration
- Frame processing pipeline
- License plate detection and recognition
- Multiple input source support (webcam, RTSP, video)
- OCR integration for text recognition

### Database (`/database`)
- PostgreSQL schema definitions
- Migration scripts
- Seed data for testing
- Backup and restore procedures

## 🔧 Configuration

Configuration files are located in the `/config` directory:
- `app.yaml` - Application settings
- `database.yaml` - Database connection settings
- `vision.yaml` - Computer vision model configurations

## 📊 API Documentation

Once the backend is running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test

# Vision module tests
cd vision
python -m pytest tests/
```

## 🚀 Deployment

See the `/scripts/deployment` directory for production deployment guides:
- Docker deployment
- Kubernetes manifests
- Cloud provider configurations

## 📖 Documentation

Detailed documentation is available in the `/docs` directory:
- API Reference
- Architecture Guide
- Deployment Instructions
- Troubleshooting Guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation in `/docs`
- Review the troubleshooting guide

## 🏷️ Version

Current version: v1.0.0

---

**Built with ❤️ using YOLOv11, FastAPI, React, and PostgreSQL**
