#!/bin/bash

# ANPR System Setup Script
# This script sets up the development environment for the ANPR system

set -e

echo "🚀 Setting up ANPR System Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is required but not installed"
        exit 1
    fi
}

# Check if Node.js is installed
check_node() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js $NODE_VERSION found"
    else
        print_error "Node.js is required but not installed"
        exit 1
    fi
}

# Check if Docker is installed
check_docker() {
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
        print_status "Docker $DOCKER_VERSION found"
    else
        print_warning "Docker not found. Database will need to be set up manually"
        return 1
    fi
}

# Setup Python virtual environment for backend
setup_backend() {
    print_status "Setting up backend environment..."
    
    cd backend
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Created Python virtual environment"
    fi
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_status "Backend dependencies installed"
    cd ..
}

# Setup Python virtual environment for vision module
setup_vision() {
    print_status "Setting up vision module environment..."
    
    cd vision
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Created Python virtual environment for vision module"
    fi
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_status "Vision module dependencies installed"
    
    # Create models directory
    mkdir -p models
    print_status "Created models directory"
    
    cd ..
}

# Setup Node.js dependencies for frontend
setup_frontend() {
    print_status "Setting up frontend environment..."
    
    cd frontend
    
    # Install Node.js dependencies
    npm install
    
    print_status "Frontend dependencies installed"
    cd ..
}

# Setup database with Docker
setup_database() {
    if check_docker; then
        print_status "Setting up database with Docker..."
        
        cd database
        docker-compose up -d
        
        # Wait for database to be ready
        print_status "Waiting for database to be ready..."
        sleep 10
        
        # Check if database is accessible
        if docker-compose exec postgres pg_isready -U anpr_user -d anpr_db; then
            print_status "Database is ready"
        else
            print_warning "Database might not be fully ready. Please check manually."
        fi
        
        cd ..
    else
        print_warning "Skipping database setup. Please set up PostgreSQL manually."
    fi
}

# Create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p uploads
    mkdir -p logs
    mkdir -p config
    
    print_status "Directories created"
}

# Copy environment configuration
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_status "Created .env file from template"
        print_warning "Please update .env file with your actual configuration values"
    else
        print_status ".env file already exists"
    fi
}

# Download sample models (placeholder)
download_models() {
    print_status "Setting up model files..."
    
    mkdir -p vision/models
    
    # Create placeholder files for models
    echo "# YOLOv11 ANPR model placeholder" > vision/models/yolov11_anpr.pt.placeholder
    echo "# OCR model placeholder" > vision/models/ocr_model.onnx.placeholder
    
    print_warning "Model files are placeholders. Please download actual YOLOv11 and OCR models"
    print_status "Model directory structure created"
}

# Main setup function
main() {
    echo "=========================================="
    echo "🚗 ANPR System Setup"
    echo "=========================================="
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    check_python
    check_node
    
    # Setup project components
    setup_directories
    setup_env
    download_models
    
    setup_backend
    setup_vision
    setup_frontend
    setup_database
    
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Update .env file with your configuration"
    echo "2. Download actual YOLOv11 and OCR model files"
    echo "3. Start the backend: cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
    echo "4. Start the frontend: cd frontend && npm start"
    echo "5. Access the application at http://localhost:3000"
    echo ""
    echo "For database management, access Adminer at http://localhost:8080"
    echo ""
    print_status "Happy coding! 🎉"
}

# Run main function
main "$@"
