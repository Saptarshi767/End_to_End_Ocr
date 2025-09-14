#!/bin/bash

# End-to-End OCR Deployment Script
# Usage: ./deploy.sh [development|production|docker]

set -e

MODE=${1:-development}
echo "🚀 Deploying OCR Analytics in $MODE mode..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 1 ]]; then
            print_status "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng libgl1-mesa-glx
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install tesseract
        else
            print_warning "Homebrew not found. Please install Tesseract manually."
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        print_warning "Please install Tesseract manually from: https://github.com/UB-Mannheim/tesseract/wiki"
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Created virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        python install_ocr_deps.py
    fi
    
    print_status "Python dependencies installed"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# UI Configuration
UI_HOST=0.0.0.0
UI_PORT=8501
UI_DEBUG=false

# OCR Settings
DEFAULT_OCR_ENGINE=auto
DEFAULT_CONFIDENCE=0.5
ENABLE_PREPROCESSING=true

# Performance
MAX_IMAGE_SIZE=10485760
PROCESSING_TIMEOUT=300
EOF
        print_status "Created .env file"
    fi
    
    # Create necessary directories
    mkdir -p uploads outputs logs
    print_status "Created necessary directories"
}

# Generate test images
generate_test_images() {
    print_status "Generating test images..."
    python create_test_table.py
}

# Development deployment
deploy_development() {
    print_status "Setting up development environment..."
    
    check_python
    install_system_deps
    install_python_deps
    setup_environment
    generate_test_images
    
    print_status "Development setup complete!"
    echo ""
    echo "🎯 To start the application:"
    echo "   source venv/bin/activate  # or venv/Scripts/activate on Windows"
    echo "   python run_ui.py"
    echo ""
    echo "📱 Then open: http://localhost:8501"
}

# Production deployment
deploy_production() {
    print_status "Setting up production environment..."
    
    check_python
    install_system_deps
    install_python_deps
    setup_environment
    
    # Create systemd service file
    cat > ocr-analytics.service << EOF
[Unit]
Description=OCR Analytics Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python run_ui.py --host 0.0.0.0 --port 8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    print_status "Production setup complete!"
    echo ""
    echo "🎯 To install as system service:"
    echo "   sudo cp ocr-analytics.service /etc/systemd/system/"
    echo "   sudo systemctl enable ocr-analytics"
    echo "   sudo systemctl start ocr-analytics"
    echo ""
    echo "📱 Service will be available at: http://your-server:8501"
}

# Docker deployment
deploy_docker() {
    print_status "Setting up Docker deployment..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Build Docker image
    print_status "Building Docker image..."
    docker build -t ocr-analytics .
    
    # Run with docker-compose if available
    if command -v docker-compose &> /dev/null && [ -f "docker-compose.yml" ]; then
        print_status "Starting with docker-compose..."
        docker-compose up -d
    else
        print_status "Starting Docker container..."
        docker run -d \
            --name ocr-analytics \
            -p 8501:8501 \
            -v $(pwd)/uploads:/app/uploads \
            -v $(pwd)/outputs:/app/outputs \
            ocr-analytics
    fi
    
    print_status "Docker deployment complete!"
    echo ""
    echo "📱 Application available at: http://localhost:8501"
    echo ""
    echo "🔧 Docker commands:"
    echo "   docker logs ocr-analytics     # View logs"
    echo "   docker stop ocr-analytics     # Stop container"
    echo "   docker start ocr-analytics    # Start container"
}

# Main deployment logic
case $MODE in
    "development"|"dev")
        deploy_development
        ;;
    "production"|"prod")
        deploy_production
        ;;
    "docker")
        deploy_docker
        ;;
    *)
        print_error "Unknown deployment mode: $MODE"
        echo "Usage: $0 [development|production|docker]"
        exit 1
        ;;
esac

print_status "Deployment completed successfully! 🎉"