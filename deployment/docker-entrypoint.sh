#!/bin/bash
set -e

# Wait for database to be ready
wait_for_db() {
    echo "Waiting for database to be ready..."
    while ! nc -z postgres 5432; do
        sleep 1
    done
    echo "Database is ready!"
}

# Wait for Redis to be ready
wait_for_redis() {
    echo "Waiting for Redis to be ready..."
    while ! nc -z redis 6379; do
        sleep 1
    done
    echo "Redis is ready!"
}

# Run database migrations
run_migrations() {
    echo "Running database migrations..."
    cd /app
    python -m alembic upgrade head
}

# Initialize application data
init_app() {
    echo "Initializing application..."
    cd /app
    python -c "
from src.core.database_init import initialize_database
initialize_database()
print('Application initialized successfully!')
"
}

# Start the application based on command
start_app() {
    case "$1" in
        "api")
            echo "Starting API server..."
            cd /app
            python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
            ;;
        "ui")
            echo "Starting Streamlit UI..."
            cd /app
            streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
            ;;
        "worker")
            echo "Starting background worker..."
            cd /app
            python -m celery -A src.core.tasks worker --loglevel=info
            ;;
        "scheduler")
            echo "Starting task scheduler..."
            cd /app
            python -m celery -A src.core.tasks beat --loglevel=info
            ;;
        "migrate")
            wait_for_db
            run_migrations
            exit 0
            ;;
        "init")
            wait_for_db
            wait_for_redis
            run_migrations
            init_app
            exit 0
            ;;
        "shell")
            echo "Starting interactive shell..."
            cd /app
            python -c "
from src.core.database import get_session
from src.core.models import *
from src.security.auth_manager import AuthManager
from src.security.privacy_manager import PrivacyManager
print('OCR Analytics Shell - All modules imported')
print('Available: get_session, AuthManager, PrivacyManager, and all models')
"
            python
            ;;
        "test")
            echo "Running tests..."
            cd /app
            python -m pytest tests/ -v --tb=short
            ;;
        *)
            echo "Usage: $0 {api|ui|worker|scheduler|migrate|init|shell|test}"
            echo "  api       - Start API server"
            echo "  ui        - Start Streamlit UI"
            echo "  worker    - Start background worker"
            echo "  scheduler - Start task scheduler"
            echo "  migrate   - Run database migrations"
            echo "  init      - Initialize application"
            echo "  shell     - Start interactive shell"
            echo "  test      - Run tests"
            exit 1
            ;;
    esac
}

# Main execution
main() {
    # Set up environment
    export PYTHONPATH="/app:$PYTHONPATH"
    
    # Create necessary directories
    mkdir -p /app/data /app/logs /app/uploads /app/exports
    
    # Handle different startup modes
    if [ "$1" = "api" ] || [ "$1" = "ui" ] || [ "$1" = "worker" ] || [ "$1" = "scheduler" ]; then
        # For main services, wait for dependencies and initialize
        wait_for_db
        wait_for_redis
        
        # Run migrations if needed
        if [ ! -f /app/data/.migrated ]; then
            run_migrations
            touch /app/data/.migrated
        fi
        
        # Initialize app if needed
        if [ ! -f /app/data/.initialized ]; then
            init_app
            touch /app/data/.initialized
        fi
    fi
    
    # Start the requested service
    start_app "$1"
}

# Execute main function with all arguments
main "$@"