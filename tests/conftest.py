"""
Shared test fixtures for the OCR Table Analytics system.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.core.database import Base
from src.core.repository import RepositoryManager
from src.core.config import SystemConfig, DatabaseConfig


@pytest.fixture
def test_db_config():
    """Test database configuration."""
    config = SystemConfig()
    config.database = DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_ocr_analytics",
        username="postgres",
        password="test_password"
    )
    return config


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Enable foreign key constraints for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal


@pytest.fixture
def db_session(in_memory_db):
    """Create database session for testing."""
    session = in_memory_db()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def repo_manager(db_session):
    """Create repository manager for testing."""
    return RepositoryManager(db_session)