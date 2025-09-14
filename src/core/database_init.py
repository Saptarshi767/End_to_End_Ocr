"""
Database initialization and setup utilities.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
import logging

from .database import Base, DatabaseManager, get_database_url
from .config import config_manager
from .models import ValidationResult

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Database initialization and setup manager."""
    
    def __init__(self, config=None):
        self.config = config or config_manager.config
        self.database_url = get_database_url(self.config)
        self.db_manager = None
    
    def check_database_connection(self) -> ValidationResult:
        """Check if database connection is working."""
        try:
            engine = create_engine(self.database_url, echo=False)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return ValidationResult(
                is_valid=True,
                confidence=1.0
            )
        
        except OperationalError as e:
            error_msg = f"Database connection failed: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                errors=[error_msg],
                confidence=0.0
            )
        
        except Exception as e:
            error_msg = f"Unexpected database error: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                errors=[error_msg],
                confidence=0.0
            )
    
    def check_database_exists(self) -> bool:
        """Check if the target database exists."""
        try:
            # Create connection to default database (usually 'postgres')
            base_url = self.database_url.rsplit('/', 1)[0] + '/postgres'
            engine = create_engine(base_url, echo=False)
            
            with engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT 1 FROM pg_database WHERE datname = :db_name"
                ), {"db_name": self.config.database.database})
                
                return result.fetchone() is not None
        
        except Exception as e:
            logger.error(f"Error checking database existence: {e}")
            return False
    
    def create_database(self) -> ValidationResult:
        """Create the database if it doesn't exist."""
        try:
            if self.check_database_exists():
                return ValidationResult(
                    is_valid=True,
                    warnings=["Database already exists"],
                    confidence=1.0
                )
            
            # Create connection to default database
            base_url = self.database_url.rsplit('/', 1)[0] + '/postgres'
            engine = create_engine(base_url, echo=False, isolation_level='AUTOCOMMIT')
            
            with engine.connect() as conn:
                conn.execute(text(f'CREATE DATABASE "{self.config.database.database}"'))
            
            logger.info(f"Database '{self.config.database.database}' created successfully")
            
            return ValidationResult(
                is_valid=True,
                confidence=1.0
            )
        
        except ProgrammingError as e:
            if "already exists" in str(e):
                return ValidationResult(
                    is_valid=True,
                    warnings=["Database already exists"],
                    confidence=1.0
                )
            else:
                error_msg = f"Error creating database: {str(e)}"
                logger.error(error_msg)
                return ValidationResult(
                    is_valid=False,
                    errors=[error_msg],
                    confidence=0.0
                )
        
        except Exception as e:
            error_msg = f"Unexpected error creating database: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                errors=[error_msg],
                confidence=0.0
            )
    
    def check_tables_exist(self) -> bool:
        """Check if database tables exist."""
        try:
            engine = create_engine(self.database_url, echo=False)
            with engine.connect() as conn:
                if 'sqlite' in self.database_url:
                    # SQLite query
                    result = conn.execute(text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
                    ))
                else:
                    # PostgreSQL query
                    result = conn.execute(text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_name = 'documents'"
                    ))
                return result.fetchone() is not None
        
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def create_tables(self) -> ValidationResult:
        """Create database tables."""
        try:
            self.db_manager = DatabaseManager(self.database_url)
            self.db_manager.create_tables()
            
            logger.info("Database tables created successfully")
            
            return ValidationResult(
                is_valid=True,
                confidence=1.0
            )
        
        except Exception as e:
            error_msg = f"Error creating tables: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                errors=[error_msg],
                confidence=0.0
            )
    
    def drop_tables(self) -> ValidationResult:
        """Drop all database tables."""
        try:
            if not self.db_manager:
                self.db_manager = DatabaseManager(self.database_url)
            
            self.db_manager.drop_tables()
            
            logger.info("Database tables dropped successfully")
            
            return ValidationResult(
                is_valid=True,
                confidence=1.0
            )
        
        except Exception as e:
            error_msg = f"Error dropping tables: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                errors=[error_msg],
                confidence=0.0
            )
    
    def run_migrations(self) -> ValidationResult:
        """Run database migrations using Alembic."""
        try:
            import subprocess
            
            # Change to project root directory
            project_root = Path(__file__).parent.parent.parent
            
            # Run alembic upgrade
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Database migrations completed successfully")
                return ValidationResult(
                    is_valid=True,
                    confidence=1.0
                )
            else:
                error_msg = f"Migration failed: {result.stderr}"
                logger.error(error_msg)
                return ValidationResult(
                    is_valid=False,
                    errors=[error_msg],
                    confidence=0.0
                )
        
        except FileNotFoundError:
            error_msg = "Alembic not found. Please install alembic: pip install alembic"
            logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                errors=[error_msg],
                confidence=0.0
            )
        
        except Exception as e:
            error_msg = f"Unexpected error running migrations: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(
                is_valid=False,
                errors=[error_msg],
                confidence=0.0
            )
    
    def initialize_database(self, force_recreate: bool = False) -> ValidationResult:
        """Initialize the complete database setup."""
        errors = []
        warnings = []
        
        logger.info("Starting database initialization...")
        
        # Step 1: Check database connection
        conn_result = self.check_database_connection()
        if not conn_result.is_valid:
            # Try to create database first
            create_result = self.create_database()
            if not create_result.is_valid:
                errors.extend(create_result.errors)
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    confidence=0.0
                )
            
            # Check connection again
            conn_result = self.check_database_connection()
            if not conn_result.is_valid:
                errors.extend(conn_result.errors)
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    confidence=0.0
                )
        
        # Step 2: Handle existing tables
        if self.check_tables_exist():
            if force_recreate:
                logger.info("Dropping existing tables...")
                drop_result = self.drop_tables()
                if not drop_result.is_valid:
                    errors.extend(drop_result.errors)
                    return ValidationResult(
                        is_valid=False,
                        errors=errors,
                        confidence=0.0
                    )
            else:
                warnings.append("Database tables already exist. Use force_recreate=True to recreate.")
                return ValidationResult(
                    is_valid=True,
                    warnings=warnings,
                    confidence=1.0
                )
        
        # Step 3: Create tables
        create_result = self.create_tables()
        if not create_result.is_valid:
            errors.extend(create_result.errors)
            return ValidationResult(
                is_valid=False,
                errors=errors,
                confidence=0.0
            )
        
        logger.info("Database initialization completed successfully")
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            confidence=1.0
        )
    
    def get_database_info(self) -> dict:
        """Get database information and statistics."""
        try:
            engine = create_engine(self.database_url, echo=False)
            
            with engine.connect() as conn:
                if 'sqlite' in self.database_url:
                    # SQLite queries
                    tables_result = conn.execute(text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    ))
                    
                    tables = [{"name": row[0], "size": "N/A"} for row in tables_result]
                    
                    return {
                        "database_name": "SQLite",
                        "database_size": "N/A",
                        "connection_count": 1,
                        "tables": tables,
                        "table_count": len(tables)
                    }
                else:
                    # PostgreSQL queries
                    tables_result = conn.execute(text(
                        "SELECT table_name, "
                        "pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size "
                        "FROM pg_tables WHERE schemaname = 'public'"
                    ))
                    
                    tables = [{"name": row[0], "size": row[1]} for row in tables_result]
                    
                    # Get database size
                    db_size_result = conn.execute(text(
                        "SELECT pg_size_pretty(pg_database_size(:db_name))"
                    ), {"db_name": self.config.database.database})
                    
                    db_size = db_size_result.fetchone()[0]
                    
                    # Get connection count
                    conn_count_result = conn.execute(text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = :db_name"
                    ), {"db_name": self.config.database.database})
                    
                    connection_count = conn_count_result.fetchone()[0]
                    
                    return {
                        "database_name": self.config.database.database,
                        "database_size": db_size,
                        "connection_count": connection_count,
                        "tables": tables,
                        "table_count": len(tables)
                    }
        
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}


def init_database_cli():
    """Command-line interface for database initialization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize OCR Analytics database")
    parser.add_argument("--force", action="store_true", 
                       help="Force recreate existing tables")
    parser.add_argument("--info", action="store_true",
                       help="Show database information")
    parser.add_argument("--check", action="store_true",
                       help="Check database connection")
    parser.add_argument("--migrate", action="store_true",
                       help="Run database migrations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    initializer = DatabaseInitializer()
    
    if args.check:
        result = initializer.check_database_connection()
        if result.is_valid:
            print("✓ Database connection successful")
        else:
            print("✗ Database connection failed:")
            for error in result.errors:
                print(f"  - {error}")
            sys.exit(1)
    
    elif args.info:
        info = initializer.get_database_info()
        if "error" in info:
            print(f"Error getting database info: {info['error']}")
            sys.exit(1)
        
        print(f"Database: {info['database_name']}")
        print(f"Size: {info['database_size']}")
        print(f"Active connections: {info['connection_count']}")
        print(f"Tables ({info['table_count']}):")
        for table in info['tables']:
            print(f"  - {table['name']}: {table['size']}")
    
    elif args.migrate:
        result = initializer.run_migrations()
        if result.is_valid:
            print("✓ Database migrations completed successfully")
        else:
            print("✗ Database migrations failed:")
            for error in result.errors:
                print(f"  - {error}")
            sys.exit(1)
    
    else:
        result = initializer.initialize_database(force_recreate=args.force)
        if result.is_valid:
            print("✓ Database initialization completed successfully")
            if result.warnings:
                print("Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
        else:
            print("✗ Database initialization failed:")
            for error in result.errors:
                print(f"  - {error}")
            sys.exit(1)


if __name__ == "__main__":
    init_database_cli()