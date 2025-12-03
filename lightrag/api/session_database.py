"""
Session History Database Configuration and Utilities

This module provides database connection and session management
for the LightRAG session history feature.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Optional
from lightrag.utils import logger
from urllib.parse import quote_plus


class SessionDatabaseConfig:
    """
    Configuration for session history database.
    
    Uses the same PostgreSQL configuration as LightRAG (POSTGRES_* env vars).
    Session history tables will be created in the same database as LightRAG data.
    """
    
    def __init__(self):
        """
        Initialize database configuration from environment variables.
        
        Uses POSTGRES_* variables directly - same database as LightRAG.
        """
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "password")
        self.database = os.getenv("POSTGRES_DATABASE", "lightrag_db")
        
        # Encode credentials to handle special characters
        encoded_user = quote_plus(self.user)
        encoded_password = quote_plus(self.password)
        
        self.database_url = f"postgresql://{encoded_user}:{encoded_password}@{self.host}:{self.port}/{self.database}"
        
        logger.info(f"Session database: {self.host}:{self.port}/{self.database}")


class SessionDatabaseManager:
    """Manages database connections for session history."""
    
    def __init__(self, config: Optional[SessionDatabaseConfig] = None):
        """
        Initialize database manager.
        
        Args:
            config: Database configuration. If None, creates default config.
        """
        self.config = config or SessionDatabaseConfig()
        self.engine = None
        self.SessionLocal = None
        
    def initialize(self):
        """Initialize database engine and session factory."""
        if self.engine is not None:
            logger.debug("Session database already initialized")
            return
            
        try:
            self.engine = create_engine(
                self.config.database_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("Session database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize session database: {e}")
            raise
    
    def create_tables(self):
        """Create all session history tables if they don't exist."""
        if self.engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
            
        try:
            from lightrag.api.session_models import Base
            Base.metadata.create_all(bind=self.engine)
            logger.info("Session history tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create session tables: {e}")
            raise
    
    def get_session(self):
        """
        Get a database session.
        
        Returns:
            SQLAlchemy session object.
            
        Raises:
            RuntimeError: If database not initialized.
        """
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope for database operations.
        
        Yields:
            Database session that will be committed on success or rolled back on error.
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Session database connections closed")


# Global database manager instance
_db_manager: Optional[SessionDatabaseManager] = None


def get_session_db_manager() -> SessionDatabaseManager:
    """
    Get the global session database manager instance.
    
    Returns:
        SessionDatabaseManager instance.
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = SessionDatabaseManager()
        _db_manager.initialize()
        _db_manager.create_tables()
    return _db_manager


def get_db():
    """
    Dependency function for FastAPI to get database session.
    
    Yields:
        Database session.
    """
    db_manager = get_session_db_manager()
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()

