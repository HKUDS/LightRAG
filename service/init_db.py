import sys
import os
import logging

# Add the service directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))

from app.core.database import engine, Base, SessionLocal
from app.models.models import ChatSession, ChatMessage, MessageCitation  # Import models to register them
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    logger.info("Checking database tables...")
    try:
        # Check if tables already exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        if existing_tables:
            logger.info(f"Database tables already exist: {existing_tables}")
            logger.info("Skipping table creation.")
        else:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=engine)
            logger.info("Tables created successfully!")
        
        logger.info("Database initialized.")
                
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    init_db()
