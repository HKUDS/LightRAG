import sys
import os
import logging

# Add the service directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))

from app.core.database import engine, Base, SessionLocal
from app.models.models import User, ChatSession, ChatMessage, MessageCitation  # Import models to register them
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    logger.info("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Tables created successfully!")
        
        # Create default users from AUTH_ACCOUNTS
        if settings.AUTH_ACCOUNTS:
            db = SessionLocal()
            try:
                accounts = settings.AUTH_ACCOUNTS.split(',')
                for account in accounts:
                    if ':' in account:
                        username, password = account.split(':', 1)
                        username = username.strip()
                        # Check if user exists
                        existing_user = db.query(User).filter(User.username == username).first()
                        if not existing_user:
                            logger.info(f"Creating default user: {username}")
                            # Note: In a real app, password should be hashed. 
                            # For now, we are just creating the user record. 
                            # The User model doesn't have a password field in the provided schema, 
                            # so we might need to add it or just store the user for now.
                            # Looking at models.py, User has: username, email, full_name. No password.
                            # I will use username as email for now if email is required.
                            new_user = User(
                                username=username,
                                email=f"{username}@example.com",
                                full_name=username
                            )
                            db.add(new_user)
                        else:
                            logger.info(f"User {username} already exists.")
                db.commit()
                logger.info("Default users processed.")
            except Exception as e:
                logger.error(f"Error creating default users: {e}")
                db.rollback()
            finally:
                db.close()
                
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise

if __name__ == "__main__":
    init_db()
