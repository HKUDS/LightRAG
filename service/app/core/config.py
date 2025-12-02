import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "LightRAG Service Wrapper"
    API_V1_STR: str = "/api/v1"
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB: str = os.getenv("POSTGRES_DATABASE", "lightrag_db")
    POSTGRES_MAX_CONNECTIONS: int = os.getenv("POSTGRES_MAX_CONNECTIONS", 12)
    
    # Encode credentials to handle special characters
    from urllib.parse import quote_plus
    _encoded_user = quote_plus(POSTGRES_USER)
    _encoded_password = quote_plus(POSTGRES_PASSWORD)
    
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"postgresql://{_encoded_user}:{_encoded_password}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    LIGHTRAG_WORKING_DIR: str = os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    AUTH_ACCOUNTS: str = os.getenv("AUTH_ACCOUNTS", "")

    class Config:
        case_sensitive = True

settings = Settings()