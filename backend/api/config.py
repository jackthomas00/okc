from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

settings = Settings()
