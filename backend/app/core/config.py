"""
Application Configuration

Centralized configuration management using environment variables
and Pydantic settings.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "ANPR System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/anpr_db"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis (for caching and task queue)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_EXPIRE: int = 3600
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # File Storage
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".mp4", ".avi"]
    
    # Vision Processing
    YOLO_MODEL_PATH: str = "models/yolov11_anpr.pt"
    OCR_MODEL_PATH: str = "models/ocr_model.onnx"
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.4
    
    # Video Processing
    MAX_CONCURRENT_STREAMS: int = 4
    FRAME_BUFFER_SIZE: int = 10
    PROCESSING_FPS: float = 30.0
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Email (for notifications)
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_USE_TLS: bool = True
    
    @validator("DEBUG", pre=True)
    def parse_debug(cls, value):
        """Parse DEBUG environment variable."""
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, value):
        """Parse ALLOWED_HOSTS environment variable."""
        if isinstance(value, str):
            return [host.strip() for host in value.split(",")]
        return value
    
    @validator("ALLOWED_EXTENSIONS", pre=True)
    def parse_allowed_extensions(cls, value):
        """Parse ALLOWED_EXTENSIONS environment variable."""
        if isinstance(value, str):
            return [ext.strip() for ext in value.split(",")]
        return value
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class DevelopmentSettings(Settings):
    """Development environment settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # Development database
    DATABASE_URL: str = "postgresql://anpr_user:anpr_pass@localhost/anpr_dev"
    
    # Allow all origins in development
    ALLOWED_HOSTS: List[str] = ["*"]


class ProductionSettings(Settings):
    """Production environment settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Production should use secure secrets
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    
    # Production database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Restricted CORS in production
    ALLOWED_HOSTS: List[str] = ["https://your-domain.com"]
    
    # Enable all production features
    ENABLE_METRICS: bool = True


class TestSettings(Settings):
    """Test environment settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # Test database
    DATABASE_URL: str = "postgresql://anpr_user:anpr_pass@localhost/anpr_test"
    
    # Disable external services in tests
    ENABLE_METRICS: bool = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings based on environment.
    
    Returns:
        Settings: Application settings instance
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()
