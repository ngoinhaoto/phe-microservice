import os
from pydantic_settings import BaseSettings
import logging
from typing import ClassVar  # Add this import

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("phe-microservice")

class Settings(BaseSettings):
    """Application settings"""
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key")
    SERVER_URL: str = "http://localhost:8000"
    
    # Paths
    KEYS_DIR: str = "keys"
    PRIVATE_KEY_PATH: str = os.path.join(KEYS_DIR, "private_key.txt")
    PUBLIC_KEY_PATH: str = os.path.join(KEYS_DIR, "public_key.txt")
    KEY_INFO_PATH: str = os.path.join(KEYS_DIR, "key_info.json")
    
    PHE_ALGORITHM: str = "Paillier"
    PHE_PRECISION: int = 14
    
    FACE_MODEL: str = "VGG-Face"
    FACE_DETECTOR: str = "yunet"
    
    SIMILARITY_THRESHOLD: float = 0.5

    # Add proper type annotation for this field
    ENABLE_ANTISPOOFING: bool = True 
    
    class Config:
        env_file = ".env"

settings = Settings()