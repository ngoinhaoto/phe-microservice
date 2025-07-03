from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Make sure this matches the PHE_API_KEY in server's config/app.py
    API_KEY: str = "your-secret-api-key"
    
    SERVER_URL: str = "http://localhost:8000"
    
    SERVER_URLS: list = [
        {
            "url": "http://localhost:8000",
            "api_key": "your-secret-api-key" 
        }
    ]
    
    model_config = {
        "env_file": ".env"
    }

settings = Settings()