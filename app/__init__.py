from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import logger
from app.services.phe_service import LightPHEWrapper

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting PHE microservice")
    try:
        LightPHEWrapper.get_instance()
        logger.info("PHE service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PHE service: {str(e)}")
    
    yield  # The app runs here
    
    # Shutdown logic
    logger.info("Shutting down PHE microservice")

def create_app():
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="PHE Client Microservice", 
        description="Client-side microservice for Partially Homomorphic Encryption operations with face extraction",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include routers
    from app.api.routes import router as api_router
    app.include_router(api_router)

    return app