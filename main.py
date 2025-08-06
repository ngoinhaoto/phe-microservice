import multiprocessing
import uvicorn
from app import create_app
from app.core.config import logger

app = create_app()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    
    logger.info("Starting PHE Client Microservice with FastAPI...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=4002,
        workers=2 * multiprocessing.cpu_count(), 
        log_level="info"
    )