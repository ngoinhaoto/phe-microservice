from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import numpy as np
import tempfile
import shutil
import time
import logging
import ast
from contextlib import asynccontextmanager
import multiprocessing
from multiprocessing import Value, Lock

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("phe-microservice")

# Use a counter to track initialization
init_counter = Value('i', 0)
init_lock = Lock()

class LightPHEWrapper:
    """A wrapper for LightPHE that only initializes once per process."""
    _instance = None
    _initialized = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with init_lock:
                if init_counter.value == 0:
                    logger.info("First PHE service initialization")
                init_counter.value += 1
            
            from lightphe import LightPHE
            private_key_path = "keys/private_key.txt"
            
            # Create keys if they don't exist (only once)
            if not os.path.exists(private_key_path):
                logger.info("Generating new PHE key pair...")
                os.makedirs("keys", exist_ok=True)
                cs = LightPHE(algorithm_name="Paillier", precision=19)
                cs.export_keys(private_key_path, public=False)
                cs.export_keys("keys/public_key.txt", public=True)
                logger.info("Key pair generated successfully.")
            else:
                if not cls._initialized:
                    logger.info("Using existing PHE key pair (first use in this process).")
                    cls._initialized = True
            
            # Initialize PHE with private key
            cls._instance = LightPHE(algorithm_name="Paillier", precision=19, key_file=private_key_path)
            
        return cls._instance

# Create startup/shutdown context for app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting PHE microservice")
    # Initialize the PHE service on the main process to create keys if needed
    LightPHEWrapper.get_instance()
    
    yield  # The app runs here
    
    # Shutdown logic
    logger.info("Shutting down PHE microservice")

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

class EmbeddingData(BaseModel):
    embedding: List[float]

class EncryptedData(BaseModel):
    encrypted_data: str

@app.get("/")
def home():
    return {
        "status": "PHE Client Microservice is running",
        "capabilities": [
            "Generate encryption keys",
            "Extract face embeddings with VGG-Face",
            "Encrypt face embeddings",
            "Decrypt similarity scores"
        ]
    }

@app.get("/public-key")
def get_public_key():
    """Get PHE public key for use in client-side operations"""
    try:
        with open("keys/public_key.txt", 'r') as f:
            public_key = f.read()
        return {"public_key": public_key}
    except Exception as e:
        logger.error(f"Error retrieving public key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving public key: {str(e)}")

@app.post("/extract-and-encrypt")
async def extract_and_encrypt(file: UploadFile = File(...)):
    """Extract face embedding using DeepFace and encrypt it using PHE"""
    # Get model parameter or default to VGG-Face
    model = "VGG-Face"
    
    try:
        # Get the PHE service instance
        phe_service = LightPHEWrapper.get_instance()
        
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name
        
        try:
            # Import DeepFace here to avoid importing in worker processes
            from deepface import DeepFace
            
            # Step 1: Extract face embedding
            logger.info(f"Extracting face embedding with {model}...")
            start_time = time.time()
            embedding_objs = DeepFace.represent(
                img_path=temp_path, 
                model_name=model,
                enforce_detection=True,
                detector_backend="yunet"
            )
            extraction_time = time.time() - start_time
            logger.info(f"Face extraction completed in {extraction_time:.2f} seconds")
            
            if not embedding_objs or len(embedding_objs) == 0:
                raise HTTPException(status_code=400, detail="No face detected in the image")
            
            embedding = embedding_objs[0]["embedding"]
            
            # Check for negative values
            embedding_array = np.array(embedding)
            if np.any(embedding_array < 0):
                raise HTTPException(
                    status_code=400,
                    detail="Embedding contains negative values which cannot be encrypted with PHE."
                )
            
            # Step 2: Encrypt the embedding
            logger.info(f"Encrypting embedding of length {len(embedding)}...")
            start_time = time.time()
            encrypted = phe_service.encrypt(embedding)
            encryption_time = time.time() - start_time
            logger.info(f"Encryption successful in {encryption_time:.2f} seconds")
            
            # Prepare response
            serialized = str(encrypted)
            
            return {
                "encrypted_embedding": serialized,
                "embedding_size": len(embedding),
                "model": model,
                "extraction_time": extraction_time,
                "encryption_time": encryption_time
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error in extract_and_encrypt: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Face extraction or encryption error: {str(e)}")

@app.post("/encrypt")
def encrypt_data(data: EmbeddingData):
    """Encrypt face embeddings using PHE"""
    try:
        phe_service = LightPHEWrapper.get_instance()
        
        embedding = np.array(data.embedding)
        
        if np.any(embedding < 0):
            raise HTTPException(
                status_code=400,
                detail="Embedding contains negative values. Only positive values can be encrypted with PHE."
            )
        
        logger.info(f"Encrypting embedding of length {len(data.embedding)}...")
        start_time = time.time()
        encrypted = phe_service.encrypt(data.embedding)
        encryption_time = time.time() - start_time
        logger.info(f"Encryption successful in {encryption_time:.2f} seconds")

        # Serialize for transport
        serialized = str(encrypted)
        
        return {
            "encrypted_embedding": serialized,
            "encryption_time": encryption_time
        }
    except Exception as e:
        logger.error(f"Error in encrypt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Encryption error: {str(e)}")

@app.post("/decrypt")
def decrypt_data(data: EncryptedData):
    """Decrypt similarity scores using PHE"""
    try:
        phe_service = LightPHEWrapper.get_instance()
        
        # Parse the encrypted data
        encrypted_data = ast.literal_eval(data.encrypted_data)
        
        # Decrypt
        decrypted = phe_service.decrypt(encrypted_data)
        
        return {"decrypted_data": decrypted}
    except Exception as e:
        logger.error(f"Error in decrypt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Decryption error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Set worker initialization method
    multiprocessing.set_start_method("spawn", force=True)
    
    # Start Uvicorn with multiple workers but with our custom wrapper that prevents reinitialization
    logger.info("Starting PHE Client Microservice with FastAPI...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8001,
        workers=2 * multiprocessing.cpu_count(), 
        log_level="info"
    )