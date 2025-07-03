from fastapi import FastAPI, HTTPException, File, UploadFile, Request
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
import requests
import base64
import pickle
from config import settings

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
    model = "VGG-Face"
    
    try:
        phe_service = LightPHEWrapper.get_instance()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name
        
        try:
            from deepface import DeepFace
            
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
            
            embedding_array = np.array(embedding)
            if np.any(embedding_array < 0):
                raise HTTPException(
                    status_code=400,
                    detail="Embedding contains negative values which cannot be encrypted with PHE."
                )
            
            logger.info(f"Encrypting embedding of length {len(embedding)}...")
            start_time = time.time()
            encrypted = phe_service.encrypt(embedding)
            encryption_time = time.time() - start_time
            logger.info(f"Encryption successful in {encryption_time:.2f} seconds")
            
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

@app.post("/register-face-direct")
async def register_face_direct(request: Request, file: UploadFile = File(...)):
    """Extract, encrypt, and register face directly with server"""
    try:
        # Get embedding and encrypt it
        phe_service = LightPHEWrapper.get_instance()
        
        # Get the auth header from the incoming request
        auth_header = request.headers.get("Authorization")
        
        # Prepare headers for the server request
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": settings.API_KEY
        }
        
        if auth_header:
            headers["Authorization"] = auth_header
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name
        
        try:
            from deepface import DeepFace
            from lightphe.models.Tensor import EncryptedTensor
            
            # Extract face embedding
            embedding_objs = DeepFace.represent(
                img_path=temp_path, 
                model_name="VGG-Face",
                enforce_detection=True,
                detector_backend="yunet"
            )
            
            if not embedding_objs or len(embedding_objs) == 0:
                raise HTTPException(status_code=400, detail="No face detected in the image")
            
            embedding = embedding_objs[0]["embedding"]
            embedding_size = len(embedding)
            
            encrypted = phe_service.encrypt(embedding)
            
            encrypted_bytes = pickle.dumps(encrypted)

            server_response = requests.post(
                f"{settings.SERVER_URL}/phe/store-encrypted-embedding",
                json={
                    "encrypted_embedding": base64.b64encode(encrypted_bytes).decode('utf-8'),
                    "aligned_face": None,
                    "embedding_size": embedding_size
                },
                headers=headers
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error in direct registration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-face-direct")
async def verify_face_direct(request: Request, file: UploadFile = File(...), session_id: Optional[int] = None):
    """Extract embedding and verify directly with server"""
    try:
        embedding_result = await extract_embedding(file)
        
        if not embedding_result or "embedding" not in embedding_result:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        logger.info(f"Extracted embedding with {len(embedding_result['embedding'])} dimensions")
        
        auth_header = request.headers.get("Authorization")
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": settings.API_KEY
        }
        
        if auth_header:
            headers["Authorization"] = auth_header
        
        params = {}
        if session_id:
            params["session_id"] = session_id
            
        server_response = requests.post(
            f"{settings.SERVER_URL}/phe/verify-with-embedding",
            json={"embedding": embedding_result["embedding"]},
            headers=headers,
            params=params
        )
        
        server_response.raise_for_status()
        server_data = server_response.json()
        
        similarity_threshold = 0.32
        highest_similarity = 0.0
        best_match = None
        
        if "results" in server_data and len(server_data["results"]) > 0:
            phe_service = LightPHEWrapper.get_instance()
            
            logger.info(f"Found {len(server_data['results'])} results to decrypt")
            
            for idx, result in enumerate(server_data["results"]):
                if "encrypted_similarity" in result:
                    try:
                        logger.info(f"Decrypting similarity for result #{idx+1}, user: {result.get('username', 'unknown')}")
                        
                        encrypted_bytes = base64.b64decode(result["encrypted_similarity"])
                        encrypted_similarity = pickle.loads(encrypted_bytes)
                        
                        logger.info(f"Successfully unpickled encrypted similarity of type: {type(encrypted_similarity).__name__}")
                        
                        decrypted_similarity = phe_service.decrypt(encrypted_similarity)
                        
                        logger.info(f"Raw decrypted value: {decrypted_similarity}")
                        
                        # Get the similarity value as a float
                        if isinstance(decrypted_similarity, list):
                            similarity_value = float(decrypted_similarity[0])
                        else:
                            similarity_value = float(decrypted_similarity)
                            
                        logger.info(f"Similarity value: {similarity_value}")
                        
                        # Update best match if this is higher
                        if similarity_value > highest_similarity:
                            highest_similarity = similarity_value
                            best_match = result
                        
                        # Store the decrypted similarity in the result
                        result["similarity"] = similarity_value
                        del result["encrypted_similarity"]
                    except Exception as decrypt_error:
                        logger.error(f"Error decrypting similarity: {str(decrypt_error)}", exc_info=True)
                        result["decrypt_error"] = str(decrypt_error)
                else:
                    logger.warning(f"Result #{idx+1} has no encrypted_similarity field")
            
            # Check if the best match exceeds the threshold
            if highest_similarity >= similarity_threshold and best_match:
                logger.info(f"Found match: {best_match.get('username')} with similarity {highest_similarity}")
                
                if session_id:
                    try:
                        check_in_response = requests.post(
                            f"{settings.SERVER_URL}/attendance/phe-check-in",
                            json={
                                "session_id": session_id,
                                "user_id": best_match.get("user_id"),
                                "verification_method": "phe"
                            },
                            headers=headers
                        )
                        
                        if check_in_response.status_code == 200:
                            logger.info(f"Successfully checked in student {best_match.get('full_name')} to session {session_id}")
                            server_data["attendance_recorded"] = True
                            server_data["check_in_details"] = check_in_response.json()
                        else:
                            logger.error(f"Failed to check in student: {check_in_response.text}")
                            server_data["attendance_recorded"] = False
                            server_data["check_in_error"] = check_in_response.text
                    except Exception as check_in_error:
                        logger.error(f"Error submitting check-in: {str(check_in_error)}")
                        server_data["attendance_recorded"] = False
                        server_data["check_in_error"] = str(check_in_error)
            else:
                logger.info(f"No match found above threshold ({similarity_threshold}). Best similarity: {highest_similarity}")
                server_data["match_found"] = False
                server_data["highest_similarity"] = highest_similarity
                server_data["threshold"] = similarity_threshold
        
        return server_data
    except Exception as e:
        logger.error(f"Error in direct face verification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def extract_embedding(file: UploadFile):
    model = "VGG-Face"
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name
        
        try:
            from deepface import DeepFace
            
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
                return None
            
            embedding = embedding_objs[0]["embedding"]
            
            return {
                "embedding": embedding,
                "embedding_size": len(embedding),
                "model": model,
                "extraction_time": extraction_time
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error in extract_embedding: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    import uvicorn
    
    multiprocessing.set_start_method("spawn", force=True)
    
    logger.info("Starting PHE Client Microservice with FastAPI...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8001,
        workers=2 * multiprocessing.cpu_count(), 
        log_level="info"
    )