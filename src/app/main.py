from fastapi import FastAPI, HTTPException, Depends, Header, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from lightphe import LightPHE
import os
import numpy as np
import ast
import tempfile
import shutil
from io import BytesIO
import base64
from deepface import DeepFace


app = FastAPI(title="PHE Client Microservice", 
              description="Client-side microservice for Partially Homomorphic Encryption operations with face extraction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
os.makedirs("keys", exist_ok=True)
private_key_path = "keys/private_key.txt"
public_key_path = "keys/public_key.txt"

# Create keys if they don't exist
if not os.path.exists(private_key_path):
    print("Generating new PHE key pair...")
    cs = LightPHE(algorithm_name="Paillier", precision=19)
    cs.export_keys(private_key_path, public=False)
    cs.export_keys(public_key_path, public=True)
    print("Key pair generated successfully.")
else:
    print("Using existing PHE key pair.")

# Initialize PHE with private key for client operations
try:
    phe_service = LightPHE(algorithm_name="Paillier", precision=19, key_file=private_key_path)
    print("PHE service initialized successfully with private key.")
except Exception as e:
    print(f"Error initializing PHE service: {str(e)}")
    raise

class EmbeddingData(BaseModel):
    embedding: List[float]

class EncryptedData(BaseModel):
    encrypted_data: str

@app.get("/")
async def read_root():
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
async def get_public_key():
    """Get PHE public key for use in client-side operations"""
    try:
        with open(public_key_path, 'r') as f:
            public_key = f.read()
        return {"public_key": public_key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving public key: {str(e)}")

@app.post("/extract-and-encrypt")
async def extract_and_encrypt(file: UploadFile = File(...)):
    """Extract face embedding using DeepFace and encrypt it using PHE"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        try:
            print("Extracting face embedding with VGG-Face...")
            embedding_objs = DeepFace.represent(
                img_path=temp_path, 
                model_name="VGG-Face",
                enforce_detection=True,
                detector_backend="yunet"
            )
            
            if not embedding_objs:
                raise HTTPException(status_code=400, detail="No face detected in the image")
            
            embedding = embedding_objs[0]["embedding"]
            
            # Verify the embedding has positive values (required for PHE)
            embedding_array = np.array(embedding)
            if np.any(embedding_array < 0):
                raise HTTPException(
                    status_code=400, 
                    detail="Embedding contains negative values. This should not happen with VGG-Face."
                )
            
            # Extract aligned face for display/verification purposes
            face_objs = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend="yunet",
                enforce_detection=True,
                align=True
            )
            
            aligned_face_base64 = None
            if face_objs and len(face_objs) > 0:
                import cv2
                face_obj = face_objs[0]
                face_img = face_obj["face"]
                if face_img is not None:
                    _, buf = cv2.imencode('.jpg', face_img)
                    aligned_face_base64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            
            # Encrypt the embedding
            print(f"Encrypting embedding of length {len(embedding)}...")
            encrypted = phe_service.encrypt(embedding_array)
            
            # Serialize for transport
            serialized = str(encrypted)
            print("Extraction and encryption successful.")
            
            return {
                "encrypted_embedding": serialized,
                "aligned_face": aligned_face_base64,
                "embedding_size": len(embedding),
                "model": "VGG-Face"
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        error_message = str(e)
        if "No face detected" in error_message:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        if "negative values" in error_message:
            raise HTTPException(status_code=400, detail=error_message)
        raise HTTPException(status_code=500, detail=f"Face extraction or encryption error: {error_message}")

@app.post("/encrypt")
async def encrypt_data(data: EmbeddingData):
    """Encrypt face embeddings using PHE"""
    try:
        # Convert to numpy array
        embedding = np.array(data.embedding)
        
        if np.any(embedding < 0):
            raise HTTPException(
                status_code=400, 
                detail="Embedding contains negative values. Only VGG-Face model with positive values is supported."
            )
        
        print(f"Encrypting embedding of length {len(embedding)}...")
        encrypted = phe_service.encrypt(embedding)
        
        # Serialize for transport
        serialized = str(encrypted)
        print("Encryption successful.")
        
        return {"encrypted_embedding": serialized}
    except Exception as e:
        if "negative values" in str(e):
            raise  # Re-raise the existing exception
        raise HTTPException(status_code=500, detail=f"Encryption error: {str(e)}")

@app.post("/decrypt")
async def decrypt_data(data: EncryptedData):
    """Decrypt similarity scores using PHE"""
    try:
        # Safely deserialize the encrypted data using ast.literal_eval instead of eval
        print("Decrypting similarity score...")
        decrypted = phe_service.decrypt(ast.literal_eval(data.encrypted_data))[0]
        print(f"Decryption successful. Result: {decrypted}")
        
        return {"decrypted_value": decrypted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decryption error: {str(e)}")

@app.post("/compare")
async def compare_faces(file: UploadFile = File(...), encrypted_data: str = None):
    """Extract embedding from image and compute similarity with encrypted embedding"""
    if not encrypted_data:
        raise HTTPException(status_code=400, detail="Missing encrypted_data parameter")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        try:
            print("Extracting face embedding from image...")
            embedding_objs = DeepFace.represent(
                img_path=temp_path, 
                model_name="VGG-Face",
                enforce_detection=True,
                detector_backend="retinaface"
            )
            
            if not embedding_objs:
                raise HTTPException(status_code=400, detail="No face detected in the image")
            
            embedding = np.array(embedding_objs[0]["embedding"])
            
            # Verify the embedding has positive values
            if np.any(embedding < 0):
                raise HTTPException(
                    status_code=400, 
                    detail="Embedding contains negative values. This should not happen with VGG-Face."
                )
            
            # Compute similarity
            print("Computing similarity with encrypted embedding...")
            encrypted_embedding = ast.literal_eval(encrypted_data)
            
            # Use dot product
            encrypted_similarity = encrypted_embedding @ embedding
            
            return {"encrypted_similarity": str(encrypted_similarity)}
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        error_message = str(e)
        if "No face detected" in error_message:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        raise HTTPException(status_code=500, detail=f"Face comparison error: {error_message}")

@app.post("/batch-encrypt")
async def batch_encrypt(data: Dict[str, List[List[float]]]):
    """Encrypt multiple embeddings in a single request"""
    try:
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise HTTPException(status_code=400, detail="No embeddings provided")
        
        print(f"Batch encrypting {len(embeddings)} embeddings...")
        encrypted_results = []
        
        for embedding in embeddings:
            np_embedding = np.array(embedding)
            # Verify positivity
            if np.any(np_embedding < 0):
                raise HTTPException(
                    status_code=400, 
                    detail="Some embeddings contain negative values. Only VGG-Face model with positive values is supported."
                )
            encrypted = phe_service.encrypt(np_embedding)
            encrypted_results.append(str(encrypted))
        
        print("Batch encryption successful.")
        return {"encrypted_embeddings": encrypted_results}
    except Exception as e:
        if "negative values" in str(e):
            raise  # Re-raise the existing exception
        raise HTTPException(status_code=500, detail=f"Batch encryption error: {str(e)}")

@app.post("/batch-decrypt")
async def batch_decrypt(data: Dict[str, List[str]]):
    """Decrypt multiple similarity scores in a single request"""
    try:
        encrypted_values = data.get("encrypted_values", [])
        if not encrypted_values:
            raise HTTPException(status_code=400, detail="No encrypted values provided")
        
        print(f"Batch decrypting {len(encrypted_values)} values...")
        decrypted_results = []
        
        for encrypted_value in encrypted_values:
            decrypted = phe_service.decrypt(ast.literal_eval(encrypted_value))[0]
            decrypted_results.append(decrypted)
        
        print("Batch decryption successful.")
        return {"decrypted_values": decrypted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch decryption error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting PHE Client Microservice...")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)