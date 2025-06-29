# src/app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from lightphe import LightPHE
import os
import numpy as np

app = FastAPI(title="PHE Microservice", 
              description="Microservice for Partially Homomorphic Encryption operations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your client URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("keys", exist_ok=True)
private_key_path = "keys/private_key.txt"
public_key_path = "keys/public_key.txt"

# Create keys if they don't exist
if not os.path.exists(private_key_path):
    cs = LightPHE(algorithm_name="Paillier", precision=19)
    cs.export_keys(private_key_path, public=False)
    cs.export_keys(public_key_path, public=True)

# Initialize PHE with private key, PHE should also have public keys too 
phe_service = LightPHE(algorithm_name="Paillier", precision=19, key_file=private_key_path)

# Define request models
class EmbeddingData(BaseModel):
    embedding: List[float]

class EncryptedData(BaseModel):
    encrypted_data: str

@app.get("/")
async def read_root():
    return {"status": "PHE Microservice is running"}

@app.get("/public-key")
async def get_public_key():
    """Get PHE public key for client-side operations"""
    with open(public_key_path, 'r') as f:
        public_key = f.read()
    return {"public_key": public_key}

@app.post("/encrypt")
async def encrypt_data(data: EmbeddingData):
    """Encrypt data using PHE"""
    try:
        embedding = np.array(data.embedding)
        encrypted = phe_service.encrypt(embedding)


        serialized = str(encrypted)  
        return {"encrypted_embedding": serialized}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption error: {str(e)}")

@app.post("/compute-similarity")
async def compute_similarity(data: Dict[str, Any]):
    """Compute similarity between encrypted and plain embeddings"""
    try:
        encrypted_embedding = data.get("encrypted_embedding")
        plain_embedding = data.get("plain_embedding")
        
        if not encrypted_embedding or not plain_embedding:
            raise HTTPException(status_code=400, detail="Missing required data")
            
        # Deserialize the encrypted embedding
        # This part depends on how you serialized it in the encrypt endpoint
        
        # Compute dot product
        similarity_encrypted = encrypted_embedding @ plain_embedding
        
        return {"encrypted_similarity": str(similarity_encrypted)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computation error: {str(e)}")

@app.post("/decrypt")
async def decrypt_data(data: EncryptedData):
    """Decrypt data using PHE"""
    try:
        # Deserialize the encrypted data (this depends on your serialization method)
        decrypted = phe_service.decrypt(eval(data.encrypted_data))[0]  # Be careful with eval
        return {"decrypted_value": decrypted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decryption error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)