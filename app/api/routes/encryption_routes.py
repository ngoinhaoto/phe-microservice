from fastapi import APIRouter, HTTPException
import numpy as np
import base64
import pickle
from app.services.phe_service import LightPHEWrapper
from app.utils.helpers import ensure_key_match
from app.core.config import settings, logger
from app.api.models import EmbeddingData, EncryptedData

router = APIRouter(tags=["Encryption Operations"])

@router.post("/encrypt")
@ensure_key_match
def encrypt_data(data: EmbeddingData):
    """Encrypt face embeddings using PHE"""
    try:
        phe_service = LightPHEWrapper.get_instance()
        embedding = data.embedding
        
        # Validate embedding
        embedding_array = np.array(embedding)
        if np.any(embedding_array < 0):
            raise HTTPException(
                status_code=400,
                detail="Embedding contains negative values which cannot be encrypted with PHE."
            )
        
        # Encrypt the embedding
        encrypted = phe_service.encrypt(embedding)
        
        # Serialize the encrypted object
        serialized = str(encrypted)
        
        return {
            "encrypted_embedding": serialized,
            "embedding_size": len(embedding)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in encrypt_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Encryption error: {str(e)}")

@router.post("/decrypt")
@ensure_key_match
def decrypt_data(data: EncryptedData):
    """Decrypt data using PHE"""
    try:
        phe_service = LightPHEWrapper.get_instance()
        encrypted_data = data.encrypted_data
        
        # Convert the string representation back to an encrypted object
        encrypted_obj = eval(encrypted_data)
        
        # Decrypt the data
        decrypted = phe_service.decrypt(encrypted_obj)
        
        return {
            "decrypted_data": decrypted,
            "type": type(decrypted).__name__
        }
    except Exception as e:
        logger.error(f"Error in decrypt_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Decryption error: {str(e)}")