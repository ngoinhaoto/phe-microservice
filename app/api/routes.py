from fastapi import APIRouter, HTTPException, File, UploadFile, Request, Depends
from typing import List, Dict, Any, Optional
import os
import requests
import base64
import pickle
import numpy as np
import tempfile
import time
from app.api.models import EmbeddingData, EncryptedData
from app.services.phe_service import LightPHEWrapper
from app.services.face_service import extract_embedding
from app.utils.helpers import ensure_key_match
from app.core.config import settings, logger

router = APIRouter()

@router.get("/")
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

@router.get("/public-key")
def get_public_key():
    """Get PHE public key for use in client-side operations"""
    try:
        with open(settings.PUBLIC_KEY_PATH, 'r') as f:
            public_key = f.read()
        
        key_info = LightPHEWrapper.get_key_info()
        if not key_info:
            # Generate new key info if missing
            import uuid
            import json
            from datetime import datetime
            
            key_id = str(uuid.uuid4())
            key_info = {
                "key_id": key_id,
                "created_at": datetime.now().isoformat(),
                "algorithm": settings.PHE_ALGORITHM,
                "precision": settings.PHE_PRECISION
            }
            
            # Make sure keys directory exists
            os.makedirs(settings.KEYS_DIR, exist_ok=True)
            
            # Write key info
            with open(settings.KEY_INFO_PATH, 'w') as f:
                json.dump(key_info, f)
        
        return {
            "public_key": public_key,
            "key_id": key_info.get("key_id", "unknown"),
            "last_updated": key_info.get("created_at")
        }
    except Exception as e:
        logger.error(f"Error retrieving public key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving public key: {str(e)}")

@router.post("/extract-and-encrypt")
@ensure_key_match
async def extract_and_encrypt(file: UploadFile = File(...)):
    """Extract face embedding using DeepFace and encrypt it using PHE"""
    try:
        phe_service = LightPHEWrapper.get_instance()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name
        
        try:
            from deepface import DeepFace
            
            logger.info(f"Extracting face embedding with {settings.FACE_MODEL}...")
            start_time = time.time()
            embedding_objs = DeepFace.represent(
                img_path=temp_path, 
                model_name=settings.FACE_MODEL,
                detector_backend=settings.FACE_DETECTOR
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
                "model": settings.FACE_MODEL,
                "extraction_time": extraction_time,
                "encryption_time": encryption_time
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_and_encrypt: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Face extraction or encryption error: {str(e)}")

@router.post("/encrypt")
@ensure_key_match
def encrypt_data(data: EmbeddingData):
    """Encrypt face embeddings using PHE"""
    try:
        embedding = np.array(data.embedding)
        
        if np.any(embedding < 0):
            raise HTTPException(
                status_code=400,
                detail="Embedding contains negative values. Only positive values can be encrypted with PHE."
            )
        
        logger.info(f"Encrypting embedding of length {len(data.embedding)}...")
        start_time = time.time()
        encrypted = LightPHEWrapper.encrypt(data.embedding)
        encryption_time = time.time() - start_time
        logger.info(f"Encryption successful in {encryption_time:.2f} seconds")

        # Serialize for transport
        serialized = str(encrypted)
        
        return {
            "encrypted_embedding": serialized,
            "encryption_time": encryption_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in encrypt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Encryption error: {str(e)}")

@router.post("/decrypt")
@ensure_key_match
def decrypt_data(data: EncryptedData):
    """Decrypt similarity scores using PHE"""
    try:
        # Decrypt
        decrypted = LightPHEWrapper.decrypt(data.encrypted_data)
        
        # Normalize result to ensure valid similarity range
        if isinstance(decrypted, list) and len(decrypted) > 0:
            normalized = [LightPHEWrapper.normalize_similarity_value(val) for val in decrypted]
        else:
            normalized = LightPHEWrapper.normalize_similarity_value(decrypted)
        
        return {"decrypted_data": normalized}
    except Exception as e:
        logger.error(f"Error in decrypt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Decryption error: {str(e)}")

@router.post("/register-face-direct")
@ensure_key_match
async def register_face_direct(request: Request, file: UploadFile = File(...)):
    try:
        # Get embedding and encrypt it
        phe_service = LightPHEWrapper.get_instance()
        
        # Get the auth header from the incoming request
        auth_header = request.headers.get("Authorization")
        
        # Prepare headers for the server request
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": settings.API_KEY  # Make sure API_KEY is defined in settings
        }
        
        # Debug log to verify API key is being sent
        logger.info(f"Using API key for server request: {settings.API_KEY[:5]}...")
        
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
                model_name=settings.FACE_MODEL,
                detector_backend=settings.FACE_DETECTOR
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
            
            server_response.raise_for_status()
            response_data = server_response.json()
            
            return {
                "message": "Face registration successful",
                "embedding_id": response_data.get("embedding_id"),
                "registration_group_id": response_data.get("registration_group_id"),
                "embedding_type": response_data.get("embedding_type", "encrypted_tensor"),
                "status": "success"
            }
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during registration: {str(e)}")
            # Pass along the server's error message
            status_code = e.response.status_code if e.response else 500
            detail = e.response.json().get("detail") if e.response and e.response.json() else str(e)
            raise HTTPException(status_code=status_code, detail=detail)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in direct registration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-face-direct")
@ensure_key_match
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
        else:
            logger.warning("No Authorization header in incoming request")
        
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
                        
                        if isinstance(decrypted_similarity, list):
                            similarity_value = float(decrypted_similarity[0])
                        else:
                            similarity_value = float(decrypted_similarity)
                            
                        # Add this normalization step to ensure proper cosine similarity range
                        similarity_value = max(min(similarity_value, 1.0), -1.0)

                        logger.info(f"Normalized similarity value: {similarity_value}")
                        
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
                        # *** IMPORTANT: For the check-in request, use API key only headers ***
                        check_in_headers = {
                            "Content-Type": "application/json",
                            "X-API-Key": settings.API_KEY  # Only use API key for this request
                        }
                        
                        # Don't include the Authorization header here
                        check_in_response = requests.post(
                            f"{settings.SERVER_URL}/attendance/phe-check-in",
                            json={
                                "session_id": session_id,
                                "user_id": best_match.get("user_id"),
                                "verification_method": "phe"
                            },
                            headers=check_in_headers  # Use the API key only headers
                        )
                        
                        logger.info(f"Check-in request sent with status: {check_in_response.status_code}")
                        
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

@router.get("/verify-key-match")
def verify_key_match(key_id: str):
    """Verify if the server's key ID matches the microservice's current key"""
    try:
        # Check if key info exists
        key_info = LightPHEWrapper.get_key_info()
        if not key_info:
            return {"match": False, "reason": "No key info found on microservice"}
        
        current_key_id = key_info.get("key_id", "unknown")
        
        # Check if key IDs match
        match = current_key_id == key_id
        
        return {
            "match": match,
            "current_key_id": current_key_id,
            "server_key_id": key_id,
            "needs_update": not match
        }
    except Exception as e:
        logger.error(f"Error verifying key match: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error verifying key match: {str(e)}")