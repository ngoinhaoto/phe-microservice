from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from typing import Optional
import tempfile
import os
import requests
import base64
import pickle
from app.services.phe_service import LightPHEWrapper
from app.services.face_service import extract_embedding
from app.utils.helpers import ensure_key_match
from app.core.config import settings, logger

router = APIRouter(tags=["Verification Operations"])

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
        
        similarity_threshold = settings.SIMILARITY_THRESHOLD
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
                            
                        logger.info(f"Normalized similarity value: {similarity_value}")
                        
                        # Update best match if this is higher
                        if similarity_value > highest_similarity:
                            highest_similarity = similarity_value
                            best_match = result
                        
                        # Store the decrypted similarity in the result
                        result["similarity"] = similarity_value
                        
                    except Exception as decrypt_error:
                        logger.error(f"Error decrypting similarity: {str(decrypt_error)}")
                        result["similarity"] = 0.0
                        result["decrypt_error"] = str(decrypt_error)
                else:
                    logger.warning(f"No encrypted_similarity found for result #{idx+1}")
            
            # Sort results by similarity
            server_data["results"].sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
            
            if highest_similarity >= similarity_threshold:
                logger.info(f"Match found: {best_match.get('full_name')} with similarity {highest_similarity}")
                server_data["match_found"] = True
                server_data["best_match"] = best_match
                server_data["highest_similarity"] = highest_similarity
            else:
                logger.info(f"No match above similarity threshold of {similarity_threshold}")
                server_data["match_found"] = False
                server_data["best_match"] = None
                server_data["highest_similarity"] = 0.0
        else:
            logger.info("No results found in server response")
            server_data["match_found"] = False
            server_data["best_match"] = None
            server_data["highest_similarity"] = 0.0
        
        return server_data
    
    except Exception as e:
        logger.error(f"Unexpected error in verify_face_direct: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")