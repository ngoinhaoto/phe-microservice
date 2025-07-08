from fastapi import APIRouter, HTTPException, UploadFile, File, Request
import tempfile
import os
import time
import requests
import base64
import pickle
import numpy as np
from app.services.phe_service import LightPHEWrapper
from app.services.face_service import extract_embedding
from app.utils.helpers import ensure_key_match
from app.core.config import settings, logger
import cv2
from app.utils.face_utils import check_face_completeness

router = APIRouter(tags=["Face Operations"])

@router.post("/register-face-direct")
@ensure_key_match
async def register_face_direct(request: Request, file: UploadFile = File(...)):
    temp_path = None
    try:
        phe_service = LightPHEWrapper.get_instance()
        
        auth_header = request.headers.get("Authorization")
        
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
        
        logger.info(f"Registering face using deepface model")
        logger.info(f"Starting DeepFace embedding extraction with anti_spoofing={settings.ENABLE_ANTISPOOFING}")
        logger.info(f"Image saved to temporary file: {temp_path}")
        
        try:
            from deepface import DeepFace
            from lightphe.models.Tensor import EncryptedTensor
            
            # Load the image
            img = cv2.imread(temp_path)
            if img is None:
                raise HTTPException(
                    status_code=400, 
                    detail="Failed to read image. Please upload a valid image file."
                )
            
            detector_to_use = settings.FACE_DETECTOR
            
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=temp_path,
                    detector_backend=detector_to_use,
                    align=True
                )
                
                if not face_objs or len(face_objs) == 0:
                    raise HTTPException(
                        status_code=400, 
                        detail="No face detected in the image. Please ensure your face is clearly visible and try again."
                    )
                
                is_complete, error_message = check_face_completeness(face_objs[0], img)
                if not is_complete:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Incomplete face detected: {error_message}. Please ensure your entire face is visible and centered in the frame."
                    )
                
                logger.info("Face completeness check passed")
                
                if settings.ENABLE_ANTISPOOFING:
                    try:
                        logger.info("Starting anti-spoofing check")                    
                        anti_spoof_faces = DeepFace.extract_faces(
                            img_path=temp_path,
                            anti_spoofing=True
                        )
                        
                        logger.info(f"Anti-spoofing check completed, found {len(anti_spoof_faces) if anti_spoof_faces else 0} faces")
                        
                        is_spoof = True  # Default to assuming it's a spoof until proven otherwise
                        
                        if anti_spoof_faces and len(anti_spoof_faces) > 0:
                            face_obj = anti_spoof_faces[0]
                            is_real = face_obj.get("is_real", False)
                            
                            if is_real:
                                is_spoof = False
                                logger.info(f"DeepFace anti-spoofing result: Face appears to be real (is_real={is_real})")
                            else:
                                logger.warning(f"DeepFace anti-spoofing result: Potential spoof detected (is_real={is_real})")
                                                            
                            if is_spoof:
                                raise HTTPException(
                                    status_code=400, 
                                    detail="Potential spoofing detected. Please use a real face for registration."
                                )
                        else:
                            logger.warning("No faces detected during anti-spoofing check, aborting registration")
                            raise HTTPException(
                                status_code=400,
                                detail="Anti-spoofing failed: No face detected during anti-spoofing check. Please use a clear, real face photo."
                            )
                    except HTTPException:
                        raise
                    except Exception as spoof_e:
                        logger.warning(f"Anti-spoofing check failed, aborting registration: {str(spoof_e)}")
                        logger.warning(f"Anti-spoofing check stack trace:", exc_info=True)
                        raise HTTPException(
                            status_code=400,
                            detail=f"Anti-spoofing check failed: {str(spoof_e)}"
                        )
                
                # Step 4: Extract the face embedding
                logger.info(f"Extracting face embedding with model={settings.FACE_MODEL}, detector={detector_to_use}")
                embedding_objs = DeepFace.represent(
                    img_path=temp_path, 
                    model_name=settings.FACE_MODEL,
                    detector_backend=detector_to_use
                )
                
                if not embedding_objs or len(embedding_objs) == 0:
                    raise HTTPException(
                        status_code=400, 
                        detail="Failed to extract face embedding. Please try again with a clearer image."
                    )
                
                logger.info(f"Successfully extracted face embedding")
                
            except HTTPException:
                raise  # Re-raise HTTP exceptions without modification
            except Exception as e:
                if "Face could not be detected" in str(e):
                    raise HTTPException(
                        status_code=400, 
                        detail="No face detected in the image. Please ensure your face is clearly visible and try again."
                    )
                logger.error(f"Error during face detection or completeness check: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))

            # Step 5: Process the embedding for PHE
            embedding = embedding_objs[0]["embedding"]
            embedding_size = len(embedding)
            
            logger.info(f"Embedding extracted with size: {embedding_size}")
            
            # Check for negative values which PHE can't encrypt
            embedding_array = np.array(embedding)
            if np.any(embedding_array < 0):
                raise HTTPException(
                    status_code=400,
                    detail="Embedding contains negative values which cannot be encrypted with PHE."
                )
            
            # Step 6: Encrypt the embedding
            encrypted = phe_service.encrypt(embedding)
            encrypted_bytes = pickle.dumps(encrypted)
            
            # Step 7: Send to server for storing
            logger.info("Sending encrypted embedding to server for storage")
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
            
            logger.info(f"Face registration successful, embedding ID: {response_data.get('embedding_id')}")
            
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
            detail = e.response.json().get("detail") if e.response and hasattr(e.response, 'json') and callable(e.response.json) else str(e)
            raise HTTPException(status_code=status_code, detail=detail)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info("Temporary file deleted")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in direct registration: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )

