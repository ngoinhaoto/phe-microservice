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
from app.utils.face_utils import check_face_completeness

router = APIRouter(tags=["Verification Operations"])

@router.post("/verify-face-direct")
@ensure_key_match
async def verify_face_direct(request: Request, file: UploadFile = File(...), session_id: Optional[int] = None):
    """Extract embedding and verify directly with server"""
    temp_path = None
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name
        
        logger.info(f"Verifying face using deepface model")
        logger.info(f"Starting DeepFace embedding extraction with anti_spoofing={settings.ENABLE_ANTISPOOFING}")
        logger.info(f"Image saved to temporary file: {temp_path}")
        
        try:
            from deepface import DeepFace
            import cv2
            import numpy as np
            
            # Load the image
            img = cv2.imread(temp_path)
            if img is None:
                raise HTTPException(
                    status_code=400, 
                    detail="Failed to read image. Please upload a valid image file."
                )
            
            detector_to_use = settings.FACE_DETECTOR
            
            # Step 1: Extract faces to check if a face is detected
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
                
                # Step 3: Perform anti-spoofing check if enabled
                if settings.ENABLE_ANTISPOOFING:
                    try:
                        logger.info("Starting anti-spoofing check")
                        # Use opencv detector for anti-spoofing as it's more reliable for this purpose
                        
                        anti_spoof_faces = DeepFace.extract_faces(
                            img_path=temp_path,
                            anti_spoofing=True
                        )
                        
                        logger.info(f"Anti-spoofing check completed, found {len(anti_spoof_faces) if anti_spoof_faces else 0} faces")
                        
                        is_spoof = True  # Default to assuming it's a spoof until proven otherwise
                        
                        if anti_spoof_faces and len(anti_spoof_faces) > 0:
                            face_obj = anti_spoof_faces[0]
                            is_real = face_obj.get("is_real", False)
                            
                            # Check if the face is real based on DeepFace result
                            if is_real:
                                is_spoof = False
                                logger.info(f"DeepFace anti-spoofing result: Face appears to be real (is_real={is_real})")
                            else:
                                logger.warning(f"DeepFace anti-spoofing result: Potential spoof detected (is_real={is_real})")
                                                            
                            if is_spoof:
                                raise HTTPException(
                                    status_code=400, 
                                    detail="Potential spoofing detected. Please use a real face for verification."
                                )
                        else:
                            # If no faces were found in anti-spoofing check but we found faces earlier,
                            # it's likely a processing error rather than an actual spoof
                            logger.warning("No faces detected during anti-spoofing check, but faces were detected earlier")
                            # We'll proceed without the anti-spoofing check in this case
                    except HTTPException:
                        raise
                    except Exception as spoof_e:
                        # More detailed logging to help diagnose the issue
                        logger.warning(f"Anti-spoofing check failed, proceeding without it: {str(spoof_e)}")
                        logger.warning(f"Anti-spoofing check stack trace:", exc_info=True)
                
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
                raise 
            except Exception as e:
                if "Face could not be detected" in str(e):
                    raise HTTPException(
                        status_code=400, 
                        detail="No face detected in the image. Please ensure your face is clearly visible and try again."
                    )
                logger.error(f"Error during face detection or completeness check: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))

            embedding = embedding_objs[0]["embedding"]
            embedding_result = {"embedding": embedding}
            
            # Call the server to verify the embedding
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
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in verification processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing face: {str(e)}")
    
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions with original status code
        logger.error(f"HTTP exception in verify_face_direct: {http_ex.detail} (status {http_ex.status_code})")
        raise http_ex
    except Exception as e:
        logger.error(f"Unexpected error in verify_face_direct: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info("Temporary file deleted")