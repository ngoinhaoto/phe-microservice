import os
import tempfile
import time
import traceback
from fastapi import UploadFile
import numpy as np
from app.core.config import settings, logger

async def extract_embedding(file: UploadFile):
    """Extract face embedding from an uploaded image file"""
    model = settings.FACE_MODEL
    
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
                detector_backend=settings.FACE_DETECTOR
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
        logger.error(traceback.format_exc())
        return None