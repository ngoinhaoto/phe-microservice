from fastapi import APIRouter, HTTPException
import os
import json
from datetime import datetime
import uuid
import requests
from app.services.phe_service import LightPHEWrapper
from app.utils.helpers import ensure_key_match
from app.core.config import settings, logger

router = APIRouter(tags=["Key Management"])

@router.get("/public")
def get_public_key():
    """Get PHE public key for use in client-side operations"""
    try:
        with open(settings.PUBLIC_KEY_PATH, 'r') as f:
            public_key = f.read()
        
        key_info = LightPHEWrapper.get_key_info()
        if not key_info:
            # Generate new key info if missing
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

@router.get("/verify-match")
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

@router.post("/test-key-compatibility")
def test_key_compatibility():
    """Test endpoint that gets a test encrypted array from the server and decrypts it to verify key compatibility"""
    try:
        logger.info("Starting key compatibility test")
        
        phe_service = LightPHEWrapper.get_instance()
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": settings.API_KEY
        }
        
        try:
            response = requests.get(
                f"{settings.SERVER_URL}/phe/key-compatibility-test",
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract data
            original_array = result.get("test_array")
            encrypted_array_str = result.get("encrypted_array")
            server_key_id = result.get("key_id")
                        
            # Get our key ID
            key_info = LightPHEWrapper.get_key_info()
            current_key_id = key_info.get("key_id", "unknown") if key_info else "unknown"
            
            # Decrypt the array
            import base64
            import pickle
            encrypted_array = pickle.loads(base64.b64decode(encrypted_array_str))
            
            decrypted_array = phe_service.decrypt(encrypted_array)
            logger.info(f"Decrypted array: {decrypted_array}")
            
            # Convert to same format for comparison
            original_array = [float(x) for x in original_array]
            
            if isinstance(decrypted_array, list):
                decrypted_values = [float(x) for x in decrypted_array]
            else:
                decrypted_values = [float(decrypted_array)]
            
            # Check if decryption was successful
            errors = []
            for i, (orig, decrypted) in enumerate(zip(original_array, decrypted_values)):
                if abs(orig - decrypted) > 0.0001:
                    errors.append(f"Value at index {i}: expected {orig}, got {decrypted}")
            
            compatibility_ok = len(errors) == 0
            
            return {
                "key_compatibility": compatibility_ok,
                "server_key_id": server_key_id,
                "microservice_key_id": current_key_id,
                "key_ids_match": server_key_id == current_key_id,
                "original_array": original_array,
                "decrypted_array": decrypted_values,
                "errors": errors if errors else None
            }
            
        except Exception as e:
            logger.error(f"Error in server communication or decryption: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error in server communication or decryption: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error in key compatibility test: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in key compatibility test: {str(e)}"
        )