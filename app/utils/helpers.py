import functools
import json
from fastapi import Request, HTTPException
import os
from app.core.config import settings, logger

def ensure_key_match(func):
    """Decorator to verify key ID matches between server and microservice"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Get request object if available
        request = next((arg for arg in args if isinstance(arg, Request)), None)
        
        if request:
            # Check if the server sent its key ID in header
            server_key_id = request.headers.get("X-PHE-Key-ID")
            
            if server_key_id:
                # Verify key match
                key_info_path = settings.KEY_INFO_PATH
                if os.path.exists(key_info_path):
                    with open(key_info_path, 'r') as f:
                        key_info = json.load(f)
                        current_key_id = key_info.get("key_id", "unknown")
                    
                    if current_key_id != server_key_id:
                        logger.warning(f"Key mismatch! Server has {server_key_id}, microservice has {current_key_id}")
                        # Throw an exception to enforce key matching
                        raise HTTPException(
                            status_code=409, 
                            detail={
                                "error": "PHE key mismatch",
                                "server_key_id": server_key_id,
                                "current_key_id": current_key_id,
                                "message": "The server is using an outdated public key. Please update your key."
                            }
                        )
        
        return await func(*args, **kwargs)
    return wrapper