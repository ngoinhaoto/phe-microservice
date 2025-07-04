import os
import json
import uuid
import pickle
from datetime import datetime
import ast
from multiprocessing import Value, Lock
from app.core.config import settings, logger

from lightphe import LightPHE

# Global shared counter for initialization
init_counter = Value('i', 0)
init_lock = Lock()

class LightPHEWrapper:
    """A wrapper for LightPHE that only initializes once per process."""
    _instance = None
    _initialized = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with init_lock:
                if init_counter.value == 0:
                    logger.info("First PHE service initialization")
                init_counter.value += 1
            
            from lightphe import LightPHE
            private_key_path = settings.PRIVATE_KEY_PATH
            key_info_path = settings.KEY_INFO_PATH
            
            # Check if keys directory exists
            os.makedirs(settings.KEYS_DIR, exist_ok=True)
            
            # Get or create key ID with file lock to prevent race conditions
            key_id = cls._get_or_create_key_id(key_info_path)
            
            if not os.path.exists(private_key_path):
                logger.info(f"Generating new PHE key pair with ID: {key_id}")
                
                # Generate keys with explicit create_keys call
                try:
                    cs = LightPHE(algorithm_name=settings.PHE_ALGORITHM, precision=settings.PHE_PRECISION)
                    cs.pk, cs.sk = cs._create_keys()
                    
                    # Export keys
                    cs.export_keys(private_key_path, public=False)
                    cs.export_keys(settings.PUBLIC_KEY_PATH, public=True)
                    
                    logger.info(f"Key pair generated and exported successfully with ID: {key_id}")
                    cls._instance = cs
                except Exception as e:
                    logger.error(f"Error generating PHE keys: {str(e)}")
                    raise
            else:
                if not cls._initialized:
                    logger.info("Using existing PHE key pair (first use in this process)")
                    cls._initialized = True
                
                try:
                    # Always load with explicit private key path
                    cls._instance = LightPHE(
                        algorithm_name=settings.PHE_ALGORITHM, 
                        precision=settings.PHE_PRECISION, 
                        key_file=private_key_path
                    )
                    logger.info(f"Successfully loaded PHE keys with ID: {key_id}")
                except Exception as e:
                    logger.error(f"Failed to load PHE keys: {str(e)}")
                    raise
        
        return cls._instance
    
    @classmethod
    def _get_or_create_key_id(cls, key_info_path):
        """Get existing key ID or create a new one with proper locking"""
        try:
            # Check if key info exists
            if os.path.exists(key_info_path):
                with open(key_info_path, 'r') as f:
                    key_info = json.load(f)
                    key_id = key_info.get("key_id", None)
                
                if key_id:
                    return key_id
            
            # If we get here, need to create a new key ID
            with init_lock:
                # Double-check after acquiring lock
                if os.path.exists(key_info_path):
                    try:
                        with open(key_info_path, 'r') as f:
                            key_info = json.load(f)
                            key_id = key_info.get("key_id", None)
                        if key_id:
                            return key_id
                    except:
                        pass  # Continue to create new if read fails
                
                # Create new key info
                key_id = str(uuid.uuid4())
                key_info = {
                    "key_id": key_id,
                    "created_at": datetime.now().isoformat(),
                    "algorithm": settings.PHE_ALGORITHM,
                    "precision": settings.PHE_PRECISION
                }
                
                with open(key_info_path, 'w') as f:
                    json.dump(key_info, f)
                
                logger.info(f"Created new key ID: {key_id}")
                return key_id
        except Exception as e:
            logger.error(f"Error in key ID management: {str(e)}")
            # Create a fallback key ID if everything fails
            return str(uuid.uuid4())

    @classmethod
    def encrypt(cls, data):
        """Encrypt data using PHE"""
        phe_service = cls.get_instance()
        return phe_service.encrypt(data)
    
    @classmethod
    def decrypt(cls, encrypted_data):
        """Decrypt data using PHE"""
        phe_service = cls.get_instance()
        if isinstance(encrypted_data, str):
            encrypted_data = ast.literal_eval(encrypted_data)
        return phe_service.decrypt(encrypted_data)
    
    @classmethod
    def get_key_info(cls):
        """Get the current key information"""
        if not os.path.exists(settings.KEY_INFO_PATH):
            return None
            
        with open(settings.KEY_INFO_PATH, 'r') as f:
            return json.load(f)
    
    @classmethod
    def normalize_similarity_value(cls, value):
        """Ensure similarity value is within -1.0 to 1.0 range"""
        if isinstance(value, list):
            value = float(value[0])
        else:
            value = float(value)
        
        # Normalize to correct cosine similarity range
        return max(min(value, 1.0), -1.0)
    
    @classmethod
    def verify_key_consistency(cls):
        """Verify that the key files match the loaded instance"""
        if not cls._instance:
            return False
            
        try:            
            # Check if keys exist
            if not os.path.exists(settings.PRIVATE_KEY_PATH) or not os.path.exists(settings.PUBLIC_KEY_PATH):
                return False
                
            # Create a new instance to load keys
            test_instance = LightPHE(
                algorithm_name=settings.PHE_ALGORITHM,
                precision=settings.PHE_PRECISION,
                key_file=settings.PRIVATE_KEY_PATH
            )
            
            # Test encryption/decryption with a simple value
            test_value = 0.5
            encrypted = cls._instance.encrypt(test_value)
            decrypted = test_instance.decrypt(encrypted)
            
            # Check if the decryption is close to original value
            return abs(float(decrypted) - test_value) < 0.0001
        except Exception as e:
            logger.error(f"Key consistency check failed: {str(e)}")
            return False