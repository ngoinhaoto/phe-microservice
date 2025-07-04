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
        
            os.makedirs(settings.KEYS_DIR, exist_ok=True)
            
            if not os.path.exists(settings.PRIVATE_KEY_PATH):
                try:
                    logger.info(f"PRIVATE PATH: {settings.PRIVATE_KEY_PATH}")
                    logger.info(f"PRIVATE PATH: {settings.PUBLIC_KEY_PATH}")
                    logger.info(f"PRIVATE_KEY_PATH type: {type(settings.PRIVATE_KEY_PATH)}, value: {settings.PRIVATE_KEY_PATH}")
                    logger.info(f"PUBLIC_KEY_PATH type: {type(settings.PUBLIC_KEY_PATH)}, value: {settings.PUBLIC_KEY_PATH}")
                    cs = LightPHE(algorithm_name=settings.PHE_ALGORITHM, precision=settings.PHE_PRECISION)
                    
                    cs.export_keys(settings.PRIVATE_KEY_PATH, public=False)
                    cs.export_keys(settings.PUBLIC_KEY_PATH, public=True)
                    
                    logger.info(f"Generated new PHE keys [Public and private]")
                    cls._instance = cs
                except Exception as e:
                    logger.error(f"Error generating PHE keys: {str(e)}")
                    raise
            else:
                if not cls._initialized:
                    logger.info("Using existing PHE key pair (first use in this process)")
                    cls._initialized = True
                
                try:
                    cls._instance = LightPHE(
                        algorithm_name=settings.PHE_ALGORITHM, 
                        precision=settings.PHE_PRECISION, 
                        key_file=settings.PRIVATE_KEY_PATH
                    )
                except Exception as e:
                    logger.error(f"Failed to load PHE keys: {str(e)}")
                    raise
        
        return cls._instance

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