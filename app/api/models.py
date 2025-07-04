from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class EmbeddingData(BaseModel):
    embedding: List[float]

class EncryptedData(BaseModel):
    encrypted_data: str