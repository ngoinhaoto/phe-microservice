from fastapi import APIRouter
from .key_routes import router as key_router
from .encryption_routes import router as encryption_router
from .face_routes import router as face_router
from .verification_routes import router as verification_router

router = APIRouter()

# Root endpoint
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

# Include all routers
router.include_router(key_router)
router.include_router(encryption_router)
router.include_router(face_router)
router.include_router(verification_router)