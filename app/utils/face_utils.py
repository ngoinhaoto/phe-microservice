import logging
from typing import Tuple, Optional
import cv2

logger = logging.getLogger(__name__)

def check_face_completeness(face_obj, img=None) -> Tuple[bool, Optional[str]]:
    """
    Check if the face is complete (entire face is visible in the frame).
    
    Args:
        face_obj: The face object from DeepFace extract_faces
        img: Original image (numpy array) if available
        
    Returns:
        Tuple containing:
        - Boolean indicating if face is complete
        - Error message if face is incomplete, None otherwise
    """
    try:
        # Initialize image dimensions
        img_width, img_height = 0, 0
        
        # Try to get image dimensions from the provided image
        if img is not None:
            img_height, img_width = img.shape[:2]
        
        # If no facial area info, can't determine completeness
        if "facial_area" not in face_obj:
            return False, "No facial area information available"
            
        facial_area = face_obj["facial_area"]
        
        # Get face coordinates
        x = facial_area.get("x", 0)
        y = facial_area.get("y", 0)
        w = facial_area.get("w", 0)
        h = facial_area.get("h", 0)
        
        # If we couldn't determine image dimensions earlier, try to infer from face
        if img_width == 0 and "img" in face_obj:
            face_img = face_obj["img"]
            if face_img is not None:
                img_height, img_width = face_img.shape[:2]
                # Since this is the face crop, we need to adjust based on known coordinates
                img_width = max(img_width, x + w)
                img_height = max(img_height, y + h)
        
        # If we still don't have image dimensions, use detection confidence only
        if img_width == 0:
            if "confidence" in face_obj and face_obj["confidence"] < 0.7:  # Adjust threshold as needed
                return False, "Face detection confidence too low"
            return True, None
        
        width_ratio = w / img_width
        height_ratio = h / img_height
        
        if width_ratio < 0.15:  # Adjust threshold as needed
            return False, "Face too small (width)"
            
        if height_ratio < 0.15:  # Adjust threshold as needed
            return False, "Face too small (height)"
        
        # Check if face is too close to the edge
        margin_ratio = 0.05  # Adjust margin as needed
        margin_x = img_width * margin_ratio
        margin_y = img_height * margin_ratio
        
        if x < margin_x or (x + w) > (img_width - margin_x) or y < margin_y or (y + h) > (img_height - margin_y):
            return False, "Face too close to image edge"
        
        if "confidence" in face_obj and face_obj["confidence"] < 0.7:  # Adjust threshold as needed
            return False, "Face detection confidence too low"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error checking face completeness: {str(e)}")
        return False, f"Error checking face completeness: {str(e)}"