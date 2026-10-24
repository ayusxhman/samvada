from fastapi import APIRouter, HTTPException
from backend.app.services.text_to_sign import text_to_sign_service

router = APIRouter(prefix="/avatar", tags=["Avatar Animations"])

@router.get("/animation/{word}")
async def get_avatar_animation(word: str):
    """
    Returns the authored VRMA clip for a sign word.
    """
    word = word.strip().lower()
    if not word:
        raise HTTPException(status_code=400, detail="Word cannot be empty.")
        
    # Check if pre-built VRMA exists
    if word in text_to_sign_service.available_vrm_clips:
        return {
            "word": word,
            "type": "vrma",
            "path": text_to_sign_service.available_vrm_clips[word]
        }
    
    raise HTTPException(
        status_code=404,
        detail=f"No authored VRMA animation is available for '{word}'.",
    )

@router.get("/animations")
async def get_all_animations():
    """
    Returns a list of all pre-built VRM animation files available.
    """
    return {
        "animations": list(text_to_sign_service.available_vrm_clips.keys())
    }
