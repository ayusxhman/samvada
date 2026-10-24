import cv2
import numpy as np
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from backend.app.services.sign_to_text import (
    ModelUnavailableError,
    RecognitionInferenceError,
)
from backend.app.orchestration import ai_orchestrator
from backend.app.models.landmark_extractor import landmark_extractor
from backend.app.services.dictionary import dictionary_service

router = APIRouter(prefix="/recognize", tags=["Sign Recognition"])

class LearnWordRequest(BaseModel):
    word: str

@router.post("/image")
async def recognize_image(
    image: UploadFile = File(...),
    target_lang: str = Form("en")
):
    """
    Accepts an uploaded hand sign image and classifies it (A-Z, space, del, nothing).
    """
    try:
        # Read image bytes
        file_bytes = np.frombuffer(await image.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
            
        # Run CNN prediction
        prediction = ai_orchestrator.recognize_fingerspell(frame)
        if not prediction.accepted:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Recognition confidence is too low. Please provide a clearer image.",
                    **prediction.as_dict(),
                },
            )
        predicted_char = prediction.label
        
        # Translate character description to target language
        translated_char = predicted_char
        if predicted_char in ["del", "nothing", "space"]:
            translated_char = ai_orchestrator.translate_language(
                text=predicted_char,
                target_lang=target_lang,
                source_lang="en",
            )

        return {
            "success": True,
            "prediction": predicted_char,
            "translated_prediction": translated_char,
            "confidence": prediction.confidence,
            "confidence_threshold": prediction.threshold,
            "alternatives": [candidate.__dict__ for candidate in prediction.alternatives],
        }
    except HTTPException:
        raise
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RecognitionInferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to process the uploaded image.") from exc

@router.post("/video")
async def recognize_video(
    video: UploadFile = File(...),
    target_lang: str = Form("en")
):
    """
    Accepts a video segment, extracts landmarks, runs sequence prediction (WLASL words).
    """
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_path = temp_video.name

    try:
        # Extract a 30-frame landmark sequence in the exact 570-feature model
        # contract: normalized position, velocity, and acceleration.
        seq_landmarks = landmark_extractor.extract_sequence_from_video(
            temp_path,
            target_frames=30,
            with_kinematics=True,
        )
        if not np.any(seq_landmarks):
            raise HTTPException(
                status_code=422,
                detail="No sign landmarks were detected in the uploaded video.",
            )
        
        # Run Transformer classification
        prediction = ai_orchestrator.recognize_word(seq_landmarks)
        if not prediction.accepted:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Recognition confidence is too low. Please record the sign again.",
                    **prediction.as_dict(),
                },
            )
        recognized_gloss = prediction.label
        
        # Correct grammar & translate
        translation_info = ai_orchestrator.translate_sign_sequence(
            gloss_sequence=[recognized_gloss],
            target_lang=target_lang,
        )
        
        return {
            "success": True,
            "gloss": recognized_gloss,
            "confidence": prediction.confidence,
            "confidence_threshold": prediction.threshold,
            "alternatives": [candidate.__dict__ for candidate in prediction.alternatives],
            "english_text": translation_info["english_text"],
            "translated_text": translation_info["translated_text"]
        }
    except HTTPException:
        raise
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RecognitionInferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to process the uploaded video.") from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/learn")
async def learn_new_word(request: LearnWordRequest):
    """
    Learns a new English word dynamically from online dictionaries.
    """
    result = await dictionary_service.learn_new_word(request.word)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
