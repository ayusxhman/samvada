from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
from pydantic import BaseModel
from backend.app.orchestration import ai_orchestrator

router = APIRouter(prefix="/translate", tags=["Translation"])

class TextToSignRequest(BaseModel):
    text: str
    target_lang: str = "en"
    source_lang: str = "auto"
    session_id: str | None = None

class LanguageTranslateRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str

import asyncio
import logging

logger = logging.getLogger(__name__)

TTS_CACHE: dict[tuple[str, str], bytes] = {}

async def pregenerate_tts_bytes(text: str, lang: str) -> bytes | None:
    clean_text = text.strip()
    if not clean_text:
        return None
    cache_key = (clean_text, lang.lower())
    if cache_key in TTS_CACHE:
        return TTS_CACHE[cache_key]
    
    voice_model = NEURAL_VOICES.get(lang.lower(), "en-US-JennyNeural")
    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean_text, voice_model)
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        if audio_data:
            res_bytes = bytes(audio_data)
            if len(TTS_CACHE) > 300:
                TTS_CACHE.clear()
            TTS_CACHE[cache_key] = res_bytes
            return res_bytes
    except Exception as e:
        logger.warning(f"Pregenerate TTS error: {e}")
    return None

@router.post("/text-to-sign")
async def translate_text_to_sign(request: TextToSignRequest):
    """
    Translates text to sign language glosses, video paths, and avatar animation instructions.
    Also translates the captions/text to target language.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        result = await ai_orchestrator.text_to_sign(
            text=request.text,
            target_lang=request.target_lang,
            source_lang=request.source_lang,
            session_id=request.session_id,
        )
        caption_text = result.get("caption_text", "")
        if caption_text:
            asyncio.create_task(pregenerate_tts_bytes(caption_text, request.target_lang))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/language")
async def translate_language(request: LanguageTranslateRequest):
    """
    Translates text between any two supported languages.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        translated = ai_orchestrator.translate_language(
            text=request.text,
            target_lang=request.target_lang,
            source_lang=request.source_lang
        )
        return {"translated_text": translated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/speech-to-text")
async def translate_speech_to_text(
    audio: UploadFile = File(...),
    target_lang: str = Form("en"),
    source_lang: str = Form("auto")
):
    """
    Transcribes uploaded audio and translates it to English + target language.
    Also returns sign language translation information.
    """
    try:
        # Read file bytes
        audio_bytes = await audio.read()
        
        # Transcribe & Translate
        result = ai_orchestrator.transcribe_and_translate(
            audio_bytes=audio_bytes,
            target_lang=target_lang,
            source_lang=source_lang,
        )
        
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "No speech detected in audio clip. Please speak clearly into the microphone."),
                "transcription": "",
                "english_text": "",
                "translated_text": "",
                "detected_lang": "en",
                "sign_language": {"glosses": [], "videos": []}
            }

        # Add Sign language translation mapping
        original_text = result["original_text"]
        detected_lang = result["detected_lang"]
        
        sign_translation = await ai_orchestrator.text_to_sign(
            text=original_text,
            target_lang=target_lang,
            source_lang=detected_lang,
            session_id=None,
        )
        
        # Combine
        return {
            "success": True,
            "transcription": original_text,
            "english_text": sign_translation.get("english_text", original_text),
            "translated_text": sign_translation.get("caption_text", original_text),
            "detected_lang": detected_lang,
            "sign_language": sign_translation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Microsoft Azure Neural Voice Inventory (52 Languages) - Stock Female Voices
NEURAL_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "bn": "bn-IN-TanishaaNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "pa": "pa-IN-GurmukhiNeural",
    "ur": "ur-PK-UzmaNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-AmalaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "it": "it-IT-ElsaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ar": "ar-SA-ZariyahNeural",
    "tr": "tr-TR-EmelNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "th": "th-TH-PremwadeeNeural",
    "id": "id-ID-GadisNeural",
    "ms": "ms-MY-YasminNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
    "sv": "sv-SE-SofieNeural",
    "cs": "cs-CZ-VlastaNeural",
    "ro": "ro-RO-AlinaNeural",
    "hu": "hu-HU-NoemiNeural",
    "el": "el-GR-AthinaNeural",
    "fi": "fi-FI-NooraNeural",
    "no": "nb-NO-PernilleNeural",
    "da": "da-DK-ChristelNeural",
    "he": "he-IL-HilaNeural",
    "fa": "fa-IR-DilaraNeural",
    "sw": "sw-KE-ZuriNeural",
    "tl": "fil-PH-BlessicaNeural",
}

@router.get("/tts")
async def get_text_to_speech_audio(text: str, lang: str = "en"):
    """
    Generates studio-quality Neural Speech for all 52 languages using edge-tts.
    Uses default stock voice pitch for natural-sounding output.
    """
    import logging
    logger = logging.getLogger(__name__)

    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    cache_key = (clean_text, lang.lower())
    if cache_key in TTS_CACHE:
        return Response(content=TTS_CACHE[cache_key], media_type="audio/mpeg")

    cached_bytes = await pregenerate_tts_bytes(clean_text, lang)
    if cached_bytes:
        return Response(content=cached_bytes, media_type="audio/mpeg")

    # 2. Fallback Google Proxy Engine
    import urllib.parse
    import urllib.request
    encoded_text = urllib.parse.quote(clean_text)
    url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_text}&tl={lang}&client=tw-ob"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://translate.google.com/"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read()
            return Response(content=content, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Proxy Error: {str(e)}")
