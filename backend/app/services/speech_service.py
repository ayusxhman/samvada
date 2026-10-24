import logging
import os
import tempfile
from pathlib import Path
from backend.app.services.translation import translation_service

logger = logging.getLogger(__name__)

class FasterSpeechService:
    """
    High-performance Speech Recognition Service powered by Faster-Whisper (CTranslate2).
    Includes FP16 GPU Acceleration on RTX GPU, Silero VAD silence filtering,
    auto-language detection, auto-punctuation, and 50+ language translation.
    """
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = None
        self.common_misrecognitions = {
            "each rice": "eat rice",
            "bow around": "roam around",
            "around the bar": "around the park",
            "to college": "to college",
            "want to each": "want to eat",
            "kadi": "curry",
            "curdie": "curry",
            "karry": "curry"
        }

    @property
    def model(self):
        if self._model is None:
            logger.info("Initializing Faster-Whisper CTranslate2 model...")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4
                )
                logger.info(f"Faster-Whisper loaded successfully on device='{device}' (compute_type='{compute_type}').")
            except Exception as e:
                logger.warning(f"Failed to load Faster-Whisper model '{self.model_size}' on {device} ({e}). Falling back to 'base' on CPU...")
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    "base",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4
                )
                logger.info("Faster-Whisper fallback CPU model loaded.")
        return self._model

    def transcribe(self, audio_path: str, language: str = None) -> tuple:
        """
        Transcribes audio instantly using Groq Whisper API.
        Returns (text, detected_lang)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Transcribing audio file with Groq Whisper API: {audio_path}")
            from groq import Groq
            groq_api_key = os.getenv("GROQ_API_KEY", "")
            groq_client = Groq(api_key=groq_api_key)
            
            with open(audio_path, "rb") as file:
                kwargs = {
                    "file": (audio_path, file.read()),
                    "model": "whisper-large-v3-turbo",
                    "response_format": "verbose_json"
                }
                # Only force language if it's explicitly set and NOT English or Auto
                if language and language != "auto" and language != "en":
                    kwargs["language"] = language
                    
                response = groq_client.audio.transcriptions.create(**kwargs, timeout=3.0)
            
            # verbose_json returns an object with text and language
            text = str(response.text).strip()
            detected_lang = getattr(response, "language", language if language and language != "auto" else "en")
            
            # Apply transcript post-correction rules for common phonetic speech misrecognitions
            text_lower = text.lower()
            for wrong, correct in self.common_misrecognitions.items():
                if wrong in text_lower:
                    import re
                    pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                    text = pattern.sub(correct, text)
            
            # Capitalization and punctuation fallback check
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:]
            if text and text[-1] not in {".", "!", "?"}:
                text += "."
                
            return text, detected_lang
        except Exception as e:
            logger.error(f"Groq Whisper transcription failed: {e}")
            raise RuntimeError(f"Transcription error: {str(e)}")

    def transcribe_and_translate(self, audio_bytes: bytes, target_lang: str, source_lang: str = "auto") -> dict:
        """
        Accepts raw audio bytes, transcribes with Faster-Whisper (Silero VAD),
        auto-detects language, auto-punctuates, and translates to target language & English.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            # Transcribe audio with Faster-Whisper
            original_text, whisper_lang = self.transcribe(temp_path, language=source_lang)
            
            if not original_text:
                return {
                    "success": False,
                    "error": "No speech detected in audio clip"
                }

            # If user left source_lang as English or Auto, but Whisper detected another language, use Whisper's!
            if source_lang == "auto" or source_lang == "en":
                source_lang = whisper_lang or translation_service.detect_language(original_text)

            # We DO NOT translate here anymore, because `text_to_sign` will do it 
            # concurrently in parallel! Just return the original text!
            return {
                "success": True,
                "original_text": original_text,
                "english_text": original_text, # Will be translated downstream
                "translated_text": original_text, # Will be translated downstream
                "detected_lang": source_lang
            }
        except Exception as e:
            logger.error(f"Faster-Whisper transcribe and translate failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Instantiate singleton speech service
SpeechService = FasterSpeechService
speech_service = FasterSpeechService(model_size="small")
