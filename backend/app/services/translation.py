import os
import json
import logging
from groq import Groq
from deep_translator import GoogleTranslator
from langdetect import detect
# NOTE: 'transformers' is imported lazily inside load_nllb_model() to avoid a 5.5s startup penalty
from backend.app.config import OFFLINE_TRANSLATION_DICT_PATH, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

# Map ISO language codes to NLLB-200 language codes
ISO_TO_NLLB = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Asme",
    "ur": "urd_Arab",
    "sd": "snd_Arab",
    "ks": "kas_Arab",
    "kok": "kok_Deva",
    "mni": "mni_Beng",
    "ne": "npi_Deva",
    "sa": "san_Deva",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "mai": "mai_Deva",
    "sat": "sat_Olck",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hant",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "sv": "swe_Latn",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "da": "dan_Latn",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
    "sw": "swh_Latn",
    "tl": "tgl_Latn"
}

class TranslationService:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_client = Groq(api_key=groq_api_key)
        
        self.offline_dict = {}
        self.load_offline_dict()
        
        self.tokenizer = None
        self.model = None
        self.device = None
        self.loading_nllb = False
        
        # Contextual Indian Regional Translation Corrections (Culinary & Common Phrasal Overrides)
        self.context_corrections = {
            "hi": {
                "rice": "चावल",
                "dhaan": "चावल",
                "eat rice": "चावल खाना",
                "i love to eat rice": "मुझे चावल खाना बहुत पसंद है",
                "i like to eat rice": "मुझे चावल खाना पसंद है",
                "i eat rice": "मैं चावल खाता हूँ"
            },
            "bn": {
                "rice": "ভাত",
                "eat rice": "ভাত খাওয়া",
                "i love to eat rice": "আমি ভাত খেতে খুব ভালোবাসি"
            }
        }
        self._cache = {}
        self._cache_file = OFFLINE_TRANSLATION_DICT_PATH.parent / "translation_cache.json"
        self._load_persistent_cache()
        logger.info("Initializing Translation Service...")

    def _load_persistent_cache(self):
        try:
            if self._cache_file.exists():
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    self._cache.update(json.load(f))
                logger.info(f"Loaded persistent translation cache with {len(self._cache)} entries.")
        except Exception as e:
            logger.warning(f"Could not load persistent translation cache: {e}")

    def _save_persistent_cache(self):
        try:
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save persistent translation cache: {e}")

    def load_offline_dict(self):
        try:
            if OFFLINE_TRANSLATION_DICT_PATH.exists():
                with open(OFFLINE_TRANSLATION_DICT_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.offline_dict = data.get("en_to_target", {})
                logger.info(f"Loaded offline translation dictionary with {len(self.offline_dict)} phrases.")
            else:
                logger.warning(f"Offline translation dictionary not found at {OFFLINE_TRANSLATION_DICT_PATH}")
        except Exception as e:
            logger.error(f"Error loading offline dictionary: {e}")

    def load_nllb_model(self):
        """Loads the NLLB-200 model on demand to avoid blocking service initialization."""
        if self.model is not None:
            return True
        if self.loading_nllb:
            return False
            
        self.loading_nllb = True
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model_name = "facebook/nllb-200-distilled-600M"
            logger.info(f"Loading SOTA NLLB-200 translation model ({model_name}) on {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            
            logger.info("NLLB-200 translation model loaded successfully.")
            self.loading_nllb = False
            return True
        except Exception as e:
            logger.error(f"Failed to load NLLB-200 model: {e}. Falling back to online engine.")
            self.loading_nllb = False
            return False

    def detect_language(self, text: str) -> str:
        try:
            clean_text = text.strip()
            if not clean_text:
                return "en"
            # Instant check: Pure ASCII text/letters default immediately to English (<1ms)
            if clean_text.isascii():
                return "en"
            lang = detect(clean_text)
            if lang in SUPPORTED_LANGUAGES:
                return lang
            return "en"
        except Exception as e:
            logger.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
            return "en"

    def translate(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        """
        Translates text to target language with instant caching (<0.1ms) and contextual accuracy.
        """
        text = text.strip()
        if not text:
            return ""

        if target_lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Target language '{target_lang}' not supported. Defaulting to 'en'.")
            target_lang = "en"

        text_lower = text.lower()
        fast_cache_key = f"{source_lang}:{target_lang}:{text_lower}"
        if fast_cache_key in self._cache:
            return self._cache[fast_cache_key]

        # Resolve auto source language
        if source_lang == "auto":
            source_lang = self.detect_language(text)

        if source_lang == target_lang:
            return text

        cache_key = f"{source_lang}:{target_lang}:{text_lower}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Check Contextual Phrasal Overrides & Offline Translation Dictionary
        text_lower = text.lower()
        if target_lang in self.context_corrections:
            if text_lower in self.context_corrections[target_lang]:
                corrected = self.context_corrections[target_lang][text_lower]
                self._cache[cache_key] = corrected
                self._save_persistent_cache()
                return corrected

        if text_lower in self.offline_dict:
            target_translations = self.offline_dict[text_lower]
            if target_lang in target_translations:
                result = target_translations[target_lang]
                self._cache[cache_key] = result
                self._save_persistent_cache()
                return result

        # 2. Fast Online Translation Engine (Groq Llama 3)
        # Skip LLM/Google for ultra-rare regional languages that only NLLB supports
        rare_languages = {"brx", "doi", "mai", "sat", "kok", "mni", "sd", "ks", "as", "or"}
        
        if target_lang not in rare_languages:
            try:
                target_lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
                source_lang_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
                
                prompt = f"Translate the following {source_lang_name} text into {target_lang_name}. Return ONLY the translated text and nothing else. Ensure the translation captures the semantic meaning perfectly and sounds natural, avoiding awkward literal translations."
                
                chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=4096,
                timeout=15.0,
            )
            
                translated = chat_completion.choices[0].message.content.strip()
            
                self._cache[fast_cache_key] = translated
                self._cache[cache_key] = translated
                self._save_persistent_cache()
                return translated
            except Exception as e:
                logger.warning(f"Groq translation fallback triggered: {e}")

            # 2.5 Free Online Translation Engine (Google Translator)
            try:
                from deep_translator import GoogleTranslator
                # deep_translator uses 'auto' instead of 'auto' for source detection, but accepts standard ISO codes
                translator = GoogleTranslator(source='auto' if source_lang == 'auto' else source_lang, target=target_lang)
                translated = translator.translate(text)
                
                if translated:
                    self._cache[fast_cache_key] = translated
                    self._cache[cache_key] = translated
                    self._save_persistent_cache()
                    return translated
            except Exception as e:
                logger.warning(f"GoogleTranslator fallback triggered: {e}")

        # 3. Offline SOTA NLLB-200 GPU translation fallback
        if self.load_nllb_model():
            import torch
            try:
                src_token = ISO_TO_NLLB.get(source_lang, "eng_Latn")
                tgt_token = ISO_TO_NLLB.get(target_lang, "eng_Latn")
                
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    translated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_token),
                        max_new_tokens=64,
                        num_beams=1
                    )
                translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                if target_lang == "hi":
                    translated_text = translated_text.replace("धान", "चावल").replace("भात", "चावल")
                elif target_lang == "bn":
                    translated_text = translated_text.replace("চাল", "ভাত")
                    
                self._cache[cache_key] = translated_text
                return translated_text
            except Exception as e:
                logger.error(f"NLLB-200 translation failed: {e}")

        return text

translation_service = TranslationService()
