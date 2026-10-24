import json
import logging
import os
import httpx
from pathlib import Path
import time
import aiofiles
from backend.app.config import DATASET_DIR, WLASL_VIDEOS_DIR

logger = logging.getLogger(__name__)

LEARNED_WORDS_FILE = DATASET_DIR / "learned_words.json"

class DictionaryService:
    def __init__(self):
        self.learned_words = {}
        self.load_learned_words()

    def load_learned_words(self):
        try:
            if LEARNED_WORDS_FILE.exists():
                with open(LEARNED_WORDS_FILE, 'r', encoding='utf-8') as f:
                    self.learned_words = json.load(f)
                logger.info(f"Loaded {len(self.learned_words)} dynamically learned words.")
            else:
                self.learned_words = {}
        except Exception as e:
            logger.error(f"Error loading learned words: {e}")
            self.learned_words = {}

    def save_learned_words(self):
        try:
            with open(LEARNED_WORDS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.learned_words, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving learned words: {e}")

    async def learn_new_word(self, word: str) -> dict:
        """
        Looks up a word online using the Free Dictionary API and adds it to the dictionary.
        """
        word = word.strip().lower()
        if not word:
            return {"success": False, "error": "Empty word"}

        if word in self.learned_words:
            return {
                "success": True,
                "word": word,
                "data": self.learned_words[word],
                "source": "cache"
            }

        try:
            from groq import Groq
            groq_api_key = os.getenv("GROQ_API_KEY", "")
            client = Groq(api_key=groq_api_key)
            
            prompt = f"Provide exactly 3 common, simple, one-word English synonyms for the word '{word}' separated by commas. Do not write anything else. Example: fast -> quick, rapid, swift"
            
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=30,
                timeout=1.5
            )
            
            response = chat.choices[0].message.content.strip()
            synonyms = [s.strip().lower() for s in response.split(",") if s.strip()]
            
            data = {"synonyms": synonyms}
            self.learned_words[word] = data
            
            # Fire and forget save to avoid blocking
            import asyncio
            asyncio.create_task(asyncio.to_thread(self.save_learned_words))
            
            return {
                "success": True,
                "word": word,
                "data": data,
                "source": "groq_llm"
            }
        except Exception as e:
            logger.warning(f"Groq synonym fetch failed: {e}")
            return {"success": False, "error": "Synonym LLM failed"}

    async def search_external_sign_resource(self, word: str) -> bool:
        """
        Search configured external sign-language resources for the word.
        If a compatible sign is found, downloads/caches it to WLASL_VIDEOS_DIR,
        registers it in the local dictionary (video_word_map.json), and returns True.
        """
        word_lower = word.lower()
        external_api = os.getenv("EXTERNAL_SIGN_API_URL")
        if not external_api:
            # No external resource configured.
            return False
            
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"{external_api}/search?word={word_lower}", timeout=2.0)
                if res.status_code == 200:
                    data = res.json()
                    video_url = data.get("video_url")
                    if video_url:
                        # Download and cache it
                        video_res = await client.get(video_url)
                        if video_res.status_code == 200:
                            # Generate a unique 5 digit ID like WLASL
                            new_id = str(int(time.time()))[-5:]
                            video_path = WLASL_VIDEOS_DIR / f"{new_id}.mp4"
                            async with aiofiles.open(video_path, "wb") as f:
                                await f.write(video_res.content)
                                
                            # Register in local dictionary
                            video_map_path = DATASET_DIR / "wlasl" / "video_word_map.json"
                            if video_map_path.exists():
                                with open(video_map_path, 'r', encoding='utf-8') as f:
                                    vmap = json.load(f)
                                vmap[new_id] = {
                                    "word": word_lower,
                                    "path": f"dataset/wlasl/videos/{new_id}.mp4"
                                }
                                with open(video_map_path, 'w', encoding='utf-8') as f:
                                    json.dump(vmap, f)
                                    
                            logger.info(f"Downloaded and registered new external sign for '{word_lower}' (ID: {new_id})")
                            return True
        except Exception as e:
            logger.error(f"Error searching external sign resource for '{word_lower}': {e}")
            
        return False

dictionary_service = DictionaryService()
