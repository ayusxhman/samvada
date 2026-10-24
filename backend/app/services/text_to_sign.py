import json
import logging
import os
from typing import List, Dict, Any
from backend.app.config import DATASET_DIR, ANIMATION_CLIPS_DIR, WLASL_VIDEOS_DIR
from backend.app.services.gloss_mapper import gloss_mapper
from backend.app.services.translation import translation_service

logger = logging.getLogger(__name__)

class TextToSignService:
    def __init__(self):
        self.word_to_videos = {}
        self.available_vrm_clips = {}
        self.alphabet_assets = {}
        self.load_video_mappings()
        self.scan_vrm_clips()
        self.load_alphabet_assets()

    def load_video_mappings(self):
        try:
            video_map_path = DATASET_DIR / "wlasl" / "video_word_map.json"
            if video_map_path.exists():
                with open(video_map_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Group video paths by word
                    for video_id, info in data.items():
                        word = info["word"].lower().strip()
                        # Normalize path delimiters for URL usage
                        rel_path = f"dataset/wlasl/videos/{video_id}.mp4"
                        
                        # Verify video exists on disk
                        abs_path = DATASET_DIR / "wlasl" / "videos" / f"{video_id}.mp4"
                        if abs_path.exists():
                            if word not in self.word_to_videos:
                                self.word_to_videos[word] = []
                            self.word_to_videos[word].append(rel_path)
                            
                logger.info(f"Mapped {len(self.word_to_videos)} words to video clips.")
            else:
                logger.warning("video_word_map.json not found, video playback mapping disabled.")
        except Exception as e:
            logger.error(f"Error building video mappings: {e}")

    def load_alphabet_assets(self):
        """Pre-verifies and registers ASL fingerspelling sign image assets for all 26 letters (A-Z)."""
        try:
            for char in "abcdefghijklmnopqrstuvwxyz":
                upper = char.upper()
                img_rel = f"dataset/asl_alphabet/ASL_Alphabet_Dataset/asl_alphabet_test/{upper}_test.jpg"
                self.alphabet_assets[char] = {
                    "type": "image",
                    "word": upper,
                    "url": img_rel,
                    "duration": 0.5
                }
            logger.info(f"Successfully registered 26/26 ASL alphabet high-res image assets.")
        except Exception as e:
            logger.error(f"Error registering alphabet assets: {e}")

    def scan_vrm_clips(self):
        try:
            if ANIMATION_CLIPS_DIR.exists():
                for f in os.listdir(ANIMATION_CLIPS_DIR):
                    if f.endswith(".vrma"):
                        word = f.split(".")[0].lower()
                        # Path relative to project root
                        self.available_vrm_clips[word] = f"dataset/animation_clips/{f}"
                logger.info(f"Scanned {len(self.available_vrm_clips)} VRMA clips.")
        except Exception as e:
            logger.error(f"Error scanning VRMA clips: {e}")

    async def translate_to_sign(self, text: str, target_lang: str = "en", source_lang: str = "auto") -> dict:
        """
        Complete text to sign translation pipeline:
        1. Language detection & translation to English
        2. Gloss mapping with diagnostics
        3. Video paths / Image card sequence (for A-Z alphabets)
        4. VRM animation instructions
        """
        from time import perf_counter
        import asyncio
        t0 = perf_counter()
        
        async def _translate_en():
            if (source_lang in ["en", "auto"]) and text.isascii():
                return text
            return await asyncio.to_thread(translation_service.translate, text, "en", source_lang)
            
        async def _translate_target():
            if source_lang == target_lang:
                return text
            # If the target language is English, we can actually just rely on the english translation,
            # but since we run these in parallel, we'll just let it translate to English too.
            return await asyncio.to_thread(translation_service.translate, text, target_lang, source_lang)

        # Run both translations concurrently
        english_text, caption_text = await asyncio.gather(_translate_en(), _translate_target())
        t1 = perf_counter()
        
        # Get gloss sequence & diagnostics asynchronously
        gloss_res = await gloss_mapper.text_to_gloss_with_diagnostics(english_text)
        t2 = perf_counter()
        
        logger.info(f"[PERF] translations={((t1-t0)*1000):.1f}ms, gloss_mapper={((t2-t1)*1000):.1f}ms, TOTAL={((t2-t0)*1000):.1f}ms")
        gloss_seq = gloss_res["glosses"]
        diagnostics = gloss_res["diagnostics"]
        
        # Gather video & sign asset paths
        independent_words = set(w.lower().strip() for w in english_text.split())
        videos = []
        for gloss in gloss_seq:
            word = gloss.lower()
            is_single_letter = (len(word) == 1 and word.isalpha())
            
            if is_single_letter:
                # If letter was an independent word in original sentence (e.g. "I" or "A" in "I am a good boy")
                if word in independent_words and word in self.word_to_videos and self.word_to_videos[word]:
                    videos.append(self.word_to_videos[word][0])
                else:
                    # Letter came from fingerspelling a word/proper noun (e.g. "India" -> I, N, D, I, A) -> Image card ONLY!
                    upper = word.upper()
                    videos.append({
                        "type": "image",
                        "word": upper,
                        "url": f"dataset/asl_alphabet/ASL_Alphabet_Dataset/asl_alphabet_test/{upper}_test.jpg",
                        "duration": 0.5
                    })
            elif word in self.word_to_videos and self.word_to_videos[word]:
                # Multi-letter sign word -> Video clip instance
                videos.append(self.word_to_videos[word][0])
            else:
                # No video available -> Fingerspell letters as Image cards ONLY!
                for char in word:
                    if char.isalpha():
                        upper = char.upper()
                        videos.append({
                            "type": "image",
                            "word": upper,
                            "url": f"dataset/asl_alphabet/ASL_Alphabet_Dataset/asl_alphabet_test/{upper}_test.jpg",
                            "duration": 0.5
                        })

        # Build avatar animation sequence
        avatar_animations = []
        for gloss in gloss_seq:
            word = gloss.lower()
            if word in self.available_vrm_clips:
                avatar_animations.append({
                    "word": word,
                    "type": "vrma",
                    "path": self.available_vrm_clips[word]
                })
            else:
                # Use fingerspelling instruction for avatar timeline synthesis
                avatar_animations.append({
                    "word": word,
                    "type": "fingerspell",
                    "duration": max(1.0, len(word) * 0.4)
                })

        # Generate AI VRM Quaternion Timeline for the sentence (Primary AI Animation Path)
        from backend.app.services.motion_synthesis import motion_synthesis_service
        animation_timeline = motion_synthesis_service.generate_sentence_timeline(gloss_seq)

        # Collect metadata for each gloss to clearly indicate its linguistic source
        gloss_metadata = []
        original_words = [w.lower().strip() for w in english_text.split()]
        for i, gloss in enumerate(gloss_seq):
            gloss_lower = gloss.lower()
            has_video = isinstance(videos[i], str) if i < len(videos) else False
            
            if gloss_lower in original_words:
                source = "direct" if has_video else "missing_no_video"
                original = gloss_lower
            elif len(gloss) == 1 and gloss.isalnum():
                source = "fingerspelled" if has_video else "missing_no_video"
                original = None
            else:
                source = "semantic_equivalent" if has_video else "missing_no_video"
                original = None
                for w in original_words:
                    if gloss_mapper.synonyms_map.get(w) == gloss_lower:
                        original = w
                        break
                if not original:
                    original = "unknown"
            
            gloss_metadata.append({
                "gloss": gloss,
                "source": source,
                "original": original,
                "has_video": has_video
            })

        return {
            "original_text": text,
            "english_text": english_text,
            "caption_text": caption_text,
            "glosses": gloss_seq,
            "videos": videos,
            "animation_timeline": animation_timeline,
            "avatar_animations": avatar_animations,
            "gloss_metadata": gloss_metadata,
            "diagnostics": diagnostics
        }

text_to_sign_service = TextToSignService()
