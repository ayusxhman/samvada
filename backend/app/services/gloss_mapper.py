import json
import logging
from typing import List
from backend.app.config import SYNONYMS_MAP_PATH, DATASET_DIR
from backend.app.services.grammar import _get_nlp

logger = logging.getLogger(__name__)

class GlossMapper:
    def __init__(self):
        self.synonyms_map = {}
        self.wlasl_words = set()
        self.load_synonyms_map()
        self.load_wlasl_vocabulary()

    def load_synonyms_map(self):
        try:
            if SYNONYMS_MAP_PATH.exists():
                with open(SYNONYMS_MAP_PATH, 'r', encoding='utf-8') as f:
                    self.synonyms_map = json.load(f)
                logger.info(f"Loaded {len(self.synonyms_map)} word synonyms.")
            else:
                logger.warning(f"Synonyms map not found at {SYNONYMS_MAP_PATH}")
        except Exception as e:
            logger.error(f"Error loading synonyms map: {e}")

    def load_wlasl_vocabulary(self):
        try:
            video_map_path = DATASET_DIR / "wlasl" / "video_word_map.json"
            if video_map_path.exists():
                with open(video_map_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract all unique words
                    self.wlasl_words = {item["word"].lower().strip() for item in data.values() if "word" in item}
                logger.info(f"Loaded {len(self.wlasl_words)} unique WLASL vocabulary words.")
            else:
                # Fallback to class list
                class_list_path = DATASET_DIR / "wlasl" / "wlasl_class_list.txt"
                if class_list_path.exists():
                    with open(class_list_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                self.wlasl_words.add(parts[1].lower().strip())
                    logger.info(f"Loaded {len(self.wlasl_words)} WLASL words from class list.")
        except Exception as e:
            logger.error(f"Error loading WLASL vocabulary: {e}")
            
        # Add local cached sign assets (VRM clips)
        try:
            from backend.app.config import ANIMATION_CLIPS_DIR
            import os
            if ANIMATION_CLIPS_DIR.exists():
                for f in os.listdir(ANIMATION_CLIPS_DIR):
                    if f.endswith(".vrma"):
                        self.wlasl_words.add(f.split(".")[0].lower())
        except Exception as e:
            logger.error(f"Error loading animation clips to vocabulary: {e}")

    def _save_synonyms_map(self):
        """Persist updated synonym mappings back to disk for instant future lookups."""
        try:
            with open(SYNONYMS_MAP_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.synonyms_map, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.synonyms_map)} synonym mappings to disk.")
        except Exception as e:
            logger.error(f"Error saving synonyms map: {e}")

    SEMANTIC_TAXONOMY = {
        # Food & Cooking
        "curry": "food",
        "gravy": "food",
        "stew": "food",
        "sauce": "food",
        "soup": "food",
        
        # Vehicles & Timepieces
        "vehicle": "car",
        "automobile": "car",
        "timepiece": "clock",
        "clockmaster": "clock",
        "watchmaker": "clock",
        "clockmaker": "clock",
        
        # Housing & Living
        "residence": "house",
        "dwelling": "house",
        "accommodation": "house",
        "apartment": "house",
        
        # Financial & Commerce
        "payment": "pay",
        "purchase": "buy",
        "acquire": "buy",
        "fiscal": "money",
        "financial": "money",
        "revenue": "money",
        "corporation": "business",
        "quarterly": "month",
        "officer": "lead",
        "chief": "head",
        "executive": "lead",
        
        # Formal, Business & Complex Concepts
        "notwithstanding": "yet",
        "mitigation": "help",
        "mitigate": "help",
        "meticulous": "careful",
        "plummet": "fall",
        "plummeted": "fall",
        "unprecedented": "new",
        "disruption": "trouble",
        "paralyze": "stop",
        "paralyzed": "stop",
        "shipping": "ship",
        "lane": "street",
        "lanes": "street",
        "supply": "provide",
        "global": "world",
        
        # People & Pronouns
        "nobody": "none",
        "no-one": "none",
        
        # Actions & Concepts
        "trick": "cheat",
        "deceive": "cheat",
        "fool": "cheat",
        "stared": "stare",
        "stare": "stare",
        "biological": "body",
        "youth": "young",
        "college": "school",
        "academy": "school",
        "university": "school",
        "football": "game",
        "soccer": "game",
        "roam": "walk",
        "wander": "walk",
        "stroll": "walk",
        "park": "park"
    }

    def decompose_compound(self, word: str) -> str | None:
        """Decomposes compound words and morphological suffixes (-ment, -master, -maker, -ness)."""
        w = word.lower().strip()
        suffixes = [
            ("ment", 4),
            ("master", 6),
            ("maker", 5),
            ("ness", 4),
            ("less", 4)
        ]
        for suf, length in suffixes:
            if w.endswith(suf) and len(w) > length + 2:
                base = w[:-length]
                if base in self.wlasl_words:
                    return base
        return None

    async def resolve_synonym_recursive(self, word: str, pos_tag: str = "NOUN", max_depth: int = 4) -> str | None:
        """
        Recursively searches for an in-vocabulary sign word by traversing synonyms up to max_depth levels.
        Uses Breadth-First Search (BFS) graph traversal across:
        1. Local static synonym maps & SEMANTIC_TAXONOMY
        2. Cached learned_words dictionary
        3. Datamuse & Free Dictionary API online lookups
        
        Returns the closest matching in-vocabulary gloss word, or None if no synonym matches.
        """
        from backend.app.services.dictionary import dictionary_service
        from collections import deque
        
        target_w = word.lower().strip()
        if not target_w or len(target_w) == 1:
            return None
            
        queue = deque([(target_w, 0)])
        visited = {target_w}
        
        while queue:
            curr_word, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
                
            candidates = []
            
            # Source 1: synonyms_map
            if curr_word in self.synonyms_map:
                candidates.append(self.synonyms_map[curr_word])
                
            # Source 2: SEMANTIC_TAXONOMY
            if curr_word in self.SEMANTIC_TAXONOMY:
                candidates.append(self.SEMANTIC_TAXONOMY[curr_word])
                
            # Source 3: decompose_compound
            decomp = self.decompose_compound(curr_word)
            if decomp:
                candidates.append(decomp)
                
            # Source 4: Online Dictionary / Datamuse API lookup (synonyms + means-like)
            # DISABLED: 1.5s per missing word latency.
                        
            # Source 5: Datamuse Means-Like (ml) Broad Concept Fallback
            # REMOVED: Datamuse API is too slow (adds 8 seconds of latency) and blocks the STT pipeline.
            # Groq Whisper and LLM pipelines are now fast enough that we don't need this slow HTTP fallback.
    
            # Evaluate all discovered candidates for this hop
            for cand in candidates:
                cand_clean = cand.lower().strip()
                if not cand_clean or cand_clean in visited:
                    continue
                visited.add(cand_clean)
                
                # Check if candidate is directly in WLASL / VRM vocabulary
                if cand_clean in self.wlasl_words:
                    logger.info(f"[RECURSIVE-SYNONYM] Resolved '{word}' -> '{cand_clean}' (Depth {depth+1})")
                    # Learn direct shortcut mapping for active learning
                    self.synonyms_map[target_w] = cand_clean
                    self._save_synonyms_map()
                    return cand_clean
                    
                # Push to queue for next depth level search
                queue.append((cand_clean, depth + 1))
            
        return None

    async def text_to_gloss_with_diagnostics(self, text: str) -> dict:
        """
        Maps English sentence or paragraph to an ASL gloss sequence using a multi-tiered semantic pipeline.
        Returns a dict containing final glosses and complete diagnostic execution traces.
        """
        from backend.app.services.grammar import grammar_service
        from backend.app.services.dictionary import dictionary_service

        # 1. Handle paragraphs by splitting into sentences using spaCy if available
        sentences = [text]
        spacy_nlp = _get_nlp()
        if spacy_nlp is not None:
            doc = spacy_nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            if sents:
                sentences = sents

        # Build token POS & Named Entity map from full sentence context
        token_info_map = {}
        if spacy_nlp is not None:
            full_doc = spacy_nlp(text)
            for token in full_doc:
                token_info_map[token.text.lower()] = {
                    "lemma": token.lemma_.lower(),
                    "pos": token.pos_,
                    "ent_type": token.ent_type_
                }

        all_glosses = []
        all_grammar_parses = []
        dictionary_hits = []
        unknown_words = []
        synonym_chosen = {}
        sign_chosen = []
        fingerspelled_reasons = {}

        for sent in sentences:
            base_glosses = grammar_service.english_to_asl_gloss(sent)
            all_grammar_parses.append(base_glosses)

            for word in base_glosses:
                w = word.lower().strip()
                if not w:
                    continue

                info = token_info_map.get(w, {})
                pos_tag = info.get("pos", "NOUN")
                ent_type = info.get("ent_type", "")
                lemma = info.get("lemma", w)

                # Step 0: Single-letter tokens (fingerspelling) MUST NOT be mutated by online synonyms (e.g. C -> LOOK, X -> NO)
                if len(w) == 1 and w.isalpha():
                    all_glosses.append(w.upper())
                    dictionary_hits.append(w)
                    sign_chosen.append(w.upper())
                    continue

                # Step 0.5: Proper Noun check (Names, Places, Cities, Countries, Entities)
                is_proper = (pos_tag == "PROPN" or ent_type in ["PERSON", "GPE", "LOC", "FAC", "ORG"])

                if is_proper and w not in self.wlasl_words:
                    spelled = [char.upper() for char in w if char.isalnum()]
                    all_glosses.extend(spelled)
                    sign_chosen.append(f"FINGERSPELL({' '.join(spelled)})")
                    reason = f"Proper noun '{w}' registered for ASL fingerspelling."
                    fingerspelled_reasons[w] = reason
                    logger.info(f"Fingerspelling proper noun '{w}': {spelled}")
                    continue

                # Step 1: Explicit static synonym map
                if w in self.synonyms_map:
                    mapped_w = self.synonyms_map[w]
                    synonym_chosen[w] = mapped_w
                    w = mapped_w

                # Step 2: Direct WLASL match
                if w in self.wlasl_words:
                    all_glosses.append(w.upper())
                    dictionary_hits.append(w)
                    sign_chosen.append(w.upper())
                    continue

                # Step 3: SpaCy Lemmatization (replaces naive suffix trimming)
                lemma_found = False
                pos_tag = "NOUN"
                if spacy_nlp is not None:
                    doc = spacy_nlp(w)
                    lemma = doc[0].lemma_.lower()
                    pos_tag = doc[0].pos_
                    if lemma in self.wlasl_words:
                        logger.info(f"Lemmatized '{w}' -> '{lemma}'")
                        all_glosses.append(lemma.upper())
                        dictionary_hits.append(lemma)
                        sign_chosen.append(lemma.upper())
                        synonym_chosen[w] = lemma
                        lemma_found = True

                if lemma_found:
                    continue

                unknown_words.append(w)

                # Step 4: Check explicit taxonomy map
                if w in self.SEMANTIC_TAXONOMY and self.SEMANTIC_TAXONOMY[w] in self.wlasl_words:
                    target_gloss = self.SEMANTIC_TAXONOMY[w].upper()
                    all_glosses.append(target_gloss)
                    synonym_chosen[w] = self.SEMANTIC_TAXONOMY[w]
                    sign_chosen.append(target_gloss)
                    logger.info(f"Taxonomy match for '{w}' -> '{target_gloss}'")
                    continue

                # Step 5: Morphological & Compound Decomposition
                decomp = self.decompose_compound(w)
                if decomp and decomp in self.wlasl_words:
                    target_gloss = decomp.upper()
                    all_glosses.append(target_gloss)
                    synonym_chosen[w] = decomp
                    sign_chosen.append(target_gloss)
                    logger.info(f"Morphological decomposition for '{w}' -> '{target_gloss}'")
                    continue

                # Step 6: Multi-Hop Recursive Semantic Synonym Search via Graph BFS (Depth 4)
                logger.info(f"Attempting multi-hop recursive synonym lookup for '{w}'...")
                recursive_match = await self.resolve_synonym_recursive(w, pos_tag=pos_tag, max_depth=4)
                if recursive_match:
                    all_glosses.append(recursive_match.upper())
                    synonym_chosen[w] = recursive_match
                    sign_chosen.append(recursive_match.upper())
                    logger.info(f"Multi-hop recursive synonym matched for '{w}' -> '{recursive_match}'")
                    continue

                # Step 7: Search external sign resources (Online Active Learning Download)
                logger.info(f"Checking external sign resources for '{w}'...")
                external_found = await dictionary_service.search_external_sign_resource(w)
                if external_found:
                    self.wlasl_words.add(w)
                    all_glosses.append(w.upper())
                    sign_chosen.append(w.upper())
                    continue

                # Step 8: Final Fallback (Proper Nouns vs Common Words)
                is_proper_noun = False
                if spacy_nlp is not None:
                    doc = spacy_nlp(w)
                    if doc[0].pos_ == 'PROPN' or doc[0].ent_type_ in ['PERSON', 'GPE', 'LOC', 'FAC', 'ORG']:
                        is_proper_noun = True

                if is_proper_noun:
                    # Proper Nouns (Names, Places) -> Genuine ASL Fingerspelling
                    spelled = [char.upper() for char in w if char.isalnum()]
                    all_glosses.extend(spelled)
                    sign_chosen.append(f"FINGERSPELL({' '.join(spelled)})")
                    reason = f"Proper noun '{w}' registered for ASL fingerspelling."
                    fingerspelled_reasons[w] = reason
                    logger.info(f"Fingerspelling proper noun '{w}': {spelled}")
                else:
                    # Common Words -> Substitute closest conceptual sign
                    fallback_concept = "THING"
                    if pos_tag == "VERB":
                        fallback_concept = "DO"
                    elif pos_tag in ["ADJ", "ADV"]:
                        fallback_concept = "VERY"
                    elif pos_tag == "NOUN":
                        fallback_concept = "THING"

                    all_glosses.append(fallback_concept)
                    synonym_chosen[w] = fallback_concept.lower()
                    sign_chosen.append(fallback_concept)
                    logger.info(f"Common word fallback substitution for '{w}' -> '{fallback_concept}'")

        diagnostics = {
            "original_english": text,
            "parsed_syntax_tree": all_grammar_parses,
            "extracted_concepts": sign_chosen,
            "dictionary_hits": dictionary_hits,
            "synonym_substitutions": synonym_chosen,
            "unknown_concepts": unknown_words,
            "fingerspelled_reasons": fingerspelled_reasons,
            "final_gloss": all_glosses
        }

        # Print detailed execution trace to logger
        logger.info(f"=== GLOSS PIPELINE DIAGNOSTICS ===")
        logger.info(f"Original English: {text}")
        logger.info(f"Parsed Syntax Tree: {all_grammar_parses}")
        logger.info(f"Extracted Concepts: {sign_chosen}")
        logger.info(f"Dictionary Hits: {dictionary_hits}")
        logger.info(f"Synonym Substitutions: {synonym_chosen}")
        logger.info(f"Unknown Concepts: {unknown_words}")
        logger.info(f"Fingerspelled Reasons: {fingerspelled_reasons}")
        logger.info(f"Final Gloss: {all_glosses}")
        logger.info(f"=================================")

        return {
            "glosses": all_glosses,
            "diagnostics": diagnostics
        }

    async def text_to_gloss(self, text: str) -> List[str]:
        res = await self.text_to_gloss_with_diagnostics(text)
        return res["glosses"]

def re_sub_punc(text: str) -> str:
    import re
    # Keep alphanumeric characters and spaces
    return re.sub(r'[^a-z0-9\s]', '', text)

gloss_mapper = GlossMapper()
