import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Lazy-loaded spacy to avoid blocking server startup (~1.7s saved)
_nlp = None
_spacy_checked = False

def _get_nlp():
    global _nlp, _spacy_checked
    if _spacy_checked:
        return _nlp
    _spacy_checked = True
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.info("Spacy en_core_web_sm loaded (lazy).")
    except ImportError:
        logger.warning("spacy is not installed. English to ASL mapping will use basic fallback rules.")
    except Exception as e:
        logger.warning(f"Could not load spacy en_core_web_sm model: {e}")
    return _nlp

# Backward-compat aliases used by gloss_mapper.py
nlp = None  # Will be set on first _get_nlp() call
SPACY_AVAILABLE = True  # Assume available; _get_nlp() will verify

class GrammarService:
    def __init__(self):
        # Common ASL gloss patterns and mappings to English sentences
        self.common_phrases = {
            "hello": "Hello",
            "thank you": "Thank you",
            "please": "Please",
            "sorry": "I am sorry",
            "good morning": "Good morning",
            "good night": "Good night",
            "how you": "How are you?",
            "how are you": "How are you?",
            "what your name": "What is your name?",
            "me love you": "I love you",
            "nice meet you": "Nice to meet you",
            "me name": "My name is",
            "where you from": "Where are you from?",
            "you understand": "Do you understand?",
            "me understand": "I understand",
            "not understand": "I don't understand",
            "help me": "Help me",
            "me help you": "I will help you",
            "you welcome": "You're welcome",
            "see you later": "See you later",
            "bye": "Goodbye",
            "yes": "Yes",
            "no": "No",
            "me hungry": "I am hungry",
            "me thirsty": "I am thirsty",
            "me tired": "I am tired",
            "me happy": "I am happy",
            "me sad": "I am sad",
            "what time": "What time is it?",
            "where bathroom": "Where is the bathroom?",
            "me go home": "I am going home",
            "you come": "Are you coming?",
            "me need help": "I need help",
        }
        
    def english_to_asl_gloss(self, sentence: str) -> List[str]:
        """
        Converts fluent English into ASL grammar structure (Time-Topic-Comment).
        Uses Dependency Parsing via SpaCy.
        """
        if not sentence or not str(sentence).strip():
            return []
            
        sentence = sentence.lower().strip()
        
        # 1. Exact common phrases bypass
        reverse_common = {v.lower(): k for k, v in self.common_phrases.items()}
        # Normalize punctuation
        norm_sent = sentence.replace("?", "").replace(".", "").replace("!", "").strip()
        if norm_sent in reverse_common:
            return reverse_common[norm_sent].upper().split()
            
        # 2. Dependency Parsing (Time-Topic-Comment)
        spacy_nlp = _get_nlp()
        if spacy_nlp is not None:
            doc = spacy_nlp(sentence)
            
            time_words = []
            topic_words = []
            comment_words = []
            question_words = []
            negation_words = []
            
            # Identify components based on dependency tags and POS
            for token in doc:
                # Stopwords that ASL often drops (articles, 'be' verbs unless emphasized)
                if token.pos_ in ['DET', 'PUNCT'] or token.lemma_ == 'be':
                    if token.dep_ != 'ROOT':  # Keep 'be' if it's the root of a short sentence
                        continue
                
                # Fix common SpaCy en_core_web_sm mis-lemmatizations
                token_lemma = token.lemma_.lower()
                token_text = token.text.lower()
                LEMMA_CORRECTIONS = {
                    "stared": "stare",
                    "staring": "stare",
                    "stares": "stare"
                }
                if token_text in LEMMA_CORRECTIONS:
                    token_lemma = LEMMA_CORRECTIONS[token_text]
                
                lemma_upper = token_lemma.upper()
                
                # Negations go at the end usually
                if token.dep_ == 'neg':
                    negation_words.append(lemma_upper)
                    continue
                    
                # Questions usually go at the end
                if token.tag_ in ['WDT', 'WP', 'WP$', 'WRB'] or token_lemma in ['who', 'what', 'where', 'when', 'why', 'how']:
                    question_words.append(lemma_upper)
                    continue
                    
                # Time phrases (temporal modifiers) go at the beginning
                if token.ent_type_ in ['TIME', 'DATE'] or token.dep_ in ['npadvmod', 'tmod']:
                    time_words.append(lemma_upper)
                    continue
                    
                # Topic: Subjects and their modifiers
                if 'subj' in token.dep_ or token.head.dep_ == 'subj':
                    topic_words.append(lemma_upper)
                    continue
                    
                # Comment: Objects, Verbs, and everything else
                comment_words.append(lemma_upper)
                
            # Maintain logical sequential sentence progression: Time -> Topic(Subject) -> Comment(Verbs, Objects) -> Negation -> Question
            glosses = time_words + topic_words + comment_words + negation_words + question_words
            
            # Keep all glosses in clean sequential order
            final_glosses = [g for g in glosses if g]
                    
            if final_glosses:
                return final_glosses

        # 3. Fallback heuristic if SpaCy fails or isn't installed
        words = norm_sent.split()
        return [w.upper() for w in words if w not in ['a', 'an', 'the', 'is', 'are', 'am', 'to']]

    def correct_grammar(self, gloss_sequence: List[str]) -> str:
        """
        Corrects ASL gloss sequences back into fluent English using Active Intelligence.
        """
        if not gloss_sequence:
            return ""

        words = [w.lower().strip() for w in gloss_sequence if w]
        raw_phrase = " ".join(words)
        
        if raw_phrase in self.common_phrases:
            return self.common_phrases[raw_phrase]

        # 1. Detect Time Anchor Words
        past_time_words = {"yesterday", "past", "ago", "last night", "last week"}
        future_time_words = {"tomorrow", "future", "next week", "later"}
        
        time_anchor = None
        is_past = False
        is_future = False
        
        filtered_words = []
        for w in words:
            if w in past_time_words:
                time_anchor = w
                is_past = True
            elif w in future_time_words:
                time_anchor = w
                is_future = True
            else:
                filtered_words.append(w)

        # 2. Verb Tense Transformation Map
        past_verb_map = {
            "go": "went", "eat": "ate", "see": "saw", "buy": "bought",
            "make": "made", "read": "read", "have": "had", "need": "needed",
            "want": "wanted", "like": "liked", "help": "helped", "come": "came",
            "is": "was", "are": "were", "am": "was",
            "give": "gave", "take": "took", "tell": "told", "think": "thought",
            "get": "got", "know": "knew", "find": "found", "say": "said",
            "run": "ran", "write": "wrote", "sit": "sat", "stand": "stood",
            "drink": "drank", "sing": "sang", "drive": "drove", "break": "broke",
            "speak": "spoke", "bring": "brought", "teach": "taught", "learn": "learned",
            "feel": "felt", "leave": "left", "meet": "met", "pay": "paid",
            "put": "put", "cut": "cut", "let": "let", "set": "set",
            "show": "showed", "try": "tried", "study": "studied", "play": "played",
            "work": "worked", "walk": "walked", "talk": "talked", "call": "called",
            "start": "started", "stop": "stopped", "open": "opened", "close": "closed"
        }

        # 3. Process Pronouns & Verbs
        result_words = []
        is_question = any(qw in words for qw in ["what", "where", "who", "when", "why", "how"])
        
        for i, word in enumerate(filtered_words):
            if word in {"me", "i"}:
                if i == 0 or i < len(filtered_words) - 1:
                    result_words.append("I")
                else:
                    result_words.append("me")
            elif is_past and word in past_verb_map:
                result_words.append(past_verb_map[word])
            elif is_future and word in past_verb_map and word not in {"is", "are", "am"}:
                result_words.append(f"will {word}")
            else:
                result_words.append(word)

        sentence = " ".join(result_words)
        
        # 4. Attach Time Anchor Prefix
        if time_anchor:
            prefix = time_anchor.capitalize()
            sentence = f"{prefix}, {sentence}"
        elif sentence:
            sentence = sentence[0].upper() + sentence[1:]

        # 5. Terminal Punctuation
        if sentence:
            if sentence[-1] not in {'.', '?', '!'}:
                sentence += "?" if is_question else "."

        return sentence

grammar_service = GrammarService()
