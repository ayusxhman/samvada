"""The sole application-level coordinator for AI request pipelines."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Awaitable, Callable, TypeVar

from backend.app.orchestration.performance import PerformanceMonitor
from backend.app.orchestration.session_memory import SessionMemory
from backend.app.services.sign_to_text import RecognitionPrediction, sign_to_text_service
from backend.app.services.grammar import grammar_service
from backend.app.services.speech_service import speech_service
from backend.app.services.text_to_sign import text_to_sign_service
from backend.app.services.translation import translation_service

logger = logging.getLogger(__name__)
T = TypeVar("T")


class AIOrchestrator:
    """Coordinates specialised modules and records measurable request outcomes.

    Module implementations remain independently replaceable behind this facade.
    The orchestrator intentionally reports unavailable capabilities instead of
    substituting generated signs, model outputs, or evaluation data.
    """

    def __init__(self) -> None:
        self.session_memory = SessionMemory()
        self.performance_monitor = PerformanceMonitor()

    async def _measure_async(self, operation: str, work: Callable[[], Awaitable[T]]) -> T:
        started_at = perf_counter()
        success = False
        try:
            result = await work()
            success = True
            return result
        finally:
            self.performance_monitor.record(operation, (perf_counter() - started_at) * 1000, success=success)

    def _measure_sync(self, operation: str, work: Callable[[], T]) -> T:
        started_at = perf_counter()
        success = False
        try:
            result = work()
            success = True
            return result
        finally:
            self.performance_monitor.record(operation, (perf_counter() - started_at) * 1000, success=success)

    def translate_language(self, *, text: str, source_lang: str, target_lang: str) -> str:
        return self._measure_sync(
            "language_translation",
            lambda: translation_service.translate(text=text, source_lang=source_lang, target_lang=target_lang),
        )

    async def text_to_sign(
        self,
        *,
        text: str,
        target_lang: str,
        source_lang: str,
        session_id: str | None,
    ) -> dict[str, Any]:
        context = self.session_memory.get_or_create(session_id)

        async def work() -> dict[str, Any]:
            result = await text_to_sign_service.translate_to_sign(
                text=text,
                target_lang=target_lang or context.preferred_language,
                source_lang=source_lang,
            )
            self.session_memory.record_translation(
                context.session_id,
                english_text=result["english_text"],
                caption_text=result["caption_text"],
                target_language=target_lang,
            )
            # Record any dynamically retrieved synonyms or fingerspelled terms as learned dictionary entries
            for item in result.get("gloss_metadata", []):
                if item["source"] in ("semantic_equivalent", "fingerspelled"):
                    orig = item["original"] or item["gloss"].lower()
                    if orig not in context.learned_dictionary_entries:
                        context.learned_dictionary_entries.append(orig)
                        
            result["session_id"] = context.session_id
            return result

        return await self._measure_async("text_to_sign", work)

    def recognize_fingerspell(self, frame: Any) -> RecognitionPrediction:
        return self._measure_sync("fingerspell_recognition", lambda: sign_to_text_service.predict_fingerspell(frame))

    def recognize_word(self, landmarks: Any) -> RecognitionPrediction:
        return self._measure_sync("word_recognition", lambda: sign_to_text_service.predict_word(landmarks))

    def recognize_sentence(self, landmarks: Any) -> RecognitionPrediction:
        return self._measure_sync("sentence_recognition", lambda: sign_to_text_service.predict_sentence(landmarks))

    def translate_sign_sequence(self, *, gloss_sequence: list[str], target_lang: str) -> dict[str, Any]:
        return self._measure_sync(
            "sign_sequence_translation",
            lambda: {
                "glosses": gloss_sequence,
                "english_text": grammar_service.correct_grammar(gloss_sequence),
                "translated_text": translation_service.translate(
                    text=grammar_service.correct_grammar(gloss_sequence),
                    target_lang=target_lang,
                    source_lang="en",
                ),
            },
        )

    def transcribe_and_translate(self, *, audio_bytes: bytes, target_lang: str, source_lang: str) -> dict[str, Any]:
        return self._measure_sync(
            "speech_transcription_translation",
            lambda: speech_service.transcribe_and_translate(audio_bytes, target_lang, source_lang),
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "operations": self.performance_monitor.snapshot(),
            "models": sign_to_text_service.readiness(),
        }


ai_orchestrator = AIOrchestrator()
