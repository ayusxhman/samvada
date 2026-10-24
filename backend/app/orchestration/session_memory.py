"""Bounded in-process context for an active translation session."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Deque
from uuid import uuid4


@dataclass
class SessionContext:
    session_id: str
    preferred_language: str = "en"
    preferred_sign_language: str = "asl"
    playback_speed: float = 1.0
    preferred_avatar_settings: dict[str, object] = field(default_factory=dict)
    learned_dictionary_entries: list[str] = field(default_factory=list)
    recent_english_text: Deque[str] = field(default_factory=lambda: deque(maxlen=10))
    translated_terms: dict[str, str] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionMemory:
    """Thread-safe, bounded memory intended for a single running API process.

    This is deliberately not a persistence layer. Multi-worker deployments must
    replace it with a shared, privacy-reviewed session store.
    """

    def __init__(self, max_sessions: int = 1_000) -> None:
        self._max_sessions = max_sessions
        self._sessions: dict[str, SessionContext] = {}
        self._lock = RLock()

    def get_or_create(self, session_id: str | None = None) -> SessionContext:
        with self._lock:
            identifier = session_id or str(uuid4())
            context = self._sessions.get(identifier)
            if context is None:
                if len(self._sessions) >= self._max_sessions:
                    oldest_id = min(self._sessions, key=lambda key: self._sessions[key].updated_at)
                    del self._sessions[oldest_id]
                context = SessionContext(session_id=identifier)
                self._sessions[identifier] = context
            context.updated_at = datetime.now(timezone.utc)
            return context

    def record_translation(
        self,
        session_id: str,
        *,
        english_text: str,
        caption_text: str,
        target_language: str,
    ) -> SessionContext:
        with self._lock:
            context = self.get_or_create(session_id)
            context.preferred_language = target_language
            context.recent_english_text.append(english_text)
            context.translated_terms[english_text.lower()] = caption_text
            context.updated_at = datetime.now(timezone.utc)
            return context

    def snapshot(self, session_id: str) -> dict[str, object] | None:
        with self._lock:
            context = self._sessions.get(session_id)
            if context is None:
                return None
            return {
                "session_id": context.session_id,
                "preferred_language": context.preferred_language,
                "preferred_sign_language": context.preferred_sign_language,
                "playback_speed": context.playback_speed,
                "preferred_avatar_settings": context.preferred_avatar_settings,
                "learned_dictionary_entries": context.learned_dictionary_entries,
                "recent_english_text": list(context.recent_english_text),
                "translated_term_count": len(context.translated_terms),
                "updated_at": context.updated_at.isoformat(),
            }
