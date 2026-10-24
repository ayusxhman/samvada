"""Small, dependency-free operation timing monitor."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock


@dataclass
class OperationSamples:
    durations_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    failures: int = 0


class PerformanceMonitor:
    def __init__(self) -> None:
        self._operations: dict[str, OperationSamples] = defaultdict(OperationSamples)
        self._lock = RLock()

    def record(self, operation: str, duration_ms: float, *, success: bool) -> None:
        with self._lock:
            samples = self._operations[operation]
            samples.durations_ms.append(duration_ms)
            if not success:
                samples.failures += 1

    def snapshot(self) -> dict[str, dict[str, float | int | None]]:
        with self._lock:
            result: dict[str, dict[str, float | int | None]] = {}
            for name, samples in self._operations.items():
                durations = sorted(samples.durations_ms)
                count = len(durations)
                result[name] = {
                    "samples": count,
                    "failures": samples.failures,
                    "last_ms": round(durations[-1], 2) if durations else None,
                    "mean_ms": round(sum(durations) / count, 2) if durations else None,
                    "p95_ms": round(durations[min(count - 1, int(count * 0.95))], 2) if durations else None,
                }
            return result
