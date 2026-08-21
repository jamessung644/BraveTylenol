"""Monotonic request-deadline accounting."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class RequestDeadline:
    expires_at: float
    clock: Callable[[], float]

    @classmethod
    def start(
        cls,
        total_seconds: float = 165.0,
        clock: Callable[[], float] = perf_counter,
    ) -> "RequestDeadline":
        return cls(expires_at=clock() + total_seconds, clock=clock)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - self.clock())

    def stage_timeout(self, cap_seconds: float, reserve_seconds: float = 0.0) -> float:
        return max(0.0, min(cap_seconds, self.remaining() - reserve_seconds))

    def can_spend(self, minimum_seconds: float) -> bool:
        return self.remaining() >= minimum_seconds
