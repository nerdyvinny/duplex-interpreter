"""Event bus connecting the pipeline to whatever is displaying it.

The pipeline never imports the UI. It emits events here; the CLI subscribes.
That keeps a future GUI a matter of adding a second subscriber rather than
rewriting the orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class Stage(str, Enum):
    """Where an utterance is in the cascade."""

    CAPTURED = "captured"  # VAD closed a segment
    TRANSCRIBED = "transcribed"  # STT returned text
    TRANSLATED = "translated"  # MT returned text
    SPEAKING = "speaking"  # first TTS bytes hit the speaker
    DONE = "done"  # playback finished
    DROPPED = "dropped"  # filtered out (self-echo, too short, no speech)
    ERROR = "error"


@dataclass
class PipelineEvent:
    """One state change for one utterance."""

    seq: int
    channel_id: str
    stage: Stage

    source_lang: str | None = None
    target_lang: str | None = None
    source_text: str | None = None
    target_text: str | None = None

    # Per-stage wall-clock durations in ms, accumulated as the utterance moves.
    timings: dict[str, float] = field(default_factory=dict)

    # Seconds of audio captured, for the "is my VAD tuned right" sanity check.
    audio_seconds: float | None = None

    detail: str | None = None  # drop reason or error message

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())


Subscriber = Callable[[PipelineEvent], None]


class EventBus:
    """Fan-out to synchronous subscribers.

    Subscribers run inline on the event loop, so they must be cheap and must
    not raise — a broken display should never take down a conversation.
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._history: list[PipelineEvent] = []
        self._history_limit = 500

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

        return unsubscribe

    def emit(self, event: PipelineEvent) -> None:
        self._history.append(event)
        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]

        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception:  # noqa: BLE001 - a bad subscriber must not kill the run
                log.exception("event subscriber failed")

    @property
    def history(self) -> list[PipelineEvent]:
        return list(self._history)


class AsyncSignal:
    """Single-value latch usable from both a thread and the event loop.

    Used to wake the orchestrator on Ctrl-C from the PortAudio callback thread
    or a signal handler.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._payload: Any = None

    def set_threadsafe(self, loop: asyncio.AbstractEventLoop, payload: Any = None) -> None:
        def _set() -> None:
            self._payload = payload
            self._event.set()

        loop.call_soon_threadsafe(_set)

    def set(self, payload: Any = None) -> None:
        self._payload = payload
        self._event.set()

    async def wait(self) -> Any:
        await self._event.wait()
        return self._payload

    def is_set(self) -> bool:
        return self._event.is_set()
