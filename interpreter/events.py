# little event system so the pipeline can tell the screen whats happening
# without importing the screen code. the pipeline just emits stuff here and
# the cli listens. means I could add a gui later without touching any of the
# audio code

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class Stage(str, Enum):
    # where a sentence is in the chain
    CAPTURED = "captured"        # vad decided a sentence ended
    TRANSCRIBED = "transcribed"  # got the text back
    TRANSLATED = "translated"    # got the translation back
    SPEAKING = "speaking"        # first audio hit the speaker
    DONE = "done"                # finished playing
    DROPPED = "dropped"          # thrown away (echo, too short, silence)
    ERROR = "error"


@dataclass
class PipelineEvent:
    seq: int
    channel_id: str
    stage: Stage

    source_lang: str = None
    target_lang: str = None
    source_text: str = None
    target_text: str = None

    # how long each stage took in ms, fills up as the sentence moves along
    timings: dict = field(default_factory=dict)

    # seconds of audio, useful for checking if the vad is set right
    audio_seconds: float = None

    detail: str = None  # why it was dropped, or the error message

    @property
    def total_ms(self):
        return sum(self.timings.values())


class EventBus:
    # subscribers get called right away on the event loop so they need to
    # be quick and they must not raise. a broken display should never take
    # down the actual conversation

    def __init__(self):
        self._subscribers = []
        self._history = []
        self._history_limit = 500

    def subscribe(self, subscriber):
        self._subscribers.append(subscriber)

        # give back a function that undoes it
        def unsubscribe():
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

        return unsubscribe

    def emit(self, event):
        self._history.append(event)
        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]

        # copy the list first in case somebody unsubscribes while we loop
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception:
                log.exception("event subscriber failed")

    @property
    def history(self):
        return list(self._history)


class AsyncSignal:
    # a flag you can set from a normal thread or from async code.
    # I use it to wake things up on ctrl-c

    def __init__(self):
        self._event = asyncio.Event()
        self._payload = None

    def set_threadsafe(self, loop, payload=None):
        def _set():
            self._payload = payload
            self._event.set()

        loop.call_soon_threadsafe(_set)

    def set(self, payload=None):
        self._payload = payload
        self._event.set()

    async def wait(self):
        await self._event.wait()
        return self._payload

    def is_set(self):
        return self._event.is_set()
