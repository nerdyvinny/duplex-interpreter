"""Deciding which way an utterance should be translated.

In dual_mic mode this is trivial: each microphone belongs to one person, so
its language is pinned and there is nothing to guess.

In single_mic mode both people share a microphone, so the app has to work out
who just spoke. Three signals, in order of trustworthiness:

1. **Text language ID** (`langid.identify`) — deciding between two *known*
   candidates from a transcript is much easier than open-set detection, and
   it is instant and offline.
2. **Audio language ID** from the STT engine — only faster-whisper reports a
   usable confidence; the OpenAI transcribe models don't expose one at all.
3. **Alternation** — in a two-person conversation, the next speaker is
   usually the other person. A weak signal, but a good tiebreak, and it is
   what a human does when they can't quite hear who spoke.

When the signals conflict, text wins: short utterances are exactly where
audio LID is worst.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import langid
from .providers.base import Transcript

log = logging.getLogger(__name__)

# Below this, a lone signal isn't trusted on its own.
_TEXT_TRUST = 0.45
_AUDIO_TRUST = 0.70


@dataclass
class RoutingDecision:
    source: str
    target: str
    confidence: float
    reason: str
    # When set, the caller should drop the utterance instead of translating
    # it. Reserved for cases where the evidence says this is not either
    # configured language at all, rather than merely being ambiguous.
    reject: bool = False


class LanguageRouter:
    """Per-channel router. Holds the alternation state for single_mic mode."""

    def __init__(
        self,
        candidates: tuple[str, str],
        *,
        pinned: str | None = None,
        channel_id: str = "A",
    ) -> None:
        if len(candidates) != 2 or candidates[0] == candidates[1]:
            raise ValueError(f"need two distinct candidate languages, got {candidates}")
        self.candidates = candidates
        self.pinned = pinned
        self.channel_id = channel_id
        self._last_source: str | None = None

    def other(self, code: str) -> str:
        first, second = self.candidates
        return second if code == first else first

    def route(self, transcript: Transcript) -> RoutingDecision:
        if self.pinned:
            return RoutingDecision(
                source=self.pinned,
                target=self.other(self.pinned),
                confidence=1.0,
                reason="pinned to this microphone",
            )

        text_guess = langid.identify(transcript.text, self.candidates)
        audio_lang = transcript.language if transcript.language in self.candidates else None
        audio_confidence = transcript.language_confidence

        rejection = self._rejection_reason(transcript, text_guess, audio_lang)
        if rejection is not None:
            # Leave `_last_source` alone: a hallucination is not a turn, and
            # letting it flip the alternation state would misroute the next
            # real utterance too.
            return RoutingDecision(
                source=self.candidates[0],
                target=self.candidates[1],
                confidence=0.0,
                reason=rejection,
                reject=True,
            )

        source, confidence, reason = self._combine(text_guess, audio_lang, audio_confidence)

        self._last_source = source
        return RoutingDecision(
            source=source,
            target=self.other(source),
            confidence=confidence,
            reason=reason,
        )

    def _rejection_reason(
        self,
        transcript: Transcript,
        text_guess: langid.LangGuess,
        audio_lang: str | None,
    ) -> str | None:
        """Should this be dropped rather than routed?

        Only for evidence that the utterance is not in either configured
        language — not for mere ambiguity, which alternation handles. In live
        testing the recognizer turned room noise into Cyrillic, and routing
        it by alternation meant "translating" noise into itself.
        """
        if text_guess.foreign:
            return f"not {self.candidates[0]} or {self.candidates[1]}: {text_guess.reason}"

        # The recognizer named a language outside the pair and the text gives
        # us nothing to overrule it with. Two weak signals both pointing away
        # from the conversation beat a coin flip.
        reported = transcript.language
        if (
            reported
            and audio_lang is None
            and text_guess.language is None
            and reported not in self.candidates
        ):
            return f"recognized as {reported!r}, which is neither configured language"

        return None

    def _combine(
        self,
        text_guess: langid.LangGuess,
        audio_lang: str | None,
        audio_confidence: float | None,
    ) -> tuple[str, float, str]:
        text_lang = text_guess.language
        text_confidence = text_guess.confidence

        # Both signals agree — the easy and most common case.
        if text_lang and audio_lang and text_lang == audio_lang:
            return text_lang, min(0.99, max(text_confidence, audio_confidence or 0.8) + 0.15), (
                f"text and audio agree on {text_lang}"
            )

        # They disagree. Text is the better signal on short conversational turns.
        if text_lang and audio_lang and text_lang != audio_lang:
            if text_confidence >= _TEXT_TRUST:
                return text_lang, text_confidence, (
                    f"text says {text_lang}, audio said {audio_lang}; trusting text"
                )
            if audio_confidence is not None and audio_confidence >= _AUDIO_TRUST:
                return audio_lang, audio_confidence, (
                    f"audio says {audio_lang} at {audio_confidence:.2f}; text unsure"
                )
            return self._alternate("both signals weak and in conflict")

        if text_lang and text_confidence >= _TEXT_TRUST:
            return text_lang, text_confidence, f"text: {text_guess.reason}"

        if audio_lang and (audio_confidence is None or audio_confidence >= _AUDIO_TRUST):
            return audio_lang, audio_confidence or 0.75, f"audio detected {audio_lang}"

        if text_lang:
            # Weak, but a weak real signal beats pure alternation.
            return text_lang, text_confidence, f"weak text signal: {text_guess.reason}"

        return self._alternate("no usable language signal")

    def _alternate(self, why: str) -> tuple[str, float, str]:
        if self._last_source is None:
            # Nothing to go on at all: assume language A opened the conversation.
            return self.candidates[0], 0.3, f"{why}; defaulting to {self.candidates[0]}"
        guessed = self.other(self._last_source)
        return guessed, 0.4, f"{why}; alternating to {guessed}"

    def note_source(self, code: str) -> None:
        """Override the alternation state, e.g. after a manual correction."""
        if code in self.candidates:
            self._last_source = code
