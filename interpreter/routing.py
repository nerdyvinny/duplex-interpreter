# figures out which way to translate a sentence
#
# in dual_mic mode this is easy, each headset belongs to one person so I
# already know what language they speak and there is nothing to guess.
#
# single_mic is the hard one. both people use the same microphone so I have
# to work out who just talked. I use 3 things:
#   1. look at the transcript text (langid.py). picking between 2 languages
#      you already know is way easier than guessing from scratch, and its
#      instant and offline
#   2. what language whisper thinks the audio was. only the local whisper
#      gives a confidence number, the openai one doesn't tell you at all
#   3. just assume its the other person's turn now. weak but its what a
#      human does when they didn't catch who spoke
#
# if 1 and 2 disagree I go with the text. short sentences like "ok" or "si"
# are exactly where the audio detection is worst

import logging
from dataclasses import dataclass

from . import langid

log = logging.getLogger(__name__)

# below these numbers I don't trust a signal on its own
_TEXT_TRUST = 0.45
_AUDIO_TRUST = 0.70


@dataclass
class RoutingDecision:
    source: str
    target: str
    confidence: float
    reason: str
    # if this is True just throw the sentence away instead of translating.
    # only for when its clearly not either language, not for "im not sure"
    reject: bool = False


class LanguageRouter:
    # one per channel. holds the whose-turn-is-it state for single_mic

    def __init__(self, candidates, *, pinned=None, channel_id="A"):
        if len(candidates) != 2 or candidates[0] == candidates[1]:
            raise ValueError(
                f"need two distinct candidate languages, got {candidates}"
            )
        self.candidates = candidates
        self.pinned = pinned
        self.channel_id = channel_id
        self._last_source = None

    def other(self, code):
        # given one language give me the other one
        first, second = self.candidates
        if code == first:
            return second
        return first

    def route(self, transcript):
        if self.pinned:
            # dual_mic, we already know
            return RoutingDecision(
                source=self.pinned,
                target=self.other(self.pinned),
                confidence=1.0,
                reason="pinned to this microphone",
            )

        text_guess = langid.identify(transcript.text, self.candidates)
        if transcript.language in self.candidates:
            audio_lang = transcript.language
        else:
            audio_lang = None
        audio_confidence = transcript.language_confidence

        rejection = self._rejection_reason(transcript, text_guess, audio_lang)
        if rejection is not None:
            # leave _last_source alone here. garbage isn't somebody's turn,
            # and if I let it flip the alternation the NEXT real sentence
            # goes the wrong way too
            return RoutingDecision(
                source=self.candidates[0],
                target=self.candidates[1],
                confidence=0.0,
                reason=rejection,
                reject=True,
            )

        source, confidence, reason = self._combine(
            text_guess, audio_lang, audio_confidence
        )

        self._last_source = source
        return RoutingDecision(
            source=source,
            target=self.other(source),
            confidence=confidence,
            reason=reason,
        )

    def _rejection_reason(self, transcript, text_guess, audio_lang):
        # should we just drop this?
        # only if it looks like neither language. being unsure is fine,
        # thats what the alternation guess is for.
        # I added this because whisper kept turning the fan noise in my room
        # into russian, and then it would "translate" noise into more noise
        if text_guess.foreign:
            return (
                f"not {self.candidates[0]} or {self.candidates[1]}: "
                f"{text_guess.reason}"
            )

        # whisper named some third language and the text doesn't argue.
        # two weak signals both saying "not this conversation" beats a
        # coin flip
        reported = transcript.language
        if (
            reported
            and audio_lang is None
            and text_guess.language is None
            and reported not in self.candidates
        ):
            return f"recognized as {reported!r}, which is neither configured language"

        return None

    def _combine(self, text_guess, audio_lang, audio_confidence):
        text_lang = text_guess.language
        text_confidence = text_guess.confidence

        # both agree, easiest case and also the most common one
        if text_lang and audio_lang and text_lang == audio_lang:
            best = max(text_confidence, audio_confidence or 0.8)
            confidence = min(0.99, best + 0.15)
            return text_lang, confidence, f"text and audio agree on {text_lang}"

        # they disagree
        if text_lang and audio_lang and text_lang != audio_lang:
            if text_confidence >= _TEXT_TRUST:
                return (
                    text_lang,
                    text_confidence,
                    f"text says {text_lang}, audio said {audio_lang}; trusting text",
                )
            if audio_confidence is not None and audio_confidence >= _AUDIO_TRUST:
                return (
                    audio_lang,
                    audio_confidence,
                    f"audio says {audio_lang} at {audio_confidence:.2f}; text unsure",
                )
            return self._alternate("both signals weak and in conflict")

        # only one of them said anything useful
        if text_lang and text_confidence >= _TEXT_TRUST:
            return text_lang, text_confidence, f"text: {text_guess.reason}"

        if audio_lang and (audio_confidence is None or audio_confidence >= _AUDIO_TRUST):
            return audio_lang, audio_confidence or 0.75, f"audio detected {audio_lang}"

        if text_lang:
            # weak, but a weak real answer is still better than guessing
            return text_lang, text_confidence, f"weak text signal: {text_guess.reason}"

        return self._alternate("no usable language signal")

    def _alternate(self, why):
        # nothing to go on, so assume people take turns
        if self._last_source is None:
            return (
                self.candidates[0],
                0.3,
                f"{why}; defaulting to {self.candidates[0]}",
            )
        guessed = self.other(self._last_source)
        return guessed, 0.4, f"{why}; alternating to {guessed}"

    def note_source(self, code):
        # lets you fix the turn state by hand if it gets stuck
        if code in self.candidates:
            self._last_source = code
