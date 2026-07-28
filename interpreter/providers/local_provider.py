"""Fully local providers: faster-whisper, Argos Translate, Piper.

No API keys, no network after the first model download, nothing leaving the
machine. Slower than the cloud path (roughly 1-2 s end to end) and the models
are large, but it runs on a plane.

Each of these is CPU/GPU bound rather than I/O bound, so every call is pushed
to a worker thread — otherwise transcribing one person's sentence would stall
the other person's audio capture.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from ..config import PIPELINE_SAMPLE_RATE, ConfigError, ProvidersConfig
from .base import (
    ProviderError,
    SpeechAudio,
    STTProvider,
    Transcript,
    TranslationProvider,
    TTSProvider,
)

log = logging.getLogger(__name__)


class FasterWhisperSTT(STTProvider):
    """Whisper via CTranslate2. Loads lazily so startup stays fast."""

    name = "faster-whisper"

    def __init__(self, providers: ProvidersConfig) -> None:
        self.size = providers.local_whisper_size
        self.device = providers.local_whisper_device
        self.compute = providers.local_whisper_compute
        self._model = None
        self._load_lock = asyncio.Lock()

    def _resolve_placement(self) -> tuple[str, str]:
        device, compute = self.device, self.compute
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return device, compute

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        except Exception:  # noqa: BLE001 - no CUDA build, no driver, ...
            return False

    @staticmethod
    def _warmup(model) -> None:
        """Force one real inference.

        Constructing a CUDA model succeeds even when the CUDA math libraries
        are missing — the failure surfaces on the first transcription instead,
        which would otherwise be the first thing somebody says. Doing it here
        turns a mid-conversation crash into a startup-time fallback, and pays
        the first-inference cost before anyone is waiting on it.
        """
        segments, _ = model.transcribe(
            np.zeros(PIPELINE_SAMPLE_RATE, dtype=np.float32),
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        list(segments)  # transcribe() is lazy; the work happens on iteration

    def _load(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ConfigError(
                "providers.stt is 'faster-whisper' but the package is missing. "
                "Run `pip install -r requirements-local.txt`."
            ) from exc

        device, compute = self._resolve_placement()
        log.info("loading faster-whisper %s on %s (%s)", self.size, device, compute)
        try:
            model = WhisperModel(self.size, device=device, compute_type=compute)
            self._warmup(model)
            return model
        except Exception as exc:  # noqa: BLE001
            if device == "cuda" and self.device == "auto":
                # Very new consumer GPUs routinely need a CUDA-matched
                # ctranslate2 build, and a missing cuBLAS only shows up here.
                log.warning(
                    "CUDA is present but unusable (%s); falling back to CPU int8. "
                    "Set local_whisper_device: cpu to silence this.",
                    exc,
                )
                model = WhisperModel(self.size, device="cpu", compute_type="int8")
                self._warmup(model)
                return model
            raise ProviderError(
                f"could not load faster-whisper on {device}: {exc}"
            ) from exc

    async def _ensure_model(self):
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load)
        return self._model

    async def transcribe(
        self,
        pcm: np.ndarray,
        *,
        language: str | None = None,
        candidates: tuple[str, ...] = (),
    ) -> Transcript:
        model = await self._ensure_model()
        audio = pcm.astype(np.float32) / 32768.0

        def _call() -> Transcript:
            try:
                segments, info = model.transcribe(
                    audio,
                    language=language,
                    beam_size=1,  # greedy: this is a latency-critical path
                    vad_filter=False,  # our own VAD already trimmed the segment
                    condition_on_previous_text=False,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"faster-whisper transcription failed: {exc}") from exc
            return Transcript(
                text=text,
                language=language or getattr(info, "language", None),
                language_confidence=getattr(info, "language_probability", None),
            )

        return await asyncio.to_thread(_call)


class ArgosTranslation(TranslationProvider):
    """Argos Translate. CPU only, ~100 MB per language pair."""

    name = "argos"

    def __init__(self, model: str | None = None) -> None:
        del model
        try:
            import argostranslate.translate  # noqa: F401
        except ImportError as exc:
            raise ConfigError(
                "providers.translation is 'argos' but the package is missing. "
                "Run `pip install -r requirements-local.txt`."
            ) from exc
        self._installed: set[tuple[str, str]] = set()
        self._install_lock = asyncio.Lock()

    def _ensure_pair(self, source: str, target: str) -> None:
        import argostranslate.package
        import argostranslate.translate

        available = argostranslate.translate.get_installed_languages()
        codes = {lang.code for lang in available}
        if source in codes and target in codes:
            from_lang = next(l for l in available if l.code == source)
            if from_lang.get_translation(next(l for l in available if l.code == target)):
                return

        log.info("installing Argos model %s->%s (first run only)", source, target)
        argostranslate.package.update_package_index()
        packages = argostranslate.package.get_available_packages()
        match = next(
            (p for p in packages if p.from_code == source and p.to_code == target), None
        )
        if match is None:
            raise ProviderError(
                f"Argos has no direct {source}->{target} model. "
                "Use providers.translation: openai for this pair."
            )
        argostranslate.package.install_from_path(match.download())

    async def translate(
        self,
        text: str,
        *,
        source: str,
        target: str,
        context: list[str] | None = None,
    ) -> str:
        del context  # Argos is stateless, sentence-at-a-time
        if not text.strip():
            return ""

        if (source, target) not in self._installed:
            async with self._install_lock:
                if (source, target) not in self._installed:
                    await asyncio.to_thread(self._ensure_pair, source, target)
                    self._installed.add((source, target))

        def _call() -> str:
            import argostranslate.translate

            try:
                return argostranslate.translate.translate(text, source, target).strip()
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"Argos translation failed: {exc}") from exc

        return await asyncio.to_thread(_call)


class PiperTTS(TTSProvider):
    """Piper ONNX voices. Fast (real-time factor ~0.05), ~60 MB per voice.

    Voices are per-language, so point at them with either:
      PIPER_VOICE_ES=/path/to/es_ES-...onnx
    or by putting the .onnx files in ./models/piper/ named `<lang>.onnx`.
    Download from https://huggingface.co/rhasspy/piper-voices
    """

    name = "piper"

    def __init__(self, model: str | None = None) -> None:
        del model
        try:
            import piper  # noqa: F401
        except ImportError as exc:
            raise ConfigError(
                "providers.tts is 'piper' but the package is missing. "
                "Run `pip install -r requirements-local.txt`."
            ) from exc
        self._voices: dict[str, object] = {}
        self._lock = asyncio.Lock()

    def _voice_path(self, language: str) -> Path:
        override = os.environ.get(f"PIPER_VOICE_{language.upper()}")
        if override:
            path = Path(override)
            if not path.exists():
                raise ProviderError(f"PIPER_VOICE_{language.upper()} points at a missing file: {path}")
            return path

        local = Path("models") / "piper" / f"{language}.onnx"
        if local.exists():
            return local

        raise ProviderError(
            f"no Piper voice for {language!r}. Download one from "
            "https://huggingface.co/rhasspy/piper-voices and save it as "
            f"models/piper/{language}.onnx, or set PIPER_VOICE_{language.upper()}."
        )

    def _load_voice(self, language: str):
        from piper import PiperVoice

        return PiperVoice.load(str(self._voice_path(language)))

    async def _ensure_voice(self, language: str):
        if language not in self._voices:
            async with self._lock:
                if language not in self._voices:
                    self._voices[language] = await asyncio.to_thread(
                        self._load_voice, language
                    )
        return self._voices[language]

    async def synthesize(self, text: str, *, language: str, voice: str) -> SpeechAudio:
        del voice  # the voice IS the model file for Piper
        loaded = await self._ensure_voice(language)
        rate = getattr(getattr(loaded, "config", None), "sample_rate", 22_050)
        return SpeechAudio(chunks=self._stream(loaded, text), sample_rate=int(rate))

    async def _stream(self, loaded, text: str) -> AsyncIterator[bytes]:
        def _synthesize() -> list[bytes]:
            try:
                # piper-tts changed its API between 1.2 and 1.3; support both.
                if hasattr(loaded, "synthesize_stream_raw"):
                    return list(loaded.synthesize_stream_raw(text))
                return [
                    chunk.audio_int16_bytes if hasattr(chunk, "audio_int16_bytes") else bytes(chunk)
                    for chunk in loaded.synthesize(text)
                ]
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"Piper synthesis failed: {exc}") from exc

        for chunk in await asyncio.to_thread(_synthesize):
            if chunk:
                yield chunk


class PassthroughTranslation(TranslationProvider):
    """No-op translation, for testing audio plumbing without an MT bill."""

    name = "passthrough"

    async def translate(
        self, text: str, *, source: str, target: str, context: list[str] | None = None
    ) -> str:
        del source, target, context
        return text


class SilentTTS(TTSProvider):
    """Emits silence proportional to the text length.

    Lets you exercise capture -> STT -> translate -> playback timing without
    paying for or waiting on synthesis.
    """

    name = "silent"

    async def synthesize(self, text: str, *, language: str, voice: str) -> SpeechAudio:
        del language, voice
        samples = int(PIPELINE_SAMPLE_RATE * min(10.0, 0.06 * max(1, len(text))))

        async def _chunks() -> AsyncIterator[bytes]:
            yield np.zeros(samples, dtype=np.int16).tobytes()

        return SpeechAudio(chunks=_chunks(), sample_rate=PIPELINE_SAMPLE_RATE)
