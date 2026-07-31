# the fully local ones: faster-whisper, argos translate, piper
#
# no api keys, no internet after the first download, nothing leaves your
# computer. slower than the cloud (more like 1-2 seconds) and the models
# are big, but it runs on a plane.
#
# all of these are cpu/gpu work rather than network waiting, so every call
# goes on a worker thread. if I didn't do that, transcribing one person's
# sentence would freeze the other person's microphone

import asyncio
import contextlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np

from ..config import PIPELINE_SAMPLE_RATE, ConfigError
from .base import (
    ProviderError,
    SpeechAudio,
    STTProvider,
    Transcript,
    TranslationProvider,
    TTSProvider,
)

log = logging.getLogger(__name__)

_cuda_libraries_registered = False


def _register_cuda_libraries():
    # windows gpu fix. this one cost me an entire evening.
    #
    # nvidia-cublas-cu12 and nvidia-cudnn-cu12 put their dlls inside
    # site-packages, but ctranslate2 asks for them by plain name, and since
    # python 3.8 windows stopped searching PATH for a module's dependencies.
    # so you get "Library cublas64_12.dll is not found" on a machine that
    # has a perfectly good gpu and all the right packages installed.
    #
    # I register the folders both ways because different loaders look in
    # different places. on my laptop this is 1.75s vs 0.15s per sentence

    global _cuda_libraries_registered
    if _cuda_libraries_registered or sys.platform != "win32":
        return
    _cuda_libraries_registered = True

    try:
        spec = importlib.util.find_spec("nvidia")
        if spec:
            roots = list(spec.submodule_search_locations)
        else:
            roots = []
    except (ImportError, ValueError, AttributeError):
        roots = []
    if not roots:
        return

    directories = [
        str(path)
        for root in roots
        for path in sorted(Path(root).glob("*/bin"))
        if path.is_dir()
    ]
    if not directories:
        return

    existing = os.environ.get("PATH", "")
    additions = [d for d in directories if d not in existing]
    if additions:
        os.environ["PATH"] = os.pathsep.join(additions) + os.pathsep + existing

    for directory in directories:
        with contextlib.suppress(OSError, AttributeError):
            os.add_dll_directory(directory)

    log.debug("registered %d NVIDIA DLL directories", len(directories))


class FasterWhisperSTT(STTProvider):
    name = "faster-whisper"

    def __init__(self, providers):
        self.size = providers.local_whisper_size
        self.device = providers.local_whisper_device
        self.compute = providers.local_whisper_compute
        self._model = None
        self._load_lock = asyncio.Lock()

    def _resolve_placement(self):
        device, compute = self.device, self.compute
        if device == "auto":
            if self._cuda_available():
                device = "cuda"
            else:
                device = "cpu"
        if compute == "auto":
            if device == "cuda":
                compute = "float16"
            else:
                compute = "int8"
        return device, compute

    @staticmethod
    def _cuda_available():
        _register_cuda_libraries()
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False  # no cuda build, no driver, whatever

    @staticmethod
    def _warmup(model):
        # run one real (empty) transcription right now.
        #
        # building a cuda model SUCCEEDS even when the cuda math libraries
        # are missing, it only blows up on the first actual transcription.
        # which would be the first thing somebody says. doing it here turns
        # a crash mid conversation into a fallback at startup, and it also
        # pays the slow first-run cost before anybody is waiting
        segments, _ = model.transcribe(
            np.zeros(PIPELINE_SAMPLE_RATE, dtype=np.float32),
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        list(segments)  # transcribe() is lazy, iterating is what runs it

    def _load(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ConfigError(
                "providers.stt is 'faster-whisper' but the package is missing. "
                "Run `pip install -r requirements-local.txt`."
            ) from exc

        device, compute = self._resolve_placement()
        if device == "cuda":
            _register_cuda_libraries()

        log.info("loading faster-whisper %s on %s (%s)", self.size, device, compute)
        try:
            model = WhisperModel(self.size, device=device, compute_type=compute)
            self._warmup(model)
            return model
        except Exception as exc:
            if device == "cuda" and self.device == "auto":
                # newer gpus often need a matching ctranslate2 build and a
                # missing cublas only shows up right here
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

    def preflight(self, cfg):
        # load it now. whisper takes a second or two to load and warm up,
        # and if I leave that lazy it lands on the first thing anybody
        # says, which is exactly when people judge whether it works
        if self._model is None:
            self._model = self._load()

    async def _ensure_model(self):
        # double check with the lock so two sentences arriving at once
        # don't both try to load the model
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load)
        return self._model

    async def transcribe(self, pcm, *, language=None, candidates=()):
        model = await self._ensure_model()
        audio = pcm.astype(np.float32) / 32768.0  # whisper wants floats

        def _call():
            try:
                segments, info = model.transcribe(
                    audio,
                    language=language,
                    beam_size=1,      # greedy, this needs to be fast
                    vad_filter=False,  # my own vad already trimmed it
                    condition_on_previous_text=False,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as exc:
                raise ProviderError(
                    f"faster-whisper transcription failed: {exc}"
                ) from exc

            return Transcript(
                text=text,
                language=language or getattr(info, "language", None),
                language_confidence=getattr(info, "language_probability", None),
            )

        return await asyncio.to_thread(_call)


class ArgosTranslation(TranslationProvider):
    # cpu only, about 100mb per language pair
    name = "argos"

    def __init__(self, model=None):
        try:
            import argostranslate.translate
        except ImportError as exc:
            raise ConfigError(
                "providers.translation is 'argos' but the package is missing. "
                "Run `pip install -r requirements-local.txt`."
            ) from exc
        self._installed = set()
        self._install_lock = asyncio.Lock()

    def preflight(self, cfg):
        # install BOTH directions now. argos downloads ~100mb per direction
        # the first time, and lazily that download happens in the middle of
        # somebody's first sentence
        import argostranslate.translate

        first, second = cfg.language_codes
        for source, target in ((first, second), (second, first)):
            try:
                self._ensure_pair(source, target)
                # force the one time setup now, one at a time. if I leave
                # it to the first real translation then two sentences at
                # once race to rename the same temp folder and one of them
                # dies with a permission error. that was a fun one
                argostranslate.translate.translate("Hello.", source, target)
            except ProviderError as exc:
                raise ConfigError(str(exc)) from exc
            except Exception as exc:
                raise ConfigError(
                    f"Argos could not prepare {source}->{target}: {exc}"
                ) from exc
            self._installed.add((source, target))

    def _ensure_pair(self, source, target):
        import argostranslate.package
        import argostranslate.translate

        available = argostranslate.translate.get_installed_languages()
        codes = {lang.code for lang in available}
        if source in codes and target in codes:
            from_lang = next(l for l in available if l.code == source)
            to_lang = next(l for l in available if l.code == target)
            if from_lang.get_translation(to_lang):
                return  # already got it

        log.info("installing Argos model %s->%s (first run only)", source, target)
        argostranslate.package.update_package_index()
        packages = argostranslate.package.get_available_packages()

        match = next(
            (p for p in packages if p.from_code == source and p.to_code == target),
            None,
        )
        if match is None:
            raise ProviderError(
                f"Argos has no direct {source}->{target} model. "
                "Use providers.translation: openai for this pair."
            )
        argostranslate.package.install_from_path(match.download())

    async def translate(self, text, *, source, target, context=None):
        # argos does one sentence at a time and has no memory, so context
        # is useless to it
        if not text.strip():
            return ""

        if (source, target) not in self._installed:
            async with self._install_lock:
                if (source, target) not in self._installed:
                    await asyncio.to_thread(self._ensure_pair, source, target)
                    self._installed.add((source, target))

        def _call():
            import argostranslate.translate
            try:
                return argostranslate.translate.translate(
                    text, source, target
                ).strip()
            except Exception as exc:
                raise ProviderError(f"Argos translation failed: {exc}") from exc

        return await asyncio.to_thread(_call)


class PiperTTS(TTSProvider):
    # piper onnx voices. fast and about 60mb each.
    #
    # the voices are per language so point at them with either
    #   PIPER_VOICE_ES=/path/to/es_ES-whatever.onnx
    # or just drop the .onnx files in models/piper/ named <lang>.onnx
    # get them from https://huggingface.co/rhasspy/piper-voices
    #
    # NOTE each voice is TWO files, the .onnx and a .onnx.json next to it.
    # I only downloaded the first one at first and spent ages confused

    name = "piper"

    def __init__(self, model=None):
        try:
            import piper
        except ImportError as exc:
            raise ConfigError(
                "providers.tts is 'piper' but the package is missing. "
                "Run `pip install -r requirements-local.txt`."
            ) from exc
        self._voices = {}
        self._lock = asyncio.Lock()

    def preflight(self, cfg):
        # load both voices now. checking the files exist isn't enough,
        # the first synthesis in each language takes about 1.3s to load
        # the model and the second language's cost lands mid conversation
        # when the other person first answers
        for code in cfg.language_codes:
            try:
                self._voices[code] = self._load_voice(code)
            except ProviderError as exc:
                raise ConfigError(str(exc)) from exc
            except Exception as exc:
                raise ConfigError(
                    f"Piper could not load the {code} voice: {exc}"
                ) from exc

    def _voice_path(self, language):
        override = os.environ.get(f"PIPER_VOICE_{language.upper()}")
        if override:
            path = Path(override)
            if not path.exists():
                raise ProviderError(
                    f"PIPER_VOICE_{language.upper()} points at a missing file: {path}"
                )
            return path

        local = Path("models") / "piper" / f"{language}.onnx"
        if local.exists():
            return local

        raise ProviderError(
            f"no Piper voice for {language!r}. Download one from "
            "https://huggingface.co/rhasspy/piper-voices and save it as "
            f"models/piper/{language}.onnx, or set PIPER_VOICE_{language.upper()}."
        )

    def _load_voice(self, language):
        from piper import PiperVoice
        return PiperVoice.load(str(self._voice_path(language)))

    async def _ensure_voice(self, language):
        if language not in self._voices:
            async with self._lock:
                if language not in self._voices:
                    self._voices[language] = await asyncio.to_thread(
                        self._load_voice, language
                    )
        return self._voices[language]

    async def synthesize(self, text, *, language, voice):
        # piper ignores the voice name, the voice IS the model file
        loaded = await self._ensure_voice(language)
        rate = getattr(getattr(loaded, "config", None), "sample_rate", 22_050)
        return SpeechAudio(chunks=self._stream(loaded, text), sample_rate=int(rate))

    async def _stream(self, loaded, text):
        def _synthesize():
            try:
                # piper changed its api between 1.2 and 1.3 so I check for
                # both. the old one has synthesize_stream_raw
                if hasattr(loaded, "synthesize_stream_raw"):
                    return list(loaded.synthesize_stream_raw(text))
                return [
                    chunk.audio_int16_bytes
                    if hasattr(chunk, "audio_int16_bytes")
                    else bytes(chunk)
                    for chunk in loaded.synthesize(text)
                ]
            except Exception as exc:
                raise ProviderError(f"Piper synthesis failed: {exc}") from exc

        for chunk in await asyncio.to_thread(_synthesize):
            if chunk:
                yield chunk


class PassthroughTranslation(TranslationProvider):
    # doesn't translate at all. for testing the audio plumbing without
    # paying for translations
    name = "passthrough"

    async def translate(self, text, *, source, target, context=None):
        return text


class SilentTTS(TTSProvider):
    # makes silence, roughly as long as the text would take to say.
    # lets me test the capture -> recognize -> translate -> play timing
    # without waiting for or paying for real speech
    name = "silent"

    async def synthesize(self, text, *, language, voice):
        seconds = min(10.0, 0.06 * max(1, len(text)))
        samples = int(PIPELINE_SAMPLE_RATE * seconds)

        async def _chunks():
            yield np.zeros(samples, dtype=np.int16).tobytes()

        return SpeechAudio(chunks=_chunks(), sample_rate=PIPELINE_SAMPLE_RATE)
