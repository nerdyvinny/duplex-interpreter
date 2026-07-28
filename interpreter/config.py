"""Configuration: dataclasses, YAML loading, .env keys, and validation.

Validation errors are raised as `ConfigError` with a message aimed at the
person editing config.yaml, not at a stack trace reader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from . import languages

# Everything upstream of TTS runs at 16 kHz mono int16: it is what Silero VAD
# and every Whisper variant expect, and it keeps upload sizes small.
PIPELINE_SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = PIPELINE_SAMPLE_RATE * FRAME_MS // 1000  # 320


class ConfigError(ValueError):
    """Raised for a config a human needs to fix."""


class Mode(str, Enum):
    SINGLE_MIC = "single_mic"
    DUAL_MIC = "dual_mic"


@dataclass
class LanguageSlot:
    code: str
    name: str
    voice: str

    @classmethod
    def parse(cls, raw: Any, slot: str, taken_voices: set[str]) -> LanguageSlot:
        if isinstance(raw, str):
            raw = {"code": raw}
        if not isinstance(raw, dict):
            raise ConfigError(
                f"languages.{slot} must be a language code or a mapping, got {type(raw).__name__}"
            )
        code = raw.get("code")
        if not code:
            raise ConfigError(f"languages.{slot}.code is required (e.g. 'en')")
        code = str(code).strip().lower().split("-")[0]
        if not languages.is_known(code):
            raise ConfigError(
                f"languages.{slot}.code={code!r} is not in the language table. "
                f"Add it to interpreter/languages.py or pick a known code."
            )
        voice = str(raw.get("voice") or languages.default_voice(code, taken_voices))
        return cls(code=code, name=str(raw.get("name") or languages.name_of(code)), voice=voice)


@dataclass
class ChannelConfig:
    """One microphone plus the speaker its *translations* come out of.

    In dual_mic mode the two channels are cross-wired by the orchestrator:
    what channel A hears is translated and played on channel B's output.
    """

    id: str
    input_device: str | int | None = None
    output_device: str | int | None = None
    language: str = "auto"  # "auto" = detect per utterance
    input_gain: float = 1.0

    @classmethod
    def parse(cls, raw: Any, index: int) -> ChannelConfig:
        if not isinstance(raw, dict):
            raise ConfigError(f"channels[{index}] must be a mapping")
        channel_id = str(raw.get("id") or chr(ord("A") + index))
        language = str(raw.get("language", "auto")).strip().lower()
        if language != "auto":
            language = language.split("-")[0]
            if not languages.is_known(language):
                raise ConfigError(
                    f"channels[{index}].language={language!r} is unknown; use 'auto' or a known code"
                )
        gain = float(raw.get("input_gain", 1.0))
        if not 0.1 <= gain <= 10.0:
            raise ConfigError(f"channels[{index}].input_gain must be between 0.1 and 10.0")
        return cls(
            id=channel_id,
            input_device=raw.get("input_device"),
            output_device=raw.get("output_device"),
            language=language,
            input_gain=gain,
        )


@dataclass
class ProvidersConfig:
    stt: str = "openai"
    translation: str = "openai"
    tts: str = "openai"

    stt_model: str = "gpt-4o-mini-transcribe"
    translation_model: str = "gpt-4o-mini"
    tts_model: str = "gpt-4o-mini-tts"

    # Local-path knobs, ignored by the cloud providers.
    local_whisper_size: str = "small"
    local_whisper_device: str = "auto"  # auto | cuda | cpu
    local_whisper_compute: str = "auto"  # auto | float16 | int8

    # How many previous turns to hand the translator for pronoun/gender context.
    context_turns: int = 3

    @classmethod
    def parse(cls, raw: Any) -> ProvidersConfig:
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ConfigError("providers must be a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(f"unknown providers keys: {', '.join(sorted(unknown))}")
        return cls(**{k: raw[k] for k in raw})


@dataclass
class VadConfig:
    backend: str = "silero"  # silero | webrtc
    # The single biggest latency lever. Lower = snappier but clips people who
    # pause mid-sentence. See README "Tuning latency".
    silence_ms_to_end: int = 600
    preroll_ms: int = 300  # audio kept from *before* the trigger, so no clipped onsets
    min_utterance_ms: int = 250  # drop coughs and clicks
    max_utterance_ms: int = 15_000  # force-flush monologues
    start_frames: int = 3  # consecutive voiced frames needed to open a segment
    threshold: float = 0.5  # silero speech probability
    aggressiveness: int = 2  # webrtc only, 0-3

    @classmethod
    def parse(cls, raw: Any) -> VadConfig:
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ConfigError("vad must be a mapping")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown vad keys: {', '.join(sorted(unknown))}")
        cfg = cls(**{k: raw[k] for k in raw})
        if cfg.backend not in {"silero", "webrtc"}:
            raise ConfigError(f"vad.backend must be 'silero' or 'webrtc', got {cfg.backend!r}")
        if not 0.0 < cfg.threshold < 1.0:
            raise ConfigError("vad.threshold must be between 0 and 1")
        if not 0 <= cfg.aggressiveness <= 3:
            raise ConfigError("vad.aggressiveness must be 0-3")
        if cfg.min_utterance_ms >= cfg.max_utterance_ms:
            raise ConfigError("vad.min_utterance_ms must be less than vad.max_utterance_ms")
        if cfg.silence_ms_to_end < 100:
            raise ConfigError("vad.silence_ms_to_end below 100 will chop words mid-sentence")
        return cfg


@dataclass
class DuplexConfig:
    """Echo control for when a mic and a speaker share a room.

    All of this is off in dual_mic mode: headphones solve it physically.
    """

    shared_audio: bool = True
    bargein: bool = True
    hangover_ms: int = 150  # keep the gate shut this long after playback ends
    bargein_rms: float = 0.08  # normalized RMS that counts as "someone interrupted"
    bargein_ms: int = 200  # ...sustained for this long
    echo_guard_similarity: float = 0.85  # transcript vs. recent TTS text
    echo_guard_window_s: float = 5.0

    @classmethod
    def parse(cls, raw: Any) -> DuplexConfig:
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ConfigError("duplex must be a mapping")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown duplex keys: {', '.join(sorted(unknown))}")
        cfg = cls(**{k: raw[k] for k in raw})
        if not 0.0 < cfg.echo_guard_similarity <= 1.0:
            raise ConfigError("duplex.echo_guard_similarity must be between 0 and 1")
        return cfg


@dataclass
class AppConfig:
    mode: Mode = Mode.SINGLE_MIC
    language_a: LanguageSlot = field(
        default_factory=lambda: LanguageSlot("en", "English", "alloy")
    )
    language_b: LanguageSlot = field(default_factory=lambda: LanguageSlot("es", "Spanish", "nova"))
    channels: list[ChannelConfig] = field(default_factory=list)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    duplex: DuplexConfig = field(default_factory=DuplexConfig)
    log_level: str = "WARNING"
    source_path: Path | None = None

    @property
    def language_codes(self) -> tuple[str, str]:
        return (self.language_a.code, self.language_b.code)

    def slot_for(self, code: str) -> LanguageSlot:
        if code == self.language_a.code:
            return self.language_a
        if code == self.language_b.code:
            return self.language_b
        raise KeyError(f"{code!r} is not one of the two configured languages")

    def other_language(self, code: str) -> str:
        """The language an utterance in `code` should be translated into."""
        return self.language_b.code if code == self.language_a.code else self.language_a.code


def _validate(cfg: AppConfig) -> AppConfig:
    if cfg.language_a.code == cfg.language_b.code:
        raise ConfigError(
            f"languages.a and languages.b are both {cfg.language_a.code!r} — "
            "there would be nothing to translate"
        )

    if not cfg.channels:
        raise ConfigError("at least one channel is required")

    seen_ids = set()
    for channel in cfg.channels:
        if channel.id in seen_ids:
            raise ConfigError(f"duplicate channel id {channel.id!r}")
        seen_ids.add(channel.id)

    if cfg.mode is Mode.SINGLE_MIC:
        if len(cfg.channels) != 1:
            raise ConfigError(
                f"mode=single_mic needs exactly 1 channel, found {len(cfg.channels)}. "
                "Use mode=dual_mic for two microphones."
            )
        if cfg.channels[0].language != "auto":
            raise ConfigError(
                "mode=single_mic requires channels[0].language: auto — with one shared "
                "mic the app has to detect who is speaking from the audio itself"
            )
    else:
        if len(cfg.channels) != 2:
            raise ConfigError(
                f"mode=dual_mic needs exactly 2 channels, found {len(cfg.channels)}"
            )
        pinned = [c.language for c in cfg.channels]
        if "auto" in pinned:
            raise ConfigError(
                "mode=dual_mic requires each channel to pin a language "
                f"(got {pinned}). That is the whole point: no guessing."
            )
        if set(pinned) != set(cfg.language_codes):
            raise ConfigError(
                f"dual_mic channel languages {pinned} must be exactly the two "
                f"configured languages {list(cfg.language_codes)}"
            )
        inputs = [c.input_device for c in cfg.channels]
        if inputs[0] is not None and inputs[0] == inputs[1]:
            raise ConfigError(
                "dual_mic channels share input_device "
                f"{inputs[0]!r} — both pipelines would hear both speakers. "
                "Give each person their own microphone, or use mode=single_mic."
            )

    # dual_mic assumes headsets, so the echo machinery is dead weight there.
    if cfg.mode is Mode.DUAL_MIC and cfg.duplex.shared_audio:
        cfg.duplex.shared_audio = False

    return cfg


def load(path: str | Path | None = None, *, env_file: str | Path | None = None) -> AppConfig:
    """Load config.yaml (falling back to defaults) plus .env keys."""
    load_dotenv(env_file) if env_file else load_dotenv()

    if path is None:
        for candidate in ("config.yaml", "config.yml", "config.example.yaml"):
            if Path(candidate).exists():
                path = candidate
                break
    if path is None:
        return _validate(_defaults())

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    cfg = from_dict(raw)
    cfg.source_path = path
    return cfg


def from_dict(raw: dict[str, Any]) -> AppConfig:
    mode_raw = str(raw.get("mode", "single_mic")).strip().lower()
    try:
        mode = Mode(mode_raw)
    except ValueError as exc:
        raise ConfigError(
            f"mode must be 'single_mic' or 'dual_mic', got {mode_raw!r}"
        ) from exc

    lang_raw = raw.get("languages") or {}
    if not isinstance(lang_raw, dict):
        raise ConfigError("languages must be a mapping with keys 'a' and 'b'")
    if "a" not in lang_raw or "b" not in lang_raw:
        raise ConfigError("languages must define both 'a' and 'b'")

    taken: set[str] = set()
    language_a = LanguageSlot.parse(lang_raw["a"], "a", taken)
    taken.add(language_a.voice)
    language_b = LanguageSlot.parse(lang_raw["b"], "b", taken)

    channels_raw = raw.get("channels")
    if channels_raw is None:
        channels = _default_channels(mode, language_a, language_b)
    else:
        if not isinstance(channels_raw, list):
            raise ConfigError("channels must be a list")
        channels = [ChannelConfig.parse(c, i) for i, c in enumerate(channels_raw)]

    cfg = AppConfig(
        mode=mode,
        language_a=language_a,
        language_b=language_b,
        channels=channels,
        providers=ProvidersConfig.parse(raw.get("providers")),
        vad=VadConfig.parse(raw.get("vad")),
        duplex=DuplexConfig.parse(raw.get("duplex")),
        log_level=str(raw.get("log_level", "WARNING")).upper(),
    )
    return _validate(cfg)


def _default_channels(
    mode: Mode, language_a: LanguageSlot, language_b: LanguageSlot
) -> list[ChannelConfig]:
    if mode is Mode.SINGLE_MIC:
        return [ChannelConfig(id="A", language="auto")]
    return [
        ChannelConfig(id="A", language=language_a.code),
        ChannelConfig(id="B", language=language_b.code),
    ]


def _defaults() -> AppConfig:
    cfg = AppConfig()
    cfg.channels = _default_channels(cfg.mode, cfg.language_a, cfg.language_b)
    return cfg


def to_dict(cfg: AppConfig) -> dict[str, Any]:
    """Serialize back to the YAML shape, for the `--setup` wizard."""
    return {
        "mode": cfg.mode.value,
        "languages": {
            "a": {"code": cfg.language_a.code, "name": cfg.language_a.name, "voice": cfg.language_a.voice},
            "b": {"code": cfg.language_b.code, "name": cfg.language_b.name, "voice": cfg.language_b.voice},
        },
        "channels": [
            {
                "id": c.id,
                "input_device": c.input_device,
                "output_device": c.output_device,
                "language": c.language,
                "input_gain": c.input_gain,
            }
            for c in cfg.channels
        ],
        "providers": {
            "stt": cfg.providers.stt,
            "translation": cfg.providers.translation,
            "tts": cfg.providers.tts,
            "stt_model": cfg.providers.stt_model,
            "translation_model": cfg.providers.translation_model,
            "tts_model": cfg.providers.tts_model,
        },
        "vad": {
            "backend": cfg.vad.backend,
            "silence_ms_to_end": cfg.vad.silence_ms_to_end,
            "preroll_ms": cfg.vad.preroll_ms,
            "min_utterance_ms": cfg.vad.min_utterance_ms,
            "max_utterance_ms": cfg.vad.max_utterance_ms,
            "threshold": cfg.vad.threshold,
        },
        "duplex": {
            "shared_audio": cfg.duplex.shared_audio,
            "bargein": cfg.duplex.bargein,
        },
        "log_level": cfg.log_level,
    }


def save(cfg: AppConfig, path: str | Path = "config.yaml") -> Path:
    path = Path(path)
    path.write_text(
        yaml.safe_dump(to_dict(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


# The literal values in .env.example. Setup says "copy .env.example to .env",
# so forgetting to edit it is the single most likely first-run mistake — and
# an unedited placeholder is truthy, which would otherwise sail past the
# emptiness check and fail much later as an opaque 401.
_PLACEHOLDER_KEYS = {"sk-...", "your-key-here", "changeme", "..."}


def require_env(name: str, provider: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set, but the {provider!r} provider needs it. "
            f"Copy .env.example to .env and add your key."
        )
    if value in _PLACEHOLDER_KEYS:
        raise ConfigError(
            f"{name} is still the placeholder from .env.example. "
            f"Edit .env and replace it with a real {provider} key."
        )
    return value
