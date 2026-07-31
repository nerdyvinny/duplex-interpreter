# reads config.yaml and the .env file, and yells at you if something is wrong
#
# I tried to make the error messages actually say what to fix instead of
# just dumping a traceback, because I kept confusing myself with my own
# config file

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import languages

# everything before the TTS runs at 16khz mono 16 bit. thats what silero
# and whisper both want, and it keeps the uploads small
PIPELINE_SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = PIPELINE_SAMPLE_RATE * FRAME_MS // 1000  # = 320


class ConfigError(ValueError):
    # something in the config a person has to go fix
    pass


class Mode(str, Enum):
    SINGLE_MIC = "single_mic"
    DUAL_MIC = "dual_mic"


@dataclass
class LanguageSlot:
    code: str
    name: str
    voice: str

    @classmethod
    def parse(cls, raw, slot, taken_voices):
        # you can write either "es" or a whole block with a voice in it
        if isinstance(raw, str):
            raw = {"code": raw}
        if not isinstance(raw, dict):
            raise ConfigError(
                f"languages.{slot} must be a language code or a mapping, "
                f"got {type(raw).__name__}"
            )

        code = raw.get("code")
        if not code:
            raise ConfigError(f"languages.{slot}.code is required (e.g. 'en')")

        # "en-US" -> "en"
        code = str(code).strip().lower().split("-")[0]
        if not languages.is_known(code):
            raise ConfigError(
                f"languages.{slot}.code={code!r} is not in the language table. "
                f"Add it to interpreter/languages.py or pick a known code."
            )

        voice = raw.get("voice")
        if not voice:
            voice = languages.default_voice(code, taken_voices)

        name = raw.get("name")
        if not name:
            name = languages.name_of(code)

        return cls(code=code, name=str(name), voice=str(voice))


@dataclass
class ChannelConfig:
    # one microphone, plus the speaker its translations come out of.
    # in dual_mic these get crossed over in orchestrator.py
    id: str
    input_device: object = None
    output_device: object = None
    language: str = "auto"   # "auto" means detect it every sentence
    input_gain: float = 1.0

    @classmethod
    def parse(cls, raw, index):
        if not isinstance(raw, dict):
            raise ConfigError(f"channels[{index}] must be a mapping")

        # if they didn't name it, call it A, B, C...
        channel_id = raw.get("id")
        if not channel_id:
            channel_id = chr(ord("A") + index)

        language = str(raw.get("language", "auto")).strip().lower()
        if language != "auto":
            language = language.split("-")[0]
            if not languages.is_known(language):
                raise ConfigError(
                    f"channels[{index}].language={language!r} is unknown; "
                    f"use 'auto' or a known code"
                )

        gain = float(raw.get("input_gain", 1.0))
        if not 0.1 <= gain <= 10.0:
            raise ConfigError(
                f"channels[{index}].input_gain must be between 0.1 and 10.0"
            )

        return cls(
            id=str(channel_id),
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

    # only used by the local stuff, the cloud providers ignore these
    local_whisper_size: str = "small"
    local_whisper_device: str = "auto"   # auto | cuda | cpu
    local_whisper_compute: str = "auto"  # auto | float16 | int8

    # how many old lines to give the translator so it can figure out
    # pronouns and he/she stuff
    context_turns: int = 3

    @classmethod
    def parse(cls, raw):
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ConfigError("providers must be a mapping")

        # catch typos instead of silently ignoring the key
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(
                f"unknown providers keys: {', '.join(sorted(unknown))}"
            )

        return cls(**{k: raw[k] for k in raw})


@dataclass
class VadConfig:
    backend: str = "silero"  # silero | webrtc

    # THIS is the setting that decides how fast the app feels. its how much
    # silence has to happen before I decide the sentence is over. lower =
    # snappier but it cuts off people who pause to think
    silence_ms_to_end: int = 600

    preroll_ms: int = 300       # audio kept from BEFORE the trigger so the
                                # first word isn't chopped off
    min_utterance_ms: int = 250  # ignore coughs and clicks
    max_utterance_ms: int = 15_000  # split up people who never stop talking
    start_frames: int = 3       # voiced frames in a row before I believe it
    threshold: float = 0.5      # silero speech probability
    aggressiveness: int = 2     # webrtc only, 0 to 3

    @classmethod
    def parse(cls, raw):
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ConfigError("vad must be a mapping")

        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown vad keys: {', '.join(sorted(unknown))}")

        cfg = cls(**{k: raw[k] for k in raw})

        if cfg.backend not in {"silero", "webrtc"}:
            raise ConfigError(
                f"vad.backend must be 'silero' or 'webrtc', got {cfg.backend!r}"
            )
        if not 0.0 < cfg.threshold < 1.0:
            raise ConfigError("vad.threshold must be between 0 and 1")
        if not 0 <= cfg.aggressiveness <= 3:
            raise ConfigError("vad.aggressiveness must be 0-3")
        if cfg.min_utterance_ms >= cfg.max_utterance_ms:
            raise ConfigError(
                "vad.min_utterance_ms must be less than vad.max_utterance_ms"
            )
        if cfg.silence_ms_to_end < 100:
            raise ConfigError(
                "vad.silence_ms_to_end below 100 will chop words mid-sentence"
            )

        return cfg


@dataclass
class DuplexConfig:
    # the echo settings. all of this turns itself off in dual_mic mode
    # because headphones already solve it

    shared_audio: bool = True
    bargein: bool = True
    hangover_ms: int = 150          # keep ignoring the mic this long after
                                    # playback stops, for the room echo
    bargein_rms: float = 0.08       # how loud counts as interrupting
    bargein_ms: int = 200           # ...for this long
    echo_guard_similarity: float = 0.85  # transcript vs what we just said
    echo_guard_window_s: float = 5.0

    @classmethod
    def parse(cls, raw):
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
    language_b: LanguageSlot = field(
        default_factory=lambda: LanguageSlot("es", "Spanish", "nova")
    )
    channels: list = field(default_factory=list)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    duplex: DuplexConfig = field(default_factory=DuplexConfig)
    log_level: str = "WARNING"
    source_path: object = None

    @property
    def language_codes(self):
        return (self.language_a.code, self.language_b.code)

    def slot_for(self, code):
        if code == self.language_a.code:
            return self.language_a
        if code == self.language_b.code:
            return self.language_b
        raise KeyError(f"{code!r} is not one of the two configured languages")

    def other_language(self, code):
        # if they spoke A translate to B, otherwise translate to A
        if code == self.language_a.code:
            return self.language_b.code
        return self.language_a.code


def _validate(cfg):
    if cfg.language_a.code == cfg.language_b.code:
        raise ConfigError(
            f"languages.a and languages.b are both {cfg.language_a.code!r} - "
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
                f"mode=single_mic needs exactly 1 channel, found "
                f"{len(cfg.channels)}. Use mode=dual_mic for two microphones."
            )
        if cfg.channels[0].language != "auto":
            raise ConfigError(
                "mode=single_mic requires channels[0].language: auto - with one "
                "shared mic the app has to detect who is speaking from the "
                "audio itself"
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
                f"{inputs[0]!r} - both pipelines would hear both speakers. "
                "Give each person their own microphone, or use mode=single_mic."
            )

    # dual_mic means headsets, so all the echo stuff is pointless there
    if cfg.mode is Mode.DUAL_MIC and cfg.duplex.shared_audio:
        cfg.duplex.shared_audio = False

    return cfg


def load(path=None, *, env_file=None):
    # loads config.yaml (or the defaults) plus the api keys from .env
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    if path is None:
        for candidate in ("config.yaml", "config.yml", "config.example.yaml"):
            if Path(candidate).exists():
                path = candidate
                break

    if path is None:
        return _validate(_defaults())  # no file anywhere, use defaults

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


def from_dict(raw):
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

    # keep track of the voices so both people don't end up sounding the same
    taken = set()
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


def _default_channels(mode, language_a, language_b):
    if mode is Mode.SINGLE_MIC:
        return [ChannelConfig(id="A", language="auto")]
    return [
        ChannelConfig(id="A", language=language_a.code),
        ChannelConfig(id="B", language=language_b.code),
    ]


def _defaults():
    cfg = AppConfig()
    cfg.channels = _default_channels(cfg.mode, cfg.language_a, cfg.language_b)
    return cfg


def to_dict(cfg):
    # turn it back into the yaml shape, used by --setup
    return {
        "mode": cfg.mode.value,
        "languages": {
            "a": {
                "code": cfg.language_a.code,
                "name": cfg.language_a.name,
                "voice": cfg.language_a.voice,
            },
            "b": {
                "code": cfg.language_b.code,
                "name": cfg.language_b.name,
                "voice": cfg.language_b.voice,
            },
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


def save(cfg, path="config.yaml"):
    path = Path(path)
    path.write_text(
        yaml.safe_dump(to_dict(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


# the exact junk that is sitting in .env.example. the setup says "copy
# .env.example to .env" so forgetting to actually edit it is the most likely
# first mistake, and "sk-..." is a non empty string so it sails right past
# the empty check and then you get a confusing 401 much later
_PLACEHOLDER_KEYS = {"sk-...", "your-key-here", "changeme", "..."}


def require_env(name, provider):
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
