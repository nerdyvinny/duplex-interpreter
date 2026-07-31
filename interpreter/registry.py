# turns the provider names from config.yaml into actual objects
#
# all the imports are inside the functions on purpose. if I put them at the
# top then installing only the cloud requirements crashes because
# faster_whisper isn't there, even if you never asked for it

from .config import ConfigError

STT_CHOICES = ("openai", "faster-whisper", "fake")
TRANSLATION_CHOICES = ("openai", "deepl", "argos", "passthrough", "fake")
TTS_CHOICES = ("openai", "elevenlabs", "piper", "silent", "fake")


def _unknown(kind, name, choices):
    return ConfigError(
        f"providers.{kind}={name!r} is not recognised. "
        f"Choose one of: {', '.join(choices)}"
    )


def build_stt(cfg):
    name = cfg.stt.strip().lower()

    if name == "openai":
        from .providers.openai_provider import OpenAISTT
        return OpenAISTT(cfg.stt_model)

    # I keep typing the underscore version so I just accept both
    if name in {"faster-whisper", "faster_whisper", "local"}:
        from .providers.local_provider import FasterWhisperSTT
        return FasterWhisperSTT(cfg)

    if name == "fake":
        from .providers.fake import FakeSTT
        return FakeSTT()

    raise _unknown("stt", name, STT_CHOICES)


def build_translation(cfg):
    name = cfg.translation.strip().lower()

    if name == "openai":
        from .providers.openai_provider import OpenAITranslation
        return OpenAITranslation(cfg.translation_model)

    if name == "deepl":
        from .providers.cloud_extras import DeepLTranslation
        return DeepLTranslation()

    if name == "argos":
        from .providers.local_provider import ArgosTranslation
        return ArgosTranslation()

    if name == "passthrough":
        from .providers.local_provider import PassthroughTranslation
        return PassthroughTranslation()

    if name == "fake":
        from .providers.fake import FakeTranslation
        return FakeTranslation()

    raise _unknown("translation", name, TRANSLATION_CHOICES)


def build_tts(cfg):
    name = cfg.tts.strip().lower()

    if name == "openai":
        from .providers.openai_provider import OpenAITTS
        return OpenAITTS(cfg.tts_model)

    if name == "elevenlabs":
        from .providers.cloud_extras import ElevenLabsTTS
        return ElevenLabsTTS()

    if name == "piper":
        from .providers.local_provider import PiperTTS
        return PiperTTS()

    if name == "silent":
        from .providers.local_provider import SilentTTS
        return SilentTTS()

    if name == "fake":
        from .providers.fake import FakeTTS
        return FakeTTS()

    raise _unknown("tts", name, TTS_CHOICES)


def build_all(cfg):
    return build_stt(cfg), build_translation(cfg), build_tts(cfg)
