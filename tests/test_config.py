"""Config parsing, validation and device resolution.

Every error here should read like advice to the person editing config.yaml.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from interpreter import config as config_module
from interpreter.audio import devices as audio_devices
from interpreter.audio.devices import DeviceInfo
from interpreter.config import AppConfig, ConfigError, Mode

SINGLE_MIC = {
    "mode": "single_mic",
    "languages": {"a": {"code": "en"}, "b": {"code": "es"}},
    "channels": [{"id": "A", "language": "auto"}],
}

DUAL_MIC = {
    "mode": "dual_mic",
    "languages": {"a": {"code": "en"}, "b": {"code": "es"}},
    "channels": [
        {"id": "A", "input_device": "Headset One", "language": "en"},
        {"id": "B", "input_device": "Headset Two", "language": "es"},
    ],
}


def test_minimal_single_mic_config_loads():
    cfg = config_module.from_dict(SINGLE_MIC)

    assert cfg.mode is Mode.SINGLE_MIC
    assert cfg.language_codes == ("en", "es")
    assert cfg.channels[0].language == "auto"
    assert cfg.duplex.shared_audio is True


def test_languages_can_be_bare_codes():
    cfg = config_module.from_dict(
        {"mode": "single_mic", "languages": {"a": "en", "b": "fr"}}
    )

    assert cfg.language_codes == ("en", "fr")
    assert cfg.language_a.name == "English"


def test_the_two_slots_get_different_default_voices():
    """Both sides sounding identical is confusing on a shared speaker."""
    cfg = config_module.from_dict({"languages": {"a": "en", "b": "nl"}})

    assert cfg.language_a.voice != cfg.language_b.voice


def test_channels_default_to_the_mode():
    single = config_module.from_dict(
        {"mode": "single_mic", "languages": {"a": "en", "b": "es"}}
    )
    dual = config_module.from_dict(
        {"mode": "dual_mic", "languages": {"a": "en", "b": "es"}}
    )

    assert len(single.channels) == 1
    assert [c.language for c in dual.channels] == ["en", "es"]


def test_headset_mode_turns_off_echo_suppression():
    cfg = config_module.from_dict({**DUAL_MIC, "duplex": {"shared_audio": True}})

    assert cfg.duplex.shared_audio is False


def test_other_language_flips_the_direction():
    cfg = config_module.from_dict(SINGLE_MIC)

    assert cfg.other_language("en") == "es"
    assert cfg.other_language("es") == "en"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_the_same_language_twice_is_rejected():
    with pytest.raises(ConfigError, match="nothing to translate"):
        config_module.from_dict({"languages": {"a": "en", "b": "en"}})


def test_unknown_language_codes_are_rejected():
    with pytest.raises(ConfigError, match="not in the language table"):
        config_module.from_dict({"languages": {"a": "en", "b": "klingon"}})


def test_single_mic_rejects_two_channels():
    with pytest.raises(ConfigError, match="exactly 1 channel"):
        config_module.from_dict(
            {**SINGLE_MIC, "channels": [{"id": "A", "language": "auto"}, {"id": "B", "language": "auto"}]}
        )


def test_single_mic_requires_auto_detection():
    with pytest.raises(ConfigError, match="requires channels\\[0\\].language: auto"):
        config_module.from_dict({**SINGLE_MIC, "channels": [{"id": "A", "language": "en"}]})


def test_dual_mic_requires_pinned_languages():
    with pytest.raises(ConfigError, match="no guessing"):
        config_module.from_dict(
            {
                **DUAL_MIC,
                "channels": [
                    {"id": "A", "input_device": 1, "language": "auto"},
                    {"id": "B", "input_device": 2, "language": "es"},
                ],
            }
        )


def test_dual_mic_rejects_a_shared_microphone():
    """The whole point of dual_mic is one microphone per person."""
    with pytest.raises(ConfigError, match="own microphone"):
        config_module.from_dict(
            {
                **DUAL_MIC,
                "channels": [
                    {"id": "A", "input_device": "Same Mic", "language": "en"},
                    {"id": "B", "input_device": "Same Mic", "language": "es"},
                ],
            }
        )


def test_dual_mic_channel_languages_must_match_the_pair():
    with pytest.raises(ConfigError, match="must be exactly the two"):
        config_module.from_dict(
            {
                **DUAL_MIC,
                "channels": [
                    {"id": "A", "input_device": 1, "language": "en"},
                    {"id": "B", "input_device": 2, "language": "fr"},
                ],
            }
        )


def test_duplicate_channel_ids_are_rejected():
    with pytest.raises(ConfigError, match="duplicate channel id"):
        config_module.from_dict(
            {
                **DUAL_MIC,
                "channels": [
                    {"id": "A", "input_device": 1, "language": "en"},
                    {"id": "A", "input_device": 2, "language": "es"},
                ],
            }
        )


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ConfigError, match="single_mic"):
        config_module.from_dict({"mode": "telepathy", "languages": {"a": "en", "b": "es"}})


def test_a_dangerously_short_vad_timeout_is_rejected():
    with pytest.raises(ConfigError, match="chop words"):
        config_module.from_dict({**SINGLE_MIC, "vad": {"silence_ms_to_end": 50}})


def test_typos_in_section_keys_are_caught():
    with pytest.raises(ConfigError, match="unknown vad keys"):
        config_module.from_dict({**SINGLE_MIC, "vad": {"silence_ms": 400}})
    with pytest.raises(ConfigError, match="unknown duplex keys"):
        config_module.from_dict({**SINGLE_MIC, "duplex": {"bargein_enabled": True}})


def test_missing_language_slots_are_reported():
    with pytest.raises(ConfigError, match="both 'a' and 'b'"):
        config_module.from_dict({"languages": {"a": "en"}})


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def test_the_shipped_example_config_is_valid():
    """A broken example is the first thing every new user would hit."""
    cfg = config_module.load("config.example.yaml")

    assert cfg.mode is Mode.SINGLE_MIC
    assert cfg.language_codes == ("en", "es")


def test_save_and_load_round_trips(tmp_path):
    original = config_module.from_dict(DUAL_MIC)
    path = tmp_path / "config.yaml"
    config_module.save(original, path)

    reloaded = config_module.load(path)
    assert reloaded.mode is original.mode
    assert reloaded.language_codes == original.language_codes
    assert [c.language for c in reloaded.channels] == ["en", "es"]


def test_malformed_yaml_is_reported_clearly(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("mode: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid YAML"):
        config_module.load(path)


def test_a_missing_config_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        config_module.load(tmp_path / "nope.yaml")


def test_a_non_mapping_config_is_reported(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(["a", "list"]), encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping at the top level"):
        config_module.load(path)


# --------------------------------------------------------------------------
# device resolution
# --------------------------------------------------------------------------


FAKE_DEVICES = [
    DeviceInfo(0, "Microphone (FHD Camera Microphone)", 2, 0, 44100.0, "MME"),
    DeviceInfo(1, "Speakers (USB2.0 Device)", 0, 2, 48000.0, "MME"),
    DeviceInfo(2, "Microphone (FHD Camera Microphone)", 2, 0, 48000.0, "Windows WASAPI"),
    DeviceInfo(3, "Headset Earphone (Jabra)", 1, 2, 16000.0, "Windows WASAPI"),
]


@pytest.fixture
def fake_devices(monkeypatch):
    monkeypatch.setattr(audio_devices, "list_devices", lambda: FAKE_DEVICES)


def test_none_means_system_default(fake_devices):
    assert audio_devices.resolve(None, kind="input") is None


def test_an_index_resolves_to_itself(fake_devices):
    assert audio_devices.resolve(1, kind="output") == 1
    assert audio_devices.resolve("3", kind="input") == 3


def test_an_exact_name_resolves(fake_devices):
    assert audio_devices.resolve("Speakers (USB2.0 Device)", kind="output") == 1


def test_a_substring_resolves_and_is_case_insensitive(fake_devices):
    assert audio_devices.resolve("jabra", kind="input") == 3
    assert audio_devices.resolve("USB2.0", kind="output") == 1


def test_an_ambiguous_name_prefers_wasapi(fake_devices):
    """Two host APIs expose the same physical mic; WASAPI is the better one."""
    assert audio_devices.resolve("FHD Camera", kind="input") == 2


def test_a_missing_device_lists_the_alternatives(fake_devices):
    with pytest.raises(audio_devices.AudioDeviceError, match="Available:"):
        audio_devices.resolve("Blue Yeti", kind="input")


def test_an_output_only_device_is_not_a_valid_input(fake_devices):
    with pytest.raises(audio_devices.AudioDeviceError):
        audio_devices.resolve(1, kind="input")


def test_shared_device_detection(fake_devices):
    # Same physical webcam housing for both -> assume it hears itself.
    assert audio_devices.same_physical_device(0, 0) is True
    # A Jabra headset and USB speakers are separate bodies.
    assert audio_devices.same_physical_device(3, 1) is False
    # Unknown wiring: assume the cautious answer.
    assert audio_devices.same_physical_device(None, None) is True


# --------------------------------------------------------------------------
# provider registry
# --------------------------------------------------------------------------


def test_unknown_providers_list_the_valid_choices():
    from interpreter import registry

    cfg = AppConfig().providers
    cfg.stt = "wishful-thinking"
    with pytest.raises(ConfigError, match="Choose one of"):
        registry.build_stt(cfg)


def test_fake_providers_build():
    from interpreter import registry
    from interpreter.config import ProvidersConfig

    stt, translation, tts = registry.build_all(
        ProvidersConfig(stt="fake", translation="fake", tts="fake")
    )
    assert (stt.name, translation.name, tts.name) == ("fake", "fake", "fake")


def test_a_missing_api_key_is_reported_as_config_advice(monkeypatch):
    from interpreter import registry
    from interpreter.config import ProvidersConfig
    from interpreter.providers import openai_provider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    openai_provider._client.cache_clear()

    provider = registry.build_stt(ProvidersConfig(stt="openai"))
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        openai_provider._client()
    assert provider.name == "openai"


def test_preflight_catches_a_missing_key_before_devices_open(monkeypatch):
    """One clear failure at startup beats every utterance failing later."""
    from interpreter.config import ProvidersConfig
    from interpreter.orchestrator import Orchestrator
    from interpreter.providers import openai_provider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    openai_provider._client.cache_clear()

    cfg = config_module.from_dict(SINGLE_MIC)
    cfg.providers = ProvidersConfig(stt="openai", translation="openai", tts="openai")

    opened = []
    orchestrator = Orchestrator(
        cfg,
        microphone_factory=lambda channel: opened.append(channel) or None,
        speaker_factory=lambda channel: opened.append(channel) or None,
    )

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        orchestrator.build()
    assert opened == [], "no audio device should be opened before credentials pass"


def test_preflight_is_a_no_op_for_providers_that_need_nothing():
    from interpreter.config import ProvidersConfig
    from interpreter.orchestrator import Orchestrator

    cfg = config_module.from_dict(SINGLE_MIC)
    cfg.providers = ProvidersConfig(stt="fake", translation="fake", tts="fake")

    from interpreter.audio.capture import ArrayMicrophone
    from interpreter.audio.playback import RecordingSpeaker

    orchestrator = Orchestrator(
        cfg,
        microphone_factory=lambda channel: ArrayMicrophone(
            np.zeros(0, dtype=np.int16), channel_id=channel.id
        ),
        speaker_factory=lambda channel: RecordingSpeaker(channel_id=channel.id),
    )
    orchestrator.build()

    assert len(orchestrator.channels) == 1
