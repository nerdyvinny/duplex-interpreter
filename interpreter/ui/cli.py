"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import wave
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .. import config as config_module
from .. import languages
from ..audio import devices as audio_devices
from ..audio.capture import ArrayMicrophone
from ..audio.playback import RecordingSpeaker
from ..audio.resample import resample_int16
from ..config import PIPELINE_SAMPLE_RATE, AppConfig, ChannelConfig, ConfigError, Mode
from ..events import EventBus, Stage
from ..orchestrator import Orchestrator
from ..providers.base import wav_bytes_to_pcm
from .display import LiveTranscript, PlainLogger

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="duplex-interpreter",
        description="Hands-free, two-way, real-time speech translation.",
    )
    parser.add_argument("-c", "--config", help="path to config.yaml")
    parser.add_argument("--devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--setup", action="store_true", help="interactive setup wizard")
    parser.add_argument(
        "--selftest",
        metavar="WAV",
        help="run one WAV file through the whole pipeline (no microphone needed)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="with --selftest, feed the audio at wall-clock speed. Slower, but "
        "the reported latencies then match live use instead of being inflated "
        "by every utterance arriving at once.",
    )
    parser.add_argument(
        "--loopback",
        type=float,
        nargs="?",
        const=5.0,
        metavar="SECONDS",
        help="pipe the microphone straight to the speaker to check device wiring",
    )
    parser.add_argument("--no-live", action="store_true", help="plain line output instead of the live view")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="repeat for more logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        if args.devices:
            return _cmd_devices()
        if args.setup:
            return _cmd_setup(args.config)

        cfg = config_module.load(args.config)
        if args.selftest:
            return asyncio.run(
                _cmd_selftest(cfg, Path(args.selftest), realtime=args.realtime)
            )
        if args.loopback is not None:
            return asyncio.run(_cmd_loopback(cfg, args.loopback))
        return asyncio.run(_cmd_run(cfg, live=not args.no_live))
    except ConfigError as exc:
        console.print(f"[red]Configuration problem:[/red] {exc}")
        return 2
    except audio_devices.AudioDeviceError as exc:
        console.print(f"[red]Audio device problem:[/red] {exc}")
        return 3
    except KeyboardInterrupt:
        console.print("\nstopped")
        return 130


def _force_utf8_output() -> None:
    """Make the console handle accented text.

    A translation app prints Spanish, Greek and Japanese by definition, and
    the legacy Windows console codepage mangles all of it into question
    marks. Harmless everywhere else.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # already redirected, or not a real stream


# Third-party loggers that would otherwise bury the conversation. Argos logs
# every tokenized sentence at INFO; stanza warns about model packaging on
# every call.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "argostranslate",
    "stanza",
    "huggingface_hub",
    "faster_whisper",
    "numba",
)


class _QuietThirdParty(logging.Filter):
    """Silence noisy libraries at the handler.

    Setting levels per logger is not enough: argostranslate calls
    `setLevel(INFO)` on its own logger when it is imported, which is after
    our configuration runs, and a logger's own level beats its parent's.
    Filtering at the handler catches the records whatever the library does.
    """

    def __init__(self, threshold: int) -> None:
        super().__init__()
        self.threshold = threshold

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith(_NOISY_LOGGERS):
            return record.levelno >= self.threshold
        return True


def _configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # -vv means the user genuinely wants the library internals.
    if verbosity < 2:
        third_party = logging.ERROR if verbosity == 0 else logging.WARNING
        for handler in logging.getLogger().handlers:
            handler.addFilter(_QuietThirdParty(third_party))


# --------------------------------------------------------------------------
# --devices
# --------------------------------------------------------------------------


def _cmd_devices() -> int:
    default_in, default_out = audio_devices.default_devices()

    for kind, entries, default in (
        ("Inputs (microphones)", audio_devices.input_devices(), default_in),
        ("Outputs (speakers)", audio_devices.output_devices(), default_out),
    ):
        table = Table(title=kind, title_justify="left", header_style="bold")
        table.add_column("#", justify="right")
        table.add_column("Name")
        table.add_column("API")
        table.add_column("Ch", justify="right")
        table.add_column("Rate", justify="right")
        for device in entries:
            marker = " (default)" if device.index == default else ""
            channels = (
                device.max_input_channels
                if "Input" in kind
                else device.max_output_channels
            )
            table.add_row(
                str(device.index),
                f"{device.name}{marker}",
                device.hostapi_name,
                str(channels),
                f"{device.default_samplerate:.0f}",
            )
        console.print(table)
        console.print()

    console.print(
        "Put either the number or any part of the name in config.yaml, "
        "e.g. [cyan]input_device: \"FHD Camera\"[/cyan]"
    )
    return 0


# --------------------------------------------------------------------------
# --setup
# --------------------------------------------------------------------------


def _cmd_setup(path: str | None) -> int:
    console.print("[bold]duplex-interpreter setup[/bold]\n")

    mode = _ask_mode()
    language_a, language_b = _ask_languages()
    channels = _ask_channels(mode, language_a, language_b)
    providers = _ask_providers()

    cfg = AppConfig(
        mode=mode,
        language_a=language_a,
        language_b=language_b,
        channels=channels,
        providers=providers,
    )
    cfg = config_module.from_dict(config_module.to_dict(cfg))  # validate the round trip

    target = Path(path or "config.yaml")
    if target.exists() and not Confirm.ask(f"\n{target} exists. Overwrite?", default=False):
        console.print("left unchanged")
        return 0

    config_module.save(cfg, target)
    console.print(f"\n[green]Wrote {target}[/green]")

    if providers.stt == "openai" or providers.translation == "openai" or providers.tts == "openai":
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            console.print(
                "\n[yellow]OPENAI_API_KEY is not set.[/yellow] "
                "Copy .env.example to .env and add your key before running."
            )

    console.print("\nNext: [cyan]python run.py --selftest samples/hello_en.wav[/cyan]")
    console.print("Then: [cyan]python run.py[/cyan]")
    return 0


def _ask_mode() -> Mode:
    console.print("How are the two people miked?\n")
    console.print("  [cyan]1[/cyan]  One shared microphone and speaker")
    console.print("     Hands-free, no buttons. The app detects which language was")
    console.print("     spoken and routes it. Works with any laptop.")
    console.print("     [dim]Genuinely overlapping speech degrades — one mic can't")
    console.print("     separate two voices.[/dim]\n")
    console.print("  [cyan]2[/cyan]  A headset each (two mics, two earpieces)")
    console.print("     True full duplex: both can talk at once, no echo, and each")
    console.print("     person's language is pinned so nothing is guessed.\n")
    choice = Prompt.ask("Choose", choices=["1", "2"], default="1")
    return Mode.SINGLE_MIC if choice == "1" else Mode.DUAL_MIC


def _ask_languages() -> tuple[config_module.LanguageSlot, config_module.LanguageSlot]:
    console.print("\n[bold]Languages[/bold]")
    known = ", ".join(sorted(languages.LANGUAGES))
    console.print(f"[dim]{known}[/dim]\n")

    code_a = _ask_language("Language A", "en")
    code_b = _ask_language("Language B", "es" if code_a != "es" else "en")
    while code_b == code_a:
        console.print("[yellow]The two languages must differ.[/yellow]")
        code_b = _ask_language("Language B", "es" if code_a != "es" else "en")

    voice_a = languages.default_voice(code_a)
    voice_b = languages.default_voice(code_b, {voice_a})
    return (
        config_module.LanguageSlot(code_a, languages.name_of(code_a), voice_a),
        config_module.LanguageSlot(code_b, languages.name_of(code_b), voice_b),
    )


def _ask_language(label: str, default: str) -> str:
    while True:
        value = Prompt.ask(label, default=default).strip().lower()
        if languages.is_known(value):
            return languages.get(value).code
        console.print(f"[yellow]{value!r} is not in the language table.[/yellow]")


def _ask_channels(
    mode: Mode,
    language_a: config_module.LanguageSlot,
    language_b: config_module.LanguageSlot,
) -> list[ChannelConfig]:
    try:
        inputs = audio_devices.input_devices()
        outputs = audio_devices.output_devices()
    except audio_devices.AudioDeviceError as exc:
        console.print(f"[yellow]Could not list audio devices ({exc}); using defaults.[/yellow]")
        inputs = outputs = []

    if not inputs or not outputs:
        if mode is Mode.SINGLE_MIC:
            return [ChannelConfig(id="A", language="auto")]
        return [
            ChannelConfig(id="A", language=language_a.code),
            ChannelConfig(id="B", language=language_b.code),
        ]

    console.print("\n[bold]Audio devices[/bold]")
    console.print("[dim]Blank = system default.[/dim]\n")
    for device in inputs:
        console.print(f"  in  [cyan]{device.index}[/cyan]  {device.name}")
    for device in outputs:
        console.print(f"  out [magenta]{device.index}[/magenta]  {device.name}")
    console.print()

    if mode is Mode.SINGLE_MIC:
        return [
            ChannelConfig(
                id="A",
                input_device=_ask_device("Microphone", inputs),
                output_device=_ask_device("Speaker", outputs),
                language="auto",
            )
        ]

    console.print(f"[cyan]{language_a.name} speaker's headset:[/cyan]")
    channel_a = ChannelConfig(
        id="A",
        input_device=_ask_device("  microphone", inputs),
        output_device=_ask_device("  earpiece", outputs),
        language=language_a.code,
    )
    console.print(f"\n[magenta]{language_b.name} speaker's headset:[/magenta]")
    channel_b = ChannelConfig(
        id="B",
        input_device=_ask_device("  microphone", inputs),
        output_device=_ask_device("  earpiece", outputs),
        language=language_b.code,
    )
    if channel_a.input_device is not None and channel_a.input_device == channel_b.input_device:
        console.print(
            "\n[yellow]Both channels use the same microphone. Each person needs "
            "their own, or both pipelines will hear both speakers.[/yellow]"
        )
    return [channel_a, channel_b]


def _ask_device(label: str, options) -> int | None:
    valid = {str(d.index) for d in options}
    while True:
        answer = Prompt.ask(f"{label} (number, or blank for default)", default="").strip()
        if not answer:
            return None
        if answer in valid:
            return int(answer)
        console.print(f"[yellow]{answer!r} is not one of the numbers listed.[/yellow]")


def _ask_providers() -> config_module.ProvidersConfig:
    console.print("\n[bold]Translation engine[/bold]\n")
    console.print("  [cyan]1[/cyan]  OpenAI  — one API key, ~0.8s, roughly $0.03/minute")
    console.print("  [cyan]2[/cyan]  Local   — free and offline, ~1-2s, multi-GB downloads")
    console.print("  [cyan]3[/cyan]  Mixed   — local speech recognition, OpenAI translation + voice\n")
    choice = Prompt.ask("Choose", choices=["1", "2", "3"], default="1")

    if choice == "2":
        return config_module.ProvidersConfig(
            stt="faster-whisper", translation="argos", tts="piper"
        )
    if choice == "3":
        return config_module.ProvidersConfig(
            stt="faster-whisper", translation="openai", tts="openai"
        )
    return config_module.ProvidersConfig()


# --------------------------------------------------------------------------
# --selftest
# --------------------------------------------------------------------------


async def _cmd_selftest(cfg: AppConfig, wav_path: Path, *, realtime: bool = False) -> int:
    if not wav_path.exists():
        console.print(f"[red]No such file:[/red] {wav_path}")
        return 4

    pcm, rate = wav_bytes_to_pcm(wav_path.read_bytes())
    if rate != PIPELINE_SAMPLE_RATE:
        pcm = resample_int16(pcm, rate, PIPELINE_SAMPLE_RATE)

    console.print(
        f"[bold]Self-test[/bold]  {wav_path.name}  "
        f"({pcm.size / PIPELINE_SAMPLE_RATE:.1f}s, {rate} Hz)\n"
    )
    providers = cfg.providers
    local_stt = providers.stt in {"faster-whisper", "faster_whisper", "local"}
    stt_model = providers.local_whisper_size if local_stt else providers.stt_model
    console.print(f"  stt         {providers.stt} ({stt_model})")
    console.print(
        f"  translation {providers.translation}"
        + (f" ({providers.translation_model})" if providers.translation == "openai" else "")
    )
    console.print(
        f"  tts         {providers.tts}"
        + (f" ({providers.tts_model})" if providers.tts == "openai" else "")
    )
    console.print(f"  languages   {cfg.language_a.name} <-> {cfg.language_b.name}")
    console.print(
        "  pacing      "
        + (
            "wall-clock (latencies match live use)"
            if realtime
            else "as fast as possible (latencies inflated by contention; "
            "add --realtime for honest numbers)"
        )
        + "\n"
    )

    bus = EventBus()
    speakers: dict[str, RecordingSpeaker] = {}

    def make_speaker(channel: ChannelConfig) -> RecordingSpeaker:
        speaker = RecordingSpeaker(channel_id=channel.id)
        speakers[channel.id] = speaker
        return speaker

    # The whole file goes into the first channel; a WAV has only one speaker.
    def make_microphone(channel: ChannelConfig):
        audio = pcm if channel.id == cfg.channels[0].id else np.zeros(0, dtype=np.int16)
        return ArrayMicrophone(audio, channel_id=channel.id, realtime=realtime)

    orchestrator = Orchestrator(
        cfg,
        bus=bus,
        microphone_factory=make_microphone,
        speaker_factory=make_speaker,
    )

    with PlainLogger(cfg, bus, console=console):
        await orchestrator.start()
        try:
            await asyncio.wait_for(orchestrator.run_until_stopped(), timeout=120.0)
        except (TimeoutError, asyncio.TimeoutError):
            console.print("[yellow]timed out after 120s[/yellow]")
        finally:
            await orchestrator.shutdown()

    history = bus.history
    spoke = [e for e in history if e.stage is Stage.DONE]
    errors = [e for e in history if e.stage is Stage.ERROR]
    captured = [e for e in history if e.stage is Stage.CAPTURED]
    dropped = [e for e in history if e.stage is Stage.DROPPED]

    console.print()
    if not captured:
        console.print(
            "[yellow]The VAD never triggered.[/yellow] Either the file has no speech, "
            "or the threshold is too high. Try `vad.threshold: 0.3`."
        )
        return 5
    if not spoke and not errors:
        console.print(
            f"[yellow]Segmented {len(captured)} utterance(s), but none produced "
            f"audio.[/yellow]"
        )
        for event in dropped:
            console.print(f"  [dim]{event.channel_id}{event.seq}: {event.detail}[/dim]")
        return 5

    for event in spoke:
        console.print(f"[bold]heard[/bold] ({event.source_lang}): {event.source_text}")
        console.print(f"[bold]said [/bold] ({event.target_lang}): {event.target_text}")
        timings = "  ".join(f"{k} {v:.0f}ms" for k, v in event.timings.items())
        console.print(f"[dim]{timings}  |  total {event.total_ms:.0f}ms[/dim]\n")

    for event in errors:
        console.print(f"[red]error:[/red] {event.detail}")

    written = 0
    for channel_id, speaker in speakers.items():
        audio = speaker.pcm()
        if audio.size == 0:
            continue
        out_path = Path(f"selftest_out_{channel_id}.wav")
        with wave.open(str(out_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(speaker.sample_rate)
            handle.writeframes(audio.tobytes())
        console.print(f"[green]wrote {out_path}[/green] ({audio.size / speaker.sample_rate:.1f}s)")
        written += 1

    if errors:
        return 6
    return 0 if written else 5


# --------------------------------------------------------------------------
# --loopback
# --------------------------------------------------------------------------


async def _cmd_loopback(cfg: AppConfig, seconds: float) -> int:
    from ..audio.capture import MicrophoneStream
    from ..audio.playback import SpeakerStream

    channel = cfg.channels[0]
    microphone = MicrophoneStream(channel.input_device, gain=channel.input_gain)
    speaker = SpeakerStream(channel.output_device)

    speaker.start()
    microphone.start()
    console.print(
        f"[bold]Loopback[/bold] for {seconds:.0f}s\n"
        f"  in  {microphone.description}\n"
        f"  out {speaker.description}\n\n"
        "[yellow]Wear headphones — speakers will feed back.[/yellow]\n"
        "Talk; you should hear yourself."
    )

    async def _pump() -> None:
        async for frame in microphone.frames():
            await speaker.play(
                _one_chunk(frame.tobytes()), source_rate=PIPELINE_SAMPLE_RATE
            )

    task = asyncio.create_task(_pump())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=seconds)
    except (TimeoutError, asyncio.TimeoutError):
        pass
    finally:
        task.cancel()
        microphone.stop()
        speaker.close()

    stats = microphone.stats
    console.print(
        f"\ndone. dropped frames: {stats['dropped_frames']}, "
        f"overflows: {stats['overflows']}"
    )
    if stats["overflows"] > 5:
        console.print(
            "[yellow]Frequent overflows — this device may not keep up. "
            "Try a different input.[/yellow]"
        )
    return 0


async def _one_chunk(data: bytes):
    yield data


# --------------------------------------------------------------------------
# default: run the conversation
# --------------------------------------------------------------------------


async def _cmd_run(cfg: AppConfig, *, live: bool) -> int:
    _preflight(cfg)

    bus = EventBus()
    orchestrator = Orchestrator(cfg, bus=bus)
    orchestrator.build()

    for line in orchestrator.describe():
        console.print(f"[dim]{line}[/dim]")
    console.print()

    loop = asyncio.get_running_loop()
    _install_stop_handler(loop, orchestrator)

    presenter = LiveTranscript(cfg, bus, console=console) if live else PlainLogger(cfg, bus, console=console)
    try:
        with presenter:
            await orchestrator.start()
            await orchestrator.run_until_stopped()
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[dim]stopping...[/dim]")
        await orchestrator.shutdown()

    _print_summary(orchestrator, bus)
    return 0


def _preflight(cfg: AppConfig) -> None:
    """Warn about a room where the microphone can hear the speaker.

    Provider credentials are checked by `Orchestrator.build()`, which covers
    the self-test path too.
    """
    if cfg.mode is Mode.SINGLE_MIC and cfg.duplex.shared_audio:
        channel = cfg.channels[0]
        try:
            shares = audio_devices.same_physical_device(
                audio_devices.resolve(channel.input_device, kind="input"),
                audio_devices.resolve(channel.output_device, kind="output"),
            )
        except audio_devices.AudioDeviceError:
            shares = True
        if shares:
            console.print(
                "[yellow]Heads up:[/yellow] the microphone can probably hear the "
                "speaker. Echo suppression is on, but headphones work much better."
            )


def _install_stop_handler(loop: asyncio.AbstractEventLoop, orchestrator: Orchestrator) -> None:
    """Ctrl-C stops cleanly. Windows has no loop.add_signal_handler."""

    def _handler(signum, frame) -> None:  # noqa: ANN001, ARG001
        loop.call_soon_threadsafe(orchestrator.stop)

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        pass  # not on the main thread; KeyboardInterrupt still works


def _print_summary(orchestrator: Orchestrator, bus: EventBus) -> None:
    history = bus.history
    spoken = [e for e in history if e.stage is Stage.DONE]
    dropped = [e for e in history if e.stage is Stage.DROPPED]
    errors = [e for e in history if e.stage is Stage.ERROR]

    console.print(
        f"\n[bold]{len(spoken)}[/bold] translated, "
        f"{len(dropped)} dropped, {len(errors)} failed"
    )
    if spoken:
        latencies = sorted(e.total_ms for e in spoken)
        median = latencies[len(latencies) // 2]
        console.print(
            f"latency: median {median:.0f}ms, "
            f"best {latencies[0]:.0f}ms, worst {latencies[-1]:.0f}ms"
        )
    for channel_id, stats in orchestrator.stats().items():
        interesting = {k: v for k, v in stats.items() if v}
        if interesting:
            console.print(f"[dim]channel {channel_id}: {interesting}[/dim]")
    for event in errors[-3:]:
        console.print(f"[red]{event.detail}[/red]")
