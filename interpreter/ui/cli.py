# the command line stuff

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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="duplex-interpreter",
        description="Hands-free, two-way, real-time speech translation.",
    )
    parser.add_argument("-c", "--config", help="path to config.yaml")
    parser.add_argument(
        "--devices", action="store_true", help="list audio devices and exit"
    )
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
        const=5.0,  # bare --loopback means 5 seconds
        metavar="SECONDS",
        help="pipe the microphone straight to the speaker to check device wiring",
    )
    parser.add_argument(
        "--no-live", action="store_true", help="plain line output instead of the live view"
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="repeat for more logging"
    )
    return parser


def main(argv=None):
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

    # catching these here so the user gets a readable message instead of a
    # wall of traceback. different exit codes so scripts can tell them apart
    except ConfigError as exc:
        console.print(f"[red]Configuration problem:[/red] {exc}")
        return 2
    except audio_devices.AudioDeviceError as exc:
        console.print(f"[red]Audio device problem:[/red] {exc}")
        return 3
    except KeyboardInterrupt:
        console.print("\nstopped")
        return 130


def _force_utf8_output():
    # this is a TRANSLATION app so it prints spanish and greek and japanese
    # by definition, and the old windows console turns all of it into
    # question marks. doesn't hurt anything on mac or linux
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # already redirected somewhere, or not a real terminal


# libraries that will bury the actual conversation if I let them.
# argos logs every single sentence it tokenizes at INFO level
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
    # just setting the level on those loggers is NOT enough. argostranslate
    # calls setLevel(INFO) on its own logger when you import it, which
    # happens after my setup runs, and a logger's own level beats the
    # parent's. filtering at the handler catches it no matter what they do

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def filter(self, record):
        if record.name.startswith(_NOISY_LOGGERS):
            return record.levelno >= self.threshold
        return True


def _configure_logging(verbosity):
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # -vv means you actually want to see the library internals
    if verbosity < 2:
        if verbosity == 0:
            third_party = logging.ERROR
        else:
            third_party = logging.WARNING
        for handler in logging.getLogger().handlers:
            handler.addFilter(_QuietThirdParty(third_party))


# ==========================================================================
# --devices
# ==========================================================================


def _cmd_devices():
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
            if "Input" in kind:
                channels = device.max_input_channels
            else:
                channels = device.max_output_channels

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
        'e.g. [cyan]input_device: "FHD Camera"[/cyan]'
    )
    return 0


# ==========================================================================
# --setup
# ==========================================================================


def _cmd_setup(path):
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
    # round trip it through the loader so I find out NOW if the answers
    # make an invalid config, not later when they try to run it
    cfg = config_module.from_dict(config_module.to_dict(cfg))

    target = Path(path or "config.yaml")
    if target.exists():
        if not Confirm.ask(f"\n{target} exists. Overwrite?", default=False):
            console.print("left unchanged")
            return 0

    config_module.save(cfg, target)
    console.print(f"\n[green]Wrote {target}[/green]")

    uses_openai = "openai" in (providers.stt, providers.translation, providers.tts)
    if uses_openai:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            console.print(
                "\n[yellow]OPENAI_API_KEY is not set.[/yellow] "
                "Copy .env.example to .env and add your key before running."
            )

    console.print("\nNext: [cyan]python run.py --selftest samples/hello_en.wav[/cyan]")
    console.print("Then: [cyan]python run.py[/cyan]")
    return 0


def _ask_mode():
    console.print("How are the two people miked?\n")
    console.print("  [cyan]1[/cyan]  One shared microphone and speaker")
    console.print("     Hands-free, no buttons. The app detects which language was")
    console.print("     spoken and routes it. Works with any laptop.")
    console.print("     [dim]If you both talk at the exact same time it gets")
    console.print("     confused, one mic can't separate two voices.[/dim]\n")
    console.print("  [cyan]2[/cyan]  A headset each (two mics, two earpieces)")
    console.print("     Both can talk at once, no echo, and each person's language")
    console.print("     is pinned so nothing is guessed.\n")

    choice = Prompt.ask("Choose", choices=["1", "2"], default="1")
    if choice == "1":
        return Mode.SINGLE_MIC
    return Mode.DUAL_MIC


def _ask_languages():
    console.print("\n[bold]Languages[/bold]")
    known = ", ".join(sorted(languages.LANGUAGES))
    console.print(f"[dim]{known}[/dim]\n")

    code_a = _ask_language("Language A", "en")
    # don't suggest the same one they just picked
    default_b = "es" if code_a != "es" else "en"

    code_b = _ask_language("Language B", default_b)
    while code_b == code_a:
        console.print("[yellow]The two languages must differ.[/yellow]")
        code_b = _ask_language("Language B", default_b)

    voice_a = languages.default_voice(code_a)
    voice_b = languages.default_voice(code_b, {voice_a})

    return (
        config_module.LanguageSlot(code_a, languages.name_of(code_a), voice_a),
        config_module.LanguageSlot(code_b, languages.name_of(code_b), voice_b),
    )


def _ask_language(label, default):
    while True:
        value = Prompt.ask(label, default=default).strip().lower()
        if languages.is_known(value):
            return languages.get(value).code
        console.print(f"[yellow]{value!r} is not in the language table.[/yellow]")


def _ask_channels(mode, language_a, language_b):
    try:
        inputs = audio_devices.input_devices()
        outputs = audio_devices.output_devices()
    except audio_devices.AudioDeviceError as exc:
        console.print(
            f"[yellow]Could not list audio devices ({exc}); using defaults.[/yellow]"
        )
        inputs = outputs = []

    # no devices to pick from, just use the system defaults
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

    if channel_a.input_device is not None:
        if channel_a.input_device == channel_b.input_device:
            console.print(
                "\n[yellow]Both channels use the same microphone. Each person "
                "needs their own, or both pipelines will hear both speakers.[/yellow]"
            )

    return [channel_a, channel_b]


def _ask_device(label, options):
    valid = {str(d.index) for d in options}
    while True:
        answer = Prompt.ask(
            f"{label} (number, or blank for default)", default=""
        ).strip()
        if not answer:
            return None
        if answer in valid:
            return int(answer)
        console.print(f"[yellow]{answer!r} is not one of the numbers listed.[/yellow]")


def _ask_providers():
    console.print("\n[bold]Translation engine[/bold]\n")
    console.print("  [cyan]1[/cyan]  OpenAI  - one API key, ~0.8s, roughly $0.03/minute")
    console.print("  [cyan]2[/cyan]  Local   - free and offline, ~1-2s, big downloads")
    console.print("  [cyan]3[/cyan]  Mixed   - local speech recognition, OpenAI translation + voice\n")

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


# ==========================================================================
# --selftest
# ==========================================================================


async def _cmd_selftest(cfg, wav_path, *, realtime=False):
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
    if local_stt:
        stt_model = providers.local_whisper_size
    else:
        stt_model = providers.stt_model

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
    speakers = {}

    def make_speaker(channel):
        speaker = RecordingSpeaker(channel_id=channel.id)
        speakers[channel.id] = speaker
        return speaker

    def make_microphone(channel):
        # the whole file goes to the first channel, a wav only has one
        # person in it. the other channel gets silence
        if channel.id == cfg.channels[0].id:
            audio = pcm
        else:
            audio = np.zeros(0, dtype=np.int16)
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
            "[yellow]The VAD never triggered.[/yellow] Either the file has no "
            "speech, or the threshold is too high. Try `vad.threshold: 0.3`."
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

    # write out the audio so you can actually listen to it
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

        console.print(
            f"[green]wrote {out_path}[/green] "
            f"({audio.size / speaker.sample_rate:.1f}s)"
        )
        written += 1

    if errors:
        return 6
    return 0 if written else 5


# ==========================================================================
# --loopback
# ==========================================================================

# 60ms. you can't hear it but it stops the speaker from starving every
# time the scheduler hiccups
_LOOPBACK_PREBUFFER_FRAMES = 3


async def _cmd_loopback(cfg, seconds):
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
        "[yellow]Wear headphones, speakers will feed back.[/yellow]\n"
        "Talk; you should hear yourself."
    )

    async def _pump():
        # ONE play() for the whole run, not one per frame.
        #
        # I had this calling play() on every 20ms frame and it was awful.
        # play() finishes by draining the buffer all the way down and
        # waiting for the device, which takes longer than the 20ms of audio
        # you just gave it, so it ran at about a third of real time and the
        # mic queue filled up and started dropping. one call also means one
        # resampler instead of a new one per frame, which is what was
        # making it click at every boundary
        async def _stream():
            prebuffer = []
            async for frame in microphone.frames():
                if len(prebuffer) < _LOOPBACK_PREBUFFER_FRAMES:
                    prebuffer.append(frame.tobytes())
                    continue
                if prebuffer:
                    yield b"".join(prebuffer)
                    prebuffer.clear()
                yield frame.tobytes()
            if prebuffer:
                yield b"".join(prebuffer)  # stopped before it even filled

        await speaker.play(_stream(), source_rate=PIPELINE_SAMPLE_RATE)

    task = asyncio.create_task(_pump())
    try:
        # shield so the timeout doesn't cancel it half way through a write,
        # I cancel it myself below
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
            "[yellow]Frequent overflows - this device may not keep up. "
            "Try a different input.[/yellow]"
        )
    return 0


# ==========================================================================
# the normal thing: run the conversation
# ==========================================================================


async def _cmd_run(cfg, *, live):
    _preflight(cfg)

    bus = EventBus()
    orchestrator = Orchestrator(cfg, bus=bus)
    orchestrator.build()

    for line in orchestrator.describe():
        console.print(f"[dim]{line}[/dim]")
    console.print()

    loop = asyncio.get_running_loop()
    _install_stop_handler(loop, orchestrator)

    if live:
        presenter = LiveTranscript(cfg, bus, console=console)
    else:
        presenter = PlainLogger(cfg, bus, console=console)

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


def _preflight(cfg):
    # warn if the mic can probably hear the speaker.
    # the api keys get checked by Orchestrator.build() instead, so that the
    # selftest path gets checked too
    if cfg.mode is Mode.SINGLE_MIC and cfg.duplex.shared_audio:
        channel = cfg.channels[0]
        try:
            shares = audio_devices.same_physical_device(
                audio_devices.resolve(channel.input_device, kind="input"),
                audio_devices.resolve(channel.output_device, kind="output"),
            )
        except audio_devices.AudioDeviceError:
            shares = True  # couldn't tell, assume the worse case

        if shares:
            console.print(
                "[yellow]Heads up:[/yellow] the microphone can probably hear the "
                "speaker. Echo suppression is on, but headphones work much better."
            )


def _install_stop_handler(loop, orchestrator):
    # ctrl-c should stop it cleanly.
    # windows doesn't have loop.add_signal_handler so I use the plain
    # signal module and bounce back onto the loop
    def _handler(signum, frame):
        loop.call_soon_threadsafe(orchestrator.stop)

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        pass  # not the main thread. KeyboardInterrupt still works anyway


def _print_summary(orchestrator, bus):
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
        # only show the counters that aren't zero, otherwise its noise
        interesting = {k: v for k, v in stats.items() if v}
        if interesting:
            console.print(f"[dim]channel {channel_id}: {interesting}[/dim]")

    for event in errors[-3:]:
        console.print(f"[red]{event.detail}[/red]")
