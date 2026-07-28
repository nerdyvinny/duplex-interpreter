"""Enumerating and resolving audio devices.

`sounddevice` is imported lazily so the config layer, the VAD and the whole
test suite work on a machine with no PortAudio at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class AudioDeviceError(RuntimeError):
    """No such device, or PortAudio is unavailable."""


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi_name: str

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    def __str__(self) -> str:
        kind = []
        if self.is_input:
            kind.append(f"in:{self.max_input_channels}")
        if self.is_output:
            kind.append(f"out:{self.max_output_channels}")
        return f"[{self.index}] {self.name} ({self.hostapi_name}, {', '.join(kind)})"


def _sounddevice() -> Any:
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio shared library missing
        raise AudioDeviceError(
            "PortAudio could not be loaded. On Linux: `apt install libportaudio2`. "
            "On Windows/macOS reinstall with `pip install --force-reinstall sounddevice`."
        ) from exc
    except ImportError as exc:
        raise AudioDeviceError(
            "sounddevice is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return sd


def list_devices() -> list[DeviceInfo]:
    sd = _sounddevice()
    hostapis = sd.query_hostapis()
    devices = []
    for index, raw in enumerate(sd.query_devices()):
        hostapi = raw.get("hostapi", 0)
        devices.append(
            DeviceInfo(
                index=index,
                name=str(raw["name"]).strip(),
                max_input_channels=int(raw["max_input_channels"]),
                max_output_channels=int(raw["max_output_channels"]),
                default_samplerate=float(raw["default_samplerate"]),
                hostapi_name=str(hostapis[hostapi]["name"]) if hostapi < len(hostapis) else "?",
            )
        )
    return devices


def input_devices() -> list[DeviceInfo]:
    return [d for d in list_devices() if d.is_input]


def output_devices() -> list[DeviceInfo]:
    return [d for d in list_devices() if d.is_output]


def default_devices() -> tuple[int | None, int | None]:
    sd = _sounddevice()
    try:
        raw_in, raw_out = sd.default.device
    except (TypeError, ValueError):
        return None, None
    to_index = lambda v: v if isinstance(v, int) and v >= 0 else None  # noqa: E731
    return to_index(raw_in), to_index(raw_out)


def resolve(spec: str | int | None, *, kind: str) -> int | None:
    """Turn a config value into a PortAudio device index.

    Accepts an index, an exact name, or a case-insensitive substring. `None`
    means "let PortAudio pick the system default".
    """
    if spec is None:
        return None
    if kind not in {"input", "output"}:
        raise ValueError(f"kind must be 'input' or 'output', got {kind!r}")

    candidates = input_devices() if kind == "input" else output_devices()
    if not candidates:
        raise AudioDeviceError(f"no {kind} devices found on this system")

    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().lstrip("-").isdigit()):
        index = int(spec)
        match = next((d for d in candidates if d.index == index), None)
        if match is None:
            raise AudioDeviceError(
                f"device index {index} is not a usable {kind} device.\n"
                + _format_options(candidates)
            )
        return match.index

    needle = str(spec).strip().lower()
    exact = [d for d in candidates if d.name.lower() == needle]
    if exact:
        return exact[0].index

    partial = [d for d in candidates if needle in d.name.lower()]
    if len(partial) == 1:
        return partial[0].index
    if len(partial) > 1:
        # Prefer the host API that behaves best per platform: WASAPI on
        # Windows gives lower latency and correct device names.
        for preferred in ("Windows WASAPI", "Core Audio", "ALSA"):
            narrowed = [d for d in partial if d.hostapi_name == preferred]
            if len(narrowed) == 1:
                return narrowed[0].index
            if narrowed:
                return narrowed[0].index
        return partial[0].index

    raise AudioDeviceError(
        f"no {kind} device matches {spec!r}.\n" + _format_options(candidates)
    )


def _format_options(candidates: list[DeviceInfo]) -> str:
    listing = "\n".join(f"  {d}" for d in candidates)
    return f"Available:\n{listing}"


def describe(index: int | None, kind: str) -> str:
    if index is None:
        return "system default"
    try:
        match = next(d for d in list_devices() if d.index == index)
    except (StopIteration, AudioDeviceError):
        return f"device {index}"
    return f"[{index}] {match.name}"


def same_physical_device(input_index: int | None, output_index: int | None) -> bool:
    """Heuristic: is this mic likely to hear this speaker?

    Used to decide whether echo suppression is needed. Being wrong in the
    cautious direction (assuming shared) only costs a little half-duplexing,
    so the heuristic leans that way for laptop-style built-in devices.
    """
    if input_index is None or output_index is None:
        return True  # system defaults on a laptop almost always share a body
    try:
        devices = {d.index: d for d in list_devices()}
    except AudioDeviceError:
        return True
    src, dst = devices.get(input_index), devices.get(output_index)
    if src is None or dst is None:
        return True

    def normalize(name: str) -> set[str]:
        drop = {"microphone", "speakers", "headphones", "input", "output", "device", "audio"}
        words = "".join(c if c.isalnum() else " " for c in name.lower()).split()
        return {w for w in words if w not in drop and len(w) > 2}

    src_words, dst_words = normalize(src.name), normalize(dst.name)
    return bool(src_words & dst_words)
