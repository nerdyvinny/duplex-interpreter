# finding and picking audio devices
#
# sounddevice is imported inside the functions instead of at the top,
# because importing it needs the portaudio library installed and I want the
# config code and all the tests to work on a machine without it

from dataclasses import dataclass


class AudioDeviceError(RuntimeError):
    # no such device, or portaudio isn't there
    pass


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi_name: str

    @property
    def is_input(self):
        return self.max_input_channels > 0

    @property
    def is_output(self):
        return self.max_output_channels > 0

    def __str__(self):
        kind = []
        if self.is_input:
            kind.append(f"in:{self.max_input_channels}")
        if self.is_output:
            kind.append(f"out:{self.max_output_channels}")
        return f"[{self.index}] {self.name} ({self.hostapi_name}, {', '.join(kind)})"


def _sounddevice():
    try:
        import sounddevice as sd
    except OSError as exc:
        # this one means the python package is fine but the actual
        # portaudio .dll/.so is missing
        raise AudioDeviceError(
            "PortAudio could not be loaded. On Linux: `apt install libportaudio2`. "
            "On Windows/macOS reinstall with `pip install --force-reinstall sounddevice`."
        ) from exc
    except ImportError as exc:
        raise AudioDeviceError(
            "sounddevice is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return sd


def list_devices():
    sd = _sounddevice()
    hostapis = sd.query_hostapis()

    devices = []
    for index, raw in enumerate(sd.query_devices()):
        hostapi = raw.get("hostapi", 0)
        if hostapi < len(hostapis):
            api_name = str(hostapis[hostapi]["name"])
        else:
            api_name = "?"

        devices.append(
            DeviceInfo(
                index=index,
                name=str(raw["name"]).strip(),
                max_input_channels=int(raw["max_input_channels"]),
                max_output_channels=int(raw["max_output_channels"]),
                default_samplerate=float(raw["default_samplerate"]),
                hostapi_name=api_name,
            )
        )
    return devices


def input_devices():
    return [d for d in list_devices() if d.is_input]


def output_devices():
    return [d for d in list_devices() if d.is_output]


def default_devices():
    sd = _sounddevice()
    try:
        raw_in, raw_out = sd.default.device
    except (TypeError, ValueError):
        return None, None

    def to_index(v):
        if isinstance(v, int) and v >= 0:
            return v
        return None

    return to_index(raw_in), to_index(raw_out)


def resolve(spec, *, kind):
    # you can put a number, the exact name, or just part of the name in
    # config.yaml. None means let the system pick
    if spec is None:
        return None
    if kind not in {"input", "output"}:
        raise ValueError(f"kind must be 'input' or 'output', got {kind!r}")

    if kind == "input":
        candidates = input_devices()
    else:
        candidates = output_devices()

    if not candidates:
        raise AudioDeviceError(f"no {kind} devices found on this system")

    # a number, or a string that looks like a number
    if isinstance(spec, int) or (
        isinstance(spec, str) and spec.strip().lstrip("-").isdigit()
    ):
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
        # the same physical device shows up several times under different
        # host apis. WASAPI is the good one on windows, its lower latency
        # and it actually gives you the full device name
        for preferred in ("Windows WASAPI", "Core Audio", "ALSA"):
            narrowed = [d for d in partial if d.hostapi_name == preferred]
            if narrowed:
                return narrowed[0].index
        return partial[0].index

    raise AudioDeviceError(
        f"no {kind} device matches {spec!r}.\n" + _format_options(candidates)
    )


def _format_options(candidates):
    listing = "\n".join(f"  {d}" for d in candidates)
    return f"Available:\n{listing}"


def describe(index, kind):
    if index is None:
        return "system default"
    try:
        match = next(d for d in list_devices() if d.index == index)
    except (StopIteration, AudioDeviceError):
        return f"device {index}"
    return f"[{index}] {match.name}"


def same_physical_device(input_index, output_index):
    # rough guess at whether this mic can hear this speaker, so I know
    # whether to turn the echo stuff on.
    #
    # I compare the device names and see if they share any words. its not
    # exactly scientific but guessing "yes they share" when they don't only
    # costs a bit of latency, and guessing "no" when they do gives you the
    # infinite echo loop, so I lean towards yes

    if input_index is None or output_index is None:
        return True  # the defaults on a laptop are almost always one body

    try:
        devices = {d.index: d for d in list_devices()}
    except AudioDeviceError:
        return True

    src = devices.get(input_index)
    dst = devices.get(output_index)
    if src is None or dst is None:
        return True

    def normalize(name):
        # throw out the generic words, otherwise every single device
        # matches every other one on the word "audio"
        drop = {
            "microphone", "speakers", "headphones",
            "input", "output", "device", "audio",
        }
        words = "".join(c if c.isalnum() else " " for c in name.lower()).split()
        return {w for w in words if w not in drop and len(w) > 2}

    src_words = normalize(src.name)
    dst_words = normalize(dst.name)
    return bool(src_words & dst_words)
