"""Sample-rate conversion between devices, the 16 kHz pipeline, and TTS output.

Uses `soxr` when available (fast, high quality); falls back to numpy linear
interpolation so nothing hard-fails on an exotic platform.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    import soxr

    _HAVE_SOXR = True
except ImportError:  # pragma: no cover - exercised only on platforms without soxr
    _HAVE_SOXR = False
    log.debug("soxr unavailable, using linear-interpolation resampling")


def resample_int16(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono int16 PCM. Returns int16."""
    if src_rate == dst_rate or pcm.size == 0:
        return pcm

    if _HAVE_SOXR:
        return soxr.resample(pcm, src_rate, dst_rate, quality="VHQ").astype(np.int16)

    duration = pcm.size / src_rate
    out_len = max(1, int(round(duration * dst_rate)))
    src_positions = np.arange(pcm.size, dtype=np.float64)
    dst_positions = np.linspace(0, pcm.size - 1, out_len, dtype=np.float64)
    return (
        np.interp(dst_positions, src_positions, pcm.astype(np.float64))
        .round()
        .clip(-32768, 32767)
        .astype(np.int16)
    )


def resample_bytes(raw: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return raw
    pcm = np.frombuffer(raw, dtype=np.int16)
    return resample_int16(pcm, src_rate, dst_rate).tobytes()


def downmix_to_mono(pcm: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved channels down to mono int16."""
    if channels <= 1:
        return pcm
    usable = (pcm.size // channels) * channels
    frames = pcm[:usable].reshape(-1, channels).astype(np.int32)
    return frames.mean(axis=1).round().clip(-32768, 32767).astype(np.int16)


def rms(pcm: np.ndarray) -> float:
    """Normalized RMS in 0..1, used for barge-in detection and level meters."""
    if pcm.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(pcm.astype(np.float64) / 32768.0))))


class StreamResampler:
    """Stateful resampler for a continuous stream.

    Chunk-at-a-time `resample_int16` produces clicks at chunk boundaries.
    soxr's streaming API carries filter state across calls; this wraps it and
    degrades to the stateless path when soxr is missing.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._stream = (
            soxr.ResampleStream(src_rate, dst_rate, 1, dtype="int16", quality="VHQ")
            if _HAVE_SOXR and src_rate != dst_rate
            else None
        )

    @property
    def passthrough(self) -> bool:
        return self.src_rate == self.dst_rate

    def process(self, pcm: np.ndarray, *, last: bool = False) -> np.ndarray:
        if self.passthrough or pcm.size == 0 and not last:
            return pcm
        if self._stream is not None:
            return self._stream.resample_chunk(pcm, last=last).astype(np.int16)
        return resample_int16(pcm, self.src_rate, self.dst_rate)
