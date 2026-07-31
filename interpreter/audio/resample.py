# changing sample rates around
#
# the mic might be at 48000, my pipeline runs at 16000, and the TTS comes
# back at 24000, so there is a lot of converting going on. uses soxr if you
# have it because its fast and sounds good, otherwise falls back to numpy

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    import soxr
    _HAVE_SOXR = True
except ImportError:
    _HAVE_SOXR = False
    log.debug("soxr unavailable, using linear-interpolation resampling")


def resample_int16(pcm, src_rate, dst_rate):
    if src_rate == dst_rate or pcm.size == 0:
        return pcm  # nothing to do

    if _HAVE_SOXR:
        return soxr.resample(pcm, src_rate, dst_rate, quality="VHQ").astype(np.int16)

    # the backup version. basically just draws a line between the samples
    # and picks new points off it. not great but better than nothing
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


def resample_bytes(raw, src_rate, dst_rate):
    if src_rate == dst_rate:
        return raw
    pcm = np.frombuffer(raw, dtype=np.int16)
    return resample_int16(pcm, src_rate, dst_rate).tobytes()


def downmix_to_mono(pcm, channels):
    # stereo comes in as LRLRLR so reshape it into pairs and average them
    if channels <= 1:
        return pcm
    # chop off any leftover samples that dont make a full frame
    usable = (pcm.size // channels) * channels
    frames = pcm[:usable].reshape(-1, channels).astype(np.int32)
    return frames.mean(axis=1).round().clip(-32768, 32767).astype(np.int16)


def rms(pcm):
    # how loud is this, 0 to 1. used for the barge in check and the meters
    if pcm.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(pcm.astype(np.float64) / 32768.0))))


class StreamResampler:
    # for a continuous stream instead of one chunk.
    #
    # I originally just called resample_int16 on every chunk and it made a
    # clicking noise at every boundary. soxr has a streaming version that
    # remembers the filter state between calls which fixes it

    def __init__(self, src_rate, dst_rate):
        self.src_rate = src_rate
        self.dst_rate = dst_rate

        if _HAVE_SOXR and src_rate != dst_rate:
            self._stream = soxr.ResampleStream(
                src_rate, dst_rate, 1, dtype="int16", quality="VHQ"
            )
        else:
            self._stream = None

    @property
    def passthrough(self):
        return self.src_rate == self.dst_rate

    def process(self, pcm, *, last=False):
        # "last" tells soxr to spit out whatever its still holding onto
        if self.passthrough or (pcm.size == 0 and not last):
            return pcm
        if self._stream is not None:
            return self._stream.resample_chunk(pcm, last=last).astype(np.int16)
        return resample_int16(pcm, self.src_rate, self.dst_rate)
