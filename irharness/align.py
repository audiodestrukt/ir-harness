"""Time alignment and level matching of a response against its stimulus.

Delay is found from the deconvolved sweep: the linear IR's peak is a far
sharper landmark than the raw cross-correlation and it survives heavy
distortion (the fundamental still correlates). Harmonic IRs land *earlier*
than the linear one, so the search is bounded below to keep them out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .stimulus import Stimulus


@dataclass
class Alignment:
    delay: int          # samples the response lags the stimulus (may be negative for a sim that leads)
    polarity: int       # +1 or -1, sign of the linear IR peak
    peak: float         # magnitude of the linear IR peak, in response units
    search_lo: int
    search_hi: int

    def as_dict(self) -> dict:
        return {"delay_samples": self.delay, "polarity": self.polarity, "ir_peak": self.peak,
                "search": [self.search_lo, self.search_hi]}


def find_alignment(stim: Stimulus, y: np.ndarray, max_delay_s: float = 2.0, min_delay_s: float = -0.01) -> Alignment:
    sweep = stim.seg("sweep")
    inv = stim.inverse()
    ir_full = fftconvolve(y, inv)
    # identity: peak at sweep.start + sweep.length - 1
    origin = sweep.start + sweep.length - 1
    lo = origin + int(min_delay_s * stim.fs)
    hi = min(len(ir_full), origin + int(max_delay_s * stim.fs))
    win = ir_full[lo:hi]
    k = int(np.argmax(np.abs(win)))
    val = float(win[k])
    return Alignment(delay=lo + k - origin, polarity=1 if val >= 0 else -1, peak=abs(val),
                     search_lo=lo - origin, search_hi=hi - origin)


def apply_alignment(y: np.ndarray, a: Alignment, n: int, fix_polarity: bool = True) -> np.ndarray:
    """Shift y so that index 0 corresponds to stimulus index 0, cut/pad to n."""
    out = np.zeros(n)
    src_lo = max(0, a.delay)
    dst_lo = max(0, -a.delay)
    m = min(n - dst_lo, len(y) - src_lo)
    if m > 0:
        out[dst_lo : dst_lo + m] = y[src_lo : src_lo + m]
    if fix_polarity and a.polarity < 0:
        out = -out
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def dc_offset(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else 0.0


def level_match_gain(ref: np.ndarray, x: np.ndarray) -> float:
    """Scalar gain that gives x the same RMS as ref (AC-coupled)."""
    r, s = rms(ref - ref.mean()), rms(x - x.mean())
    return r / s if s > 0 else 1.0
