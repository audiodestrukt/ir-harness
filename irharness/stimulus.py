"""Stimulus generation.

A Stimulus is one long signal made of named segments (silence, sync burst,
exponential sine sweep, stepped tones, a DI clip). The segment table travels
with the signal so analysis can cut the *response* at the same offsets once it
has been aligned to the stimulus.

Units: the signal is nominally full scale, |x| <= 1. What a full-scale sample
means in volts is the DUT adapter's business (see dut/sim.py volts_per_fs).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from fractions import Fraction


def db(x: float) -> float:
    return 10.0 ** (x / 20.0)


def raised_cosine_fade(x: np.ndarray, fs: int, ms_in: float, ms_out: float) -> np.ndarray:
    x = x.copy()
    n_in = int(round(ms_in * 1e-3 * fs))
    n_out = int(round(ms_out * 1e-3 * fs))
    if n_in > 0:
        w = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_in) / n_in)
        x[:n_in] *= w
    if n_out > 0:
        w = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_out) / n_out)
        x[-n_out:] *= w[::-1]
    return x


# ------------------------------------------------------------------ sweeps
def synchronized_ess(fs: int, f1: float, f2: float, seconds: float) -> tuple[np.ndarray, float]:
    """Exponential sine sweep, Novak's synchronized form.

    L is chosen so that f1 * L is an integer; then every harmonic's impulse
    response comes out of the deconvolution with a well-defined phase, which
    is what lets the harmonic IRs be reused as a nonlinear model later.
    Returns (sweep, L). The actual length is L * ln(f2/f1), close to `seconds`.
    """
    ratio = np.log(f2 / f1)
    L = round(f1 * seconds / ratio) / f1
    n = int(round(L * ratio * fs))
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f1 * L * (np.exp(t / L) - 1.0))
    return x, L


def ess_inverse(fs: int, f1: float, f2: float, L: float, n: int, level_db: float = 0.0) -> np.ndarray:
    """Inverse filter for the sweep: time-reversed sweep with a -6 dB/oct envelope,
    scaled so that an identity DUT yields a linear IR whose passband magnitude
    is 0 dB (i.e. the IR is in units of output per full-scale input)."""
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f1 * L * (np.exp(t / L) - 1.0))
    inv = x[::-1] * np.exp(-t / L)
    ref = fftconvolve(x, inv)
    nfft = 1 << int(np.ceil(np.log2(len(ref))))
    R = np.abs(np.fft.rfft(ref, nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = (f >= 2 * f1) & (f <= f2 / 2)
    inv /= np.median(R[band]) * db(level_db)
    return inv


# ---------------------------------------------------------------- segments
@dataclass
class Segment:
    name: str
    kind: str  # silence | sync | sweep | tone | clip
    start: int
    length: int
    meta: dict = field(default_factory=dict)

    @property
    def stop(self) -> int:
        return self.start + self.length


@dataclass
class Stimulus:
    fs: int
    signal: np.ndarray
    segments: list[Segment]
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.signal)

    @property
    def seconds(self) -> float:
        return len(self.signal) / self.fs

    def seg(self, name: str) -> Segment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(name)

    def segs(self, kind: str) -> list[Segment]:
        return [s for s in self.segments if s.kind == kind]

    def cut(self, x: np.ndarray, name: str) -> np.ndarray:
        s = self.seg(name)
        return x[s.start : s.stop]

    def inverse(self) -> np.ndarray:
        s = self.seg("sweep")
        return ess_inverse(self.fs, s.meta["f1"], s.meta["f2"], s.meta["L"], s.length, s.meta.get("level_db", 0.0))

    # persistence -----------------------------------------------------------
    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / "stimulus.wav", self.signal.astype(np.float32), self.fs, subtype="FLOAT")
        meta = {
            "fs": self.fs,
            "length": len(self.signal),
            "params": self.params,
            "segments": [
                {"name": s.name, "kind": s.kind, "start": s.start, "length": s.length, "meta": s.meta}
                for s in self.segments
            ],
        }
        (d / "stimulus.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, d: Path) -> "Stimulus":
        d = Path(d)
        meta = json.loads((d / "stimulus.json").read_text())
        x, fs = sf.read(d / "stimulus.wav", dtype="float64", always_2d=False)
        if fs != meta["fs"]:
            raise ValueError(f"{d}: wav fs {fs} != json fs {meta['fs']}")
        segs = [Segment(s["name"], s["kind"], s["start"], s["length"], s.get("meta", {})) for s in meta["segments"]]
        return cls(fs=fs, signal=x, segments=segs, params=meta.get("params", {}))


# ------------------------------------------------------------------- build
def load_clip(path: Path, fs: int) -> np.ndarray:
    x, fs_in = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs_in != fs:
        r = Fraction(fs, fs_in).limit_denominator(1000)
        x = resample_poly(x, r.numerator, r.denominator)
    return x


def build(
    fs: int = 192000,
    sweep_seconds: float = 10.0,
    f1: float = 20.0,
    f2: float = 20000.0,
    sweep_db: float = -12.0,
    tone_freqs: tuple[float, ...] = (82.0, 220.0, 440.0, 1000.0, 3000.0),
    tone_dbs: tuple[float, ...] = (-40.0, -32.0, -24.0, -16.0, -8.0, 0.0),
    tone_seconds: float = 0.25,
    gap_seconds: float = 0.1,
    di: Path | None = None,
    di_db: float = -6.0,
    di_max_seconds: float | None = 20.0,
    pre_seconds: float = 0.5,
    tail_seconds: float = 1.0,
    post_seconds: float = 0.5,
) -> Stimulus:
    parts: list[np.ndarray] = []
    segs: list[Segment] = []
    pos = 0

    def add(name: str, kind: str, x: np.ndarray, meta: dict | None = None) -> None:
        nonlocal pos
        parts.append(x)
        segs.append(Segment(name, kind, pos, len(x), meta or {}))
        pos += len(x)

    def silence(name: str, seconds: float) -> None:
        add(name, "silence", np.zeros(int(round(seconds * fs))))

    silence("pre", pre_seconds)

    # sync burst: 20 ms linear chirp, a coarse landmark for eyeballing alignment
    n = int(0.02 * fs)
    t = np.arange(n) / fs
    sync = np.sin(2 * np.pi * (500 * t + 0.5 * (5000 - 500) / (n / fs) * t * t))
    add("sync", "sync", raised_cosine_fade(sync, fs, 2, 2) * db(-20), {"level_db": -20})
    silence("sync_gap", 0.2)

    sweep, L = synchronized_ess(fs, f1, f2, sweep_seconds)
    fade_in_ms = 2000.0 / f1  # two cycles of f1
    sweep = raised_cosine_fade(sweep, fs, fade_in_ms, 5.0) * db(sweep_db)
    add("sweep", "sweep", sweep, {"f1": f1, "f2": f2, "L": L, "level_db": sweep_db, "seconds": len(sweep) / fs})
    silence("tail", tail_seconds)

    for lvl in tone_dbs:
        for f in tone_freqs:
            n = int(round(tone_seconds * fs))
            t = np.arange(n) / fs
            tone = raised_cosine_fade(np.sin(2 * np.pi * f * t), fs, 10, 10) * db(lvl)
            add(f"tone_{f:g}Hz_{lvl:g}dB", "tone", tone, {"freq": f, "level_db": lvl})
            silence(f"gap_{f:g}Hz_{lvl:g}dB", gap_seconds)

    if di is not None:
        clip = load_clip(Path(di), fs)
        if di_max_seconds is not None:
            clip = clip[: int(di_max_seconds * fs)]
        peak = np.max(np.abs(clip)) or 1.0
        clip = raised_cosine_fade(clip / peak * db(di_db), fs, 5, 20)
        add("clip", "clip", clip, {"source": str(di), "level_db": di_db})
        silence("clip_tail", tail_seconds)

    silence("post", post_seconds)

    params = {
        "fs": fs, "sweep_seconds": sweep_seconds, "f1": f1, "f2": f2, "sweep_db": sweep_db,
        "tone_freqs": list(tone_freqs), "tone_dbs": list(tone_dbs), "tone_seconds": tone_seconds,
        "gap_seconds": gap_seconds, "di": str(di) if di else None, "di_db": di_db,
        "pre_seconds": pre_seconds, "tail_seconds": tail_seconds, "post_seconds": post_seconds,
    }
    return Stimulus(fs=fs, signal=np.concatenate(parts), segments=segs, params=params)
