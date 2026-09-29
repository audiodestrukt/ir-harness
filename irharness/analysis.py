"""Measurements on an aligned response.

  linear + harmonic IRs   from the sweep (Farina deconvolution, synchronized sweep)
  frequency response      of the linear IR, fractional-octave smoothed
  harmonic response       |H_k(k f)| / |H_1(f)|: distortion order k vs input frequency
  tone metrics            THD, harmonic levels, output RMS, DC per stepped tone
  clip metrics            residual between two responses to the same DI clip
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from .align import rms, dc_offset
from .stimulus import Stimulus


# ------------------------------------------------------------ deconvolution
@dataclass
class IRSet:
    fs: int
    L: float
    pre: int                       # samples of pre-ring kept before each IR's nominal t=0
    ir: dict[int, np.ndarray] = field(default_factory=dict)   # order -> windowed IR

    @property
    def linear(self) -> np.ndarray:
        return self.ir[1]


def deconvolve(stim: Stimulus, y: np.ndarray, sweep=None) -> tuple[np.ndarray, int]:
    """Convolve the aligned response with the inverse sweep. Returns
    (ir_full, origin) where ir_full[origin] is the linear IR's t=0."""
    sweep = sweep or stim.ref_sweep()
    inv = stim.inverse(sweep)
    ir_full = fftconvolve(y, inv)
    origin = sweep.start + sweep.length - 1
    return ir_full, origin


def extract_harmonics(ir_full: np.ndarray, origin: int, stim: Stimulus, max_order: int = 5,
                      ir_seconds: float = 0.2, pre_ms: float = 50.0, sweep=None) -> IRSet:
    """Window each order's IR out of the deconvolution. The pre-window matters:
    band-limiting at f1 rings symmetrically around t=0 for ~1/f1 seconds, and
    cutting that off costs low-frequency accuracy. Every order shares one `pre`
    so the IRs stay time-aligned to each other."""
    sweep = sweep or stim.ref_sweep()
    L, fs = sweep.meta["L"], stim.fs
    # the next order above k lands L*ln((k+1)/k) earlier; keep clear of it
    tightest = int(L * np.log((max_order + 1) / max_order) * fs)
    pre = min(int(pre_ms * 1e-3 * fs), int(0.3 * tightest))
    out = IRSet(fs=fs, L=L, pre=pre)
    for k in range(1, max_order + 1):
        t0 = origin - int(round(L * np.log(k) * fs))
        spacing = int(L * np.log((k + 1) / k) * fs)
        length = min(int(ir_seconds * fs), int(0.5 * spacing) - pre)
        lo, hi = t0 - pre, t0 + length
        seg = np.zeros(pre + length)
        a, b = max(0, lo), min(len(ir_full), hi)
        if b > a:
            seg[a - lo : a - lo + (b - a)] = ir_full[a:b]
        out.ir[k] = seg
    return out


# ------------------------------------------------------ frequency responses
def spectrum(h: np.ndarray, fs: int, nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    nfft = nfft or max(1 << 16, 1 << int(np.ceil(np.log2(len(h)))))
    H = np.fft.rfft(h, nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f, H


def log_grid(f_lo: float, f_hi: float, per_octave: int = 96) -> np.ndarray:
    n = int(np.ceil(np.log2(f_hi / f_lo) * per_octave)) + 1
    return f_lo * 2 ** (np.arange(n) / per_octave)


def smoothed_response(f: np.ndarray, H: np.ndarray, f_lo: float, f_hi: float,
                      frac: int = 12, per_octave: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude (dB) and unwrapped phase (rad) on a log grid, magnitude
    power-averaged over 1/frac octave."""
    g = log_grid(f_lo, min(f_hi, f[-1]), per_octave)
    mag = np.interp(g, f, np.abs(H) ** 2)
    w = max(1, int(round(per_octave / frac)))
    kern = np.ones(w) / w
    sm = np.convolve(np.pad(mag, w // 2, mode="edge"), kern, mode="same")[w // 2 : w // 2 + len(mag)]
    ph = np.interp(g, f, np.unwrap(np.angle(H)))
    return g, 10 * np.log10(np.maximum(sm, 1e-30)), ph


def harmonic_response(irs: IRSet, f1: float, f2: float, frac: int = 6) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """For each order k >= 2: input-frequency grid and level of the k-th
    harmonic relative to the fundamental, in dB. Output grid runs f1 .. f2/k."""
    f, H1 = spectrum(irs.ir[1], irs.fs)
    _, m1, _ = smoothed_response(f, H1, f1, f2, frac)
    g1 = log_grid(f1, min(f2, f[-1]))
    out = {}
    for k, h in irs.ir.items():
        if k == 1:
            continue
        fk, Hk = spectrum(h, irs.fs)
        hi = min(f2, irs.fs / 2 / k)
        if hi <= f1:
            continue
        gk, mk, _ = smoothed_response(fk, Hk, f1 * k, hi * k, frac)   # output-frequency grid k*f_in
        fin = gk / k
        fund = np.interp(fin, g1, m1)
        out[k] = (fin, mk - fund)
    return out


# ------------------------------------------------------------ stepped tones
def tone_metrics(stim: Stimulus, y: np.ndarray, max_order: int = 10, trim_ms: float = 30.0) -> list[dict]:
    fs = stim.fs
    rows = []
    for s in stim.segs("tone"):
        f, lvl = s.meta["freq"], s.meta["level_db"]
        seg = y[s.start : s.stop]
        trim = int(trim_ms * 1e-3 * fs)
        seg = seg[trim : len(seg) - trim]
        cycles = int(len(seg) * f / fs)
        n = int(round(cycles * fs / f))
        seg = seg[:n]
        t = np.arange(n) / fs
        amps = []
        for k in range(1, max_order + 1):
            if k * f >= fs / 2:
                break
            amps.append(2 * abs(np.mean(seg * np.exp(-2j * np.pi * k * f * t))))
        amps = np.array(amps)
        fund = amps[0] if len(amps) else 0.0
        thd = float(np.sqrt(np.sum(amps[1:] ** 2)) / fund) if fund > 0 else float("nan")
        rows.append({
            "name": s.name, "freq": f, "level_db": lvl,
            "in_rms_db": 20 * np.log10(rms(stim.signal[s.start : s.stop][trim:-trim]) + 1e-30),
            "out_rms": rms(seg), "out_rms_db": 20 * np.log10(rms(seg) + 1e-30),
            "dc": dc_offset(seg), "fundamental": float(fund),
            "thd": thd, "thd_db": 20 * np.log10(thd + 1e-30) if thd == thd else float("nan"),
            "harmonics_db": [float(20 * np.log10(a / fund + 1e-30)) for a in amps[1:]] if fund > 0 else [],
        })
    return rows


# --------------------------------------------------------------- residuals
def residual_metrics(a: np.ndarray, b: np.ndarray, fs: int, bands_per_octave: int = 3,
                     f_lo: float = 20.0, f_hi: float | None = None) -> dict:
    """How far b is from a: ESR (NAM's error-to-signal ratio), null depth,
    and the per-band level difference."""
    a = a - a.mean()
    b = b - b.mean()
    g = float(np.dot(a, b) / np.dot(b, b)) if np.dot(b, b) > 0 else 1.0   # least-squares gain
    e = a - g * b
    esr = float(np.sum(e * e) / np.sum(a * a)) if np.sum(a * a) > 0 else float("nan")
    f_hi = f_hi or fs / 2
    A, B = np.fft.rfft(a), np.fft.rfft(b * g)
    f = np.fft.rfftfreq(len(a), 1 / fs)
    edges = f_lo * 2 ** (np.arange(int(np.log2(f_hi / f_lo) * bands_per_octave) + 1) / bands_per_octave)
    band_db, centers = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if not m.any():
            continue
        pa, pb = np.sum(np.abs(A[m]) ** 2), np.sum(np.abs(B[m]) ** 2)
        band_db.append(10 * np.log10((pb + 1e-30) / (pa + 1e-30)))
        centers.append(float(np.sqrt(lo * hi)))
    return {"ls_gain": g, "ls_gain_db": 20 * np.log10(abs(g) + 1e-30), "esr": esr,
            "null_depth_db": 10 * np.log10(esr + 1e-30) if esr == esr else float("nan"),
            "band_centers": centers, "band_diff_db": [float(x) for x in band_db]}


# ------------------------------------------------------------------ spikes
def spike_metrics(stim: Stimulus, y: np.ndarray, factor: float = 2.0) -> dict:
    """Single-sample outliers: samples whose magnitude exceeds `factor` times the
    99.9th percentile of their own segment, only counting isolated ones (both
    neighbours below the threshold). Catches solver glitches in a sim and
    dropouts or clicks in a recording; a clean DUT reports zero everywhere."""
    out = {"count": 0, "max_ratio": 0.0, "by_segment": {}}
    for s in stim.segments:
        if s.kind == "silence":
            continue
        seg = y[s.start : s.stop]
        if len(seg) < 3:
            continue
        thr = factor * np.percentile(np.abs(seg), 99.9)
        if thr <= 0:
            continue
        a = np.abs(seg)
        hit = (a[1:-1] > thr) & (a[:-2] <= thr) & (a[2:] <= thr)
        n = int(hit.sum())
        if n:
            ratio = float(np.max(a[1:-1][hit]) / (thr / factor))
            out["by_segment"][s.name] = {"count": n, "max_ratio": ratio}
            out["count"] += n
            out["max_ratio"] = max(out["max_ratio"], ratio)
    return out


# ------------------------------------------------------------------ bursts
def _env_db(x: np.ndarray, fs: int, win_s: float = 0.01, freq: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Short-time RMS in dB. With `freq`, the window is an integer number of
    cycles near win_s so a steady tone reads steady."""
    if freq:
        w = int(round(max(1, round(win_s * freq)) * fs / freq))
    else:
        w = max(1, int(win_s * fs))
    n = len(x) // w
    seg = x[: n * w].reshape(n, w)
    lvl = 20 * np.log10(np.sqrt(np.mean(seg ** 2, axis=1)) + 1e-30)
    t = (np.arange(n) + 0.5) * w / fs
    return t, lvl


def burst_metrics(stim: Stimulus, y: np.ndarray) -> list[dict]:
    """Sag and recovery. For the burst: output level over time relative to its
    first 20 ms (negative = compression setting in). For the probe: level over
    time relative to its final 300 ms (negative = still recovering)."""
    fs = stim.fs
    rows = []
    for b in stim.segs("burst"):
        p = next((s for s in stim.segs("probe") if s.meta.get("after") == b.name), None)
        yb = y[b.start : b.stop]
        tb, lb = _env_db(yb, fs, freq=b.meta["freq"])
        start = np.mean(lb[1:3])                      # after the fade-in
        end = np.median(lb[-31:-1])                   # last ~300 ms, excluding the fade-out window
        row = {"name": b.name, "freq": b.meta["freq"], "level_db": b.meta["level_db"],
               "sag_db": float(end - start), "min_db": float(np.min(lb[1:]) - start),
               "dc_start": float(np.mean(yb[: int(0.05 * fs)])), "dc_end": float(np.mean(yb[-int(0.3 * fs):])),
               "envelope_t": tb.tolist(), "envelope_db": (lb - start).tolist()}
        if p is not None:
            yp = y[p.start : p.stop]
            tp, lp = _env_db(yp, fs, freq=p.meta["freq"])
            final = np.median(lp[-31:-1])
            rel = (lp - final)[:-1]
            tp = tp[:-1]
            settled = np.where(np.abs(rel) < 0.5)[0]
            rec_t = float(tp[settled[0]]) if len(settled) else float(tp[-1])
            # first index after which it stays within 0.5 dB
            for i in range(len(rel)):
                if np.all(np.abs(rel[i:]) < 0.5):
                    rec_t = float(tp[i]); break
            row.update({"probe": p.name, "probe_deficit_db": float(rel[1]), "recovery_s": rec_t,
                        "probe_envelope_t": tp.tolist(), "probe_envelope_db": rel.tolist()})
        rows.append(row)
    return rows


# -------------------------------------------------------------- knob ramps
def ramp_metrics(stim: Stimulus, y: np.ndarray, schedule: dict, win_s: float = 0.05, max_order: int = 10) -> list[dict]:
    """For every ramp segment a parameter moves across: short-time level, DC,
    THD and harmonics of the tone against the parameter's value."""
    from .controls import value_at
    fs = stim.fs
    out = []
    for s in stim.segs("ramp"):
        f = s.meta["freq"]
        w = int(round(max(1, round(win_s * f)) * fs / f))
        n = (s.length // w)
        t_abs = s.start / fs + (np.arange(n) + 0.5) * w / fs
        moving = {name: value_at(pts, t_abs) for name, pts in schedule.items()
                  if np.ptp(value_at(pts, np.array([s.start / fs, s.stop / fs]))) > 0}
        if not moving:
            continue
        seg = y[s.start : s.start + n * w].reshape(n, w)
        tw = np.arange(w) / fs
        basis = [np.exp(-2j * np.pi * k * f * tw) for k in range(1, max_order + 1) if k * f < fs / 2]
        rows = []
        for i in range(n):
            amps = np.array([2 * abs(np.mean(seg[i] * b)) for b in basis])
            fund = amps[0]
            rows.append({"t": float(t_abs[i]), **{k: float(v[i]) for k, v in moving.items()},
                         "out_rms_db": float(20 * np.log10(rms(seg[i]) + 1e-30)), "dc": float(seg[i].mean()),
                         "thd": float(np.sqrt(np.sum(amps[1:] ** 2)) / fund) if fund > 0 else float("nan"),
                         "h2_db": float(20 * np.log10(amps[1] / fund + 1e-30)) if fund > 0 and len(amps) > 1 else float("nan"),
                         "h3_db": float(20 * np.log10(amps[2] / fund + 1e-30)) if fund > 0 and len(amps) > 2 else float("nan")})
        out.append({"segment": s.name, "freq": f, "level_db": s.meta["level_db"], "params": list(moving), "rows": rows})
    return out
