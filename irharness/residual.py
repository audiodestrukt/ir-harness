"""Where does the model fail? Localize a residual by frequency, by input
level, by signal history, and per stepped tone. The output is a ranked list
of what to measure next.

Residual-to-signal ratio (RSR) is used throughout: residual energy over the
DUT's actual output energy in the same bin, in dB. 0 dB means the model
explains nothing there, -20 dB means it explains 99% of the energy.
"""
from __future__ import annotations

import numpy as np

from .stimulus import Stimulus


def _rsr_db(e: np.ndarray, y: np.ndarray) -> float:
    ey, ee = float(np.sum(y * y)), float(np.sum(e * e))
    return 10 * np.log10((ee + 1e-30) / (ey + 1e-30))


def ls_gain(y: np.ndarray, yhat: np.ndarray) -> float:
    d = float(np.dot(yhat, yhat))
    return float(np.dot(y, yhat) / d) if d > 0 else 1.0


def by_band(e: np.ndarray, y: np.ndarray, fs: int, f_lo=20.0, f_hi=20000.0, per_octave=3) -> tuple[list[float], list[float]]:
    E, Y = np.fft.rfft(e), np.fft.rfft(y)
    f = np.fft.rfftfreq(len(e), 1 / fs)
    edges = f_lo * 2 ** (np.arange(int(np.log2(f_hi / f_lo) * per_octave) + 1) / per_octave)
    centers, rsr = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if m.any():
            centers.append(float(np.sqrt(lo * hi)))
            rsr.append(10 * np.log10((np.sum(np.abs(E[m]) ** 2) + 1e-30) / (np.sum(np.abs(Y[m]) ** 2) + 1e-30)))
    return centers, rsr


def windows(n: int, fs: int, seconds: float = 0.01) -> np.ndarray:
    w = max(1, int(seconds * fs))
    return np.arange(0, n - w + 1, w)[:, None] + np.arange(w)[None, :]


def by_level(x: np.ndarray, e: np.ndarray, y: np.ndarray, fs: int, edges_db=(-60, -40, -30, -24, -18, -12, -6, 0)) -> dict:
    """RSR binned by the input's short-time level (10 ms RMS, sine-peak dB)."""
    idx = windows(len(x), fs)
    lvl = 20 * np.log10(np.sqrt(np.mean(x[idx] ** 2, axis=1)) * np.sqrt(2) + 1e-9)
    ee, ey = np.sum(e[idx] ** 2, axis=1), np.sum(y[idx] ** 2, axis=1)
    rows = []
    for lo, hi in zip(edges_db[:-1], edges_db[1:]):
        m = (lvl >= lo) & (lvl < hi)
        if m.sum() < 3:
            continue
        rows.append({"lo_db": lo, "hi_db": hi, "windows": int(m.sum()),
                     "rsr_db": 10 * np.log10((ee[m].sum() + 1e-30) / (ey[m].sum() + 1e-30)),
                     "share_of_residual": float(ee[m].sum() / (ee.sum() + 1e-30))})
    return {"bins": rows}


def by_history(x: np.ndarray, e: np.ndarray, y: np.ndarray, fs: int, history_s: float = 0.3,
               level_edges=(-40, -24, -12, 0), hist_edges=(-60, -30, -18, -6, 0)) -> dict:
    """Does the residual depend on what came *before*, at a fixed current level?
    RSR in a grid of (current 10 ms level) x (peak level over the previous
    `history_s`). Spread along the history axis at fixed current level is the
    memory signature: sag, bias recovery, thermal, anything stateful."""
    idx = windows(len(x), fs)
    w = idx.shape[1]
    lvl = 20 * np.log10(np.sqrt(np.mean(x[idx] ** 2, axis=1)) * np.sqrt(2) + 1e-9)
    nh = max(1, int(history_s * fs / w))
    hist = np.full(len(lvl), -120.0)
    for i in range(len(lvl)):
        if i > 0:
            hist[i] = np.max(lvl[max(0, i - nh) : i])
    ee, ey = np.sum(e[idx] ** 2, axis=1), np.sum(y[idx] ** 2, axis=1)
    grid = []
    spreads = []
    for lo, hi in zip(level_edges[:-1], level_edges[1:]):
        row = []
        for hlo, hhi in zip(hist_edges[:-1], hist_edges[1:]):
            m = (lvl >= lo) & (lvl < hi) & (hist >= hlo) & (hist < hhi)
            row.append(10 * np.log10((ee[m].sum() + 1e-30) / (ey[m].sum() + 1e-30)) if m.sum() >= 5 else None)
        grid.append(row)
        vals = [v for v in row if v is not None]
        if len(vals) >= 2:
            spreads.append(max(vals) - min(vals))
    return {"level_edges": list(level_edges), "history_edges": list(hist_edges), "rsr_db": grid,
            "memory_spread_db": float(np.mean(spreads)) if spreads else None}


def by_tone(stim: Stimulus, y: np.ndarray, yhat: np.ndarray, trim_ms: float = 30.0) -> list[dict]:
    fs = stim.fs
    rows = []
    for s in stim.segs("tone"):
        a, b = s.start + int(trim_ms * 1e-3 * fs), s.stop - int(trim_ms * 1e-3 * fs)
        ys, yh = y[a:b], yhat[a:b]
        ys, yh = ys - ys.mean(), yh - yh.mean()
        rows.append({"name": s.name, "freq": s.meta["freq"], "level_db": s.meta["level_db"],
                     "rsr_db": _rsr_db(ys - yh, ys), "gain_err_db": 20 * np.log10(abs(ls_gain(ys, yh)) + 1e-30)})
    return rows


def by_burst(stim: Stimulus, y: np.ndarray, yhat: np.ndarray) -> list[dict]:
    rows = []
    for s in stim.segs("burst") + stim.segs("probe"):
        ys, yh = y[s.start : s.stop], yhat[s.start : s.stop]
        rows.append({"name": s.name, "kind": s.kind, "rsr_db": _rsr_db(ys - yh, ys)})
    return rows


def focus(rep: dict) -> list[str]:
    """Turn the localized residual into ranked advice."""
    out = []
    total = rep["clip"]["rsr_db"] if "clip" in rep else None
    if total is not None and total < -30:
        out.append(f"clip residual {total:.1f} dB: the sweep-derived model already explains this DUT; "
                   "remaining error is at the noise/precision floor")
        return out
    lv = rep.get("clip", {}).get("by_level", {}).get("bins", [])
    if lv:
        worst = max(lv, key=lambda r: r["share_of_residual"])
        hi_bins = [r for r in lv if r["lo_db"] >= rep["model_level_db"]]
        lo_bins = [r for r in lv if r["hi_db"] <= rep["model_level_db"]]
        if hi_bins and lo_bins:
            d = np.mean([r["rsr_db"] for r in hi_bins]) - np.mean([r["rsr_db"] for r in lo_bins])
            if d > 6:
                out.append(f"level dependence: RSR is {d:.0f} dB worse above the sweep level ({rep['model_level_db']:g} dBFS) "
                           f"than below; {worst['share_of_residual']*100:.0f}% of residual energy sits in "
                           f"[{worst['lo_db']}, {worst['hi_db']}) dBFS -> add sweeps at those levels")
            elif d < -6:
                out.append(f"small-signal mismatch: RSR is {-d:.0f} dB worse below the sweep level than above "
                           "-> add a quieter sweep; the loud one is a describing function, not a linear response")
    mem = rep.get("clip", {}).get("by_history", {}).get("memory_spread_db")
    if mem is not None and mem > 6:
        out.append(f"memory: at fixed current level the residual moves {mem:.0f} dB with the previous 300 ms "
                   "-> stateful behaviour (sag, bias recovery, thermal); use tone bursts and the burst residual")
    bands = rep.get("clip", {}).get("by_band")
    if bands:
        c, r = bands
        r = np.array(r)
        top = np.argsort(r)[-3:][::-1]
        worst_band = ", ".join(f"{c[i]:.0f} Hz ({r[i]:.0f} dB)" for i in top)
        if r.max() - np.median(r) > 8:
            out.append(f"frequency-localized: residual is worst at {worst_band} -> check aliasing above 20 kHz, "
                       "harmonic IR window length, or a resonance the sweep level didn't excite")
    tones = rep.get("tones", [])
    if tones:
        bad = [t for t in tones if t["rsr_db"] > -10]
        if bad:
            fr = sorted({t["freq"] for t in bad})
            lv_ = sorted({t["level_db"] for t in bad})
            out.append(f"tones with RSR above -10 dB: {len(bad)}/{len(tones)}, frequencies {fr}, levels {lv_} dBFS")
    if not out:
        out.append("residual is spread evenly; no single mechanism dominates. Raise model order or compare against "
                   "a second measurement of the same DUT to find the repeatability floor")
    return out


def localize(stim: Stimulus, y: np.ndarray, yhat: np.ndarray, model_level_db: float) -> dict:
    """Full residual report for one run: the DUT's aligned response y against
    the model's prediction yhat of the same stimulus."""
    fs = stim.fs
    rep = {"model_level_db": model_level_db}
    if any(s.kind == "clip" for s in stim.segments):
        s = stim.seg("clip")
        x = stim.signal[s.start : s.stop]
        ys, yh = y[s.start : s.stop], yhat[s.start : s.stop]
        ys, yh = ys - ys.mean(), yh - yh.mean()
        g = ls_gain(ys, yh)
        e = ys - g * yh
        rep["clip"] = {"rsr_db": _rsr_db(e, ys), "ls_gain_db": 20 * np.log10(abs(g) + 1e-30),
                       "by_band": by_band(e, ys, fs), "by_level": by_level(x, e, ys, fs),
                       "by_history": by_history(x, e, ys, fs)}
    rep["tones"] = by_tone(stim, y, yhat)
    rep["bursts"] = by_burst(stim, y, yhat)
    rep["focus"] = focus(rep)
    return rep
