"""Knob control during a run.

A control schedule is {param: [(time_s, value), ...]}: breakpoints, linearly
interpolated between the first and last; outside that span the parameter is
whatever --set or the device default says. It is attached at run time (parameter names are
the DUT's own), not baked into the stimulus, so one stimulus serves every
device. Ramps are expressed against stimulus segments, which is what makes
them measurable: `fuzz:0.05:1@knob_ramp` ramps `fuzz` across the segment
named knob_ramp and the analysis can then plot distortion against knob value.

Three ways a DUT can honour a schedule:
  native     the adapter takes the schedule itself (ff_render --automation)
  stepwise   the run is split into pieces with constant parameters, each run
             with a settling pre-roll and stitched; works for any adapter
  prompt     stepwise, but pauses for a human to set the knob between pieces
             (hardware with real knobs)
"""
from __future__ import annotations

import numpy as np

from .stimulus import Stimulus

Schedule = dict[str, list[tuple[float, float]]]


def parse_ramp(spec: str, stim: Stimulus) -> tuple[str, list[tuple[float, float]]]:
    """'name:from:to@segment' or 'name:from:to' (whole stimulus), or
    'name:v0@t0,v1@t1,...' explicit breakpoints in seconds."""
    name, _, rest = spec.partition(":")
    if "@" in rest and "," in rest:
        pts = []
        for item in rest.split(","):
            v, _, t = item.partition("@")
            pts.append((float(t), float(v)))
        return name, sorted(pts)
    body, _, seg = rest.partition("@")
    a, _, b = body.partition(":")
    if seg:
        s = stim.seg(seg)
        t0, t1 = s.start / stim.fs, s.stop / stim.fs
    else:
        t0, t1 = 0.0, stim.seconds
    return name, [(t0, float(a)), (t1, float(b))]


def value_at(pts: list[tuple[float, float]], t: np.ndarray, base: float | None = None) -> np.ndarray:
    """Interpolated value; outside the span, `base` (or the end values if None)."""
    ts = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts])
    v = np.interp(t, ts, vs)
    if base is not None:
        v = np.where((t < ts[0]) | (t > ts[-1]), base, v)
    return v


def write_automation(path, schedule: Schedule) -> None:
    with open(path, "w") as f:
        for name, pts in schedule.items():
            for t, v in pts:
                f.write(f"{t:.9g} {name} {v:.9g}\n")


def stepwise(dut, x: np.ndarray, fs: int, schedule: Schedule, steps: int = 16,
             settle_s: float = 0.5, prompt: bool = False) -> np.ndarray:
    """Run a DUT that can't automate: hold parameters constant over pieces of
    the timeline. Each piece is rendered with `settle_s` of the preceding
    signal as pre-roll (discarded) so the device's state is close to what a
    continuous run would have. Only the ramped spans are subdivided."""
    n = len(x)
    # breakpoint times of all params + piece boundaries inside ramps
    cuts = {0.0, n / fs}
    for pts in schedule.values():
        for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
            cuts.add(t0); cuts.add(t1)
            if v0 != v1:
                for i in range(1, steps):
                    cuts.add(t0 + (t1 - t0) * i / steps)
    edges = sorted(min(max(c, 0.0), n / fs) for c in cuts)
    y = np.zeros(n)
    base = dict(getattr(dut, "sets", {}))
    for t0, t1 in zip(edges[:-1], edges[1:]):
        a, b = int(round(t0 * fs)), int(round(t1 * fs))
        if b <= a:
            continue
        tm = 0.5 * (t0 + t1)
        vals = {}
        for name, pts in schedule.items():
            if pts[0][0] <= tm <= pts[-1][0]:
                vals[name] = float(value_at(pts, np.array([tm]))[0])
        if hasattr(dut, "sets"):
            dut.sets = {**base, **vals}
        if prompt:
            input(f"set {', '.join(f'{k}={v:.4g}' for k, v in vals.items())} for {t0:.2f}..{t1:.2f} s, then press enter: ")
        pre = min(a, int(settle_s * fs))
        piece = dut.run(x[a - pre : b], fs)
        y[a:b] = piece[pre : pre + (b - a)]
    if hasattr(dut, "sets"):
        dut.sets = base
    return y


def run_with_controls(dut, x: np.ndarray, fs: int, schedule: Schedule | None, mode: str = "auto",
                      steps: int = 16) -> np.ndarray:
    if not schedule:
        return dut.run(x, fs)
    native = getattr(dut, "supports_controls", False)
    if mode == "auto":
        mode = "native" if native else "stepwise"
    if mode == "native":
        if not native:
            raise ValueError(f"{dut.spec} cannot automate natively; use --knob-mode stepwise or prompt")
        return dut.run(x, fs, controls=schedule)
    return stepwise(dut, x, fs, schedule, steps=steps, prompt=(mode == "prompt"))
