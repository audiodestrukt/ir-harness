"""circuit-decimator adapters: the simulation as a device under test.

The renderers live in ../circuit-decimator/build (override with the
CIRCUIT_DECIMATOR env var). They read "time value" text in volts and write
"time v(out) ..." text at exactly fs, so the adapter's only real job is unit
conversion: full scale -> volts_per_fs volts at the circuit input, and the
output node voltage returned as-is (the analysis level-matches later).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import DUT

CD_ROOT = Path(os.environ.get("CIRCUIT_DECIMATOR", Path(__file__).resolve().parents[3] / "circuit-decimator"))
BUILD = CD_ROOT / "build"


def _write_input(path: Path, x: np.ndarray, fs: int, volts_per_fs: float) -> None:
    t = np.arange(len(x)) / fs
    np.savetxt(path, np.column_stack([t, x * volts_per_fs]), fmt="%.9g")


def _read_output(path: Path, n: int) -> np.ndarray:
    y = np.loadtxt(path, skiprows=1, usecols=1)
    if len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y))])
    return y[:n]


class _TextRenderDUT(DUT):
    """Base for the text-in/text-out renderers."""

    binary: Path
    extra: list[str]

    def __init__(self, spec, sets, volts_per_fs=0.25, keep_files=False):
        super().__init__(spec)
        self.sets = dict(sets)
        self.volts_per_fs = float(volts_per_fs)
        self.keep_files = keep_files
        self.last_log = ""
        self.last_seconds = 0.0
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} not built (cmake --build {BUILD})")

    def args(self, inp: Path, out: Path, fs: int) -> list[str]:
        raise NotImplementedError

    def run(self, x, fs):
        d = Path(tempfile.mkdtemp(prefix="irh_"))
        try:
            inp, out = d / "in.txt", d / "out.dat"
            _write_input(inp, x, fs, self.volts_per_fs)
            t0 = time.time()
            p = subprocess.run(self.args(inp, out, fs), capture_output=True, text=True)
            self.last_seconds = time.time() - t0
            self.last_log = p.stdout + p.stderr
            if p.returncode != 0:
                raise RuntimeError(f"{self.binary.name} failed:\n{self.last_log}")
            return _read_output(out, len(x))
        finally:
            if self.keep_files:
                print(f"sim files kept in {d}")
            else:
                shutil.rmtree(d, ignore_errors=True)

    def describe(self):
        return {"spec": self.spec, "sets": self.sets, "volts_per_fs": self.volts_per_fs,
                "binary": str(self.binary), "sim_seconds": self.last_seconds, "log": self.last_log.strip()}


class FuzzFaceDUT(_TextRenderDUT):
    binary = BUILD / "ff_render"

    def __init__(self, spec, sets, solver="dk", **kw):
        self.solver = solver
        super().__init__(spec, sets, **kw)

    def args(self, inp, out, fs):
        a = [str(self.binary), str(inp), str(out), "--fs", str(fs), "--solver", self.solver]
        a += [f"{k}={v}" for k, v in self.sets.items()]
        return a

    def describe(self):
        return {**super().describe(), "solver": self.solver}


class CircuitDUT(_TextRenderDUT):
    binary = BUILD / "circuit_render"

    def __init__(self, spec, circuit, sets, **kw):
        self.circuit = circuit
        super().__init__(spec, sets, **kw)

    def args(self, inp, out, fs):
        a = [str(self.binary), self.circuit, "file", str(inp), str(out), "--fs", str(fs)]
        for k, v in self.sets.items():
            a += ["--set", f"{k}={v}"]
        return a

    def describe(self):
        return {**super().describe(), "circuit": self.circuit}


class TubePreDUT(_TextRenderDUT):
    binary = BUILD / "tubepre_render"

    def args(self, inp, out, fs):
        a = [str(self.binary), "file", str(inp), str(out), "--fs", str(fs)]
        if self.sets.get("linear"):
            a.append("--linear")
        return a


class PluginDUT(DUT):
    """Any VST3 through circuit-decimator's headless host. Values are in the
    parameter's own units (battery=6); names match the display name."""

    binary = BUILD / "plugin_render_artefacts" / "Release" / "plugin_render"

    def __init__(self, spec, path, sets, block=256, **kw):
        super().__init__(spec)
        self.path = Path(path).expanduser()
        self.sets = dict(sets)
        self.block = block
        self.last_log = ""
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} not built")
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def run(self, x, fs):
        with tempfile.TemporaryDirectory(prefix="irh_") as d:
            inp, out = Path(d) / "in.wav", Path(d) / "out.wav"
            sf.write(inp, x.astype(np.float32), fs, subtype="FLOAT")
            a = [str(self.binary), str(self.path), str(inp), str(out), "--block", str(self.block)]
            a += [f"{k}={v}" for k, v in self.sets.items()]
            p = subprocess.run(a, capture_output=True, text=True)
            self.last_log = p.stdout + p.stderr
            if p.returncode != 0:
                raise RuntimeError(f"plugin_render failed:\n{self.last_log}")
            y, fs_out = sf.read(out, dtype="float64", always_2d=True)
            if fs_out != fs:
                raise RuntimeError(f"plugin_render returned fs {fs_out}, wanted {fs}")
            return y[:, 0]

    def describe(self):
        return {"spec": self.spec, "plugin": str(self.path), "sets": self.sets, "block": self.block,
                "log": self.last_log.strip()}


def make_sim_dut(spec: str, which: str, sets: dict, **opts) -> DUT:
    if which == "ff":
        return FuzzFaceDUT(spec, sets, **opts)
    if which in ("seout", "shinei"):
        return CircuitDUT(spec, which, sets, **opts)
    if which == "tubepre":
        return TubePreDUT(spec, sets, **opts)
    raise ValueError(f"unknown sim {which!r}; have ff, seout, shinei, tubepre")
