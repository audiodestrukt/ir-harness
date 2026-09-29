"""Device-under-test adapters.

A DUT takes a full-scale float signal at a sample rate and returns the
response, at least as long as the input. Everything downstream (alignment,
analysis, comparison) is DUT-agnostic, so a circuit simulation and a real box
through an audio interface are interchangeable here.

Spec strings (irh run --dut ...):
  ident                   pass-through
  gain:<db>               scalar gain
  delay:<samples>         pure delay (alignment fixture)
  lowpass:<hz>[:order]    Butterworth, a known linear DUT
  tanh:<drive>            static nonlinearity, a known harmonic generator
  sim:ff                  circuit-decimator Fuzz Face (ff_render), --set = .param names
  sim:seout | sim:shinei  circuit-decimator netlist circuits (circuit_render)
  sim:tubepre             circuit-decimator tube mic pre (tubepre_render)
  plugin:<path.vst3>      any VST3 through circuit-decimator's plugin_render
  audio:<out>:<in>        sounddevice play+record through an audio interface
"""
from __future__ import annotations

from .base import DUT
from .synthetic import IdentityDUT, GainDUT, DelayDUT, LowpassDUT, TanhDUT


def make_dut(spec: str, sets: dict[str, float] | None = None, **opts) -> DUT:
    sets = dict(sets or {})
    head, _, rest = spec.partition(":")
    if head == "ident":
        return IdentityDUT(spec)
    if head == "gain":
        return GainDUT(spec, float(rest))
    if head == "delay":
        return DelayDUT(spec, int(rest))
    if head == "lowpass":
        hz, _, order = rest.partition(":")
        return LowpassDUT(spec, float(hz), int(order or 2))
    if head == "tanh":
        return TanhDUT(spec, float(rest or 1.0))
    if head == "sim":
        from .sim import make_sim_dut
        return make_sim_dut(spec, rest, sets, **opts)
    if head == "plugin":
        from .sim import PluginDUT
        return PluginDUT(spec, rest, sets, **opts)
    if head == "audio":
        from .interface import AudioDUT
        out_dev, _, in_dev = rest.partition(":")
        return AudioDUT(spec, out_dev or None, in_dev or None, **opts)
    raise ValueError(f"unknown DUT spec {spec!r}")
