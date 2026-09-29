"""Audio-interface adapter: play the stimulus out of one channel and record
the DUT's return on another, in one synchronous playrec call.

Loopback (output cabled straight to input) is the first real-world DUT: it
measures the interface's own converters, latency and level, which is the
floor every later measurement sits on.
"""
from __future__ import annotations

import numpy as np

from .base import DUT


def list_devices() -> str:
    import sounddevice as sd
    return str(sd.query_devices())


def _resolve(dev):
    if dev is None:
        return None
    try:
        return int(dev)
    except ValueError:
        return dev


class AudioDUT(DUT):
    def __init__(self, spec, out_dev=None, in_dev=None, out_channel=1, in_channel=1,
                 out_db=0.0, extra_seconds=0.5, **_):
        super().__init__(spec)
        self.out_dev, self.in_dev = _resolve(out_dev), _resolve(in_dev)
        self.out_channel, self.in_channel = int(out_channel), int(in_channel)
        self.out_db = float(out_db)
        self.extra_seconds = float(extra_seconds)

    def run(self, x, fs):
        import sounddevice as sd
        pad = np.zeros(int(self.extra_seconds * fs))
        play = np.concatenate([x, pad]) * 10 ** (self.out_db / 20)
        if np.max(np.abs(play)) > 1.0:
            raise ValueError("stimulus exceeds full scale; lower --out-db")
        y = sd.playrec(play.astype(np.float32)[:, None], samplerate=fs,
                       device=(self.in_dev, self.out_dev),
                       input_mapping=[self.in_channel], output_mapping=[self.out_channel],
                       blocking=True)
        return y[:, 0].astype(np.float64)

    def describe(self):
        return {"spec": self.spec, "out_dev": self.out_dev, "in_dev": self.in_dev,
                "out_channel": self.out_channel, "in_channel": self.in_channel,
                "out_db": self.out_db}
