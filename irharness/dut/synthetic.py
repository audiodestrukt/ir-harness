"""Known-answer DUTs. They exist so the analysis can be tested against
closed-form expectations before any real device is involved."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .base import DUT


class IdentityDUT(DUT):
    def run(self, x, fs):
        return x.copy()


class GainDUT(DUT):
    def __init__(self, spec, gain_db):
        super().__init__(spec)
        self.gain_db = gain_db

    def run(self, x, fs):
        return x * 10 ** (self.gain_db / 20)

    def describe(self):
        return {"spec": self.spec, "gain_db": self.gain_db}


class DelayDUT(DUT):
    def __init__(self, spec, samples):
        super().__init__(spec)
        self.samples = samples

    def run(self, x, fs):
        return np.concatenate([np.zeros(self.samples), x])

    def describe(self):
        return {"spec": self.spec, "samples": self.samples}


class LowpassDUT(DUT):
    def __init__(self, spec, hz, order=2):
        super().__init__(spec)
        self.hz, self.order = hz, order

    def run(self, x, fs):
        sos = butter(self.order, self.hz, fs=fs, output="sos")
        return sosfilt(sos, x)

    def describe(self):
        return {"spec": self.spec, "hz": self.hz, "order": self.order}


class TanhDUT(DUT):
    """y = tanh(d x) / d: unity small-signal gain, odd harmonics only."""

    def __init__(self, spec, drive=1.0):
        super().__init__(spec)
        self.drive = drive

    def run(self, x, fs):
        return np.tanh(self.drive * x) / self.drive

    def describe(self):
        return {"spec": self.spec, "drive": self.drive}
