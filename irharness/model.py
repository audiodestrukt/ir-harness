"""Sweep-derived nonlinear model: a generalized Hammerstein built from the
harmonic impulse responses (Novak, Simon, Lotton 2010).

    y = sum_n  g_n * x^n            n = 1..N, '*' is convolution

The synchronized sweep x = A sin(phi) produces harmonic k from every branch
n >= k of the same parity, weighted by the sine-power expansion coefficients:

    sin^n  = 2^(1-n) sum_{k odd}  (-1)^((k-1)/2) C(n,(n-k)/2) sin(k phi)   n odd
    sin^n  = 2^(1-n) sum_{k even} (-1)^(k/2)     C(n,(n-k)/2) cos(k phi)   n even  (+ a DC term)

With the synchronized sweep, sin(k phi(t)) is exactly the sweep advanced by
L ln k, and cos(k phi) is that with a +90 degree phase (a factor j). The
deconvolution (inverse normalized by 1/A) therefore measures

    H_k(w) = sum_n A^(n-1) T[n,k] G_n(w),   T real for n odd, imaginary for n even

at every output frequency w, a frequency-independent triangular system that
inverts directly. Branch filters G_n come out in the frequency domain and are
applied by FFT convolution.

The model is exact for a memoryless nonlinearity between linear filters, at
the sweep's drive level. Everything it cannot explain lands in the residual,
which is the point: residual.py localizes what is left.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.signal import fftconvolve

from .analysis import IRSet


def sine_power_matrix(N: int, A: float) -> np.ndarray:
    """M[k-1, n-1] = A^(n-1) T[n,k] for k, n in 1..N."""
    M = np.zeros((N, N), dtype=complex)
    for n in range(1, N + 1):
        for k in range(1, n + 1):
            if (n - k) % 2:
                continue
            c = comb(n, (n - k) // 2) / 2 ** (n - 1)
            if n % 2:
                t = (-1) ** ((k - 1) // 2) * c
            else:
                t = 1j * (-1) ** (k // 2) * c
            M[k - 1, n - 1] = A ** (n - 1) * t
    return M


@dataclass
class Hammerstein:
    fs: int
    level_db: float             # sweep level the kernels were measured at (dBFS peak)
    kernels: np.ndarray         # (N, nfft) real branch IRs, t=0 at index `pre`
    pre: int

    @property
    def order(self) -> int:
        return len(self.kernels)

    @property
    def amplitude(self) -> float:
        return 10 ** (self.level_db / 20)

    def apply(self, x: np.ndarray, clamp: bool = True) -> np.ndarray:
        """A polynomial fitted at amplitude A says nothing above A, and past it
        the high powers explode; by default the input is clamped to the fit
        amplitude so the model stays inside what it measured."""
        if clamp:
            a = 1.05 * self.amplitude
            x = np.clip(x, -a, a)
        y = np.zeros(len(x))
        for n, g in enumerate(self.kernels, start=1):
            y += fftconvolve(x ** n, g)[self.pre : self.pre + len(x)]
        return y

    def branch(self, x: np.ndarray, n: int) -> np.ndarray:
        g = self.kernels[n - 1]
        return fftconvolve(x ** n, g)[self.pre : self.pre + len(x)]


def choose_order(irs: IRSet, floor_db: float = -45.0, cap: int = 9) -> int:
    """Highest harmonic order whose IR energy is above `floor_db` re the linear
    IR, capped: the inversion's condition number grows like (2/A)^(N-1), so
    orders beyond ~9 amplify measurement noise into the kernels."""
    e1 = float(np.sum(irs.ir[1] ** 2)) or 1.0
    N = 1
    for k in sorted(irs.ir):
        if k > cap:
            break
        if 10 * np.log10(np.sum(irs.ir[k] ** 2) / e1 + 1e-30) > floor_db:
            N = k
    return max(N, 3) if 3 in irs.ir else N


def fit_hammerstein(irs: IRSet, level_db: float, order: int | None = None) -> Hammerstein:
    N = order or choose_order(irs)
    A = 10 ** (level_db / 20)
    n_ir = len(irs.ir[1])
    nfft = 1 << int(np.ceil(np.log2(2 * n_ir)))
    f = np.fft.rfftfreq(nfft, 1 / irs.fs)
    # spectra with t=0 moved to the origin (the shared pre-window becomes negative time)
    adv = np.exp(2j * np.pi * f * irs.pre / irs.fs)
    H = np.stack([np.fft.rfft(irs.ir[k], nfft) * adv for k in range(1, N + 1)])
    Minv = np.linalg.inv(sine_power_matrix(N, A))
    G = Minv @ H                                   # (N, nbins)
    g = np.fft.irfft(G, nfft)                      # negative time wraps to the end
    g = np.roll(g, irs.pre, axis=1)                # t=0 back at index pre
    return Hammerstein(fs=irs.fs, level_db=level_db, kernels=g, pre=irs.pre)


@dataclass
class LevelIndexedModel:
    """A family of Hammerstein kernels measured at several sweep levels,
    blended per sample by the input's short-time envelope (in dB peak).
    With one level it is just that model."""
    models: list[Hammerstein]
    window_s: float = 0.01

    def envelope_db(self, x: np.ndarray, fs: int) -> np.ndarray:
        n = max(1, int(self.window_s * fs))
        w = np.ones(n) / n
        env = np.sqrt(np.convolve(x * x, w, mode="same")) * np.sqrt(2)   # RMS -> sine-peak equivalent
        return 20 * np.log10(env + 1e-9)

    def apply(self, x: np.ndarray, fs: int) -> np.ndarray:
        if len(self.models) == 1:
            return self.models[0].apply(x)
        levels = np.array([m.level_db for m in self.models])
        order = np.argsort(levels)
        levels, models = levels[order], [self.models[i] for i in order]
        outs = np.stack([m.apply(x) for m in models])
        env = np.clip(self.envelope_db(x, fs), levels[0], levels[-1])
        idx = np.clip(np.searchsorted(levels, env, side="right") - 1, 0, len(levels) - 2)
        lo, hi = levels[idx], levels[idx + 1]
        t = (env - lo) / np.maximum(hi - lo, 1e-9)
        y = (1 - t) * outs[idx, np.arange(len(x))] + t * outs[idx + 1, np.arange(len(x))]
        return y
