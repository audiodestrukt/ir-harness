import numpy as np
import pytest

from irharness import analysis
from irharness.align import apply_alignment, find_alignment
from irharness.dut import make_dut
from irharness.stimulus import build

FS = 48000


@pytest.fixture(scope="module")
def stim():
    return build(fs=FS, sweep_seconds=3.0, f2=20000, tone_dbs=(-20.0, 0.0), tone_freqs=(220.0, 1000.0), tone_seconds=0.2)


def _analyze(stim, dut):
    y = dut.run(stim.signal, stim.fs)
    al = find_alignment(stim, y)
    ya = apply_alignment(y, al, len(stim))
    ir_full, origin = analysis.deconvolve(stim, ya)
    irs = analysis.extract_harmonics(ir_full, origin, stim, max_order=5, ir_seconds=0.05)
    return al, ya, irs


def test_segments_are_contiguous(stim):
    pos = 0
    for s in stim.segments:
        assert s.start == pos
        pos += s.length
    assert pos == len(stim)
    assert np.max(np.abs(stim.signal)) <= 1.0


def test_save_load_roundtrip(stim, tmp_path):
    from irharness.stimulus import Stimulus
    stim.save(tmp_path)
    s2 = Stimulus.load(tmp_path)
    assert s2.fs == stim.fs and len(s2) == len(stim)
    assert s2.seg("sweep").meta["L"] == stim.seg("sweep").meta["L"]
    assert np.allclose(s2.signal, stim.signal, atol=1e-6)


def test_delay_and_polarity(stim):
    al, _, _ = _analyze(stim, make_dut("delay:1234"))
    assert al.delay == 1234 and al.polarity == 1

    class Inv:
        def run(self, x, fs):
            return -x
    al, ya, irs = _analyze(stim, Inv())
    assert al.polarity == -1
    assert np.allclose(ya, stim.signal, atol=1e-9)


def test_identity_ir_is_flat_unit_gain(stim):
    _, _, irs = _analyze(stim, make_dut("ident"))
    h = irs.linear
    assert int(np.argmax(np.abs(h))) == irs.pre
    f, H = analysis.spectrum(h, FS)
    g, mag, _ = analysis.smoothed_response(f, H, 20, 20000)
    band = (g > 40) & (g < 10000)
    assert np.max(np.abs(mag[band])) < 0.3, (mag[band].min(), mag[band].max())
    # energy outside the main lobe is sinc ripple from band limiting, small
    assert np.sum(h[irs.pre + 50:] ** 2) < 0.01 * np.sum(h ** 2)


def test_lowpass_magnitude(stim):
    from scipy.signal import butter, sosfreqz
    _, _, irs = _analyze(stim, make_dut("lowpass:1000:2"))
    f, H = analysis.spectrum(irs.linear, FS)
    g, mag, _ = analysis.smoothed_response(f, H, 20, 20000)
    sos = butter(2, 1000, fs=FS, output="sos")
    for fq, tol in ((100, 0.1), (1000, 0.3), (3000, 0.5), (10000, 1.0)):
        _, h = sosfreqz(sos, worN=[fq], fs=FS)
        ref = 20 * np.log10(abs(h[0]))
        got = float(np.interp(fq, g, mag))
        assert abs(got - ref) < tol, (fq, got, ref)


def test_tanh_third_harmonic(stim):
    d, A = 2.0, 10 ** (-12 / 20)   # sweep level
    _, ya, irs = _analyze(stim, make_dut(f"tanh:{d}"))
    harm = analysis.harmonic_response(irs, 20, 20000)
    fin, h3 = harm[3]
    expect = 20 * np.log10(d * d * A * A / 12)   # small-signal series: -33.6 dB
    mid = (fin > 200) & (fin < 2000)
    assert np.all(np.abs(h3[mid] - expect) < 1.5), (h3[mid].min(), h3[mid].max(), expect)
    fin2, h2 = harm[2]
    assert np.all(h2[(fin2 > 200) & (fin2 < 2000)] < -70)   # odd function, no H2

    rows = analysis.tone_metrics(stim, ya)
    r = next(r for r in rows if r["freq"] == 1000 and r["level_db"] == 0)
    # direct numeric reference for tanh(2 sin)
    t = np.arange(FS) / FS
    y = np.tanh(d * np.sin(2 * np.pi * 1000 * t)) / d
    amps = [2 * abs(np.mean(y * np.exp(-2j * np.pi * k * 1000 * t))) for k in range(1, 11)]
    thd_ref = np.sqrt(sum(a * a for a in amps[1:])) / amps[0]
    assert abs(r["thd"] - thd_ref) < 1e-3
    assert r["harmonics_db"][0] < -80


def test_residual_metrics():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(FS)
    r = analysis.residual_metrics(a, 0.5 * a, FS)
    assert abs(r["ls_gain"] - 2.0) < 1e-9 and r["esr"] < 1e-20
    r = analysis.residual_metrics(a, a + 0.1 * rng.standard_normal(FS), FS)
    assert abs(r["null_depth_db"] - (-20)) < 1.0
