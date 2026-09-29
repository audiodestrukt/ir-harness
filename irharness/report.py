"""Figures and the markdown report. Matplotlib, Agg backend, one measure per axis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

# categorical slots, fixed order (validated palette, light surface)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "legend.frameon": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.dpi": 120,
})


def _logx(ax, f_lo, f_hi):
    ax.set_xscale("log")
    ax.set_xlim(f_lo, f_hi)
    ticks = [t for t in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000) if f_lo <= t <= f_hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t/1000:g}k" if t >= 1000 else str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Hz")


def _logy_percent(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_run(an: dict, out: Path, title: str) -> list[Path]:
    """an: the dict analyze_run builds (arrays included)."""
    out = Path(out)
    fs, f1, f2 = an["fs"], an["f1"], an["f2"]
    paths = []

    # linear IR + magnitude + harmonic response
    fig, axs = plt.subplots(3, 1, figsize=(8, 9))
    ir = an["ir_linear"]
    pre = an["ir_pre"]
    lo, hi = max(0, pre - int(0.001 * fs)), min(len(ir), pre + int(0.01 * fs))
    t = (np.arange(lo, hi) - pre) / fs * 1000
    axs[0].plot(t, ir[lo:hi] / (np.max(np.abs(ir)) or 1), color=SERIES[0])
    axs[0].set_title(f"{title}: linear impulse response (first 10 ms, normalized)")
    axs[0].set_xlabel("ms")
    axs[0].axhline(0, color=GRID, lw=0.8)

    axs[1].plot(an["fr_f"], an["fr_mag_db"], color=SERIES[0])
    axs[1].set_title("magnitude, 1/12 oct smoothed")
    axs[1].set_ylabel("dB")
    _logx(axs[1], f1, f2)

    for i, (k, (fin, rel)) in enumerate(sorted(an["harm"].items())):
        axs[2].plot(fin, rel, color=SERIES[i % len(SERIES)], label=f"H{k}")
    axs[2].set_title("harmonic distortion from the sweep, order k relative to fundamental")
    axs[2].set_ylabel("dB re fundamental")
    axs[2].set_ylim(-100, 10)
    axs[2].legend(loc="upper right", ncol=4)
    _logx(axs[2], f1, f2)
    fig.tight_layout()
    p = out / "fig_sweep.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # stepped tones
    tones = an["tones"]
    if tones:
        freqs = sorted({r["freq"] for r in tones})
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        for i, f in enumerate(freqs):
            rows = sorted((r for r in tones if r["freq"] == f), key=lambda r: r["level_db"])
            lv = [r["level_db"] for r in rows]
            axs[0].plot(lv, [r["thd"] * 100 for r in rows], "o-", ms=4, color=SERIES[i % 8], label=f"{f:g} Hz")
            axs[1].plot(lv, [r["out_rms_db"] for r in rows], "o-", ms=4, color=SERIES[i % 8], label=f"{f:g} Hz")
        axs[0].set_title("THD vs input level")
        axs[0].set_ylabel("THD %")
        _logy_percent(axs[0])
        axs[0].set_xlabel("input dBFS")
        axs[1].set_title("output level vs input level")
        axs[1].set_ylabel("output dB (RMS, DUT units)")
        axs[1].set_xlabel("input dBFS")
        axs[1].legend(loc="lower right")
        fig.tight_layout()
        p = out / "fig_tones.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_compare(a: dict, b: dict, cmp: dict, out: Path, la: str, lb: str) -> list[Path]:
    out = Path(out)
    f1, f2 = a["f1"], a["f2"]
    paths = []

    fig, axs = plt.subplots(3, 1, figsize=(8, 9))
    axs[0].plot(a["fr_f"], a["fr_mag_db"] + cmp["gain_a_db"], color=SERIES[0], label=la)
    axs[0].plot(b["fr_f"], b["fr_mag_db"] + cmp["gain_b_db"], color=SERIES[1], label=lb)
    axs[0].set_title("magnitude, level-matched on the sweep")
    axs[0].set_ylabel("dB")
    axs[0].legend(loc="lower left")
    _logx(axs[0], f1, f2)

    axs[1].plot(cmp["diff_f"], cmp["diff_mag_db"], color=SERIES[6])
    axs[1].axhline(0, color=GRID, lw=0.8)
    axs[1].set_title(f"difference {lb} minus {la}")
    axs[1].set_ylabel("dB")
    _logx(axs[1], f1, f2)

    for i, k in enumerate(sorted(set(a["harm"]) & set(b["harm"]))):
        if k > 3:
            break
        fa, ra = a["harm"][k]
        fb, rb = b["harm"][k]
        axs[2].plot(fa, ra, color=SERIES[i * 2], label=f"H{k} {la}")
        axs[2].plot(fb, rb, color=SERIES[i * 2 + 1], ls="--", label=f"H{k} {lb}")
    axs[2].set_title("harmonics from the sweep, relative to fundamental")
    axs[2].set_ylabel("dB")
    axs[2].set_ylim(-100, 10)
    axs[2].legend(loc="upper right", ncol=2)
    _logx(axs[2], f1, f2)
    fig.tight_layout()
    p = out / "fig_cmp_sweep.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    if a["tones"] and b["tones"]:
        freqs = sorted({r["freq"] for r in a["tones"]})
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for i, f in enumerate(freqs):
            for src, ls in ((a, "-"), (b, "--")):
                rows = sorted((r for r in src["tones"] if r["freq"] == f), key=lambda r: r["level_db"])
                lv = [r["level_db"] for r in rows]
                axs[0].plot(lv, [r["thd"] * 100 for r in rows], ls, color=SERIES[i % 8],
                            label=f"{f:g} Hz" if ls == "-" else None)
                g = cmp["gain_a_db"] if src is a else cmp["gain_b_db"]
                axs[1].plot(lv, [r["out_rms_db"] + g for r in rows], ls, color=SERIES[i % 8])
        fig.suptitle(f"stepped tones: {la} solid, {lb} dashed", x=0.01, ha="left", fontsize=10, fontweight="bold")
        axs[0].set_title("THD vs input level")
        axs[0].set_ylabel("THD %")
        _logy_percent(axs[0])
        axs[0].set_xlabel("input dBFS")
        axs[0].legend(loc="lower right")
        axs[1].set_title("output level vs input, level-matched")
        axs[1].set_ylabel("dB")
        axs[1].set_xlabel("input dBFS")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = out / "fig_cmp_tones.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    if "clip" in cmp:
        c = cmp["clip"]
        fs = a["fs"]
        fig, axs = plt.subplots(2, 1, figsize=(9, 6))
        n0, n1 = c["excerpt"]
        t = np.arange(n0, n1) / fs
        axs[0].plot(t, c["a"], color=SERIES[0], lw=1, label=la)
        axs[0].plot(t, c["b"], color=SERIES[1], lw=1, label=lb)
        axs[0].plot(t, c["a"] - c["b"], color=SERIES[6], lw=0.8, label="residual")
        axs[0].set_title(f"DI clip excerpt, level-matched; null depth {c['null_depth_db']:.1f} dB, ESR {c['esr']:.4f}")
        axs[0].set_xlabel("s")
        axs[0].legend(loc="upper right", ncol=3)
        axs[1].bar(range(len(c["band_centers"])), c["band_diff_db"], color=SERIES[6], width=0.7)
        axs[1].set_xticks(range(len(c["band_centers"])))
        axs[1].set_xticklabels([f"{x/1000:.2g}k" if x >= 1000 else f"{x:.0f}" for x in c["band_centers"]], rotation=60)
        axs[1].axhline(0, color=GRID, lw=0.8)
        axs[1].set_title(f"1/3-octave band level, {lb} minus {la}, on the DI clip")
        axs[1].set_ylabel("dB")
        axs[1].grid(axis="x", visible=False)
        fig.tight_layout()
        p = out / "fig_cmp_clip.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths
