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
    lo, hi = ax.get_ylim()
    if hi / max(lo, 1e-9) < 10:      # under a decade: label the minor ticks too or there are none
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    else:
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


SEQ = "Blues"   # sequential, one hue


def plot_residual(res: dict, out: Path, title: str) -> Path:
    """Where the sweep-derived model fails: by band, by level, by history, per tone."""
    rep = res.get("all_levels") or res["ref_only"]
    rep0 = res["ref_only"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axs[0, 0]
    if "clip" in rep0:
        c, r = rep0["clip"]["by_band"]
        xi = np.arange(len(c))
        ax.bar(xi, r, color=SERIES[0], width=0.8, label=f"sweep model at {rep0['model_level_db']:g} dBFS")
        if "all_levels" in res and "clip" in res["all_levels"]:
            c2, r2 = res["all_levels"]["clip"]["by_band"]
            ax.plot(np.arange(len(c2)), r2, "o-", ms=3, color=SERIES[1], label="all sweep levels")
        ax.set_xticks(xi[::3])
        ax.set_xticklabels([f"{x/1000:.2g}k" if x >= 1000 else f"{x:.0f}" for x in c[::3]])
        ax.legend(loc="lower right")
        ax.set_title(f"{title}: residual on the DI clip, by 1/3 octave")
        ax.set_ylabel("residual / output, dB")
        ax.grid(axis="x", visible=False)
    else:
        ax.set_title("no DI clip in this stimulus")

    ax = axs[0, 1]
    if "clip" in rep0:
        for i, (key, lab) in enumerate((("ref_only", f"at {rep0['model_level_db']:g} dBFS"), ("all_levels", "all levels"))):
            if key not in res or "clip" not in res[key]:
                continue
            bins = res[key]["clip"]["by_level"]["bins"]
            xs = [f"{b['lo_db']}..{b['hi_db']}" for b in bins]
            ax.plot(xs, [b["rsr_db"] for b in bins], "o-", ms=4, color=SERIES[i], label=lab)
        ax.axvline(-0.5, color=GRID)
        ax.set_title("residual by input level (10 ms windows)")
        ax.set_ylabel("residual / output, dB")
        ax.set_xlabel("input level, dBFS")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="lower right")

    ax = axs[1, 0]
    if "clip" in rep:
        h = rep["clip"]["by_history"]
        grid = np.array([[np.nan if v is None else v for v in row] for row in h["rsr_db"]], dtype=float)
        im = ax.imshow(grid, cmap=SEQ, aspect="auto", origin="lower", vmin=-40, vmax=0)
        le, he = h["level_edges"], h["history_edges"]
        ax.set_yticks(range(len(le) - 1)); ax.set_yticklabels([f"{a}..{b}" for a, b in zip(le[:-1], le[1:])])
        ax.set_xticks(range(len(he) - 1)); ax.set_xticklabels([f"{a}..{b}" for a, b in zip(he[:-1], he[1:])])
        ax.set_xlabel("peak level of previous 300 ms, dBFS")
        ax.set_ylabel("current level, dBFS")
        for (i, j), v in np.ndenumerate(grid):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=INK if v > -20 else "#ffffff")
        ms = h.get("memory_spread_db")
        ax.set_title(f"residual vs signal history (memory spread {ms:.1f} dB)" if ms is not None else "residual vs history")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="residual / output, dB", shrink=0.8)

    ax = axs[1, 1]
    tones = rep0["tones"]
    if tones:
        freqs = sorted({t["freq"] for t in tones}); lvls = sorted({t["level_db"] for t in tones})
        grid = np.full((len(lvls), len(freqs)), np.nan)
        for t in tones:
            grid[lvls.index(t["level_db"]), freqs.index(t["freq"])] = t["rsr_db"]
        im = ax.imshow(grid, cmap=SEQ, aspect="auto", origin="lower", vmin=-60, vmax=0)
        ax.set_xticks(range(len(freqs))); ax.set_xticklabels([f"{f:g}" for f in freqs])
        ax.set_yticks(range(len(lvls))); ax.set_yticklabels([f"{l:g}" for l in lvls])
        ax.set_xlabel("tone Hz"); ax.set_ylabel("tone level, dBFS")
        for (i, j), v in np.ndenumerate(grid):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=INK if v > -30 else "#ffffff")
        ax.set_title(f"stepped tones: residual of the {rep0['model_level_db']:g} dBFS model")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="dB", shrink=0.8)
    fig.tight_layout()
    p = Path(out) / "fig_residual.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_bursts(bursts: list[dict], out: Path, title: str) -> Path:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, b in enumerate(bursts):
        axs[0].plot(b["envelope_t"], b["envelope_db"], color=SERIES[i % 8], label=f"{b['freq']:g} Hz burst at {b['level_db']:g} dBFS")
        if "probe_envelope_t" in b:
            axs[1].plot(b["probe_envelope_t"], b["probe_envelope_db"], color=SERIES[i % 8], label=f"{b['freq']:g} Hz probe")
    axs[0].set_title(f"{title}: burst envelope re first 20 ms (sag)")
    axs[0].set_xlabel("s"); axs[0].set_ylabel("dB"); axs[0].legend(loc="lower right")
    axs[1].set_title("probe envelope re final (recovery)")
    axs[1].set_xlabel("s"); axs[1].set_ylabel("dB"); axs[1].legend(loc="lower right")
    fig.tight_layout()
    p = Path(out) / "fig_bursts.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_levels(per_level: dict, out: Path, title: str) -> Path:
    levels = sorted(per_level)
    f1 = min(per_level[levels[0]]["fr_f"]); f2 = max(per_level[levels[0]]["fr_f"])
    fig, axs = plt.subplots(2, 1, figsize=(8, 7))
    for i, lvl in enumerate(levels):
        v = per_level[lvl]
        axs[0].plot(v["fr_f"], v["fr_mag_db"], color=SERIES[i % 8], label=f"{lvl:g} dBFS")
        if 3 in v["harm"]:
            fin, rel = v["harm"][3]
            axs[1].plot(fin, rel, color=SERIES[i % 8], label=f"H3 at {lvl:g} dBFS")
    axs[0].set_title(f"{title}: fundamental response vs sweep level (a describing function when it moves)")
    axs[0].set_ylabel("dB"); axs[0].legend(loc="lower left"); _logx(axs[0], f1, f2)
    axs[1].set_title("third harmonic vs sweep level")
    axs[1].set_ylabel("dB re fundamental"); axs[1].set_ylim(-100, 10); axs[1].legend(loc="lower left"); _logx(axs[1], f1, f2)
    fig.tight_layout()
    p = Path(out) / "fig_levels.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_ramps(ramps: list[dict], out: Path, title: str) -> Path:
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for i, r in enumerate(ramps):
        prm = r["params"][0]
        xs = [row[prm] for row in r["rows"]]
        lab = f"{r['segment']} ({r['freq']:g} Hz at {r['level_db']:g} dBFS)"
        axs[0].plot(xs, [row["thd"] * 100 for row in r["rows"]], color=SERIES[i % 8], label=lab)
        axs[1].plot(xs, [row["h2_db"] for row in r["rows"]], color=SERIES[i % 8], label="H2")
        axs[1].plot(xs, [row["h3_db"] for row in r["rows"]], color=SERIES[(i + 1) % 8], ls="--", label="H3")
        axs[2].plot(xs, [row["out_rms_db"] for row in r["rows"]], color=SERIES[i % 8], label=lab)
        for ax in axs:
            ax.set_xlabel(prm)
    axs[0].set_title(f"{title}: THD vs knob"); axs[0].set_ylabel("THD %"); _logy_percent(axs[0]); axs[0].legend(loc="best")
    axs[1].set_title("H2 (solid) and H3 (dashed) vs knob"); axs[1].set_ylabel("dB re fundamental"); axs[1].legend(loc="best")
    axs[2].set_title("output level vs knob"); axs[2].set_ylabel("dB (DUT units)")
    fig.tight_layout()
    p = Path(out) / "fig_ramps.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_surface(g: dict, out: Path, metrics=("h3_1k_db", "h2_1k_db", "gain_1k_db", "hf_drop_10k_db", "clip_rsr_db", "sag_db")) -> list[Path]:
    names = list(g["grid"])
    xa, ya = names[0], names[1]
    xs, ys = list(g["grid"][xa]), list(g["grid"][ya])
    metrics = [m for m in metrics if any(c["summary"].get(m) is not None for c in g["cells"])]
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.6))
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, metrics):
        grid = np.full((len(ys), len(xs)), np.nan)
        for c in g["cells"]:
            v = c["summary"].get(m)
            if v is not None:
                grid[ys.index(c["cell"][ya]), xs.index(c["cell"][xa])] = v
        im = ax.imshow(grid, cmap=SEQ, aspect="auto", origin="lower")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{v:g}" for v in xs])
        ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{v:g}" for v in ys])
        ax.set_xlabel(xa); ax.set_ylabel(ya); ax.set_title(m); ax.grid(False)
        for (i, j), v in np.ndenumerate(grid):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=7,
                        color=INK if (v - np.nanmin(grid)) < 0.6 * (np.nanmax(grid) - np.nanmin(grid) + 1e-9) else "#ffffff")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    p = Path(out) / "fig_surface.png"
    fig.savefig(p)
    plt.close(fig)
    return [p]
