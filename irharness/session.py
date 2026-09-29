"""Orchestration: run a DUT on a stimulus, analyze a run, compare two runs.

Run directory layout:
  run.json        DUT description, stimulus path, timing, alignment (after analyze)
  response.wav    raw DUT output, unaligned, float32 in DUT units
  aligned.wav     shifted to stimulus time, polarity fixed, cut to stimulus length
  metrics.json    scalar results (levels, alignment, tones, noise floor)
  analysis.npz    arrays (IRs, frequency responses, harmonic responses)
  fig_*.png
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from . import analysis, report, residual
from .align import Alignment, apply_alignment, find_alignment, rms, level_match_gain
from .dut import make_dut
from .model import LevelIndexedModel, fit_hammerstein
from .stimulus import Stimulus


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_dut(stim_dir: Path, spec: str, sets: dict, out: Path, ramps: list[str] | None = None,
            knob_mode: str = "auto", knob_steps: int = 16, **opts) -> Path:
    from .controls import parse_ramp, run_with_controls
    stim = Stimulus.load(stim_dir)
    dut = make_dut(spec, sets, **opts)
    schedule = dict(parse_ramp(r, stim) for r in (ramps or []))
    t0 = time.time()
    y = run_with_controls(dut, stim.signal, stim.fs, schedule, mode=knob_mode, steps=knob_steps)
    wall = time.time() - t0
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "response.wav", y.astype(np.float32), stim.fs, subtype="FLOAT")
    meta = {"dut": dut.describe(), "stimulus": str(Path(stim_dir).resolve()), "fs": stim.fs,
            "controls": schedule, "knob_mode": knob_mode if schedule else None,
            "stimulus_seconds": stim.seconds, "response_seconds": len(y) / stim.fs,
            "wall_seconds": wall, "created": datetime.now().isoformat(timespec="seconds")}
    (out / "run.json").write_text(json.dumps(meta, indent=1, default=_json))
    return out


def analyze_run(run_dir: Path, max_order: int = 9, ir_seconds: float = 0.2) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    stim = Stimulus.load(meta["stimulus"])
    y, fs = sf.read(run_dir / "response.wav", dtype="float64")
    assert fs == stim.fs

    al = find_alignment(stim, y)
    ya = apply_alignment(y, al, len(stim))
    sf.write(run_dir / "aligned.wav", ya.astype(np.float32), fs, subtype="FLOAT")

    # one IR set per sweep level; the reference level carries the headline numbers
    sw = stim.ref_sweep()
    f1, f2 = sw.meta["f1"], sw.meta["f2"]
    per_level = {}
    for seg in stim.sweeps():
        ir_full, origin = analysis.deconvolve(stim, ya, seg)
        irs_l = analysis.extract_harmonics(ir_full, origin, stim, max_order, ir_seconds, sweep=seg)
        f, H = analysis.spectrum(irs_l.linear, fs)
        g, m, ph = analysis.smoothed_response(f, H, f1, f2)
        per_level[seg.meta["level_db"]] = {"irs": irs_l, "fr_f": g, "fr_mag_db": m, "fr_phase": ph,
                                          "harm": analysis.harmonic_response(irs_l, f1, f2), "seg": seg}
    ref = per_level[sw.meta["level_db"]]
    irs, fr_f, fr_mag, fr_ph, harm = ref["irs"], ref["fr_f"], ref["fr_mag_db"], ref["fr_phase"], ref["harm"]
    tones = analysis.tone_metrics(stim, ya)
    bursts = analysis.burst_metrics(stim, ya)
    schedule = {k: [tuple(p) for p in v] for k, v in (meta.get("controls") or {}).items()}
    ramps = analysis.ramp_metrics(stim, ya, schedule) if schedule else []

    # sweep-derived model and where it fails
    models = [fit_hammerstein(v["irs"], lvl) for lvl, v in per_level.items()]
    ref_model = next(m for m in models if m.level_db == sw.meta["level_db"])
    yhat_ref = ref_model.apply(stim.signal)
    res = {"ref_only": residual.localize(stim, ya, yhat_ref, ref_model.level_db)}
    if len(models) > 1:
        yhat_all = LevelIndexedModel(models).apply(stim.signal, fs)
        res["all_levels"] = residual.localize(stim, ya, yhat_all, ref_model.level_db)
        res["all_levels"]["levels_db"] = sorted(m.level_db for m in models)
    (run_dir / "residual.json").write_text(json.dumps(res, indent=1, default=_json))
    np.savez_compressed(run_dir / "model.npz", fs=fs, pre=ref_model.pre,
                        levels_db=np.array([m.level_db for m in models]),
                        **{f"kernels_{m.level_db:g}dB": m.kernels for m in models})

    x_sw, y_sw = stim.cut(stim.signal, sw.name), stim.cut(ya, sw.name)
    noise = rms(stim.cut(ya, "pre")[int(0.1 * fs):])
    tail = ya[sw.stop : sw.stop + int(1.0 * fs)]
    an = {
        "fs": fs, "f1": f1, "f2": f2, "ir_linear": irs.linear, "ir_pre": irs.pre, "irs": irs.ir,
        "fr_f": fr_f, "fr_mag_db": fr_mag, "fr_phase": fr_ph, "harm": harm, "tones": tones,
        "bursts": bursts, "per_level": per_level, "residual": res,
    }
    ref_1k = float(np.interp(1000.0, fr_f, fr_mag))
    metrics = {
        "alignment": al.as_dict(),
        "sweep": {"in_rms_db": 20 * np.log10(rms(x_sw)), "out_rms": rms(y_sw), "out_rms_db": 20 * np.log10(rms(y_sw) + 1e-30),
                  "out_peak": float(np.max(np.abs(y_sw))), "out_dc": float(np.mean(y_sw)),
                  "gain_1k_db": ref_1k, "mag_rel_1k_db": {str(fq): float(np.interp(fq, fr_f, fr_mag) - ref_1k)
                                                            for fq in (50, 100, 200, 500, 2000, 5000, 10000) if f1 <= fq <= f2}},
        "noise": {"pre_rms": noise, "pre_rms_db": 20 * np.log10(noise + 1e-30),
                  "tail_rms_db": 20 * np.log10(rms(tail[len(tail) // 2:]) + 1e-30),
                  "snr_db": 20 * np.log10((rms(y_sw) + 1e-30) / (noise + 1e-30))},
        "harmonics_at_1k_db": {str(k): float(np.interp(1000.0, fin, rel)) for k, (fin, rel) in harm.items()},
        "levels": {f"{lvl:g}": {"gain_1k_db": float(np.interp(1000.0, v["fr_f"], v["fr_mag_db"])),
                                "harmonics_at_1k_db": {str(k): float(np.interp(1000.0, fin, rel)) for k, (fin, rel) in v["harm"].items()}}
                   for lvl, v in per_level.items()},
        "spikes": analysis.spike_metrics(stim, ya),
        "bursts": [{k: v for k, v in b.items() if not k.startswith(("envelope", "probe_envelope"))} for b in bursts],
        "model": {"ref_level_db": ref_model.level_db, "order": ref_model.order,
                  "clip_rsr_db": res["ref_only"].get("clip", {}).get("rsr_db"),
                  "clip_rsr_all_levels_db": res.get("all_levels", {}).get("clip", {}).get("rsr_db"),
                  "focus": res["ref_only"]["focus"],
                  "focus_all_levels": res.get("all_levels", {}).get("focus")},
        "ramps": ramps,
        "tones": tones,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=_json))
    np.savez_compressed(run_dir / "analysis.npz", fr_f=fr_f, fr_mag_db=fr_mag, fr_phase=fr_ph,
                        ir_linear=irs.linear, ir_pre=irs.pre, fs=fs, f1=f1, f2=f2,
                        **{f"ir_{k}": v for k, v in irs.ir.items()},
                        **{f"harm_{k}_f": v[0] for k, v in harm.items()},
                        **{f"harm_{k}_db": v[1] for k, v in harm.items()})
    sf.write(run_dir / "ir_linear.wav", (irs.linear / (np.max(np.abs(irs.linear)) or 1)).astype(np.float32), fs, subtype="FLOAT")
    meta["alignment"] = al.as_dict()
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1, default=_json))
    report.plot_run(an, run_dir, run_dir.name)
    report.plot_residual(res, run_dir, run_dir.name)
    if bursts:
        report.plot_bursts(bursts, run_dir, run_dir.name)
    if len(per_level) > 1:
        report.plot_levels(per_level, run_dir, run_dir.name)
    if ramps:
        report.plot_ramps(ramps, run_dir, run_dir.name)
    return metrics


def load_analysis(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    z = np.load(run_dir / "analysis.npz")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    harm = {}
    for k in range(2, 20):
        if f"harm_{k}_f" in z:
            harm[k] = (z[f"harm_{k}_f"], z[f"harm_{k}_db"])
    return {"fs": int(z["fs"]), "f1": float(z["f1"]), "f2": float(z["f2"]), "fr_f": z["fr_f"],
            "fr_mag_db": z["fr_mag_db"], "fr_phase": z["fr_phase"], "ir_linear": z["ir_linear"],
            "ir_pre": int(z["ir_pre"]), "harm": harm, "tones": metrics["tones"], "metrics": metrics,
            "run": json.loads((run_dir / "run.json").read_text())}


def compare_runs(run_a: Path, run_b: Path, out: Path, label_a: str | None = None, label_b: str | None = None) -> dict:
    run_a, run_b, out = Path(run_a), Path(run_b), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    la, lb = label_a or run_a.name, label_b or run_b.name
    a, b = load_analysis(run_a), load_analysis(run_b)
    if a["run"]["stimulus"] != b["run"]["stimulus"]:
        print(f"warning: runs used different stimuli\n  {a['run']['stimulus']}\n  {b['run']['stimulus']}")
    stim = Stimulus.load(a["run"]["stimulus"])
    ya, _ = sf.read(run_a / "aligned.wav", dtype="float64")
    yb, _ = sf.read(run_b / "aligned.wav", dtype="float64")

    # level-match both to unit RMS on the sweep so gain differences become one number
    ra, rb = rms(stim.cut(ya, "sweep")), rms(stim.cut(yb, "sweep"))
    gain_a_db, gain_b_db = -20 * np.log10(ra + 1e-30), -20 * np.log10(rb + 1e-30)
    fgrid = a["fr_f"]
    mb = np.interp(fgrid, b["fr_f"], b["fr_mag_db"])
    diff = (mb + gain_b_db) - (a["fr_mag_db"] + gain_a_db)
    cmp = {
        "a": la, "b": lb, "gain_a_db": gain_a_db, "gain_b_db": gain_b_db,
        "level_diff_db": float(gain_a_db - gain_b_db),   # b louder than a by this much on the sweep
        "diff_f": fgrid, "diff_mag_db": diff,
        "mag_diff_rms_db": float(np.sqrt(np.mean(diff ** 2))),
        "mag_diff_max_db": float(np.max(np.abs(diff))),
        "delay_diff_samples": b["metrics"]["alignment"]["delay_samples"] - a["metrics"]["alignment"]["delay_samples"],
        "sweep_residual": analysis.residual_metrics(stim.cut(ya, "sweep"), stim.cut(yb, "sweep"), stim.fs, f_hi=a["f2"]),
    }
    if any(s.kind == "clip" for s in stim.segments):
        ca, cb = stim.cut(ya, "clip"), stim.cut(yb, "clip")
        r = analysis.residual_metrics(ca, cb, stim.fs, f_hi=a["f2"])
        g = r["ls_gain"]
        n0 = int(0.5 * stim.fs)
        n1 = min(len(ca), n0 + int(0.05 * stim.fs))
        cmp["clip"] = {**r, "excerpt": [n0, n1], "a": (ca - ca.mean())[n0:n1] / ra, "b": (g * (cb - cb.mean()))[n0:n1] / ra}
    figs = report.plot_compare(a, b, cmp, out, la, lb)

    # markdown
    lines = [f"# {la} vs {lb}", "", f"stimulus: `{a['run']['stimulus']}`", "",
             "| | " + la + " | " + lb + " |", "|---|---|---|",
             f"| DUT | `{a['run']['dut'].get('spec')}` {a['run']['dut'].get('sets', '')} | `{b['run']['dut'].get('spec')}` {b['run']['dut'].get('sets', '')} |",
             f"| sweep out RMS (DUT units) | {ra:.4g} | {rb:.4g} |",
             f"| delay (samples) | {a['metrics']['alignment']['delay_samples']} | {b['metrics']['alignment']['delay_samples']} |",
             f"| polarity | {a['metrics']['alignment']['polarity']:+d} | {b['metrics']['alignment']['polarity']:+d} |",
             f"| noise floor / SNR (dB) | {a['metrics']['noise']['snr_db']:.1f} | {b['metrics']['noise']['snr_db']:.1f} |",
             f"| single-sample spikes | {a['metrics']['spikes']['count']} (max {a['metrics']['spikes']['max_ratio']:.1f}x) | {b['metrics']['spikes']['count']} (max {b['metrics']['spikes']['max_ratio']:.1f}x) |",
             f"| H2 at 1 kHz (dB) | {a['metrics']['harmonics_at_1k_db'].get('2', float('nan')):.1f} | {b['metrics']['harmonics_at_1k_db'].get('2', float('nan')):.1f} |",
             f"| H3 at 1 kHz (dB) | {a['metrics']['harmonics_at_1k_db'].get('3', float('nan')):.1f} | {b['metrics']['harmonics_at_1k_db'].get('3', float('nan')):.1f} |",
             "", "## Differences (b relative to a)", "",
             f"- level on the sweep: {cmp['level_diff_db']:+.2f} dB",
             f"- magnitude response after level match: RMS {cmp['mag_diff_rms_db']:.2f} dB, max {cmp['mag_diff_max_db']:.2f} dB",
             f"- delay: {cmp['delay_diff_samples']:+d} samples",
             f"- sweep residual after least-squares gain: ESR {cmp['sweep_residual']['esr']:.4g}, null depth {cmp['sweep_residual']['null_depth_db']:.1f} dB"]
    if "clip" in cmp:
        c = cmp["clip"]
        lines.append(f"- DI clip residual: ESR {c['esr']:.4g}, null depth {c['null_depth_db']:.1f} dB, LS gain {c['ls_gain_db']:+.2f} dB")
    if a["tones"] and b["tones"]:
        lines += ["", "## THD by tone (%)", "", "| freq | level | " + la + " | " + lb + " |", "|---|---|---|---|"]
        tb = {r["name"]: r for r in b["tones"]}
        for r in a["tones"]:
            s = tb.get(r["name"])
            lines.append(f"| {r['freq']:g} | {r['level_db']:g} | {r['thd']*100:.2f} | {s['thd']*100 if s else float('nan'):.2f} |")
    lines += ["", "## Figures", ""] + [f"![{p.name}]({p.name})" for p in figs]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    summary = {k: v for k, v in cmp.items() if not isinstance(v, np.ndarray) and k != "clip"}
    if "clip" in cmp:
        summary["clip"] = {k: v for k, v in cmp["clip"].items() if k not in ("a", "b")}
    (out / "compare.json").write_text(json.dumps(summary, indent=1, default=_json))
    return summary


# --------------------------------------------------------------------- grid
def _cells(grid: dict[str, list[float]]) -> list[dict]:
    import itertools
    names = list(grid)
    return [dict(zip(names, vals)) for vals in itertools.product(*(grid[n] for n in names))]


def run_grid(stim_dir: Path, spec: str, sets: dict, grid: dict[str, list[float]], out: Path,
             prompt: bool = False, **opts) -> Path:
    """One sub-run per combination of grid values, analyzed, indexed in grid.json.
    With prompt=True, waits for a human between cells (real knobs)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cells = _cells(grid)
    index = []
    for i, cell in enumerate(cells):
        name = "_".join(f"{k}={v:g}" for k, v in cell.items())
        if prompt:
            input(f"[{i+1}/{len(cells)}] set {name.replace('_', ', ')} then press enter: ")
        d = run_dut(stim_dir, spec, {**sets, **cell}, out / name, **opts)
        m = analyze_run(d)
        index.append({"cell": cell, "dir": str(d), "summary": summarize(m)})
        print(f"[{i+1}/{len(cells)}] {name}: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in index[-1]["summary"].items()))
    (out / "grid.json").write_text(json.dumps({"grid": grid, "spec": spec, "sets": sets, "cells": index}, indent=1, default=_json))
    surface(out)
    return out


def summarize(m: dict) -> dict:
    """The numbers a knob map is drawn from."""
    h = m["harmonics_at_1k_db"]
    out = {"gain_1k_db": m["sweep"]["gain_1k_db"], "out_rms": m["sweep"]["out_rms"],
           "h2_1k_db": h.get("2"), "h3_1k_db": h.get("3"),
           "hf_drop_10k_db": m["sweep"]["mag_rel_1k_db"].get("10000"),
           "clip_rsr_db": (m.get("model") or {}).get("clip_rsr_db"),
           "spikes": m["spikes"]["count"]}
    if m.get("bursts"):
        out["sag_db"] = m["bursts"][0]["sag_db"]
    return out


def surface(grid_dir: Path) -> Path:
    """grid.json -> surface.md table (+ heatmaps for two-parameter grids)."""
    grid_dir = Path(grid_dir)
    g = json.loads((grid_dir / "grid.json").read_text())
    names = list(g["grid"])
    keys = list(g["cells"][0]["summary"])
    lines = [f"# knob surface: `{g['spec']}` over {', '.join(names)}", "",
             "| " + " | ".join(names + keys) + " |", "|" + "---|" * (len(names) + len(keys))]
    for c in g["cells"]:
        vals = [f"{c['cell'][n]:g}" for n in names] + [("" if c["summary"][k] is None else f"{c['summary'][k]:.3g}") for k in keys]
        lines.append("| " + " | ".join(vals) + " |")
    if len(names) == 2:
        figs = report.plot_surface(g, grid_dir)
        lines += ["", "## Maps", ""] + [f"![{p.name}]({p.name})" for p in figs]
    (grid_dir / "surface.md").write_text("\n".join(lines) + "\n")
    return grid_dir / "surface.md"
