"""irh: the command line.

  irh gen  -o stim/default [--fs 192000] [--sweep-seconds 10] [--di clip.wav] ...
  irh run  --stim stim/default --dut sim:ff --set vcc=6 -o runs/ff_vcc6
  irh analyze runs/ff_vcc6
  irh compare runs/ff_baseline runs/ff_vcc6 -o compare/baseline_vs_vcc6
  irh devices
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = float(v)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="irh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a stimulus set")
    g.add_argument("-o", "--out", required=True)
    g.add_argument("--fs", type=int, default=192000)
    g.add_argument("--sweep-seconds", type=float, default=10.0)
    g.add_argument("--f1", type=float, default=20.0)
    g.add_argument("--f2", type=float, default=20000.0)
    g.add_argument("--sweep-db", type=float, default=-12.0, help="reference sweep level (alignment, headline numbers)")
    g.add_argument("--sweep-dbs", default="", help="extra sweep levels, e.g. -36,-24,0 (each adds ~11 s)")
    g.add_argument("--burst-freqs", default="110,1000", help="tone-burst frequencies; empty to skip")
    g.add_argument("--burst-db", type=float, default=0.0)
    g.add_argument("--probe-db", type=float, default=-30.0)
    g.add_argument("--tone-freqs", default="82,220,440,1000,3000")
    g.add_argument("--tone-dbs", default="-40,-32,-24,-16,-8,0")
    g.add_argument("--tone-seconds", type=float, default=0.25)
    g.add_argument("--di", type=Path, default=None, help="DI clip WAV (mono-mixed, resampled)")
    g.add_argument("--di-db", type=float, default=-6.0)
    g.add_argument("--di-max-seconds", type=float, default=20.0)

    r = sub.add_parser("run", help="run a DUT on a stimulus (and analyze)")
    r.add_argument("--stim", required=True, type=Path)
    r.add_argument("--dut", required=True)
    r.add_argument("--set", action="append", default=[], help="name=value, repeatable")
    r.add_argument("--ramp", action="append", default=[], help="name:from:to@segment (e.g. fuzz:0.05:1@knob_ramp), repeatable")
    r.add_argument("--knob-mode", default="auto", choices=["auto", "native", "stepwise", "prompt"],
                   help="how ramps are honoured: native automation, stepwise re-runs, or prompting a human")
    r.add_argument("--knob-steps", type=int, default=16, help="stepwise: pieces per ramp")
    r.add_argument("--volts-per-fs", type=float, default=0.25, help="sim adapters: volts at the circuit input for a full-scale sample")
    r.add_argument("--solver", default="dk", help="sim:ff only: mna|dk|dkf")
    r.add_argument("--out-db", type=float, default=0.0, help="audio: output attenuation")
    r.add_argument("--out-channel", type=int, default=1)
    r.add_argument("--in-channel", type=int, default=1)
    r.add_argument("--keep-files", action="store_true", help="sim: keep the text in/out files")
    r.add_argument("--no-analyze", action="store_true")
    r.add_argument("-o", "--out", required=True, type=Path)

    gr = sub.add_parser("grid", help="run every combination of knob values, one sub-run each, then map the surface")
    gr.add_argument("--stim", required=True, type=Path)
    gr.add_argument("--dut", required=True)
    gr.add_argument("--set", action="append", default=[], help="fixed name=value")
    gr.add_argument("--grid", action="append", required=True, help="name=v1,v2,v3 (repeat for more axes)")
    gr.add_argument("--prompt", action="store_true", help="wait for a human between cells (real knobs)")
    gr.add_argument("--volts-per-fs", type=float, default=0.25)
    gr.add_argument("--solver", default="dk")
    gr.add_argument("--out-db", type=float, default=0.0)
    gr.add_argument("--out-channel", type=int, default=1)
    gr.add_argument("--in-channel", type=int, default=1)
    gr.add_argument("-o", "--out", required=True, type=Path)

    su = sub.add_parser("surface", help="rebuild surface.md and maps from a grid directory")
    su.add_argument("grid", type=Path)

    a = sub.add_parser("analyze", help="(re)analyze a run directory")
    a.add_argument("run", type=Path)
    a.add_argument("--max-order", type=int, default=9, help="harmonic IRs to extract; the model picks its own order from what is above the floor")

    c = sub.add_parser("compare", help="compare two analyzed runs")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("-o", "--out", required=True, type=Path)
    c.add_argument("--label-a")
    c.add_argument("--label-b")

    sub.add_parser("devices", help="list audio devices")

    args = p.parse_args(argv)

    if args.cmd == "gen":
        from .stimulus import build
        stim = build(fs=args.fs, sweep_seconds=args.sweep_seconds, f1=args.f1, f2=args.f2, sweep_db=args.sweep_db,
                     sweep_dbs=tuple(float(x) for x in args.sweep_dbs.split(",") if x.strip()),
                     burst_freqs=tuple(float(x) for x in args.burst_freqs.split(",") if x.strip()),
                     burst_db=args.burst_db, probe_db=args.probe_db,
                     tone_freqs=tuple(float(x) for x in args.tone_freqs.split(",")),
                     tone_dbs=tuple(float(x) for x in args.tone_dbs.split(",")),
                     tone_seconds=args.tone_seconds, di=args.di, di_db=args.di_db, di_max_seconds=args.di_max_seconds)
        stim.save(Path(args.out))
        print(f"{args.out}: {stim.seconds:.2f} s at {stim.fs} Hz, {len(stim.segments)} segments, "
              f"sweep L={stim.ref_sweep().meta['L']:.6f} s, levels {[s.meta['level_db'] for s in stim.sweeps()]} dBFS")
        return 0

    if args.cmd in ("run", "grid"):
        from .session import run_dut, analyze_run, run_grid
        opts = {}
        if args.dut.startswith("sim:"):
            opts = {"volts_per_fs": args.volts_per_fs, "keep_files": getattr(args, "keep_files", False)}
            if args.dut == "sim:ff":
                opts["solver"] = args.solver
        elif args.dut.startswith("audio:"):
            opts = {"out_db": args.out_db, "out_channel": args.out_channel, "in_channel": args.in_channel}
        if args.cmd == "grid":
            grid = {}
            for gspec in args.grid:
                k, _, v = gspec.partition("=")
                grid[k] = [float(x) for x in v.split(",")]
            d = run_grid(args.stim, args.dut, _kv(args.set), grid, args.out, prompt=args.prompt, **opts)
            print(f"{d}/surface.md")
            return 0
        out = run_dut(args.stim, args.dut, _kv(args.set), args.out, ramps=args.ramp,
                      knob_mode=args.knob_mode, knob_steps=args.knob_steps, **opts)
        meta = json.loads((out / "run.json").read_text())
        print(f"{out}: {meta['response_seconds']:.2f} s response in {meta['wall_seconds']:.1f} s wall")
        if not args.no_analyze:
            m = analyze_run(out)
            _print_metrics(m)
        return 0

    if args.cmd == "surface":
        from .session import surface
        print(surface(args.grid))
        return 0

    if args.cmd == "analyze":
        from .session import analyze_run
        _print_metrics(analyze_run(args.run, max_order=args.max_order))
        return 0

    if args.cmd == "compare":
        from .session import compare_runs
        s = compare_runs(args.a, args.b, args.out, args.label_a, args.label_b)
        print(f"{args.out}/report.md")
        print(f"  level diff {s['level_diff_db']:+.2f} dB, magnitude diff RMS {s['mag_diff_rms_db']:.2f} dB max {s['mag_diff_max_db']:.2f} dB, "
              f"delay diff {s['delay_diff_samples']:+d}")
        if "clip" in s:
            print(f"  DI clip: ESR {s['clip']['esr']:.4g}, null depth {s['clip']['null_depth_db']:.1f} dB")
        return 0

    if args.cmd == "devices":
        from .dut.interface import list_devices
        print(list_devices())
        return 0
    return 1


def _print_metrics(m):
    al, sw, nz = m["alignment"], m["sweep"], m["noise"]
    print(f"  delay {al['delay_samples']} samples, polarity {al['polarity']:+d}, IR peak {al['ir_peak']:.4g}")
    print(f"  sweep out RMS {sw['out_rms']:.4g} (peak {sw['out_peak']:.4g}, dc {sw['out_dc']:+.4g}), SNR {nz['snr_db']:.1f} dB")
    h = m["harmonics_at_1k_db"]
    print("  harmonics at 1 kHz: " + ", ".join(f"H{k} {v:.1f} dB" for k, v in h.items()))
    sp = m["spikes"]
    if sp["count"]:
        print(f"  single-sample spikes: {sp['count']} (largest {sp['max_ratio']:.1f}x its segment's 99.9th percentile) in "
              + ", ".join(f"{k} x{v['count']}" for k, v in sp["by_segment"].items()))
    rel = sw["mag_rel_1k_db"]
    print("  magnitude re 1 kHz: " + ", ".join(f"{k} Hz {v:+.1f}" for k, v in rel.items()))
    for b in m.get("bursts", []):
        print(f"  burst {b['freq']:g} Hz: sag {b['sag_db']:+.2f} dB (min {b['min_db']:+.2f}), dc {b['dc_start']:+.3f} -> {b['dc_end']:+.3f}; "
              f"probe deficit {b.get('probe_deficit_db', float('nan')):+.2f} dB, recovered in {b.get('recovery_s', float('nan')):.2f} s")
    for r in m.get("ramps", []):
        prm = r["params"][0]
        rows = r["rows"]
        print(f"  ramp {prm} over {r['segment']}: THD {rows[0]['thd']*100:.1f}% -> {rows[-1]['thd']*100:.1f}%, "
              f"level {rows[0]['out_rms_db']:.1f} -> {rows[-1]['out_rms_db']:.1f} dB, H3 {rows[0]['h3_db']:.1f} -> {rows[-1]['h3_db']:.1f} dB")
    mo = m.get("model")
    if mo:
        line = f"  sweep model (order {mo['order']} at {mo['ref_level_db']:g} dBFS) on the DI clip: RSR {mo['clip_rsr_db']:.1f} dB" if mo.get("clip_rsr_db") is not None else f"  sweep model (order {mo['order']})"
        if mo.get("clip_rsr_all_levels_db") is not None:
            line += f"; with all sweep levels {mo['clip_rsr_all_levels_db']:.1f} dB"
        print(line)
        for f_ in mo["focus"]:
            print(f"    focus: {f_}")
        if mo.get("focus_all_levels"):
            for f_ in mo["focus_all_levels"]:
                print(f"    focus (all levels): {f_}")


if __name__ == "__main__":
    sys.exit(main())
