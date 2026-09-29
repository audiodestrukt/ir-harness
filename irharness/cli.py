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
    g.add_argument("--sweep-db", type=float, default=-12.0)
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
    r.add_argument("--volts-per-fs", type=float, default=0.25, help="sim adapters: volts at the circuit input for a full-scale sample")
    r.add_argument("--solver", default="dk", help="sim:ff only: mna|dk|dkf")
    r.add_argument("--out-db", type=float, default=0.0, help="audio: output attenuation")
    r.add_argument("--out-channel", type=int, default=1)
    r.add_argument("--in-channel", type=int, default=1)
    r.add_argument("--keep-files", action="store_true", help="sim: keep the text in/out files")
    r.add_argument("--no-analyze", action="store_true")
    r.add_argument("-o", "--out", required=True, type=Path)

    a = sub.add_parser("analyze", help="(re)analyze a run directory")
    a.add_argument("run", type=Path)
    a.add_argument("--max-order", type=int, default=5)

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
                     tone_freqs=tuple(float(x) for x in args.tone_freqs.split(",")),
                     tone_dbs=tuple(float(x) for x in args.tone_dbs.split(",")),
                     tone_seconds=args.tone_seconds, di=args.di, di_db=args.di_db, di_max_seconds=args.di_max_seconds)
        stim.save(Path(args.out))
        print(f"{args.out}: {stim.seconds:.2f} s at {stim.fs} Hz, {len(stim.segments)} segments, "
              f"sweep L={stim.seg('sweep').meta['L']:.6f} s")
        return 0

    if args.cmd == "run":
        from .session import run_dut, analyze_run
        opts = {}
        if args.dut.startswith("sim:"):
            opts = {"volts_per_fs": args.volts_per_fs, "keep_files": args.keep_files}
            if args.dut == "sim:ff":
                opts["solver"] = args.solver
        elif args.dut.startswith("audio:"):
            opts = {"out_db": args.out_db, "out_channel": args.out_channel, "in_channel": args.in_channel}
        out = run_dut(args.stim, args.dut, _kv(args.set), args.out, **opts)
        meta = json.loads((out / "run.json").read_text())
        print(f"{out}: {meta['response_seconds']:.2f} s response in {meta['wall_seconds']:.1f} s wall")
        if not args.no_analyze:
            m = analyze_run(out)
            _print_metrics(m)
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


if __name__ == "__main__":
    sys.exit(main())
