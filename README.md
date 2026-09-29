# ir-harness

Round-trip audio measurement for anything with an input and an output: a
circuit simulation, a pedal, an amp into a load box, or an audio interface
looped back on itself. One stimulus goes in, the response is aligned and
measured, and two runs are compared with the same numbers regardless of what
produced them. The point is to put circuit-decimator's models and real
hardware on the same axes. See `BRIEF.md` for scope.

## Install

```
uv venv .venv && uv pip install -e ".[dev]"      # add ",audio" for the interface adapter
.venv/bin/python -m pytest tests
```

The `sim:*` adapters need `../circuit-decimator/build` (override with
`CIRCUIT_DECIMATOR=/path`). The `audio:*` adapter needs `sounddevice`.

## Use

```
irh gen -o stim/default --di path/to/di.wav          # 192 kHz, 10 s sweep, stepped tones, DI clip
irh run --stim stim/default --dut sim:ff -o runs/ff_baseline
irh run --stim stim/default --dut sim:ff --set vcc=4.5 -o runs/ff_starve
irh compare runs/ff_baseline runs/ff_starve -o compare/baseline_vs_starve
irh devices                                          # list audio devices
irh run --stim stim/default --dut audio:3:3 --out-db -12 -o runs/loopback
```

`irh run` writes the raw response, aligns it, and analyzes it. `irh compare`
level-matches two analyzed runs on the sweep and writes `report.md` plus
figures.

### What a stimulus contains

| segment | what it measures |
|---|---|
| sync burst | coarse landmark for eyeballing alignment |
| synchronized exponential sweep, 20 Hz to 20 kHz, -12 dBFS | linear IR and, from the same deconvolution, the harmonic IRs of order 2..5 (Farina / Novak) |
| stepped tones, 5 frequencies x 6 levels from -40 to 0 dBFS | THD, per-harmonic level, output level and DC versus drive: the level-dependent part sweeps can't see |
| DI clip | the thing you actually hear; two runs are nulled against each other on it (ESR, null depth, 1/3-octave band difference) |

Silence before, between and after gives noise floor and settling time.

### DUT specs

| spec | device |
|---|---|
| `ident`, `gain:<db>`, `delay:<n>`, `lowpass:<hz>[:order]`, `tanh:<drive>` | known-answer fixtures; the tests check the analysis against them in closed form |
| `sim:ff` | circuit-decimator Fuzz Face via `ff_render`; `--set` takes the `.param` names from `sim/fuzzface.cir`; `--solver mna\|dk\|dkf` |
| `sim:seout`, `sim:shinei`, `sim:tubepre` | circuit-decimator netlist circuits |
| `plugin:<path.vst3>` | any VST3 through circuit-decimator's headless host |
| `audio:<out_dev>:<in_dev>` | play and record through an interface (`--out-db`, `--out-channel`, `--in-channel`) |

Sim adapters take `--volts-per-fs` (default 0.25 V at the circuit input for a
full-scale sample). Sim output is in volts; the comparison level-matches, so
absolute units only matter for drive.

### Run directory

```
run.json       DUT description and parameters, stimulus path, alignment
response.wav   raw output, unaligned, DUT units
aligned.wav    shifted to stimulus time, polarity corrected
metrics.json   alignment, sweep level and response points, noise/SNR, harmonics at 1 kHz, spikes, tones
analysis.npz   IRs (linear + harmonic), smoothed frequency response, harmonic response curves
ir_linear.wav  normalized linear IR
fig_sweep.png  IR, magnitude, harmonic distortion vs frequency
fig_tones.png  THD and output level vs input level
```

## How the measurement works

Alignment finds the linear IR peak in the deconvolved sweep; it survives
heavy distortion and yields delay and polarity in one step. The inverse sweep
is normalized so an identity DUT gives a 0 dB passband, so IRs read as
output-per-full-scale-input. Harmonic IRs land `L ln(k)` seconds before the
linear one and are windowed out with a shared pre-window of up to 50 ms,
because band-limiting at 20 Hz rings symmetrically and cutting it costs
low-frequency accuracy. Tone metrics project each steady-state tone onto its
harmonics over an integer number of cycles. Residuals use a least-squares
gain then NAM's error-to-signal ratio.

## First results (circuit-decimator Fuzz Face, 192 kHz)

- DK solver vs the reference MNA solver: identical to a -92 dB null on the
  DI clip. The realtime solver is not the source of any model error.
- The DK solver emits single-sample spikes at hard clipping edges: 16 in the
  0 dBFS 440 Hz tone at 192 kHz, up to 2.8x the segment's 99.9th percentile.
  At 48 kHz without oversampling the same glitches reach 27 V on a 9 V
  circuit. The `spikes` metric in `metrics.json` counts these.
- Starving to 4.5 V: -9 dB level, a 40 dB loss at 20 kHz, H2 rising from
  -35 dB to -2 dB at 1 kHz, and bias drift visible in the DI clip residual.

## Not yet

Instrument control (the Rigol gear here isn't USB), motorized knobs, NAM
comparison, a JMP model, psychoacoustics. They plug into the DUT slot and the
report, not into the measurement code.
