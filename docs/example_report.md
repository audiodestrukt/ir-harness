# ff_baseline vs ff_starve_4v5

stimulus: `/home/dan/sandbox/audiodestrukt/ir-harness/stim/default`

| | ff_baseline | ff_starve_4v5 |
|---|---|---|
| DUT | `sim:ff` {} | `sim:ff` {'vcc': 4.5} |
| sweep out RMS (DUT units) | 1.574 | 0.5488 |
| delay (samples) | 5 | 7 |
| polarity | +1 | +1 |
| noise floor / SNR (dB) | 146.7 | 182.7 |
| single-sample spikes | 16 (max 2.8x) | 2 (max 2.6x) |
| H2 at 1 kHz (dB) | -35.5 | -1.8 |
| H3 at 1 kHz (dB) | -9.6 | -5.2 |

## Differences (b relative to a)

- level on the sweep: -9.15 dB
- magnitude response after level match: RMS 9.41 dB, max 40.02 dB
- delay: +2 samples
- sweep residual after least-squares gain: ESR 0.7724, null depth -1.1 dB
- DI clip residual: ESR 0.868, null depth -0.6 dB, LS gain +2.66 dB

## THD by tone (%)

| freq | level | ff_baseline | ff_starve_4v5 |
|---|---|---|---|
| 82 | -40 | 0.61 | 18.10 |
| 220 | -40 | 12.20 | 17.03 |
| 440 | -40 | 19.91 | 16.73 |
| 1000 | -40 | 21.83 | 15.85 |
| 3000 | -40 | 12.41 | 11.74 |
| 82 | -32 | 27.35 | 148.23 |
| 220 | -32 | 37.68 | 141.46 |
| 440 | -32 | 39.14 | 141.57 |
| 1000 | -32 | 39.53 | 135.89 |
| 3000 | -32 | 34.89 | 43.23 |
| 82 | -24 | 46.01 | 166.34 |
| 220 | -24 | 43.75 | 160.09 |
| 440 | -24 | 43.22 | 160.91 |
| 1000 | -24 | 42.93 | 163.72 |
| 3000 | -24 | 42.41 | 167.80 |
| 82 | -16 | 48.51 | 125.38 |
| 220 | -16 | 44.43 | 119.29 |
| 440 | -16 | 43.29 | 118.99 |
| 1000 | -16 | 42.77 | 120.30 |
| 3000 | -16 | 42.98 | 123.63 |
| 82 | -8 | 51.54 | 89.99 |
| 220 | -8 | 46.22 | 81.26 |
| 440 | -8 | 44.12 | 77.15 |
| 1000 | -8 | 42.64 | 74.18 |
| 3000 | -8 | 42.79 | 69.21 |
| 82 | 0 | 58.40 | 77.87 |
| 220 | 0 | 49.67 | 68.78 |
| 440 | 0 | 45.51 | 61.72 |
| 1000 | 0 | 42.12 | 54.02 |
| 3000 | 0 | 43.21 | 50.89 |

## Figures

![fig_cmp_sweep.png](fig_cmp_sweep.png)
![fig_cmp_tones.png](fig_cmp_tones.png)
![fig_cmp_clip.png](fig_cmp_clip.png)
