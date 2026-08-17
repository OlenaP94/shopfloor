# ShopFloor

Industrial fault diagnosis: multi-output classification of hydraulic component
condition from raw sensor data, with a retrieval-augmented agent that explains the
diagnosis and cites the relevant maintenance procedure.

**Status:** week 3 of 13 — data pipeline, resampling and exploratory analysis done;
feature baselines next.

## Data

[UCI 447 — Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
(Helwig, Pignanelli & Schütze, Saarland University, CC BY 4.0).

2205 cycles of 60 s from a hydraulic test rig. 17 sensors at three sampling rates:

| Sensors | Quantity | Unit | Rate | Points per cycle |
|---|---|---|---|---|
| PS1–PS6 | pressure | bar | 100 Hz | 6000 |
| EPS1 | motor power | W | 100 Hz | 6000 |
| FS1, FS2 | volume flow | l/min | 10 Hz | 600 |
| TS1–TS4 | temperature | °C | 1 Hz | 60 |
| VS1 | vibration | mm/s | 1 Hz | 60 |
| CE, CP, SE | cooling efficiency, cooling power, efficiency factor | %, kW, % | 1 Hz | 60 |

Four components are graded independently:

| Component | States | Healthy cycles |
|---|---|---|
| Cooler | 100% / 20% / 3% | 741 (34%) |
| Valve | 100 / 90 / 80 / 73 | 1125 (51%) |
| Pump leakage | none / weak / severe | 1221 (55%) |
| Accumulator | 130 / 115 / 100 / 90 bar | 599 (27%) |
| **All four at once** | — | **21 (0.95%)** |

## Why supervised classification, not anomaly detection

Only 21 of 2205 cycles are healthy in all four components — 10 if the stable flag
is also required. The rig was run as a factorial experiment across 3 × 4 × 3 × 4 =
144 fault combinations at roughly 15 cycles each, so a "normal" population was
never collected. Reconstruction-based anomaly detection has nothing to train on.

The supervised framing is also the more useful one here: the output is a
diagnosis ("valve at 80 — severe lag, pump leaking weakly") rather than an alarm,
which is what the downstream agent needs in order to retrieve the right procedure.
Severity grades give prioritisation: 90 means schedule it, 73 means stop now.

An autoencoder is planned as a second layer for faults outside the four known
components — a classifier for what is known, an anomaly detector for the rest.

## Sampling rates

`make tensor` reconciles the three rates into one array of shape
`(2205, 24, 600)`, float32, 127 MB — down from roughly 3 GB as Python lists.

The target is 10 Hz, the middle of the three rates. Downsampling everything to
1 Hz was rejected: valve faults are switching-lag faults measured in
milliseconds, and 1 Hz destroys exactly that signal. Keeping 100 Hz would have
produced a 1.3 GB tensor for no gain on the slow channels.

17 sensors become 24 channels:

- **100 Hz** — each cycle is reshaped into 600 blocks of 10 samples, and both the
  block **mean** and the block **standard deviation** are kept. Plain decimation
  would discard the within-block ripple; the std channel preserves it in one
  number per block. Seven sensors, fourteen channels.
- **10 Hz** — passed through unchanged. Two channels.
- **1 Hz** — each reading repeated ten times rather than interpolated, since the
  sensor genuinely did not measure in between and interpolation would invent
  values. Eight channels.

The std channels were not free insurance: `PS1_std` turns out to be the single
most responsive channel for accumulator wear (see below).

## What the sensors say

`make eda` scores every channel against every component by the standardised
difference (Cohen's *d*) between healthy cycles and the most degraded ones, using
the cycle-wide mean of each channel.

| Component | Most responsive channels | *d* |
|---|---|---|
| Cooler | CE, PS6_mean, PS5_mean, TS4, TS1 | 32.5 · 11.9 · 11.9 · 10.5 · 9.3 |
| Valve | SE, FS1, EPS1_mean, PS3_mean, PS6_std | 0.57 · 0.45 · 0.41 · 0.38 · 0.38 |
| Pump leakage | SE, EPS1_mean, FS1, PS3_mean, PS1_mean | 1.16 · 1.09 · 1.00 · 0.83 · 0.70 |
| Accumulator | PS1_std, EPS1_mean, PS1_mean, CP, PS2_mean | 0.96 · 0.70 · 0.70 · 0.66 · 0.65 |

![Most responsive channels per component](reports/figures/rankings.png)

Three findings shape the modelling that follows.

**The accumulator is diagnosed by ripple, not by level.** Its job is to damp
pressure pulsations, so a worn one shows up as increased variability rather than
a shifted mean — which is why `PS1_std` leads and `PS1_mean` trails it. This
channel exists only because the resampling step kept block standard deviations.

**CE, CP and SE are computed, not measured.** The dataset marks CE and CP as
*virtual*, and SE is an efficiency factor derived the same way. `CE` is literally
cooling efficiency in percent — using it to predict cooler condition is close to
circular, and its *d* of 32.5 (three perfectly separated bands, no overlap) says
so. This is not label leakage in the strict sense, since the value is available
at inference time, but it makes one of the four tasks trivial and hides whether
the model learned anything. **Every result is therefore reported twice: with and
without the three virtual channels.**

**Cycle-wide averages are the wrong summary.** The valve scores *d* = 0.57 at
best, which reads as "barely detectable" — but the waveform plot shows the four
severity levels diverging sharply during the switching transients at 9–11 s and
19–21 s, and running together the rest of the time. Two informative seconds are
being diluted across sixty. The same holds for the accumulator, whose `PS1_std`
signal is concentrated in pulsation spikes at 20 s and 50 s. Features must
therefore be computed over windows, not over whole cycles — and this is the
argument for the convolutional model, which locates the informative window
itself.

![Mean waveform per component state](reports/figures/waveforms.png)

## Quick start

```
git clone https://github.com/OlenaP94/shopfloor && cd shopfloor
uv sync
make data      # downloads and validates 73 MB from UCI
make tensor    # resamples 17 sensors into a 24-channel tensor
make eda       # scores every channel against every fault
make check     # format, lint, tests
```

## Layout

```
src/shopfloor/data.py       readers for the sensor matrices and profile labels
src/shopfloor/dataset.py    HydraulicDataset — validated access to one experiment
src/shopfloor/arrays.py     resampling to one (cycles, channels, timepoints) tensor
src/shopfloor/config.py     settings, read from the environment or .env
scripts/download_data.py    download, checksum and structural validation
scripts/eda.py              which channels respond to which fault
reports/figures/            plots, regenerated by `make eda`
tests/                      unit tests, no dataset required
```

## Licence

Apache-2.0. See `NOTICE`.

Dataset: Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring
of hydraulic systems* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5CW21>
