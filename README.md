# ShopFloor

Industrial fault diagnosis: multi-output classification of hydraulic component
condition from raw sensor data, with a retrieval-augmented agent that explains the
diagnosis and cites the relevant maintenance procedure.

**Status:** week 1 of 13 — data pipeline and loaders.

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

The three rates are reconciled differently depending on the model:

- **Feature baselines** — statistics computed per sensor at its native rate, no
  alignment needed. The standard deviation of a 100 Hz pressure trace retains
  high-frequency content that downsampling would destroy.
- **Convolutional model** — aligned to 10 Hz: fast channels block-averaged, with
  the block standard deviation kept as an extra channel; slow channels repeated
  rather than interpolated, since the sensor genuinely did not measure in between.

Downsampling everything to 1 Hz was rejected: valve faults are switching-lag
faults measured in milliseconds, and 1 Hz destroys exactly that signal.

## Quick start

```
git clone https://github.com/OlenaP94/shopfloor && cd shopfloor
uv sync
make data      # downloads and validates 73 MB from UCI
make check     # format, lint, tests
```

## Layout

```
src/shopfloor/data.py       readers for the sensor matrices and profile labels
src/shopfloor/dataset.py    HydraulicDataset — validated access to one experiment
scripts/download_data.py    download, checksum and structural validation
tests/                      unit tests, no dataset required
```

## Licence

Apache-2.0. See `NOTICE`.

Dataset: Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition monitoring
of hydraulic systems* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5CW21>
