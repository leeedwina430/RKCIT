# RHSIC and RKCIT

| File | Purpose |
|---|---|
| `RHSIC.py` | Unconditional independence test: `RHSIC_RFF_test(x, y, s)` |
| `RKCIT.py` | Conditional independence test: `RKCIT_test(x, y, z, s)` |
| `demo.py` | DG I, DG II and conditional data generation, method calls and repeated-trial summaries |

## Main difference from the previous version: Gaussian kernel implementation

The original [RHSIC.py](https://github.com/leeedwina430/RKCIT/blob/main/RHSIC.py) uses `kerpy.GaussianKernel`; this version uses an inlined Gaussian-kernel formula matching `causallearn`.

## Install and run

Use Python 3.12 with the versions in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python demo.py
```

The algorithms require NumPy, SciPy and scikit-learn; the demo also uses threadpoolctl. No external generator file, experiment directory, kerpy, causal-learn, PyTorch or directory is needed.

Edit the settings at the bottom of `demo.py`, following the original demo's style.

| Setting | Meaning / default |
|---|---|
| `gen_type="dataset_1"` | DG I with RHSIC; `tau` is Gaussian noise standard deviation, default 0.5 |
| `gen_type="dataset_2"` | DG II with RHSIC; `tau` is the lower-tail fraction, default 0.01 |
| `gen_type="conditional"` | Conditional data with RKCIT; `tau` defaults to 0.03 |
| `num_samples=1000` | Number of observations; CI requires an even count |
| `test_type=False` | H1 / power; set `True` for H0 / Type I error |
| `noise="gaussian"` | Noise used in the comparison; Laplace is also supported |
| `seed=0`, `num_trials=1` | One example; use `seed=0`, `num_trials=100` for all 100 trial indices |
| `dim_z=10` | CI conditioning dimension |
| `ci_max_iter=100` | CI optimizer limit |
| `alpha=0.05` | Significance threshold for the summary |

`tau` is selected from `gen_type` in the settings block; edit its value to run a different setting. Each trial prints its p-value and statistic. `run_experiment(...)` returns rows `[seed, p_value, statistic]`; an optional CSV export is shown at the bottom of the file.

## Reproduce the experiment settings

All settings were included in the completed comparison, with 100 planned trial indices per setting.


## Call from your own code

```python
from RHSIC import RHSIC_RFF_test
from RKCIT import RKCIT_test

p_ui, statistic_ui = RHSIC_RFF_test(x, y, s)
p_ci, statistic_ci = RKCIT_test(x, y, z, s)
p_ci200, statistic_ci200 = RKCIT_test(x, y, z, s, max_iter=200)
```

Inputs must be finite numeric arrays with matching row counts. Inputs are converted to float64; one-dimensional inputs become single columns. The caller controls the global NumPy random state; neither entry point seeds it internally. Use the generation/seeding order described above when matching the demo.

RHSIC can also return held-out weights and data:

```python
p_value, statistic, test_weights, x_test, y_test = RHSIC_RFF_test(
    x, y, s, return_betas=True)
```
