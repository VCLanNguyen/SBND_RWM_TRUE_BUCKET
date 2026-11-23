"""Input / output paths (str, absolute) -- every file path goes through here; set them with `set_paths` in each notebook's Setup cell.

- Inputs: `STORE[run]` = folder of the stored raw RWM traces (one `run_<DAQ run>.npz` per DAQ run); `CALIB[run]` = calibration CSV of the run
  (`sbnd_fine_calibration.csv` from SBND_RWM_CALIBRATION: time chunks, periods, segments); `INPUTS` = small inputs shipped with the repo
- Outputs: `OUT` (default `<repo>/output`): `OUT/run{1,2}/` = Step 1 trace fits, `OUT/` = extraction products (`out_file`)
- CSV output: `write_csv(df, path)` -- one timestamp format everywhere (pandas `to_csv` picks it per chunk of rows)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
INPUTS = f'{ROOT}/inputs'

STORE = {1: None, 2: None}       # raw-trace store folder per run (`set_paths`)
CALIB = {1: None, 2: None}       # calibration CSV per run (`set_paths`)
OUT = f'{ROOT}/output'           # `set_paths(out=)`


def set_paths(store=None, calib=None, out=None):
    """`store` = {run: raw-trace store folder}, `calib` = {run: calibration CSV}, `out` = output folder (default `<repo>/output`);
    checks that the calibration CSVs exist, creates `out/run1`, `out/run2` (and the store folders, so a fetch can fill them)."""
    global OUT
    for run, p in (calib or {}).items():
        if not p or not os.path.isfile(p):
            raise FileNotFoundError(f'Run {run} calibration CSV not found: {p!r} -- set CALIB_CSV in the notebook Setup cell')
        CALIB[run] = os.path.abspath(p)
    for run, p in (store or {}).items():
        if not p:
            raise FileNotFoundError(f'Run {run} trace store not set -- set STORE in the notebook Setup cell')
        STORE[run] = os.path.abspath(p)
        os.makedirs(STORE[run], exist_ok=True)
    OUT = os.path.abspath(out) if out else OUT
    for run in (1, 2):
        os.makedirs(f'{OUT}/run{run}', exist_ok=True)


def fits_csv(run):
    """Step 1 per-trace fits of the run (`trace_fits` notebook)."""
    return f'{OUT}/run{run}/trace_fits.csv'


def buckets_npz(run):
    """Step 1 per-bucket fit results of the run, rows = `fits_csv` rows."""
    return f'{OUT}/run{run}/trace_buckets.npz'


def out_file(name):
    return f'{OUT}/{name}'


def write_csv(df, path):
    """`df.to_csv(path, index=False)` with every datetime column (datetime64 or object of Timestamps) as 'YYYY-MM-DD HH:MM:SS.fffffffff',
    empty for NaT; other columns unchanged."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype == object and len(s.dropna()) and s.dropna().map(lambda v: isinstance(v, pd.Timestamp)).all():
            s = pd.to_datetime(s)
        if pd.api.types.is_datetime64_any_dtype(s):
            txt = np.char.replace(np.datetime_as_string(s.values.astype('datetime64[ns]'), unit='ns'), 'T', ' ')
            out[c] = np.where(s.isna().values, '', txt)
    out.to_csv(path, index=False)
