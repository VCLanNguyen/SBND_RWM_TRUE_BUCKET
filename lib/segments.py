"""Time chunks and segments from the calibration CSV, the traces in them, and the per-segment inputs of the extraction.

- Calibration CSV (SBND_RWM_CALIBRATION, one per run): one row per time chunk (start, end, period, segment number, segment status, skip reason)
- `SEGMENT_NAMES`: segment number → name; the extraction uses the stable segments `ad_response.SEGMENTS` (Run 1 Before Rotation B is unstable)
- A trace enters a segment if it is good (`qc`), unclipped and inside one of its time chunks, and the chunk has ≥ `MIN_AD` such traces
"""
import numpy as np
import pandas as pd

import paths
from ad_response import SEGMENTS, MIN_AD, AD_MAX_CLIPPED, AD_CONFIGS, ad_config, Grid

SEGMENT_NAMES = {1: {1: 'before_a', 2: 'before_b', 3: 'before_c', 4: 'during', 5: 'after'},
                 2: {1: 'wide', 2: 'normal_a', 3: 'normal_b', 4: 'narrow'}}


def load_calibration(run):
    """Fitted time chunks of the run (rows with a skip reason left out), sorted by start; `seg` = segment name (NaN: no segment)."""
    db = pd.read_csv(paths.CALIB[run], parse_dates=['start', 'end'])
    db = db[db['skip_reason'].fillna('') == ''].sort_values('start').reset_index(drop=True)
    db['seg'] = db['segment'].map(SEGMENT_NAMES[run])
    bad = [s for r, s in SEGMENTS if r == run and not (db.loc[db['seg'] == s, 'segment_status'] == 'Stable').all()]
    assert not bad, f'segments not Stable in {paths.CALIB[run]}: {bad}'
    return db


def select_chunks(run, ad_only=True):
    """{key: dict(sel = chunk rows, ad = good unclipped trace fits with `chunk`, P = median chunk period)} for the run's segments;
    `ad_only`: only chunks with ≥ `MIN_AD` traces. `ad['i']` = row of the trace in the Step 1 fits (and bucket npz)."""
    db = load_calibration(run)
    f = pd.read_csv(paths.fits_csv(run), parse_dates=['ts'])
    f['i'] = np.arange(len(f))     # row order = buckets npz
    f = f[(f['qc'] == 'good') & (f['n_clipped'] <= AD_MAX_CLIPPED)].copy()
    out = {}
    for r_, seg in SEGMENTS:
        if r_ != run:
            continue
        sel = db[db['seg'] == seg]
        fc = f.copy()
        fc['chunk'] = -1
        for ci, r in sel.iterrows():
            fc.loc[(fc['ts'] >= r['start']) & (fc['ts'] <= r['end']), 'chunk'] = ci
        n_ad = fc['chunk'].value_counts()
        if ad_only:
            sel = sel.loc[[ci for ci in sel.index if n_ad.get(ci, 0) >= MIN_AD]]
        out[(run, seg)] = dict(sel=sel, ad=fc[fc['chunk'].isin(sel.index)], P=sel['period'].median())
    return out


def segment_inputs(chunks):
    """Per segment: period P and its `Grid`, AD config (`grp_config`, one per segment), trace times and σ (`ad_ts`, `ad_sigma`: the fit window)."""
    out = {}
    for key, ch in chunks.items():
        ts = ch['ad']['ts'].values
        n = pd.Series(ad_config(ts)).value_counts()
        pns = [pn for pn in AD_CONFIGS if n.get(pn, 0) >= MIN_AD]
        assert len(pns) == 1, f'{key}: one AD config per segment, found {dict(n)}'
        P = np.float64(ch['P'])
        out[key] = dict(P=P, grid=Grid(P), grp_config=np.array([pns[0]]), ad_ts=ts, ad_sigma=ch['ad']['sigma'].values)
    return out
