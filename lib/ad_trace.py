"""Raw RWM traces from IFBeam (device `E:NBMI12WV[]`): IFBeam queries, the local trace store, and the Step 1 fit of each trace.

- One trace = one BNB spill: 12000 signed 8-bit ADC samples at 0.16 ns + a 3-value trailer (V/count, offset [V], sample interval [s])
- Store = one npz per DAQ run (`store_path`), key (run, event, ts_ns); event −1 = time-sampled trace (query time ts_ns), else the trace matched
  to an SBND event (record `CLOCK_OFFSET_MS` before the event time); `status` 0 = no IFBeam record
- `ensure` queries IFBeam only for keys not in the store (`fetch_trace` for SBND events, `fetch_trace_at` for time-sampled keys)
- `fit_trace` (Step 1): baseline, 81 buckets, per-bucket fits, period line, σ of the stacked buckets, guards; clipped samples (`ADC_MAX`) left out of every fit
"""
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import urllib3
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# edge buckets: partly filled buckets just outside the 81-window (Bucket Filling study)
EDGE_MAX_BUCKETS = 2        # per side
EDGE_MIN_OCCUPANCY = 0.1    # AD only: peak / median core amplitude (AD empty / partial dip)


def linear(x, m, b):
    return m*x + b


def gaussian(x, H, a, x0, sigma):
    return H + a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def asymmetric_gaussian(x, H, a, x0, sigma_left, sigma_right):
    """Gaussian with `sigma_left` below x0, `sigma_right` above."""
    sigma = np.where(x < x0, sigma_left, sigma_right)
    return H + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def find_dynamic_buckets(hist, centers, expected_n=81, min_distance=9, window=8):
    """Find `expected_n` evenly spaced buckets in `hist`.
    - Height-cut search over loosening steps; median spacing merges double detections
    - Exactly `expected_n` found is kept only if evenly spaced (`_regular_spacing`), else the comb decides
    - 84 RF buckets, 81 filled: weak satellites sit just outside, so one height cut fails at the margin
    - Returns the `expected_n` contiguous comb grid points with the most counts (`_select_best_window`)
    - No gap-fill: `_require_good_buckets` raises instead
    """
    noise = np.std(hist)

    best_under = None
    over = None
    buckets = None

    for step in [0.15, 0.2, 0.3, 0.5, 0.8, 1, 1.3, 1.5, 2, 2.5, 3]:
        found, _ = find_peaks(hist, height=step*noise, distance=min_distance)
        n = len(found)
        if n == expected_n and buckets is None:
            buckets = found
        if n < expected_n:
            if best_under is None or n > best_under[0]:
                best_under = (n, found)
        else:
            # smallest over-detection: loose cuts add noise bumps (124 vs 82 real case)
            if over is None or n < over[0]:
                over = (n, found)

    if buckets is None:
        if over is not None:
            buckets = over[1]
        elif best_under is not None and best_under[0] >= 2:
            buckets = best_under[1]
        else:
            return best_under[1] if best_under is not None else np.array([], dtype=int)

    if len(buckets) < 2:
        return buckets

    # merge double detections of one peak
    spacing_guess = np.median(np.diff(centers[buckets]))
    merged = [buckets[0]]
    for p in buckets[1:]:
        if centers[p] - centers[merged[-1]] < 0.6 * spacing_guess:
            if hist[p] > hist[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    buckets = np.array(merged)

    # an exact count can still be wrong: a missed bucket + a side bump (Run 2 run 20040, gaps 37 / 23 + 16 ns)
    if len(buckets) == expected_n and _regular_spacing(centers[buckets]):
        return buckets

    comb = _select_best_window(hist, centers, buckets, expected_n, window)
    if comb is not None:
        return comb

    # fallback: height-based trim
    if len(buckets) > expected_n:
        keep = np.argsort(hist[buckets])[::-1][:expected_n]
        return np.array(sorted(buckets[keep]))
    return buckets


def _regular_spacing(positions, lo=0.6, hi=1.4):
    """True if every gap between neighbouring `positions` is within [`lo`, `hi`] x the median gap."""
    gaps = np.diff(positions)
    med = np.median(gaps)
    return bool(np.all((gaps >= lo * med) & (gaps <= hi * med)))


def _select_best_window(hist, centers, buckets, expected_n, window, margin=4):
    """Comb at the median spacing, `margin` grid points past each end; the `expected_n` contiguous grid points with the most counts, or None."""
    spacing = np.median(np.diff(centers[buckets]))
    if not np.isfinite(spacing) or spacing <= 0:
        return None

    grid = centers[buckets[0]] - margin * spacing + spacing * np.arange(
        len(buckets) + 2 * margin)

    points = []
    for pos in grid:
        nbhd = np.where(np.abs(centers - pos) < 0.4 * spacing)[0]
        if len(nbhd) == 0:
            continue
        best = nbhd[np.argmax(hist[nbhd])]
        count = hist[max(0, best - window):best + window].sum()
        points.append((best, count))

    if len(points) < expected_n:
        return None

    idxs = np.array([s[0] for s in points])
    counts = np.array([s[1] for s in points], dtype=float)
    window_sums = np.array([counts[i:i + expected_n].sum()
                             for i in range(len(counts) - expected_n + 1)])
    best_start = np.argmax(window_sums)
    return idxs[best_start:best_start + expected_n]


def _fit_gaussian_windowed(bins_mid, hist, raw_counts, guess, window_sigma, min_window_bins=8, poisson_weight=False):
    """Gaussian fit, then refit within `window_sigma`·sigma of the center; returns (popt, pcov, mask).
    - Keeps the bucket tails / histogram edge out of the fit; full fit if < `min_window_bins`
    - Poisson weights from `raw_counts`, scaled to the density `hist`
    """
    if poisson_weight:
        nz = raw_counts > 0
        scale = np.median(hist[nz] / raw_counts[nz]) if nz.any() else 1.0
        poisson_sigma = np.sqrt(np.clip(raw_counts, 1, None)) * scale
        fit_kwargs = dict(sigma=poisson_sigma, absolute_sigma=True)
    else:
        fit_kwargs = {}
    popt0, pcov0 = curve_fit(gaussian, bins_mid, hist, p0=guess, **fit_kwargs)
    full_mask = np.ones(len(bins_mid), dtype=bool)
    if not window_sigma:
        return popt0, pcov0, full_mask

    center0, sigma0 = popt0[2], abs(popt0[3])
    mask = np.abs(bins_mid - center0) <= window_sigma * sigma0
    if mask.sum() < min_window_bins:
        return popt0, pcov0, full_mask

    try:
        w_kwargs = dict(sigma=poisson_sigma[mask], absolute_sigma=True) if poisson_weight else {}
        popt_w, pcov_w = curve_fit(gaussian, bins_mid[mask], hist[mask],
                                    p0=[guess[0], max(hist[mask]), center0, sigma0], **w_kwargs)
        return popt_w, pcov_w, mask
    except Exception:
        return popt0, pcov0, full_mask


def _fit_asym_gaussian(bins_mid, hist, raw_counts, mask, sym_popt, poisson_weight=False):
    """`asymmetric_gaussian` on the sym fit's mask and units, seeded from `sym_popt`."""
    guess = [sym_popt[0], sym_popt[1], sym_popt[2], abs(sym_popt[3]), abs(sym_popt[3])]
    if poisson_weight:
        nz = raw_counts > 0
        scale = np.median(hist[nz] / raw_counts[nz]) if nz.any() else 1.0
        poisson_sigma = np.sqrt(np.clip(raw_counts[mask], 1, None)) * scale
        fit_kwargs = dict(sigma=poisson_sigma, absolute_sigma=True)
    else:
        fit_kwargs = {}
    popt, pcov = curve_fit(asymmetric_gaussian, bins_mid[mask], hist[mask], p0=guess, **fit_kwargs)
    return popt, pcov


def line_terms(popt, pcov):
    """Period-line terms of one fit (popt = (period, value at fit index 0)) → `line0`, `line0_err`, `line_cov` (cov period, line0)."""
    return {'line0': popt[1], 'line0_err': np.sqrt(pcov[1, 1]), 'line_cov': pcov[0, 1]}


def unwrap_bucket0(bucket0, period):
    """Rows in time order → (phase-continuous bucket-0 value, window offset in buckets); bucket0 = unwrapped + offset · period.
    - The 81-window can start one bucket early / late (partly filled edge buckets) → the bucket-0 value jumps by one period
    - Row to row, a jump > period / 2 = window shift; smaller = real phase step (e.g. Run 1 LLRF switch 04-23, −8.5 ns)
    - Offsets are relative to the first row's window
    """
    fp, P = np.asarray(bucket0, float), np.asarray(period, float)
    off = np.zeros(len(fp), dtype=int)
    un = fp.copy()
    for i in range(1, len(fp)):
        off[i] = int(np.rint((fp[i] - un[i - 1]) / P[i]))
        un[i] = fp[i] - off[i] * P[i]
    return un, off


def add_window_offset(df, time_col='start', good=None):
    """`window_offset` + `intercept` / `intercept_err` columns, frame fixed over the rows (`unwrap_bucket0` of `line0`, `time_col` order).
    - `window_offset`: fixed-frame bucket number of fit index 0; fixed-frame bucket number = fit index (or edge index) + `window_offset`; same number = same bucket in every row
    - `intercept`: period line at bucket 0 of the fixed frame = `line0` − `window_offset` · period; error from the line covariance
      (= spill position from all 81 buckets: no one-period window jumps, not pulled by a half-filled first bucket)
    - `good`: rows to unwrap over (default all); others get NA
    """
    m = np.ones(len(df), dtype=bool) if good is None else np.asarray(good, dtype=bool)
    d = df[m].sort_values(time_col)
    _, off = unwrap_bucket0(d['line0'], d['period'])
    df['window_offset'] = pd.Series(off, index=d.index).reindex(df.index).astype('Int64')
    o = df['window_offset'].astype(float)
    df['intercept'] = df['line0'] - o * df['period']
    df['intercept_err'] = np.sqrt(df['line0_err'] ** 2 + o ** 2 * df['period_err'] ** 2 - 2 * o * df['line_cov'])
    return df


IFBEAM_URL = 'https://dbweb2.fnal.gov:8443/ifbeam/data/data'
IFBEAM_VAR = 'E:NBMI12WV[]'
CLOCK_OFFSET_MS = -35.5
MATCH_TOL_MS = 10.0
N_SAMPLES = 12000
ADC_MAX = 127
AT_WINDOW_MS = 300
BASELINE_SAMPLES = 400
MIN_AMP_COUNTS = 30
BUCKET_WINDOW_NS = 8.0
MIN_BUCKET_DIST_NS = 9.0
WINDOW_SIGMA = 3
MAX_POS_ERR = 5.0
MAX_SIGMA = 5.0
MAX_PERIOD_CHI2NDOF = 5000


def _utc(ns):
    return pd.Timestamp(int(ns), unit='ns', tz='UTC').strftime('%Y-%m-%dT%H:%M:%S.%f') + '+00:00'


def fetch_trace(ts_ns, session=None, retries=2):
    """Trace matched to one SBND event, or None (no matching record); raises `requests.RequestException` on a failed query.
    - Queries [ts − 60 ms, ts − 10 ms] (UTC); keeps the record nearest `CLOCK_OFFSET_MS`, within `MATCH_TOL_MS`
    - Returns dict: clock_ms, dt_ms (clock − event), samples (int16), trailer (3)
    """
    s = session or requests
    params = f'e=e,1d&v={IFBEAM_VAR}&t0={_utc(ts_ns - 60e6)}&t1={_utc(ts_ns - 10e6)}&f=csv'
    for attempt in range(retries + 1):
        try:
            r = s.get(f'{IFBEAM_URL}?{params}', timeout=60, verify=False)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries:
                raise
    best = None
    for line in r.text.splitlines()[1:]:
        f = line.split(',')
        if len(f) < 5 + N_SAMPLES:
            continue
        dt = int(f[3]) - ts_ns / 1e6
        if best is None or abs(dt - CLOCK_OFFSET_MS) < abs(best[1] - CLOCK_OFFSET_MS):
            best = (f, dt)
    if best is None or abs(best[1] - CLOCK_OFFSET_MS) > MATCH_TOL_MS:
        return None
    f, dt = best
    vals = np.array(f[5:], dtype=float)
    return dict(clock_ms=int(f[3]), dt_ms=dt, samples=vals[:N_SAMPLES].astype(np.int16), trailer=vals[N_SAMPLES:])


def fetch_trace_at(ts_ns, session=None, retries=2):
    """First trace in [ts, ts + `AT_WINDOW_MS`] (UTC), or None (no record); time-sampled, no SBND event; raises on a failed query.
    - Returns the same dict as `fetch_trace`; dt_ms = clock − ts
    """
    s = session or requests
    params = f'e=e,1d&v={IFBEAM_VAR}&t0={_utc(ts_ns)}&t1={_utc(ts_ns + AT_WINDOW_MS * 1e6)}&f=csv'
    for attempt in range(retries + 1):
        try:
            r = s.get(f'{IFBEAM_URL}?{params}', timeout=60, verify=False)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries:
                raise
    recs = [f for f in (line.split(',') for line in r.text.splitlines()[1:]) if len(f) >= 5 + N_SAMPLES]
    if not recs:
        return None
    f = min(recs, key=lambda f: int(f[3]))
    vals = np.array(f[5:], dtype=float)
    return dict(clock_ms=int(f[3]), dt_ms=int(f[3]) - ts_ns / 1e6,
                samples=vals[:N_SAMPLES].astype(np.int16), trailer=vals[N_SAMPLES:])


STORE_KEYS = ('run', 'event', 'ts_ns', 'status', 'clock_ms', 'dt_ms', 'samples', 'trailer')


def _empty_entries():
    return dict(run=np.zeros(0, np.int64), event=np.zeros(0, np.int64), ts_ns=np.zeros(0, np.int64),
                status=np.zeros(0, np.int8), clock_ms=np.zeros(0, np.int64), dt_ms=np.zeros(0),
                samples=np.zeros((0, N_SAMPLES), np.int16), trailer=np.zeros((0, 3)))


def _take(d, idx):
    return {k: d[k][idx] for k in STORE_KEYS}


def _concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in STORE_KEYS}


def store_path(store_dir, run):
    return os.path.join(store_dir, f'run_{int(run)}.npz')


def read_store(store_dir, runs):
    """Stored entries of `runs` (one npz per DAQ run): run, event (−1 = time-sampled), ts_ns (query time),
    status (1 record / 0 no record), clock_ms, dt_ms, samples (zeros if no record), trailer."""
    parts = [dict(np.load(store_path(store_dir, r))) for r in np.unique(runs) if os.path.exists(store_path(store_dir, r))]
    return _concat([_empty_entries()] + parts)


def store_add(store_dir, new):
    """Merge `new` entries into the per-run files; key (run, event, ts_ns), existing entries kept."""
    os.makedirs(store_dir, exist_ok=True)
    for run in np.unique(new['run']):
        old = read_store(store_dir, [run])
        add = _take(new, new['run'] == run)
        have = set(zip(old['event'], old['ts_ns']))
        keep = np.array([(e, t) not in have for e, t in zip(add['event'], add['ts_ns'])], dtype=bool)
        if keep.any():
            m = _concat([old, _take(add, keep)])
            np.savez_compressed(store_path(store_dir, run), **_take(m, np.argsort(m['ts_ns'], kind='stable')))


def fetch_entries(sel, fetch=fetch_trace, n_threads=8):
    """Query IFBEAM for the rows of `sel` (run, event, ts_ns) -> entries; failed queries left out (warned, retried next time)."""
    def one(ts, sess):
        try:
            return fetch(ts, sess)
        except requests.RequestException:
            return 'error'

    with requests.Session() as sess, ThreadPoolExecutor(n_threads) as pool:
        res = list(pool.map(lambda t: one(t, sess), sel['ts_ns'].values))
    ok = [i for i, r in enumerate(res) if not isinstance(r, str)]
    if len(ok) < len(res):
        print(f'WARNING: {len(res) - len(ok)}/{len(res)} IFBEAM queries failed (not stored, retried on the next call)')
    rec = [res[i] for i in ok]
    return dict(
        run=sel['run'].values[ok].astype(np.int64), event=sel['event'].values[ok].astype(np.int64),
        ts_ns=sel['ts_ns'].values[ok].astype(np.int64),
        status=np.array([r is not None for r in rec], dtype=np.int8),
        clock_ms=np.array([r['clock_ms'] if r else -1 for r in rec], dtype=np.int64),
        dt_ms=np.array([r['dt_ms'] if r else np.nan for r in rec], dtype=float),
        samples=np.array([r['samples'] if r else np.zeros(N_SAMPLES, np.int16) for r in rec], dtype=np.int16).reshape(len(rec), N_SAMPLES),
        trailer=np.array([r['trailer'] if r else np.full(3, np.nan) for r in rec], dtype=float).reshape(len(rec), 3),
    )


def ensure(store_dir, sel, fetch=fetch_trace, allow_fetch=True):
    """Stored entries for the rows of `sel` (run, event, ts_ns), in `sel` order; only rows not yet stored are queried.
    - `fetch`: `fetch_trace` (matched to SBND events) or `fetch_trace_at` (time-sampled)
    - `allow_fetch=False`: never queries IFBEAM; rows not stored are left out
    - Includes no-record entries (`status` 0)
    """
    have = read_store(store_dir, sel['run'].values)
    keys = list(zip(sel['run'], sel['event'], sel['ts_ns']))
    idx = {k: i for i, k in enumerate(zip(have['run'], have['event'], have['ts_ns']))}
    todo = np.array([k not in idx for k in keys], dtype=bool)
    if todo.any() and allow_fetch:
        new = fetch_entries(sel[todo], fetch)
        store_add(store_dir, new)
        have = read_store(store_dir, sel['run'].values)
        idx = {k: i for i, k in enumerate(zip(have['run'], have['event'], have['ts_ns']))}
    return _take(have, np.array([idx[k] for k in keys if k in idx], dtype=int))


def trace_qc(samples):
    """Per-trace QC scalars: baseline + rms (first `BASELINE_SAMPLES`), amplitude, min, max,
    `n_clipped` (samples at the 8-bit ceiling `ADC_MAX`), md5 (stuck-readout check)."""
    y = samples.astype(float)
    pre = y[:BASELINE_SAMPLES]
    base = np.median(pre)
    return dict(baseline=base, rms=pre.std(), amp=y.max() - base, ymin=y.min(), ymax=y.max(),
                n_clipped=int((samples >= ADC_MAX).sum()), md5=hashlib.md5(samples.tobytes()).hexdigest())


def _fit_bucket(t, s, valid, p, half, model):
    """One bucket: pass 1 free-H sym on ±`half` samples (seed), pass 2 H = 0 (baseline) on ±`WINDOW_SIGMA`·σ.
    - Clipped samples (`valid` False) are left out of both passes
    """
    v = valid[p - half:p + half]
    x, y = t[p - half:p + half][v], s[p - half:p + half][v]
    popt1, _, _ = _fit_gaussian_windowed(x, y, y, [0, y.max(), x[np.argmax(y)], 1.5], None)
    x0, sig = popt1[2], abs(popt1[3])
    m = np.abs(x - x0) <= WINDOW_SIGMA * sig
    if m.sum() < 8:
        m = np.ones(len(x), bool)
    if model == 'sym':
        f = lambda xx, a, c, w: gaussian(xx, 0.0, a, c, w)
        popt, pcov = curve_fit(f, x[m], y[m], p0=[popt1[1], x0, sig])
        popt = np.array([0.0, popt[0], popt[1], abs(popt[2])])
        perr = np.r_[0.0, np.sqrt(np.diag(pcov))]
    else:
        f = lambda xx, a, c, wl, wr: asymmetric_gaussian(xx, 0.0, a, c, wl, wr)
        popt, pcov = curve_fit(f, x[m], y[m], p0=[popt1[1], x0, sig, sig])
        popt = np.array([0.0, popt[0], popt[1], abs(popt[2]), abs(popt[3])])
        perr = np.r_[0.0, np.sqrt(np.diag(pcov))]
    return popt, perr


def stacked_profile(t, s, positions, period, valid=None, bin_ns=0.16):
    """Mean waveform vs time to the nearest bucket position, ±period/2 (the waveform analogue of the SBND stacked bunch).
    - `valid`: sample mask (clipped samples left out of the mean)
    """
    xp = np.asarray(positions)
    lo, hi = xp[0] - period / 2, xp[-1] + period / 2
    m = (t >= lo) & (t < hi) & (valid if valid is not None else True)
    tt, yy = t[m], s[m]
    idx = np.clip(np.searchsorted(xp, tt), 1, len(xp) - 1)
    left, right = xp[idx - 1], xp[idx]
    resid = tt - np.where(np.abs(tt - left) < np.abs(tt - right), left, right)
    edges = np.arange(-period / 2, period / 2 + bin_ns, bin_ns)
    num, _ = np.histogram(resid, edges, weights=yy)
    cnt, _ = np.histogram(resid, edges)
    ok = cnt > 0
    return ((edges[:-1] + edges[1:]) / 2)[ok], num[ok] / cnt[ok], cnt[ok]


EDGE_WIDTH = 2 * EDGE_MAX_BUCKETS
EDGE_ARRAY_KEYS = ('edge_index', 'edge_positions', 'edge_pos_err', 'edge_amp', 'edge_sigma')


def _fit_edge_buckets(t, s, valid, half, line, n_core, amp_ref, model):
    """Partly filled buckets outside the core 81 on one trace → dict of `EDGE_ARRAY_KEYS`, `EDGE_WIDTH` wide, NaN-padded.
    - Candidate bucket indices k = −1, −2 … / 81, 82 … at the period line `line` = (period, intercept); walk outward, stop at the first candidate
      with peak < `EDGE_MIN_OCCUPANCY` × `amp_ref`, a failed `_fit_bucket`, pos err / σ > 5 ns or > 0.3 period off the line
    """
    period, intercept = line
    found = []
    for side in (np.arange(-1, -EDGE_MAX_BUCKETS - 1, -1), np.arange(n_core, n_core + EDGE_MAX_BUCKETS)):
        for kc in side:
            pred = intercept + period * kc
            p = int(np.argmin(np.abs(t - pred)))
            if p - half < 0 or p + half > len(s):
                break
            v = valid[p - half:p + half]
            if not v.any() or s[p - half:p + half][v].max() < EDGE_MIN_OCCUPANCY * amp_ref:
                break
            try:
                popt, perr = _fit_bucket(t, s, valid, p, half, model)
            except (RuntimeError, ValueError):
                break
            if not (np.isfinite(perr[2]) and perr[2] <= MAX_POS_ERR and np.abs(popt[3:]).max() <= MAX_SIGMA
                    and abs(popt[2] - pred) <= 0.3 * period):
                break
            found.append((kc, popt[2], perr[2], popt[1], np.abs(popt[3:]).mean()))
    found.sort()
    out = {k: np.full(EDGE_WIDTH, np.nan) for k in EDGE_ARRAY_KEYS}
    for j, f in enumerate(found):
        for k, x in zip(EDGE_ARRAY_KEYS, f):
            out[k][j] = x
    return out


def fit_trace(samples, trailer, model='sym'):
    """Per-trace calibration; returns (row dict, per-bucket dict) or raises ValueError with the reason.
    - `model`: per-bucket shape for the positions ('sym' | 'asym'); the stacked buckets are fit with both
    """
    dt_ns = trailer[-1] * 1e9 if len(trailer) and 1e-11 < trailer[-1] < 1e-9 else 0.16
    t = np.arange(len(samples)) * dt_ns
    y = samples.astype(float)
    s = y - np.median(y[:BASELINE_SAMPLES])
    rms = y[:BASELINE_SAMPLES].std()
    valid = samples < ADC_MAX
    if s.max() < MIN_AMP_COUNTS:
        raise ValueError('no beam')

    half = int(round(BUCKET_WINDOW_NS / dt_ns))
    buckets = find_dynamic_buckets(s, t, expected_n=81, min_distance=int(MIN_BUCKET_DIST_NS / dt_ns), window=half)
    if len(buckets) != 81:
        raise ValueError(f'{len(buckets)}/81 buckets')
    if buckets[0] < half or buckets[-1] + half > len(s):
        raise ValueError('bucket at trace edge')

    popts, perrs = [], []
    for p in buckets:
        try:
            popt, perr = _fit_bucket(t, s, valid, p, half, model)
        except (RuntimeError, ValueError):
            raise ValueError('bucket fit failed')
        popts.append(popt)
        perrs.append(perr)
    popts, perrs = np.array(popts), np.array(perrs)
    pos, pos_err = popts[:, 2], perrs[:, 2]
    if not np.all(np.isfinite(pos_err)) or pos_err.max() > MAX_POS_ERR:
        raise ValueError('bucket position error')
    if np.abs(popts[:, 3:]).max() > MAX_SIGMA:
        raise ValueError('bucket sigma > 5 ns')

    k = np.arange(81)
    lp, lc = curve_fit(linear, k, pos, sigma=pos_err, absolute_sigma=True)
    resid = pos - linear(k, *lp)
    period = lp[0]
    period_chi2ndof = np.sum((resid / pos_err) ** 2) / 79
    if period_chi2ndof > MAX_PERIOD_CHI2NDOF:
        raise ValueError('bucket index slip')

    ox, oy, on = stacked_profile(t, s, pos, period, valid)
    osym, oc, om = _fit_gaussian_windowed(ox, oy, oy, [0, oy.max(), 0, 1.5], WINDOW_SIGMA)
    oasym, oac = _fit_asym_gaussian(ox, oy, oy, om, osym)

    row = dict(
        period=period, period_err=np.sqrt(lc[0, 0]), period_chi2ndof=period_chi2ndof, **line_terms(lp, lc),
        firstpeak=pos[0], firstpeak_err=pos_err[0],
        sigma=abs(osym[3]), sigma_err=np.sqrt(oc[3, 3]),
        sigma_l=abs(oasym[3]), sigma_l_err=np.sqrt(oac[3, 3]), sigma_r=abs(oasym[4]), sigma_r_err=np.sqrt(oac[4, 4]),
        skew=abs(oasym[4]) - abs(oasym[3]),
        stacked_H=osym[0], stacked_amp=osym[1], stacked_center=osym[2],
        stacked_asym_H=oasym[0], stacked_asym_amp=oasym[1], stacked_asym_center=oasym[2],
        bucket_sigma_mean=np.mean(popts[:, 3] if model == 'sym' else popts[:, 3:5].mean(axis=1)),
        amp_median=np.median(popts[:, 1]), rms=rms,
    )
    edges = _fit_edge_buckets(t, s, valid, half, lp, len(pos), np.median(popts[:, 1]), model)
    row['n_buckets'] = len(pos) + int(np.isfinite(edges['edge_index']).sum())
    per_bucket = dict(peak_positions=pos, peak_pos_err=pos_err, period_resid=resid,
                      bucket_amp=popts[:, 1], bucket_sigma=popts[:, 3],
                      bucket_sigma_r=popts[:, 4] if model == 'asym' else np.full(81, np.nan), **edges)
    return row, per_bucket


PER_BUCKET_KEYS = ('peak_positions', 'peak_pos_err', 'period_resid', 'bucket_amp', 'bucket_sigma', 'bucket_sigma_r',
                   *EDGE_ARRAY_KEYS)


def fit_entries(d, model='sym', **meta):
    """Fit every record (`status` 1) in entries `d`; returns (rows, per-bucket dict of lists); `meta` is copied into every row."""
    rows, pb = [], {k: [] for k in PER_BUCKET_KEYS}
    for i in np.flatnonzero(d['status'] == 1):
        r = dict(**meta, run=int(d['run'][i]), event=int(d['event'][i]),
                 ts=pd.Timestamp(int(d['ts_ns'][i]), unit='ns'), clock_ms=int(d['clock_ms'][i]),
                 dt_ms=float(d['dt_ms'][i]), **trace_qc(d['samples'][i]), fail='')
        try:
            fit, bucket = fit_trace(d['samples'][i], d['trailer'][i], model)
            r.update(fit)
        except ValueError as e:
            r['fail'] = str(e)
            bucket = {k: np.full(EDGE_WIDTH if k in EDGE_ARRAY_KEYS else 81, np.nan) for k in PER_BUCKET_KEYS}
        for k in PER_BUCKET_KEYS:
            pb[k].append(bucket[k])
        rows.append(r)
    return rows, pb


def add_qc(fits):
    """`qc` column: 'good'; else the fit failure, or 'stuck readout' (trace bytes identical to another trace).
    - `window_offset` + `intercept` (good traces, time order, `add_window_offset`): trace-frame bucket number of bucket 0 and the period line
      there; a trace-frame jump (06-05 Run 2: 167 → 67 ns) shows up as a multi-bucket offset
    - `firstpeak` (bucket-0 fit) kept per trace as a check; chunk tables + plots use `intercept`
    """
    stuck = fits['md5'].duplicated(keep=False)
    fits['qc'] = np.where(fits['fail'] != '', fits['fail'], np.where(stuck, 'stuck readout', 'good'))
    good = (fits['qc'] == 'good').values
    if 'line0' in fits and good.any():
        add_window_offset(fits, time_col='ts', good=good)
    return fits
