"""True bucket b over the full train (Steps 2-6) and its errors (Step 6 item 3).

- Step 2, reflections per AD config (b-free): round trip (RWM ↔ scope over the Heliax, 2 × 198 ns = 21 P − 1.5 ns, lands on slot k + 21) = FIR h(τ)
  fitted to the train-end probes (`ad_response.ad_term_products`); short reflection (≈ 17-19 ns) = δ at T, ρ fitted on the mean trace after the last bucket
  (`reflection_models`); per trace D(t) = Y(t) − ρ D(t − T) − (h ⊛ Y)(t − 21 P + 1.5) (`bootstrap_trace`), bucket positions refit on D (`refit_positions`)
- Step 3: buckets aligned on their positions, averaged per trace into 4 groups (head = slots `FT_HEAD_FIRST`-20, 22-42, 43-63, 64-81; `group_blocks`)
- Steps 4-5: H (1 MΩ) and correction taps fitted, R_AD per segment (`ft_context`); per group forward fit of agg2 ⊗ R_AD under `AD_B_COND` (`ft_fits`)
- Step 6: b = amplitude-weighted mixture of the 4 group b (`group_mixture`) ⊗ G(s²) (`b_smear`, `AD_B_S2`); errors = bootstrap (`ft_bootstrap`),
  extraction-choice variants with s² retuned per variant (`ft_systematics`, `s2_tune`), time slices, ripple (`ft_ripple_closure`) → `ft_uncertainty`, `ft_covariance`
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gamma

import paths
import ad_trace as at
from segments import select_chunks
from ad_response import (AD_B_TIE, BIN_NS, N_SLOTS, OV_BINS, AD_SKIP_SLOTS, N_GROUPS, ad_period, rt_group, trace_profiles,
                         Grid, bunch_shape, bunch_rms, ad_blocks, ad_kernels_periods, ad_fit_fwd, ad_window,
                         AD_FWD_N, SEGMENTS, BL, ECHOES, AD_BOOT, AD_H1M, AD_H1M_HANDLE, AD_H1M_TAU_VAR, AD_H1M_NECHO_VAR,
                         AD_TAPS_PROC, AD_B_MIN, fit_ad_taps, fit_ad_h1m, h1m_instrument, bunch_side_rms, _resampler,
                         AD_B_COND, AD_B_S2, AD_TAPS_SHARE, AD_SCOPE_FORM, TAP_LAM, TAP_MU, TAP_SETS, RC_TAU, PAR, _fit_ad_taps_3b, SUB)

FT_HEAD_FIRST = 6                                   # first AD slot (1-based) of the head: slots 1-5 = bucket-filling ramp after the notch
FT_RT_TAU = np.arange(-3.0, 15.0 + 1e-9, BIN_NS)    # round-trip FIR lags [ns] after 21 P − 1.5 (< P: no aliasing with the next bunch)
FT_X = np.arange(-40.0, 130.0, BIN_NS)              # train-end window [ns from the last core bunch]
FT_SHORT_T = (14.0, 20.0)                           # short-reflection delay scan [ns]
# § Forward Fold checks per configuration period (b-free levers on each step of the fold)
FF_PERIODS = ('Run 1.1', 'Run 1.3', 'Run 2.1', 'Run 2.2')   # periods with AD traces in the fit (Run 1.2 clipped)
FT_SAVE_PARS = ('sl', 'sr', 'bl', 'br', 'rms_left', 'rms_right', 'rms', 'side_L', 'side_R', 'LmR')   # full-train product: agg2 to the mixture, its per-side RMS, mixture full RMS
FT_WIN_SYST = {'lo': 2.25, 'hi': 2.75}                        # window systematic ±0.25 σ_AD around `AD_FWD_N` (b only, taps fixed), kept per side
FT_FLOOR = 0.01                                               # 'weights' variant: error floor (f · block peak)² of the conditioned fits, b only
FT_SCOPE_VAR = ('butter4', 'butter8')                         # 'scope' variants around `AD_SCOPE_FORM` (taps refit, `ad_taps_cond`)
FT_RFL_VAR = (('round trip × 0.8', dict(s_rt=0.8)), ('round trip × 1.2', dict(s_rt=1.2)), ('round-trip lag −0.05 ns', dict(delta=-0.05)),
              ('round-trip lag +0.05 ns', dict(delta=0.05)), ('short ρ × 0.5', dict(rho=0.5)), ('short ρ × 1.5', dict(rho=1.5)))   # 'reflection' variants around the current model (all periods at once)
FT_ERR_GROUPS = ('window', 'weights', 'taps', 'taps procedure', 'hardware', 'echo model', 'scope', 'alignment', 'H(2.1)', 'H skew', 's2', 'reflection', 'ripple', 'time')
FT_DIAG_GROUPS = ('ripple', 'time')                           # per-segment groups (+ stat): diagonal in the cross-segment covariance
S2_SCAN = np.arange(0.0, 0.5001, 0.025)                       # s² scan of `s2_tune` [ns²]
FT_RIPPLE_DONORS = 6                                          # ripple closure: donor groups (other segments) injected per group


def group_mask(P, m):
    """Slots of round-trip group m (`rt_group`) without `AD_SKIP_SLOTS`; group 0 = the head (from `FT_HEAD_FIRST`)."""
    msk = (rt_group(P) == m) & ~np.isin(np.arange(N_SLOTS), AD_SKIP_SLOTS)
    return msk & (np.arange(N_SLOTS) >= FT_HEAD_FIRST - 1) if m == 0 else msk


def _overlay(t, s, pos, P, valid):
    # overlay binning: nearest bucket, BIN_NS bins on ± P/2, per slot → (S, N)
    m = (t >= pos[0] - P / 2) & (t < pos[-1] + P / 2) & valid
    tt, yy = t[m], s[m]
    ii = np.clip(np.searchsorted(pos, tt), 1, len(pos) - 1)
    near = np.where(np.abs(tt - pos[ii - 1]) < np.abs(tt - pos[ii]), ii - 1, ii)
    edges = np.arange(-P / 2, P / 2 + BIN_NS, BIN_NS)
    bi = np.searchsorted(edges, tt - pos[near], side='right') - 1
    bi[(tt - pos[near]) == edges[-1]] = len(edges) - 2
    ok = (bi >= 0) & (bi < len(edges) - 1)
    flat = near[ok] * OV_BINS + bi[ok]
    return (np.bincount(flat, yy[ok], N_SLOTS * OV_BINS).reshape(N_SLOTS, OV_BINS).astype(np.float32),
            np.bincount(flat, minlength=N_SLOTS * OV_BINS).reshape(N_SLOTS, OV_BINS).astype(np.uint8))


def _traces(run):
    """Yield (segment key, chunk row index j, row, baseline-subtracted samples, raw, bucket positions, amplitudes, trailing edge) for every
    usable trace of the analysis segments (`segments.select_chunks`); tail reference = the last bucket (core or fitted edge) above half the
    median bucket amplitude, None if a weaker bucket follows it (no clean tail)."""
    bk = np.load(paths.buckets_npz(run), allow_pickle=True)
    for key, ch in select_chunks(run).items():
        rows = ch['ad']
        st = at.read_store(paths.STORE[run], sorted(rows['run'].unique()))
        idx = {(int(r), int(e), int(ts)): k for k, (r, e, ts) in enumerate(zip(st['run'], st['event'], st['ts_ns']))}
        for j, (_, r) in enumerate(rows.iterrows()):
            k = idx.get((int(r['run']), int(r['event']), int(r['ts'].value)))
            pos = bk['peak_positions'][r['i']]
            if k is None or not np.isfinite(pos).all() or abs(np.median(np.diff(pos)) - r['period']) > 0.2:
                continue
            raw = st['samples'][k]
            ep, ea = bk['edge_positions'][r['i']], bk['edge_amp'][r['i']]
            fe = np.isfinite(ep) & np.isfinite(ea)
            tk, ak = np.r_[pos, ep[fe]], np.r_[bk['bucket_amp'][r['i']], ea[fe]]
            ok = np.isfinite(tk) & np.isfinite(ak)
            tk, ak = tk[ok], ak[ok]
            last = tk[ak > 0.5 * np.median(ak)].max()
            last = None if (tk > last + 5).any() else last
            yield key, j, r, raw.astype(float) - np.median(raw[:at.BASELINE_SAMPLES]), raw, pos, bk['bucket_amp'][r['i']], last
        del st


def train_end_tails(runs=(1, 2)):
    """Mean measured trace after the last bucket per configuration period (÷ median bucket amplitude; reference and selection as `_traces`),
    full `FT_X` window recorded, clipped samples out → {period: dict(y, e, n, P)}."""
    t = np.arange(at.N_SAMPLES) * BIN_NS
    acc = {}
    for run in runs:
        for key, j, r, s, raw, pos, amp, last in _traces(run):
            if last is None or last + FT_X[-1] >= t[-1]:
                continue
            pn = str(ad_period(np.array([r['ts'].to_datetime64()]))[0])
            y = s.copy(); y[raw >= at.ADC_MAX] = np.nan
            acc.setdefault(pn, []).append((np.interp(last + FT_X, t, y) / np.nanmedian(amp), r['period']))
    return {pn: dict(y=np.nanmean([a for a, _ in v], 0), e=np.nanstd([a for a, _ in v], 0) / np.sqrt(len(v)), n=len(v), P=float(np.median([p for _, p in v])))
            for pn, v in acc.items()}


def reflected_train(y, e, P):
    """Reflected-train template after the train end: Fourier series in k / P (k ≤ 8) fitted on +45 … +130 ns of `FT_X`."""
    K = np.arange(9)
    B = np.c_[np.cos(2 * np.pi * np.outer(FT_X, K) / P), np.sin(2 * np.pi * np.outer(FT_X, K[1:]) / P)]
    m = (FT_X > 45) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    return B @ np.linalg.lstsq(B[m] / e[m, None], y[m] / e[m], rcond=None)[0]


def round_trip_fir(T, period):
    """Round-trip FIR h on `FT_RT_TAU` from the `ad_term` probe (`load_ad_term`): R(u) = Σ h(τ) S_per(u − τ), weighted, small ridge → (h, model R(u))."""
    S_, R_, U = T[period]['S'].mean(0), T[period]['R'].mean(0), T['U']
    eR = T[period]['R'].std(0) / np.sqrt(len(T[period]['R']))
    P = float(np.median(T[period]['P']))
    Uw, Sw = np.r_[U, U[0] + P], np.r_[S_, S_[0]]
    A = np.array([np.interp((U - tau - U[0]) % P + U[0], Uw, Sw) for tau in FT_RT_TAU]).T
    w = 1 / np.maximum(eR, 1e-6)
    Aw = A * w[:, None]
    lam = 1e-5 * np.trace(Aw.T @ Aw) / len(FT_RT_TAU)
    h = np.linalg.solve(Aw.T @ Aw + lam * np.eye(len(FT_RT_TAU)), Aw.T @ (R_ * w))
    return h, A @ h


def short_reflection(tail):
    """δ short reflection (T, ρ) on a `train_end_tails` entry: residual after the reflected-train template, x = 6 … 45 ns, = ρ · y(x − T) + smooth
    direct tail (1, x^-1/2, x^-3/2); T scanned over `FT_SHORT_T` → dict(T, rho, chi2, ndf)."""
    y, e = tail['y'], tail['e']
    r = y - reflected_train(y, e, tail['P'])
    fm = (FT_X > 6) & (FT_X < 45) & np.isfinite(e) & (e > 0)
    yv = np.nan_to_num(y)
    xp = np.clip(FT_X, 1e-3, None)
    sm = [np.ones_like(FT_X), xp ** -0.5 * (FT_X > 0), xp ** -1.5 * (FT_X > 0)]

    def lin(T_):
        Am = np.array([np.interp(FT_X - T_, FT_X, yv)] + sm).T[fm] / e[fm, None]
        c, *_ = np.linalg.lstsq(Am, r[fm] / e[fm], rcond=None)
        return c, float(np.sum((Am @ c - r[fm] / e[fm]) ** 2))
    Tb = min(np.arange(*FT_SHORT_T, 0.04), key=lambda a: lin(a)[1])
    c, chi = lin(Tb)
    return dict(T=float(Tb), rho=float(c[0]), chi2=chi, ndf=int(fm.sum()) - 4)


def rt_delay(T, period, leftover=0.0):
    """Fixed round-trip delay of a period [ns] = the probe frame of `round_trip_fir` (21 × median probe P − 1.5, `_term_trace`) + `leftover`
    (in-train lag left after the bootstrap, slot-22 step): one frame for the h fit and its use."""
    return 21 * float(np.median(T[period]['P'])) - 1.5 + leftover


FT_RT_LEFTOVER = {'Run 2.1': -0.027263}             # in-train round-trip lag left after the bootstrap [ns]


def reflection_models(T, tails, periods, leftover=FT_RT_LEFTOVER):
    """{period: dict(h (round-trip FIR), short (δ T, ρ))} from the b-free probes; `leftover` ({period: ns}, {} = none) adds the fixed
    round-trip delay `delay` (`rt_delay`) used by `bootstrap_trace`; None = legacy per-trace 21 P − 1.5."""
    out = {pn: dict(h=round_trip_fir(T, pn)[0], short=short_reflection(tails[pn])) for pn in periods if pn in tails and pn in T}
    if leftover is not None:
        for pn, m in out.items():
            m['delay'] = rt_delay(T, pn, leftover.get(pn, 0.0))
    return out


def _taps(tau0, h, shift):
    # FIR on the BIN_NS grid starting at delay shift + tau0 [ns] → (integer start delay [samples], taps incl. the fractional sample)
    k = (shift + tau0) / BIN_NS
    ki, f = int(np.floor(k)), k - np.floor(k)
    hh = np.zeros(len(h) + 1); hh[:-1] += (1 - f) * h; hh[1:] += f * h
    return ki, hh


def bootstrap_trace(s, P, model):
    """Corrected trace D = Y − ρ D(t − T) − (h ⊛ Y)(t − d); causal blocks < T (the first T of the train has no short reflection); round-trip
    delay d = `model['delay']` (fixed per period, `rt_delay`) or, without it, the legacy per-trace 21 P − 1.5; optional `model` keys `s_rt`
    (round-trip scale, default 1) and `delta` (round-trip lag shift [ns], default 0): § Forward Fold checks."""
    D = s.copy(); n = len(s)
    d_rt = model['delay'] if 'delay' in model else 21 * P - 1.5
    kr, hr = _taps(FT_RT_TAU[0], model.get('s_rt', 1.0) * model['h'], d_rt + model.get('delta', 0.0))
    ks, hs = _taps(0.0, np.array([model['short']['rho']]), model['short']['T'])
    RT = np.convolve(s, hr)                       # RT[m] = Σ_j hr_j Y[m − j]
    jj = np.arange(n) - kr
    rt = np.where(jj >= 0, RT[np.clip(jj, 0, len(RT) - 1)], 0.0)
    L = ks - 1
    for n0 in range(at.BASELINE_SAMPLES, n, L):
        n1 = min(n0 + L, n)
        a0 = n0 - ks - (len(hs) - 1)
        seg = D[max(a0, 0):n1 - ks]
        if a0 < 0:
            seg = np.r_[np.zeros(-a0), seg]
        D[n0:n1] = s[n0:n1] - rt[n0:n1] - np.convolve(seg, hs, mode='valid')[:n1 - n0]
    return D


def refit_positions(t, D, valid, pos, sig):
    """Bucket positions on the corrected trace D with the `fit_trace` per-bucket estimator (sym Gaussian, no baseline, ±`WINDOW_SIGMA` σ),
    window from the raw-trace fit (positions `pos`, σ `sig`) → (positions, period = slope of their linear fit); a failed fit keeps `pos`."""
    from scipy.optimize import curve_fit
    out = np.array(pos, float)
    for i, (p0, s0) in enumerate(zip(pos, sig)):
        m = (np.abs(t - p0) <= at.WINDOW_SIGMA * s0) & valid
        if m.sum() < 8:
            continue
        try:
            out[i] = curve_fit(lambda x, a, c, w: a * np.exp(-0.5 * ((x - c) / w) ** 2), t[m], D[m], p0=[np.interp(p0, t, D), p0, s0])[0][1]
        except RuntimeError:
            pass
    return out, float(np.polyfit(np.arange(len(out)), out, 1)[0])


def ad_waveforms_bootstrap(models, runs=(1, 2), positions='corrected', keep_sub=False):
    """Per trace × bucket slot overlays of the bootstrap-corrected traces (S, N [trace, slot, bin], amp, period, amp_med, amp_sum, chunk, ts, row) + the corrected train-end tails per period
    → ({key: W}, {period: dict(y, n)}).
    - `positions`: 'raw' = overlays on the raw-trace bucket positions (legacy); 'corrected' = positions and period refit on D
      (`refit_positions`, corrected traces only), W['dpos'] = corrected − raw position per slot
    - `keep_sub`: W['S_sub'] = overlay sums of the removed part Y − D on the same positions (diagnostics: the subtracted reflections)"""
    t = np.arange(at.N_SAMPLES) * BIN_NS
    out, tl = {}, {}
    for run in runs:
        bk = np.load(paths.buckets_npz(run), allow_pickle=True) if positions == 'corrected' else None
        cur, W = None, None
        for key, j, r, s, raw, pos, amp, last in _traces(run):
            if key != cur:
                if cur is not None:
                    out[cur] = {n: np.array(v) for n, v in W.items()}
                cur, W = key, {n: [] for n in ('S', 'N', 'amp', 'period', 'amp_med', 'amp_sum', 'chunk', 'ts', 'row')
                               + (('dpos',) if positions == 'corrected' else ()) + (('S_sub',) if keep_sub else ())}
            pn = str(ad_period(np.array([r['ts'].to_datetime64()]))[0])
            D = bootstrap_trace(s, r['period'], models[pn]) if pn in models else s
            pc, Pc = (refit_positions(t, D, raw < at.ADC_MAX, pos, bk['bucket_sigma'][r['i']]) if positions == 'corrected' and pn in models
                      else (pos, r['period']))
            S_, N_ = _overlay(t, D, pc, Pc, raw < at.ADC_MAX)
            if positions == 'corrected':
                W['dpos'].append(pc - pos)
            if keep_sub:
                W['S_sub'].append(_overlay(t, s - D, pc, Pc, raw < at.ADC_MAX)[0])
            W['S'].append(S_); W['N'].append(N_); W['amp'].append(amp); W['period'].append(Pc)
            W['amp_med'].append(np.nanmedian(amp)); W['amp_sum'].append(np.nansum(amp)); W['chunk'].append(r['chunk'])
            W['ts'].append(r['ts'].to_datetime64()); W['row'].append(j)
            if pn in models and last is not None and last + FT_X[-1] < t[-1]:
                Dn = D.copy(); Dn[raw >= at.ADC_MAX] = np.nan
                tl.setdefault(pn, []).append(np.interp(last + FT_X, t, Dn) / np.nanmedian(amp))
        if cur is not None:
            out[cur] = {n: np.array(v) for n, v in W.items()}
    return out, {pn: dict(y=np.nanmean(v, 0), n=len(v)) for pn, v in tl.items()}


def group_blocks(d, w):
    """Segment `d` (`segments.segment_inputs`, one AD config) with its blocks rebuilt from the overlays `w` (corrected or not): one block per round-trip group
    (`group_mask`: head from `FT_HEAD_FIRST`), mean ÷ median amplitude ± scatter / √n; `blk_m` = group."""
    g, per = d['grid'], ad_period(w['ts'])
    pn = str(d['blk_period'][0])
    tr = per == pn
    Y, E = [], []
    for m in range(N_GROUPS):
        pm = trace_profiles({n: v[tr] for n, v in w.items()}, g, group_mask(d['P'], m))
        Y.append(pm.mean(0)); E.append(pm.std(0, ddof=1) / np.sqrt(len(pm)))
    return dict(d, blk_Y=np.array(Y), blk_E=np.array(E), blk_period=np.array([pn] * N_GROUPS), blk_m=np.arange(N_GROUPS),
                blk_n=np.full(N_GROUPS, tr.sum()), blk_c=np.full(N_GROUPS, 1 / N_GROUPS))


def _mix_moments(t, f):
    # mode-free moments of a density on `t`: side RMS about the mean, 2 × left − 2 × right semi-variance
    f = f / f.sum(); d = t - t @ f; L = d < 0
    return dict(side_L=float(np.sqrt(f[L] @ d[L] ** 2 / f[L].sum())), side_R=float(np.sqrt(f[~L] @ d[~L] ** 2 / f[~L].sum())), LmR=float(2 * f[L] @ d[L] ** 2 - 2 * f[~L] @ d[~L] ** 2))


def _mix_on_grid(G, mix):
    # mixture density `mix` (normalised) on `G.tf` → tt (recentred on the mode), moments, agg2 refit with a free location
    tt = G.tf - G.tf[np.argmax(mix)]
    shf = lambda q: (lambda d: d / d.sum())(bunch_shape(G, q[4], dict(sl=q[0], sr=q[1], bl=q[2], br=q[3]), 'agg2'))
    qf = least_squares(lambda q: (shf(q) - mix) * 1e3, [1.2, 1.2, 2.2, 2.2, G.tf[np.argmax(mix)]], bounds=([.2, .2, AD_B_MIN, AD_B_MIN, -6], [5, 5, 6, 6, 6]), x_scale='jac').x
    agg = dict(sl=qf[0], sr=qf[1], bl=qf[2], br=qf[3])
    return dict(tt=tt, ff=mix, rms=bunch_rms(dict(tt=tt, ff=mix)), agg2=agg, rms_agg2=bunch_rms(agg), **_mix_moments(tt, mix))


def b_smear(mix, s2, P):
    """Full-train b of the MC: `group_mixture` result (x₀ frame) ⊗ G(s2) (`AD_B_S2` of the period, periodic on `Grid(P)`), recentred on the mode, moments and agg2 refit;
    s2 = 0 → `mix` unchanged."""
    if not s2:
        return mix
    G = Grid(P)
    m = np.asarray(mix['ff'], float)       # indexed on G.tf (tt = G.tf − mode)
    fr = np.fft.fftfreq(len(m), G.tf[1] - G.tf[0])
    g = np.clip(np.real(np.fft.ifft(np.fft.fft(m) * np.exp(-2 * (np.pi * fr) ** 2 * s2))), 0, None)
    return _mix_on_grid(G, g / g.sum())


def group_mixture(P, ps, wt, x0s=None):
    """Amplitude-weighted mixture of per-group agg2 b `ps` (mode at 0) with weights `wt` on `Grid(P)` → dict(tt, ff (density), rms (of the mixture),
    agg2 (σL, σR, β tied or βL / βR per `AD_B_TIE`, fitted to the mixture density), rms_agg2)."""
    G = Grid(P)
    if x0s is not None:      # x₀ frame: each group's b where the fit placed it in the overlay frame (mean offset removed), recentred on the mixture mode; agg2 refit with a free location
        w = np.asarray(wt, float); xb = float(np.sum(w * np.asarray(x0s)) / w.sum())
        mix = sum(wi * (lambda d: d / d.sum())(bunch_shape(G, x - xb, p, 'agg2')) for wi, x, p in zip(w, x0s, ps))
        mix = mix / mix.sum()
        return _mix_on_grid(G, mix)
    dens = [bunch_shape(G, 0.0, p, 'agg2') for p in ps]
    mix = sum(w * d / d.sum() for w, d in zip(wt, dens))
    shape = lambda q: bunch_shape(G, 0.0, dict(sl=q[0], sr=q[1], bl=q[2], br=q[2] if AD_B_TIE else q[3]), 'agg2')
    q0, lo, hi = [ps[0]['sl'], ps[0]['sr'], ps[0]['bl']], [0.2, 0.2, 1.0], [5, 5, 6]
    if not AD_B_TIE:
        q0, lo, hi = q0 + [ps[0]['br']], lo + [1.0], hi + [6]
    q = least_squares(lambda q: shape(q) / shape(q).sum() - mix, q0, bounds=(lo, hi)).x
    agg = dict(sl=q[0], sr=q[1], bl=q[2], br=q[2] if AD_B_TIE else q[3])
    return dict(tt=G.tf, ff=mix, rms=bunch_rms(dict(tt=G.tf, ff=mix)), agg2=agg, rms_agg2=bunch_rms(agg), **_mix_moments(G.tf, mix))


# ---------------- § Forward Fold checks per configuration period ----------------
def ff_traces(run):
    """Raw traces of the analysis segments once per run (selection / tail reference as `_traces`) → {key: [dict(raw (int16), pos, amp, P, chunk,
    ts, row, last, headroom (`ADC_MAX` − max raw), period)]}: every variant of the fold is rebuilt from these without rereading the store."""
    out = {}
    with np.load(paths.buckets_npz(run), allow_pickle=True) as bk:
        bsig = bk['bucket_sigma']
    for key, j, r, s, raw, pos, amp, last in _traces(run):
        ts = r['ts'].to_datetime64()
        out.setdefault(key, []).append(dict(raw=raw.astype(np.int16), pos=np.array(pos, float), amp=np.array(amp, float),   # copies: no view of the npz arrays
                                            sig=np.array(bsig[r['i']], float), P=r['period'], chunk=r['chunk'], ts=ts, row=j, last=last,
                                            headroom=int(at.ADC_MAX - raw.max()), period=str(ad_period(np.array([ts]))[0])))
    return out


def ff_overlays(trs, model, refit=False):
    """Steps F1-F3 for one variant: baseline, bootstrap (`bootstrap_trace`, `model` None = uncorrected), overlay per slot (`_overlay`)
    → (W as `ad_waveforms_bootstrap` + headroom, train end: dict(n, rms (+45 … +130 ns), y17 (16.5-17.5 ns)) of the corrected mean trace);
    `refit`: positions and period refit on the corrected trace (`refit_positions`, as `ad_waveforms_bootstrap(positions='corrected')`; reflection variants of the errors)."""
    t = np.arange(at.N_SAMPLES) * BIN_NS
    W = {n: [] for n in ('S', 'N', 'amp', 'period', 'amp_med', 'amp_sum', 'chunk', 'ts', 'row', 'headroom')}
    tl = []
    for tr in trs:
        raw = tr['raw']
        s = raw.astype(float) - np.median(raw[:at.BASELINE_SAMPLES])
        D = s if model is None else bootstrap_trace(s, tr['P'], model)
        pc, Pc = refit_positions(t, D, raw < at.ADC_MAX, tr['pos'], tr['sig']) if refit and model is not None else (tr['pos'], tr['P'])
        S_, N_ = _overlay(t, D, pc, Pc, raw < at.ADC_MAX)
        for n, v in (('S', S_), ('N', N_), ('amp', tr['amp']), ('period', Pc), ('amp_med', np.nanmedian(tr['amp'])), ('amp_sum', np.nansum(tr['amp'])),
                     ('chunk', tr['chunk']), ('ts', tr['ts']), ('row', tr['row']), ('headroom', tr['headroom'])):
            W[n].append(v)
        if tr['last'] is not None and tr['last'] + FT_X[-1] < t[-1]:
            Dn = D.copy(); Dn[raw >= at.ADC_MAX] = np.nan
            tl.append(np.interp(tr['last'] + FT_X, t, Dn) / np.nanmedian(tr['amp']))
    W = {n: np.array(v) for n, v in W.items()}
    y = np.nanmean(tl, 0) if tl else np.full(len(FT_X), np.nan)
    return W, dict(n=len(tl), rms=float(np.sqrt(np.nanmean(y[FT_X > 45] ** 2))), y17=float(np.nanmean(y[(FT_X > 16.5) & (FT_X < 17.5)])))


# ---------------- full-train b: context, fits, uncertainty ----------------
def b_mean_mode(p):
    """Mean of the agg2 b `p` − its mode x₀ [ns], analytic (unwrapped)."""
    a = [p[s] * 2 ** (1 / p[b]) * gamma(1 + 1 / p[b]) for s, b in (('sl', 'bl'), ('sr', 'br'))]
    m = [p[s] ** 2 * 2 ** (2 / p[b]) * gamma(2 / p[b]) / p[b] for s, b in (('sl', 'bl'), ('sr', 'br'))]
    return (m[1] - m[0]) / (a[0] + a[1])


def block_centroids(d, K, n=AD_FWD_N):
    """Per block: centroid of the block in the AD fit window − mean of its kernel R_AD → the frame of the mean of b (position condition of `ad_fit_cond`) [ns]."""
    g = d['grid']
    lo, hi = ad_window(d, n)
    m = (g.x >= lo) & (g.x <= hi)
    t = np.fft.fftfreq(K.shape[1], 1) * K.shape[1] * g.h
    return np.array([(g.x[m] * Y[m]).sum() / Y[m].sum() - (t * k).sum() / k.sum() for Y, k in zip(d['blk_Y'], K)])


def _cond_p(rho, bl, br, sr):
    return dict(sl=rho * sr, sr=sr, bl=bl, br=br)


def ad_fit_cond(d, K, cond, starts, n=AD_FWD_N):
    """agg2 b ⊗ R_AD on the blocks of `d` under the conditions `cond` (`AD_B_COND`), diagonal χ² over ±n σ_AD, lowest χ² of `starts` (`cond_starts`) → dict(ps, x0, A, M, chi, npar, nx, cent).
    Parameters ρ = σL / σR, βL, βR (shared or per block), σR, x₀ (`pos` set: mean of b, x₀ = mean − `b_mean_mode`), A per block."""
    g, Y, E = d['grid'], d['blk_Y'], d['blk_E']
    nbk = len(Y)
    lo, hi = ad_window(d, n)
    mk = (g.x >= lo) & (g.x <= hi)
    FK = np.fft.fft(K, axis=1)
    sh, pos = cond['shared'], cond['pos']
    cent = block_centroids(d, K, n) if pos is not None else None
    nsh = 3 if sh else 3 * nbk

    def unpack(q):
        s = np.repeat(q[:3][None], nbk, 0) if sh else q[:nsh].reshape(nbk, 3)
        gp = q[nsh:].reshape(nbk, 3)
        ps = [_cond_p(s[i, 0], s[i, 1], s[i, 2], gp[i, 0]) for i in range(nbk)]
        x0 = gp[:, 1] - np.array([b_mean_mode(p) for p in ps]) if pos is not None else gp[:, 1]
        return ps, x0, gp[:, 2]

    def model(i, p, x, a):
        return a * g.nb * g.rebin(np.real(np.fft.ifft(np.fft.fft(bunch_shape(g, x, p, 'agg2')) * FK[i])))

    def resid(q):
        ps, x0, A = unpack(q)
        return np.concatenate([((Y[i] - model(i, ps[i], x0[i], A[i])) / E[i])[mk] for i in range(nbk)])

    l1, h1 = [0.05, *[cond['beta'][0]] * 2], [cond['rmax'], *[cond['beta'][1]] * 2]
    slo, shi = np.array(l1 * (1 if sh else nbk)), np.array(h1 * (1 if sh else nbk))
    loc_lo = np.full(nbk, PAR['xa'][1]) if pos is None else cent - pos
    loc_hi = np.full(nbk, PAR['xa'][2]) if pos is None else cent + pos
    lb = np.r_[slo, np.ravel([[PAR['sr'][1], loc_lo[i], 0.0] for i in range(nbk)])]
    ub = np.r_[shi, np.ravel([[PAR['sr'][2], loc_hi[i], PAR['aa'][2]] for i in range(nbk)])]
    best, first = None, None
    for s_, sr_, x_, A_ in starts:
        s_ = np.asarray(s_, float)
        loc = np.array([x_[i] + b_mean_mode(_cond_p(*(s_ if sh else s_[i]), sr_[i])) for i in range(nbk)]) if pos is not None else np.asarray(x_, float)
        q0 = np.r_[np.ravel(s_), np.ravel([[sr_[i], loc[i], A_[i]] for i in range(nbk)])]
        try:
            r = least_squares(resid, np.clip(q0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub), x_scale='jac')
        except ValueError:
            continue
        first = r if first is None else first
        if best is None or (r.fun ** 2).sum() < (best.fun ** 2).sum() - 1e-6:
            best = r
    ps, x0, A = unpack(best.x)
    M = np.array([model(i, ps[i], x0[i], A[i]) for i in range(nbk)])
    shp = np.r_[np.arange(nsh), nsh + 3 * np.arange(nbk)]      # ρ, βL, βR (shared or per block), σR per block
    sflag = bool(np.abs(first.x[shp] - best.x[shp]).max() > 0.03)   # the first (warm) start ended on a different b than the best start
    return dict(ps=ps, x0=x0, A=A, M=M, chi=[float((((Y[i] - M[i]) / E[i])[mk] ** 2).sum()) for i in range(nbk)], npar=best.x.size, nx=int(mk.sum()), cent=cent, wA=(lo, hi),
                start_flag=sflag)


def cond_starts(base, cond):
    """Starts of `ad_fit_cond` from the per-block fits `base` (list of dict(p, x0, A)): base (warm), production `PAR`, mirrored, `FT_START_*` grid in units of the block b RMS; x₀, A from `base` (shared: medians of ρ, β)."""
    sh, nbk = cond['shared'], len(base)
    lim = lambda r, a, b: (min(max(r, 0.05), cond['rmax']), *[min(max(v, cond['beta'][0]), cond['beta'][1]) for v in (a, b)])
    x0, A, rg = [b['x0'] for b in base], [b['A'] for b in base], [bunch_rms(b['p']) for b in base]
    rho, bL, bR, sr = ([b['p']['sl'] / b['p']['sr'] for b in base], [b['p']['bl'] for b in base], [b['p']['br'] for b in base], [b['p']['sr'] for b in base])
    out = []

    def add(rm, bl_, br_, srm):
        s = np.array(lim(np.median(rm), np.median(bl_), np.median(br_))) if sh else np.array([lim(rm[i], bl_[i], br_[i]) for i in range(nbk)])
        out.append((s, list(srm), x0, A))

    add(rho, bL, bR, sr)
    add([PAR['sl'][0] / PAR['sr'][0]] * nbk, [PAR['bl'][0]] * nbk, [PAR['br'][0]] * nbk, [PAR['sr'][0]] * nbk)
    add([1 / r for r in rho], bR, bL, [s_ * r for s_, r in zip(sr, rho)])
    for a, b in FT_START_SIG:
        for c, e in FT_START_BET:
            add([a / b] * nbk, [c] * nbk, [e] * nbk, [b * r_ for r_ in rg])
    return out


def cond_pins(p, x0, cent, cond, tol=1e-3):
    """Names of the conditions a block's b sits on (within `tol`): 'bl_lo', 'br_hi', 'rho', 'pos', …"""
    a = [f'{n}_{e}' for n in ('bl', 'br') for e, v in (('lo', cond['beta'][0]), ('hi', cond['beta'][1])) if abs(p[n] - v) < tol]
    a += ['rho'] if abs(p['sl'] / p['sr'] - cond['rmax']) < tol else []
    return a + (['pos'] if cond['pos'] is not None and abs(x0 + b_mean_mode(p) - cent) > cond['pos'] - tol else [])


def ad_cond_fits(d, K, base, cond=AD_B_COND, n=AD_FWD_N, n_start=None):
    """Group fits of a segment under `cond` (shared shape: one joint fit, else per group), `base` = per-group free fits (`best_group_fit`) as warm starts → list of `ad_fit_fwd`-like dicts per group (p, q, blk, M, chi2, ndof, wA, kind, pins); `n_start` = first starts only."""
    b = [dict(p=f['p'], x0=f['blk'][0][0], A=f['blk'][0][1]) for f in base]
    nk = len(d['blk_Y'])

    def one(dd, KK, bb):
        st = cond_starts(bb, cond)
        return ad_fit_cond(dd, KK, cond, st if n_start is None else st[:n_start], n)

    if cond['shared']:
        fj, idx = one(d, K, b), range(nk)
        fs = [(fj, i) for i in idx]
    else:
        fs = [(one(*ad_blocks(d, K, groups=(m,)), [b[m]]), 0) for m in range(nk)]
    out = []
    for f, i in fs:
        p = f['ps'][i]
        cent = f['cent'][i] if f['cent'] is not None else np.nan
        npg = f['npar'] / len(f['ps'])
        out.append(dict(p=p, q=np.r_[p['sl'], p['sr'], p['bl'], p['br'], f['x0'][i], f['A'][i]], blk=np.array([[f['x0'][i], f['A'][i]]]), M=f['M'][i:i + 1], chi2=f['chi'][i], ndof=f['nx'] - npg, wA=f['wA'], kind='forward', pins=cond_pins(p, f['x0'][i], cent, cond),
                        start_flag=f['start_flag']))
    return out


def ad_taps_cond(head, keys, rt, cond=AD_B_COND, lam=TAP_LAM, mu=TAP_MU, scope=AD_SCOPE_FORM, echoes=ECHOES):
    """Taps refit (`_fit_ad_taps_3b` from `rt`, shared 50 Ω groups) with the head b under `cond` per head (β range, σL ≤ σR, position; shared shape does not apply to one group), 3 starts;
    `lam`, `mu`, `scope`, `echoes` as the `fit_ad_taps` call that made `rt` (error variants: `TAP_SETS`, scope form, echo sets)."""
    c = dict(cond, shared=False)

    def b_fit(d, K, n):
        return dict(p=ad_cond_fits(d, K, [ad_fit_fwd(d, K, n)], c, n, n_start=3)[0]['p'])

    return _fit_ad_taps_3b(head, [k for k in keys if str(head[k]['blk_period'][0]) in AD_TAPS_SHARE], rt, lam, mu, scope, echoes, RC_TAU, AD_TAPS_SHARE, None, b_fit=b_fit)


def ft_context(data, wb, keys=None, fitted=None):
    """Steps F4-F5 of the full train: group blocks from the corrected overlays `wb` (`group_blocks`), corrected heads, H refit on the corrected
    handle heads (`fit_ad_h1m`; R_AD uses H as fitted, `h1m_instrument`), taps refit on the corrected heads (`fit_ad_taps`, shared 50 Ω `AD_TAPS_SHARE`, then `ad_taps_cond`), kernels `KF` (no reflection term), mixture weights `WTG`
    (AD bucket amplitude per group, all traces of the segment) → dict; `fitted` = dict(HF, RTF) of a previous call (`F_AD_FT_FITS`): no H / taps refit"""
    keys = SEGMENTS if keys is None else keys
    DSF = {k: group_blocks(data[k], wb[k]) for k in keys}
    HEAD = {k: ad_blocks(DSF[k], groups=(0,))[0] for k in keys}
    HF = fit_ad_h1m(HEAD[AD_H1M_HANDLE[0]], HEAD[AD_H1M_HANDLE[1]]) if fitted is None else fitted['HF']
    H1MF = {AD_H1M['period']: (AD_H1M['base'], h1m_instrument(HF['p']))}
    RTF = ad_taps_cond(HEAD, keys, fit_ad_taps(HEAD, keys)) if fitted is None else fitted['RTF']
    RRF = {pn: RTF for pn in RTF['C']}
    WTG = {}
    for k in keys:
        w = np.array([np.nansum(wb[k]['amp'][:, group_mask(DSF[k]['P'], m)]) for m in range(N_GROUPS)])
        WTG[k] = w / w.sum()
    return dict(keys=list(keys), DSF=DSF, HEAD=HEAD, HF=HF, H1MF=H1MF, RTF=RTF, RRF=RRF, KF=ad_kernels_periods(DSF, keys, RRF, h1m=H1MF), WTG=WTG)


FT_START_SIG = ((0.55, 1.3), (0.85, 0.85), (1.3, 0.55), (0.4, 1.7), (1.7, 0.4))     # multi-start grid: (σL, σR) in units of the group's b RMS
FT_START_BET = ((2, 2), (1.3, 4), (4, 1.3))                                         # (βL, βR)


def best_group_fit(dm, km, n=AD_FWD_N, q0=None):
    """`ad_fit_fwd` with the multi-start policy, lowest χ²: `q0` None (nominal) = production start, mirrored, 15-point grid;
    `q0` given (systematic variants, bootstrap) = warm start and production start, `start_flag` = they ended on different b (σ, β differ by > 0.03)."""
    f0 = ad_fit_fwd(dm, km, n)
    if q0 is not None:
        fw = ad_fit_fwd(dm, km, n, q0=q0)
        out = dict(min((fw, f0), key=lambda f: f['chi2']))
        out['start_flag'] = max(abs(fw['p'][p_] - f0['p'][p_]) for p_ in ('sl', 'sr', 'bl', 'br')) > 0.03
        return out
    q, rg, best = f0['q'], bunch_rms(f0['p']), f0
    for qs in [np.r_[q[1], q[0], q[3], q[2], q[4:]]] + [np.r_[a * rg, b * rg, c, e, q[4:]] for a, b in FT_START_SIG for c, e in FT_START_BET]:
        try:
            f = ad_fit_fwd(dm, km, n, q0=qs)
        except ValueError:
            continue
        if f['chi2'] < best['chi2'] - 1e-6:
            best = f
    return best


def floored(d, f):
    """Blocks of `d` with the error floor E_eff² = E² + (f · block peak)² ('weights' variant of the errors)."""
    pk = np.asarray(d['blk_Y']).max(1)
    return dict(d, blk_E=np.sqrt(np.asarray(d['blk_E']) ** 2 + (f * pk[:, None]) ** 2))


def mean_offsets(P, ps):
    """Per group b `ps` (agg2, mode at 0): −mean on `Grid(P)` → `group_mixture` x0s that place every group's mean at 0 ('alignment' variant of the errors)."""
    G = Grid(P)
    return [-float(G.tf @ (lambda q: q / q.sum())(bunch_shape(G, 0.0, p, 'agg2'))) for p in ps]


def ft_fits(ctx, k, K=None, d=None, wt=None, n=AD_FWD_N, q0=None, cond=AD_B_COND, s2=None, floor=0.0, align='x0'):
    """Steps F6-F8 for segment `k`: agg2 ⊗ R_AD per group at ±`n` σ_AD under `cond` (`ad_cond_fits`, warm starts = free fits `best_group_fit`), mixture
    of the group b (`group_mixture`, weights `wt`) ⊗ G(s2) (`b_smear`) → dict(fg, mix, mix_ad (before the smear), s2);
    `K` / `d` / `wt` default to the nominal context; `s2` None = `AD_B_S2` of the period; error variants: `floor` (error floor of the conditioned fits, `floored`; start fits unfloored),
    `align` 'x0' (nominal: x₀ frame) or 'mean' (each group's mean at 0, `mean_offsets`)"""
    K = ctx['KF'][k] if K is None else K
    d = ctx['DSF'][k] if d is None else d
    wt = ctx['WTG'][k] if wt is None else wt
    s2 = AD_B_S2.get(str(d['blk_period'][0]), 0.0) if s2 is None else s2
    bk = [ad_blocks(d, K, groups=(m,)) for m in range(N_GROUPS)]
    base = [best_group_fit(dm, km, n, None if q0 is None else q0[m]) for m, (dm, km) in enumerate(bk)]
    fg = ad_cond_fits(floored(d, floor) if floor else d, K, base, cond, n, None if q0 is None else 3)
    x0s = (lambda fs: [f['blk'][0][0] for f in fs] if align == 'x0' else mean_offsets(d['P'], [f['p'] for f in fs]))
    ma = group_mixture(d['P'], [f['p'] for f in fg], wt, x0s=x0s(fg))
    return dict(fg=fg, mix=b_smear(ma, s2, d['P']), mix_ad=ma, s2=s2)


def ft_values(mix):
    """Saved full-train parameters (`FT_SAVE_PARS`) of a `group_mixture` result: agg2 to the mixture, its per-side RMS, the mixture's full RMS."""
    a = mix['agg2']
    l, r = bunch_side_rms(a)
    return dict(sl=a['sl'], sr=a['sr'], bl=a['bl'], br=a['br'], rms_left=l, rms_right=r, rms=mix['rms'], side_L=mix['side_L'], side_R=mix['side_R'], LmR=mix['LmR'])


def _ft_trace_profiles(d, w):
    # per-trace group profiles [group, trace, bin] of the segment's period, AD amplitude per trace and group, chunk per trace
    tr = ad_period(w['ts']) == str(d['blk_period'][0])
    wt = {n: v[tr] for n, v in w.items()}
    prof = np.array([trace_profiles(wt, d['grid'], group_mask(d['P'], m)) for m in range(N_GROUPS)])
    amp = np.array([np.nansum(wt['amp'][:, group_mask(d['P'], m)], 1) for m in range(N_GROUPS)]).T
    return prof, amp, wt['chunk']


def _ft_block(d, prof, idx):
    return dict(d, blk_Y=prof[:, idx].mean(1), blk_E=prof[:, idx].std(1, ddof=1) / np.sqrt(len(idx)))


def ft_bootstrap(ctx, wb, fits, n_boot=AD_BOOT, seed=0, level='auto'):
    """Statistical error of the full-train b: per resample the traces of every segment are redrawn (chunk level, trace level below
    `AD_BOOT_MIN_CHUNKS`), group blocks rebuilt, H refit on the resampled corrected handle heads (warm start), per-group b refit (warm start),
    mixture with the resampled amplitude weights → {key: array [n_boot, `FT_SAVE_PARS`]}; taps fixed (their choice is a systematic)."""
    rng = np.random.default_rng(seed)
    keys = ctx['keys']
    TP = {k: _ft_trace_profiles(ctx['DSF'][k], wb[k]) for k in keys}
    draw = {k: _resampler(TP[k][2], level, rng) for k in keys}
    k21 = [k for k in keys if str(ctx['DSF'][k]['blk_period'][0]) == AD_H1M['period']]
    out = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = {k: draw[k]() for k in keys}
        Db = {k: _ft_block(ctx['DSF'][k], TP[k][0], idx[k]) for k in keys}
        hp = fit_ad_h1m(ad_blocks(Db[AD_H1M_HANDLE[0]], groups=(0,))[0], ad_blocks(Db[AD_H1M_HANDLE[1]], groups=(0,))[0], ctx['HF']['p']['tin'],
                        start=ctx['HF']['p'])['p']
        K21 = ad_kernels_periods(Db, k21, ctx['RRF'], h1m={AD_H1M['period']: (AD_H1M['base'], h1m_instrument(hp))})
        for k in keys:
            wt = np.nansum(TP[k][1][idx[k]], 0)
            f = ft_fits(ctx, k, K=K21.get(k, ctx['KF'][k]), d=Db[k], wt=wt / wt.sum(), q0=[g['q'] for g in fits[k]['fg']])
            out[k].append([ft_values(f['mix'])[p] for p in FT_SAVE_PARS])
    return {k: np.array(v) for k, v in out.items()}


# ---------------- s² tune (Run 2.1 width; port of the 11-12 scratch `sa_ref5.py`) ----------------
def _s2_mom(x, y, W):
    # windowed moments ±W about the iterated centroid → dict(var, LmR, m3, m4, n)
    c = x[np.argmax(y)]
    for _ in range(40):
        m = np.abs(x - c) <= W
        c = (x[m] * y[m]).sum() / y[m].sum()
    m = np.abs(x - c) <= W
    d, w = x[m] - c, y[m]
    s = w.sum()
    return dict(var=(d ** 2 * w).sum() / s, LmR=(2 * (d[d < 0] ** 2 * w[d < 0]).sum() - 2 * (d[d > 0] ** 2 * w[d > 0]).sum()) / s, m4=(d ** 4 * w).sum() / s, n=s)


def _s2_fold(t, f, s2):
    # density f on the fine grid t ⊗ G(s2) → native bins (`SUB` fine points per bin)
    n = len(f)
    fr = np.fft.rfftfreq(n, t[1] - t[0])
    g = np.fft.irfft(np.fft.rfft(f / f.sum()) * np.exp(-2 * (np.pi * fr) ** 2 * max(s2, 0)), n)
    nb = n // SUB
    return t[:nb * SUB].reshape(nb, SUB).mean(1), np.clip(g[:nb * SUB], 0, None).reshape(nb, SUB).sum(1)


def _s2_implied(t, f, st):
    # σ_det² that makes the windowed variance of f ⊗ G(σ_det²) equal the SBND stack's (NaN: no root in 0-4 ns²)
    from scipy.optimize import brentq
    fn = lambda s2: _s2_mom(*_s2_fold(t, f, s2), st['W'])['var'] - st['var']
    try:
        return brentq(fn, 0.0, 4.0, xtol=1e-4)
    except ValueError:
        return np.nan


def _s2_gconv(tt, ff, s2):
    # periodic ⊗ G(s2) of a density on its own grid (no recentring: windowed moments are frame-free)
    fr = np.fft.fftfreq(len(ff), tt[1] - tt[0])
    g = np.clip(np.real(np.fft.ifft(np.fft.fft(ff / ff.sum()) * np.exp(-2 * (np.pi * fr) ** 2 * s2))), 0, None)
    return g / g.sum()


def _s2_dnu(tt, ff, dn):
    # density ⊗ D_ν (MC ν reco − true time, frame median removed), periodic on the density grid
    h, n = tt[1] - tt[0], len(tt)
    kd = np.bincount(np.round(dn / h).astype(int) % n, minlength=n).astype(float)
    return tt, np.clip(np.real(np.fft.ifft(np.fft.fft(ff) * np.fft.fft(kd / kd.sum()))), 0, None)


def s2_tune(mix, inp, tuned, scan=S2_SCAN):
    """One s² for the `tuned` segments (the `AD_B_S2` periods: Run 2.1 wide, normal_a, narrow) from the pre-smear full-train b `mix` {key: dict(tt, ff)}:
    implied σ_det² per segment (b ⊗ D_ν ⊗ G vs the SBND AD-matched ±2.5σ windowed variance, native bins); reference = error-weighted mean over the other
    (50 Ω, s² = 0) segments, w = 1 / (SBND evar² + AD-stat²) (frozen `inp['ead']`); s² = zero of the mean (implied − reference) of the tuned segments along `scan`
    → dict(s2 (NaN: no sign change), per {seg: own zero}, ref, ref_err, chi2, ok)."""
    tuned = [k for k in mix if k in tuned]
    refk = [k for k in mix if k not in tuned]
    dd = lambda tt, ff: _s2_dnu(np.asarray(tt), np.asarray(ff), inp['dnu'])
    S = {k: _s2_implied(*dd(mix[k]['tt'], mix[k]['ff']), inp['stk'][k]) for k in refk}
    w = {k: 1 / (inp['stk'][k]['evar'] ** 2 + inp['ead'][k[1]] ** 2) for k in refk}
    ref = sum(w[k] * S[k] for k in refk) / sum(w.values())
    steps = np.array([[_s2_implied(*dd(mix[k]['tt'], _s2_gconv(np.asarray(mix[k]['tt']), np.asarray(mix[k]['ff']), s2)), inp['stk'][k]) - ref for k in tuned] for s2 in scan])

    def zero(y):
        i = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
        return float(np.interp(0, [y[i[0] + 1], y[i[0]]], [scan[i[0] + 1], scan[i[0]]])) if len(i) else np.nan

    s2 = zero(steps.mean(1))
    return dict(s2=s2, per={k[1]: zero(steps[:, j]) for j, k in enumerate(tuned)}, ref=float(ref), ref_err=float(1 / np.sqrt(sum(w.values()))),
                chi2=float(sum(w[k] * (S[k] - ref) ** 2 for k in refk)), ok=bool(np.isfinite(s2)), tuned=[k[1] for k in tuned])


# ---------------- errors: extraction-choice variants, bootstrap, ripple ----------------
def taps_variants_cond(head, keys, rt):
    """Taps systematic under the b conditions (`ad_taps_cond` after every refit, as `ft_context`): 'strong' / 'from5' (`TAP_SETS`), 'leave-one-period-out'
    (each shared 50 Ω period with the taps fitted on the group's other periods) → {name: {period: set}}."""
    out = {}
    for n in ('strong', 'from5'):
        kw = TAP_SETS[n]
        r = ad_taps_cond(head, keys, fit_ad_taps(head, keys, **kw), lam=kw.get('lam', TAP_LAM), mu=kw.get('mu', TAP_MU))
        out[n] = {pn: r for pn in r['C']}
    lo = {pn: rt for pn in rt['C']}
    per = {str(head[k]['blk_period'][0]): rt['group'][str(head[k]['blk_period'][0])] for k in keys if str(head[k]['blk_period'][0]) in rt['group']}
    for pn, g in per.items():
        others = [p for p, g_ in per.items() if g_ == g and p != pn]
        if not others:
            continue
        ko = [k for k in keys if str(head[k]['blk_period'][0]) in others]
        r = ad_taps_cond(head, ko, fit_ad_taps(head, ko))
        lo[pn] = dict(r, C={pn: r['C'][others[0]]}, rho={pn: r['rho'][others[0]]})
    out['leave-one-period-out'] = lo
    return out


def ft_systematics(ctx, data, wb, fits, hw_var, echo_variants, n_slices, s2in, traces=None, rmod=None, log=print):
    """Extraction-choice variants of the full-train b. Every model variant refits all segments it touches (warm starts from `fits`), then retunes the Run 2.1 s²
    (`s2_tune`) against that variant's own 50 Ω reference; Δ is taken against the nominal retuned by the same code (`ref`).
    - window: `FT_WIN_SYST` lo / hi (b only); weights: `FT_FLOOR` (b only); taps: `taps_variants_cond`; taps procedure: 'joint' `fit_ad_taps`;
      hardware: `hw_var` {name: [dict(bl, ec) × nominal]} (taps fixed); echo model: `echo_variants` {name: echoes} (taps refit + `ad_taps_cond`);
      scope: `FT_SCOPE_VAR` (taps refit + `ad_taps_cond`); alignment: mean-aligned mixture; H(2.1): τ_in `AD_H1M_TAU_VAR`, `AD_H1M_NECHO_VAR` echoes;
      H skew: H refit with the k3 nuisance; s2: the lowest / highest per-segment zero of the nominal tune (smear only);
      reflection (`traces` = `ff_traces`, `rmod` = `reflection_models`): `FT_RFL_VAR` on every period at once, overlays with positions refit, then H and taps refit (`ft_context`)
    - time: `n_slices` chunk slices per segment, s² = `AD_B_S2` (per segment, no retune); bootstrap and ripple: separate (`ft_bootstrap`, `ft_ripple_closure`)
    → dict(nominal (tune), ref {key: values}, variants [dict(group, name, side, s2, s2_ok, vals {key: values}, pins {key: [pins per group]}, start {key: bool})], slc)"""
    keys = ctx['keys']
    DSF, HEAD, RRF, H1MF = ctx['DSF'], ctx['HEAD'], ctx['RRF'], ctx['H1MF']
    q0 = {k: [g['q'] for g in fits[k]['fg']] for k in keys}
    pn_of = {k: str(DSF[k]['blk_period'][0]) for k in keys}
    smeared = [k for k in keys if AD_B_S2.get(pn_of[k], 0.0)]
    P = {k: DSF[k]['P'] for k in keys}
    t0 = __import__('time').time()

    def tune(F):
        return s2_tune({k: F[k]['mix_ad'] for k in keys}, s2in, smeared)

    def values(F, s2):
        return {k: (ft_values(b_smear(F[k]['mix_ad'], s2, P[k])) if np.isfinite(s2) else {p: np.nan for p in FT_SAVE_PARS}) if k in smeared
                else ft_values(F[k]['mix_ad']) for k in keys}

    nom = tune(fits)
    ref = values(fits, nom['s2'])
    out = []

    def record(group, name, F, side=None, s2=None):
        tn = tune(F) if s2 is None else dict(s2=s2, ok=True)
        out.append(dict(group=group, name=name, side=side, s2=tn['s2'], s2_ok=tn['ok'], vals=values(F, tn['s2']),
                        pins={k: [g['pins'] for g in F[k]['fg']] for k in keys}, start={k: any(g.get('start_flag', False) for g in F[k]['fg']) for k in keys}))
        log(f"[{__import__('time').time() - t0:.0f} s] {group}: {name}: s² {tn['s2']:.4f}" + ('' if tn['ok'] else ' (tune FAILED)')
            + ' | Δrms ' + ' '.join(f"{k[1]} {out[-1]['vals'][k]['rms'] - ref[k]['rms']:+.4f}" for k in keys))

    def refit(K=None, d=None, wt=None, ks=None, **kw):
        ks = keys if ks is None else ks
        return {k: (ft_fits(ctx, k, K=None if K is None else K[k], d=None if d is None else d[k], wt=None if wt is None else wt[k], q0=q0[k], s2=0.0, **kw)
                    if k in ks else fits[k]) for k in keys}

    def kern(rt, echoes=ECHOES, ks=None, h1m=H1MF, **kw):
        ks = keys if ks is None else ks
        return ad_kernels_periods(DSF, ks, rt if all(isinstance(v, dict) and 'C' in v for v in rt.values()) else {pn: rt for pn in rt['C']}, echoes=echoes, h1m=h1m, **kw)

    log(f"nominal tune: s² {nom['s2']:.4f}, zeros " + ' '.join(f'{s} {v:.3f}' for s, v in nom['per'].items()) + f", ref {nom['ref']:.4f} ± {nom['ref_err']:.4f}")
    for side, n_ in FT_WIN_SYST.items():
        record('window', f'±{n_:g} σ_AD', refit(n=n_), side=side)
    record('weights', f'error floor {FT_FLOOR:g} · peak', refit(floor=FT_FLOOR))
    record('alignment', 'mean-aligned mixture', refit(align='mean'))
    zs = [v for v in nom['per'].values() if np.isfinite(v)]
    for nm, s2v in (('lowest per-segment zero', min(zs)), ('highest per-segment zero', max(zs))):
        record('s2', f'{nm} {s2v:.3f}', fits, s2=s2v)
    k21 = [k for k in keys if pn_of[k] == AD_H1M['period']]
    h21, h22 = HEAD[AD_H1M_HANDLE[0]], HEAD[AD_H1M_HANDLE[1]]
    HV = [(f'τ_in {t:g} ns', fit_ad_h1m(h21, h22, t, start=ctx['HF']['p'])) for t in AD_H1M_TAU_VAR]
    HV += [(f'{ne} echo differences', fit_ad_h1m(h21, h22, n_echo=ne, start=ctx['HF']['p'])) for ne in AD_H1M_NECHO_VAR]
    for nm, h in HV:
        record('H(2.1)', nm, refit(K=kern(RRF, ks=k21, h1m={AD_H1M['period']: (AD_H1M['base'], h1m_instrument(h['p']))}), ks=k21))
    hs = fit_ad_h1m(h21, h22, start=ctx['HF']['p'], skew=True)
    record('H skew', f"k3 {hs['p']['k3']:+.3f} ns³ free", refit(K=kern(RRF, ks=k21, h1m={AD_H1M['period']: (AD_H1M['base'], h1m_instrument(hs['p']))}), ks=k21))
    for nm, vs in hw_var.items():
        for v in vs:
            ec = tuple(x * v.get('ec', 1.0) if i % 2 == 0 else x for i, x in enumerate(ECHOES))
            record('hardware', f"{nm}: {', '.join(f'{a} × {b:g}' for a, b in v.items())}", refit(K=kern(RRF, echoes=ec, bl=v.get('bl', 1.0) * BL)))
    rtp = fit_ad_taps(HEAD, keys, proc='joint' if AD_TAPS_PROC == '3b' else '3b')
    record('taps procedure', 'joint taps fit', refit(K=kern(rtp)))
    for nm, rs in taps_variants_cond(HEAD, keys, ctx['RTF']).items():
        record('taps', nm, refit(K=kern(rs)))
    for nm, ec in echo_variants.items():
        rt = ad_taps_cond(HEAD, keys, fit_ad_taps(HEAD, keys, echoes=ec), echoes=ec)
        record('echo model', nm, refit(K=kern(rt, echoes=ec)))
    for form in FT_SCOPE_VAR:
        rt = ad_taps_cond(HEAD, keys, fit_ad_taps(HEAD, keys, scope=form), scope=form)
        record('scope', form, refit(K=kern(rt)))
    if traces is not None:      # full production chain per variant: corrected overlays → blocks, weights, H and taps refit (`ft_context`) → b → s²
        for nm, v in FT_RFL_VAR:
            wv = {}
            for k in keys:
                pn = pn_of[k]
                mdl = dict(rmod[pn], short=dict(rmod[pn]['short'], rho=v.get('rho', 1.0) * rmod[pn]['short']['rho']), **{a: b for a, b in v.items() if a != 'rho'})
                wv[k] = ff_overlays(traces[k], mdl, refit=True)[0]      # all traces: `group_blocks` keeps the block period, WTG sums every trace (= production)
            cv = ft_context(data, wv, keys)
            record('reflection', nm, {k: ft_fits(cv, k, q0=q0[k], s2=0.0) for k in keys})
            hp, hp0 = cv['HF']['p'], ctx['HF']['p']
            dp = max((abs(float(hp[a]) - float(hp0[a])) for a in hp0 if a in hp), default=np.nan)
            dc = max(float(np.max(np.abs(np.asarray(cv['RTF']['C'][pn]) - np.asarray(ctx['RTF']['C'][pn])))) for pn in ctx['RTF']['C'])
            out[-1].update(hf=dict(chi2=cv['HF']['chi2'], p=cv['HF']['p']), rtf_c=cv['RTF']['C'])
            log(f"    {nm}: H χ² {cv['HF']['chi2']:.4g} (nominal {ctx['HF']['chi2']:.4g}), H max |Δp| {dp:.3g} ({len(hp)} / {len(hp0)} pars), taps max |ΔC| {dc:.3g}")
    slc = {}
    for k in keys:
        prof, amp, ch = _ft_trace_profiles(DSF[k], wb[k])
        slc[k] = []
        for part in np.array_split(np.unique(ch), n_slices):
            idx = np.where(np.isin(ch, part))[0]
            wt = np.nansum(amp[idx], 0)
            v = ft_values(ft_fits(ctx, k, d=_ft_block(DSF[k], prof, idx), wt=wt / wt.sum(), q0=q0[k])['mix'])
            slc[k].append(dict(v, traces=len(idx), frac=len(idx) / len(ch)))
    log(f"[{__import__('time').time() - t0:.0f} s] time slices done")
    return dict(nominal=nom, ref=ref, variants=out, slc=slc, nominal_pins={k: [g['pins'] for g in fits[k]['fg']] for k in keys})


def ft_ripple_closure(ctx, fits, n_donor=FT_RIPPLE_DONORS, seed=11):
    """Pull of the ~2 ns ripple on the group b (diagonal χ², `ad_fit_fwd` at ±`AD_FWD_N`): truth = each group's nominal fit; injected = the in-window
    residual of a group (1-3) of another segment, resampled in units of its b RMS onto the target RMS and scaled to the target's residual rms, + block
    noise → refit → {key: dict(rms, rms_left, rms_right: rms of the bias over groups × donors)}"""
    rng = np.random.default_rng(seed)
    keys = ctx['keys']
    S = {}
    for k in keys:
        d, K = ctx['DSF'][k], ctx['KF'][k]
        g = d['grid']; wA = ad_window(d, AD_FWD_N)
        blk = [ad_blocks(d, K, groups=(m,)) for m in range(N_GROUPS)]
        S[k] = dict(g=g, blk=blk, w=(g.x >= wA[0]) & (g.x <= wA[1]), res=[dm['blk_Y'][0] - fits[k]['fg'][m]['M'][0] for m, (dm, _) in enumerate(blk)],
                    rms=[bunch_rms(f['p']) for f in fits[k]['fg']])
    out = {}
    for k in keys:
        s = S[k]; g, w = s['g'], s['w']
        donors = [(k2, m2) for k2 in keys if k2 != k for m2 in range(1, N_GROUPS)]
        dv = []
        for m in range(N_GROUPS):
            (dm, km), ft = s['blk'][m], fits[k]['fg'][m]
            Mt, rms_t, side_t = ft['M'][0], s['rms'][m], bunch_side_rms(ft['p'])
            target = np.sqrt(np.mean(s['res'][m][w] ** 2))
            for i in rng.choice(len(donors), n_donor, replace=False):
                k2, m2 = donors[i]; s2 = S[k2]
                ri = np.interp(g.x / rms_t, s2['g'].x / s2['rms'][m2], np.where(s2['w'], s2['res'][m2] / s2['blk'][m2][0]['blk_Y'][0].max(), 0.0), left=0, right=0)
                ri = ri * target / max(np.sqrt(np.mean((ri * Mt.max())[w] ** 2)), 1e-30) * Mt.max()
                f = ad_fit_fwd(dict(dm, blk_Y=(Mt + ri + rng.standard_normal(g.nb) * dm['blk_E'][0])[None]), km, AD_FWD_N, q0=ft['q'])
                sr = bunch_side_rms(f['p'])
                dv.append((bunch_rms(f['p']) - rms_t, sr[0] - side_t[0], sr[1] - side_t[1]))
        dv = np.array(dv)
        out[k] = dict(zip(('rms', 'rms_left', 'rms_right'), np.sqrt(np.mean(dv ** 2, 0))))
    return out


def _ft_deltas(syst, k, p, group, side=None):
    # signed Δ of every usable variant of `group` (tune ok) for segment k, parameter p, vs the nominal retuned `ref`
    return [v['vals'][k][p] - syst['ref'][k][p] for v in syst['variants'] if v['group'] == group and v['s2_ok'] and (side is None or v['side'] == side)]


def ft_uncertainty(nominal, syst, boot, ripple):
    """Error table per segment: stat = bootstrap std; per systematic group max |Δ| over its variants (Δ vs the nominal retuned by the same code; variants whose
    s² tune failed are left out and listed in `s2_fail`); window per side (`_window_lo` = the `FT_WIN_SYST['lo']` variant, `_window_hi`), `_window` = the larger; ripple =
    `ft_ripple_closure` (rms, rms_left, rms_right only; other parameters NaN, not in the total); time = slice scatter beyond the slices' statistics
    → {key: {p, p_stat, p_<group>, p_window_lo / hi, p_tot (window = larger), p_tot_lo / hi, time_flag, pins_flag, start_flag, s2_fail}}."""
    out = {}
    grp = [g for g in FT_ERR_GROUPS if g not in FT_DIAG_GROUPS]
    fail = ';'.join(v['name'] for v in syst['variants'] if not v['s2_ok'])
    nom_pins = {k: v for k, v in syst.get('nominal_pins', {}).items()}
    for k, v0 in nominal.items():
        r = {}
        for i, p in enumerate(FT_SAVE_PARS):
            stat = float(np.std(boot[k][:, i]))
            e = {g: max(np.abs(_ft_deltas(syst, k, p, g)), default=0.0) for g in grp}
            lo, hi = (max(np.abs(_ft_deltas(syst, k, p, 'window', s_)), default=0.0) for s_ in ('lo', 'hi'))
            e['window'] = max(lo, hi)
            sl = np.array([s[p] for s in syst['slc'][k]])
            se2 = np.array([stat ** 2 / s['frac'] for s in syst['slc'][k]])
            e['time'] = float(np.sqrt(max(sl.var(ddof=1) - se2.mean(), 0.0)))
            e['ripple'] = float(ripple[k][p]) if p in ripple[k] else np.nan
            rest = stat ** 2 + sum(x ** 2 for g, x in e.items() if g != 'window' and np.isfinite(x))
            r.update({p: v0[p], f'{p}_stat': stat, **{f'{p}_{g}': float(e[g]) for g in FT_ERR_GROUPS}, f'{p}_window_lo': float(lo), f'{p}_window_hi': float(hi),
                      f'{p}_tot': float(np.sqrt(rest + e['window'] ** 2)), f'{p}_tot_lo': float(np.sqrt(rest + lo ** 2)), f'{p}_tot_hi': float(np.sqrt(rest + hi ** 2))})
        r['time_flag'] = any(r[f'{p}_time'] > r[f'{p}_stat'] for p in FT_SAVE_PARS)
        r['pins_flag'] = ';'.join(v['name'] for v in syst['variants'] if k in nom_pins and v['pins'][k] != nom_pins[k])
        r['start_flag'] = ';'.join(v['name'] for v in syst['variants'] if v['start'][k])
        r['s2_fail'] = fail
        out[k] = r
    return out


def ft_covariance(syst, unc, keys, pars=FT_SAVE_PARS):
    """Cross-segment covariance of the full-train b errors: per systematic group ρ_ij from the signed variant shifts, Σ_v Δ_v,i Δ_v,j normalised
    (0 where a segment does not move), cov_ij = ρ_ij σ_i σ_j with σ = the table's group error (`unc`; window = the larger side); stat, ripple and time diagonal
    → {p: dict(cov [n, n] (diagonal = `_tot`²), rho {group: [n, n]})}, keys order = `keys`."""
    grp = [g for g in FT_ERR_GROUPS if g not in FT_DIAG_GROUPS]
    out = {}
    for p in pars:
        diag = np.array([unc[k][f'{p}_stat'] ** 2 + sum(unc[k][f'{p}_{g}'] ** 2 for g in FT_DIAG_GROUPS if np.isfinite(unc[k][f'{p}_{g}'])) for k in keys])
        C, R = np.diag(diag), {}
        for g in grp:
            vs = [v for v in syst['variants'] if v['group'] == g and v['s2_ok']]
            sg = np.array([unc[k][f'{p}_{g}'] for k in keys])
            if not vs:
                continue
            D = np.array([[v['vals'][k][p] - syst['ref'][k][p] for k in keys] for v in vs])
            S = D.T @ D
            nn = np.sqrt(np.diag(S))
            rho = np.where(np.outer(nn, nn) > 0, S / np.where(np.outer(nn, nn) > 0, np.outer(nn, nn), 1.0), 0.0)
            np.fill_diagonal(rho, 1.0)
            R[g] = rho
            C = C + rho * np.outer(sg, sg)
        out[p] = dict(cov=C, rho=R)
    return out


def save_ft_covariance(cov, keys, f):
    """`ft_covariance` → npz: 'keys' (run|seg), '<p>' = cov, '<p>|rho|<group>' = per-group correlation."""
    z = {'keys': np.array([f'{k[0]}|{k[1]}' for k in keys])}
    for p, c in cov.items():
        z[p] = c['cov']
        z.update({f'{p}|rho|{g}': r for g, r in c['rho'].items()})
    np.savez(f, **z)


# ---------------- products on disk ----------------
def save_npz(path, prod, compress=False):
    """{(run, seg): {field: array}} → npz keyed 'run|seg|field'."""
    (np.savez_compressed if compress else np.savez)(path, **{f'{k[0]}|{k[1]}|{n}': np.asarray(v) for k, d in prod.items() for n, v in d.items()})


def load_npz(path):
    """`save_npz` file → {(run, seg): {field: value}} (numpy scalars / arrays)."""
    out = {}
    with np.load(path) as z:
        for name in z.files:
            r, s, n = name.split('|')
            out.setdefault((int(r), s), {})[n] = z[name][()]
    return out


def save_tails(path, tails):
    """`train_end_tails` result {period: {field: value}} → npz keyed 'period|field'."""
    np.savez(path, **{f'{pn}|{n}': np.asarray(v) for pn, d in tails.items() for n, v in d.items()})


def load_tails(path):
    out = {}
    with np.load(path) as z:
        for key in z.files:
            pn, n = key.split('|')
            out.setdefault(pn, {})[n] = z[key][()]
    return out


def load_s2_inputs(path):
    """Frozen inputs of `s2_tune` (`inputs/s2_tune_inputs.npz`): SBND AD-matched stack moments per segment (`stk`: n, var, m4, W (±2.5σ window),
    evar, LmR), D_ν = MC ν reco − true time (`dnu`, median removed), AD-stat errors of the b variance of the 50 Ω segments (`ead`) → dict(stk, dnu, ead)."""
    stk, ead = {}, {}
    with np.load(path) as z:
        dnu = z['dnu']
        for name in z.files:
            p = name.split('|')
            if p[0] == 'stk':
                stk.setdefault((int(p[1]), p[2]), {})[p[3]] = z[name][()]
            elif p[0] == 'ead':
                ead[p[1]] = float(z[name])
    return dict(stk=stk, dnu=dnu, ead=ead)
