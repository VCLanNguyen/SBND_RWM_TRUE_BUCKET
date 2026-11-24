"""AD hardware response R_AD and the true-bucket fit machinery (Steps 4-5).

- Model: periodic fine grid over one bunch period (`Grid`), kernels indexed by lag, circular convolution by FFT
- R_AD per AD config (`AD_CONFIGS`): cable skin effect ⊗ scope low-pass ⊗ echoes ⊗ (δ + correction taps); AD config 2.1 (1 MΩ) = the
  50 Ω response ⊗ H (`AD_H1M`, `fit_ad_h1m`)
- True bucket b = asymmetric generalised Gaussian 'agg2' (σL, σR, βL, βR; `bunch_shape`), fitted forward: A · rebin(b ⊗ R_AD) in ±`AD_FWD_N` σ_AD
- End-of-spill probes (`ad_term_products`): b-free round-trip transfer per AD config from every stored trace (Step 2)
"""
import glob

import numpy as np
import pandas as pd
from scipy.special import erf, gamma
from scipy.optimize import least_squares

import paths
import ad_trace as at

BIN_NS = 0.16                      # = AD sample interval
SUB = 8                            # fine-grid points per bin
MIN_AD = 5                         # good AD traces per chunk
BL = 1.45e-8 * 1.254 * 174         # skin-effect b·l [√s]: LDF5-50A 1.254 dB/100 ft at 1 GHz × 174 ft
CABLE_DELAY = 198.0                # RWM → MI12 Heliax one-way delay [ns] (~53 m, 89 % c; TM-2556)
SIG_SCOPE = 0.133 / 0.5            # MSO54 500 MHz → Gaussian σ [ns]
F_SCOPE = 0.5                      # MSO54 band limit [GHz] (`scope_kernel` low-pass forms)
AD_SCOPE_FORM = 'butter6'          # scope response of the AD fits (causal Butterworth at `F_SCOPE`)
ECHOES = (0.0150, 9.98, -0.0310, 14.16, 0.0083, 12.70)
ECHO_SETS = {                      # δ echoes (amplitude, delay [ns]) × 3 per AD config, two fit methods: echo-model variants of the errors only
    'unfolded': {'1.1': (0.0143, 9.43, -0.0518, 13.94, 0.0127, 12.42), '1.3': (0.0115, 9.11, -0.0368, 14.27, 0.0114, 11.93),
                 '2.1': (0.0108, 7.32, -0.0409, 14.03, 0.0400, 12.15)},
    'folded': {'1.1': (0.0127, 9.40, -0.0705, 13.74, 0.0536, 12.68), '1.3': (0.0111, 9.01, -0.0643, 13.87, 0.0476, 12.53),
               '2.1': (0.0109, 7.38, -0.0957, 13.57, 0.1057, 12.73)}}
AD_CORE_N = 2.0                    # AD true-bucket fit window: ±n · σ_AD (segment median AD trace σ); core, β not pulled by the AD ringing / late excess
B_KNEE = 2.0                       # 'aggs' b: core → tail knee at B_KNEE σ per side (fixed knee); free knee: kL / kR fitted
B_SHARP = 4.0                      # 'aggs' b: knee sharpness s (fixed)
AD_BOOT = 200                      # `ad_bootstrap` chunk resamples per segment (`ad_neighbour_response`: trace resamples)
AD_BOOT_MIN_CHUNKS = 8             # `ad_bootstrap` 'auto': fewer chunks → trace-level resampling (chunk bootstrap degenerate)
AD_FWD_N = 2.5                     # `ad_fit_fwd` nominal window ±n · σ_AD: the full RMS (the width b is quoted by) needs ±2.5 σ (narrower windows extrapolate the core shape)
AD_B_MIN = 1.8                     # lower bound of βL, βR of the AD b fits (`AD_PAR`): b tuned on SBND data has β 1.85-2.35, lowest − 2 σ = 1.82
AD_B_COND = dict(beta=(AD_B_MIN, 2.4), rmax=1.0, pos=0.3, shared=True)     # conditions of the true bucket b (`ad_cond_fits`): βL, βR range; σL / σR ≤ rmax; |mean of b + kernel mean − group centroid| ≤ pos [ns]; βL, βR, σL / σR shared along the spill (width, x₀, A per group)
AD_B_S2 = {'1.1': 0.0, '1.3': 0.0, '2.1': 0.175, '2.2': 0.0}   # symmetric width added to the true bucket b per AD config [ns²] (`b_smear`): corrects the 1 MΩ kernel H only (H as fitted, τ_in not measurable on the AD), 50 Ω = 0; tuned on the SBND ±2.5σ windowed variance (AD-matched, ⊗ D_ν) so the mean AD config 2.1 σ_det² step vs the error-weighted mean of the five 50 Ω segments (SBND ⊕ AD b stat) is zero; effective, not an AD measurement
AD_B_TIE = False                   # AD agg2 b fits (`ad_fit_fwd`, `fit_ad_taps`, `group_mixture`): True = βL = βR one β per segment, False = βL, βR free per side
AD_H1M_HANDLE = ((2, 'normal_a'), (2, 'normal_b'))   # 1 MΩ handle: Run 2 normal_a (AD config 2.1 traces, 1 MΩ) vs normal_b (AD config 2.2, 50 Ω)
AD_H1M = dict(config='2.1', base='2.2', n_echo=3, t_echo=(2.5, 16.0), half=9.4)
"""1 MΩ kernel H (`h1m_response`, `fit_ad_h1m`): R_AD(period) = R_AD(base) ⊗ H; H fitted on the raw Normal handle (2.1 vs 2.2 traces) on |x − peak| ≤ half,
`n_echo` echo differences on lags `t_echo` [ns]; τ_in = `RC_TAU[period]` (datasheet capacitance, the handle cannot separate τ_in from the beam change);
handle segments `AD_H1M_HANDLE`."""
AD_H1M_TAU_VAR = (0.575, 0.725)    # τ_in systematic [ns]: C_IN_1M ± 1.5 pF (MSO54 tolerance); the only τ_in variants (Narrow's χ² preference not used: its b RMS ≈ the RC smearing τ_in → degenerate)
AD_H1M_NECHO_VAR = (4,)            # echo-difference count variants of H (2 echoes: rejected by the handle, χ² 7384 vs 1014)
AD_MAX_CLIPPED = 0                 # clipped samples per AD trace
TAP_MIN, TAP_END = 4.0, 4.0        # AD correction-tap lags [ns]: TAP_MIN … P − TAP_END (outside both bunch cores)
TAP_LAM, TAP_MU = 100.0, 10.0      # 2nd-difference smoothness, ridge
TAP_SETS = {'nominal': {}, 'strong': dict(lam=1000.0, mu=30.0), 'from5': dict(tap_min=5.0, tap_end=5.0)}   # robustness variants
AD_TAPS_SHARE = {'1.1': '50 Ω', '1.2': '50 Ω', '1.3': '50 Ω', '2.2': '50 Ω'}
AD_TAPS_PROC = '3b'                # `fit_ad_taps` procedure: '3b' = b per segment from the ±`AD_FWD_N` σ_AD window only, alternating with the taps on the full period (outside-window structure → shared taps, never b); 'joint' = b and taps fitted together on the full period
AD_TAPS_3B_TOL, AD_TAPS_3B_MAX = 5e-5, 40   # 3b outer loop: max |Δ tap| to stop, max iterations
"""Taps groups of `fit_ad_taps` (AD config → group; AD configs not listed: own taps): one taps set for every 50 Ω AD config (same RWM → Heliax → splitter →
MSO54 chain in Run 1 and Run 2), fitted on all their segments together (b per segment free) → separable from the b tails also for one-segment AD configs;
no vertical-scale (V / count) term: echo amplitude × f null in AD config 1.3 (0.94 ± 0.06) and in the AD config 2.2 control (0.92 ± 0.08), scope σ degenerate with
the taps; AD config 2.1 (1 MΩ) = the 50 Ω set ⊗ H (`AD_H1M`)"""
RESP_ITER = 20                     # alternating iterations of `fit_ad_taps`
N_BUCKETS = 81                       # filled buckets per AD trace (core batch)
AD_SKIP_BUCKETS = (0, 20)            # buckets left out of the groups: k = 0 = first bunch of the spill (after the notch), no bunch one bunch period earlier; k = 20 = reached by the cable round trip (2 × `CABLE_DELAY` = 21 P − 1.5 ns) of the partial edge bucket before k = 0
OV_BINS = 121                      # histogram bins per bucket (BIN_NS, ± P / 2; padded)
N_GROUPS = 4                       # round-trip groups: buckets k = 0-20, 21-41, 42-62, 63-80 (`rt_group`)
AD_CONFIGS = {                     # AD configs (lasting amplitude steps or a scope-input change in the traces)
    '1.1': ('2025-01-01', '2025-04-07 18:00'),        # amplitude ~110-125 (Before Rotation A-C, Rotation)
    '1.2': ('2025-04-07 18:00', '2025-05-12 17:00'),  # ~165, clipped (no traces used)
    '1.3': ('2025-05-12 17:00', '2025-10-01'),        # ~65-70 (After Rotation)
    '2.1': ('2025-10-01', '2026-05-15'),              # ~60-65 (Wide, Normal A, Narrow)
    '2.2': ('2026-05-15', '2027-01-01')}              # ~180 (Normal B, June 2026)
AD_TERMINATION = {'1.1': '50 Ω', '1.2': '50 Ω', '1.3': '50 Ω', '2.1': '1 MΩ', '2.2': '50 Ω'}
"""Scope input termination per AD config (per-trace scan): gain per proton (V·ns / 1e12 p, Σ area of buckets k = 1-19 ×
V/count ÷ E:TOR860) ≈ 18 in Run 1 (V/count 0.008 / 0.02 / 0.004) and June 2026 (0.008), ≈ 36 (× 2) in AD config 2.1 (0.04) = unterminated input; AD config 2.1
round trip (buckets after the end of the spill ÷ their source buckets) ≈ −0.2 … −0.33 vs ≈ −0.01 at 50 Ω; 1 MΩ → 50 Ω inside the AD gap 2026-04-27 → 06-05."""
Z0 = 50.0                          # splitter output / cable impedance [Ω]
C_IN_1M = 13.0e-3                  # scope input capacitance at 1 MΩ [nF] (MSO54: 13.0 ± 1.5 pF typical, Tektronix 5 Series MSO datasheet)
RC_TAU = {pn: Z0 * C_IN_1M if t == '1 MΩ' else 0.0 for pn, t in AD_TERMINATION.items()}
"""Input-RC time constant per AD config [ns]: 1 MΩ = matched source (Z0) into C_IN_1M → causal exp(−t / τ), τ = 0.65 ns (below the 4 ns taps);
50 Ω: ≈ 0.1 ns, inside `SIG_SCOPE` → 0."""
SEGMENTS = [(1, 'before_a'), (1, 'before_c'), (1, 'during'), (1, 'after'), (2, 'wide'), (2, 'normal_a'), (2, 'normal_b'), (2, 'narrow')]   # stable segments of the calibration (Run 1 Before Rotation B is unstable: no b)
SEG_NAMES = {'before_a': 'Before Rotation A', 'before_b': 'Before Rotation B', 'before_c': 'Before Rotation C', 'during': 'During Rotation', 'after': 'After Rotation',
             'wide': 'Wide', 'normal_a': 'Normal A', 'normal_b': 'Normal B', 'narrow': 'Narrow'}
SEG_TITLE = {(r, s): f'Run {r} {SEG_NAMES[s]}' for r, s in SEGMENTS}


def ad_config(ts):
    """AD config name (`AD_CONFIGS`) per timestamp."""
    ts = pd.to_datetime(np.atleast_1d(ts))
    out = np.full(len(ts), '', dtype=object)
    for name, (a, b) in AD_CONFIGS.items():
        out[(ts >= pd.Timestamp(a)) & (ts < pd.Timestamp(b))] = name
    return out


def rt_group(P):
    """Round-trip group per bucket k (0 … `N_GROUPS` − 1): number of cable round trips 2 `CABLE_DELAY` already arrived from earlier buckets."""
    return np.minimum(np.arange(N_BUCKETS) // int(round(2 * CABLE_DELAY / P)), N_GROUPS - 1)


def trace_profiles(w, g, buckets=None):
    """Stacked profile per trace on `g.x` / median bucket amplitude, from the buckets `buckets` (bool mask over `N_BUCKETS`, None = all) of the per-bucket histograms `w` (`true_bucket.corrected_histograms`)."""
    S, N = (w['S'], w['N']) if buckets is None else (w['S'][:, buckets], w['N'][:, buckets])
    num, cnt = S.sum(1, dtype=float), N.sum(1, dtype=float)
    prof = np.empty((len(num), g.nb))
    for i, P in enumerate(w['period']):
        edges = np.arange(-P / 2, P / 2 + BIN_NS, BIN_NS)
        nb = len(edges) - 1
        ok = cnt[i, :nb] > 0
        prof[i] = np.interp(g.x, ((edges[:-1] + edges[1:]) / 2)[ok], num[i, :nb][ok] / cnt[i, :nb][ok]) / w['amp_med'][i]
    return prof


class Grid:
    def __init__(self, P):
        self.P = P
        self.nb = int(round(P / BIN_NS))
        self.n = self.nb * SUB
        self.h = P / self.n
        self.edges = np.linspace(-P / 2, P / 2, self.nb + 1)
        self.x = (self.edges[:-1] + self.edges[1:]) / 2
        self.tf = -P / 2 + (np.arange(self.n) + 0.5) * self.h
        self.slag = (np.arange(self.n) * self.h + P / 2) % P - P / 2

    def wrap_hist(self, d):
        k = np.histogram(d % self.P, bins=self.n, range=(0, self.P))[0].astype(float)
        return k / k.sum()

    def gauss(self, s):
        g = np.exp(-0.5 * (self.slag / s) ** 2)
        return g / g.sum()

    def shift(self, dl):
        k = np.zeros(self.n)
        u = (dl % self.P) / self.h
        i = int(np.floor(u))
        k[i % self.n] += 1 - (u - i)
        k[(i + 1) % self.n] += u - i
        return k

    def cable(self, bl=BL, ncyc=40):
        tt = np.arange(0, ncyc * self.P, self.h / 4)
        step = np.where(tt > 0, 1 - erf(bl / np.sqrt(2 * np.maximum(tt, 1e-9) * 1e-9)), 0.0)
        k = np.zeros(self.n)
        np.add.at(k, np.floor(((tt + self.h / 2) % self.P) / self.h).astype(int) % self.n, np.diff(np.r_[0.0, step]))
        return k / k.sum()

    def conv(self, *ks):
        F = np.ones(self.n, complex)
        for k in ks:
            F = F * np.fft.fft(k)
        return np.real(np.fft.ifft(F))

    def bunch(self, x0, sl, sr):
        f = 0
        for kk in range(-2, 3):
            t = self.tf - x0 + kk * self.P
            f = f + np.exp(-0.5 * (t / np.where(t < 0, sl, sr)) ** 2)
        return f / f.sum()

    def rebin(self, f):
        return f.reshape(self.nb, SUB).sum(1)

    def bin_kernel(self, k):
        # fine lag kernel → bin lag kernel (lag 0 centred in bin 0)
        return np.roll(k, SUB // 2).reshape(self.nb, SUB).sum(1)


def echoes_of(echoes, config):
    """Echo set of one AD config: `echoes` = (amplitude, delay, …) tuple for every AD config, or {AD config: tuple} (AD configs without a set: `ECHOES`)."""
    return echoes.get(str(config), ECHOES) if isinstance(echoes, dict) else echoes


def echo_kernel(g, echoes=ECHOES):
    # δ + cable reflections (amplitude, delay) pairs
    e = np.zeros(g.n)
    e[0] = 1
    for r, dl in zip(echoes[0::2], echoes[1::2]):
        e += r * g.shift(dl)
    return e


def rc_of(rc, config):
    """Input-RC τ of one AD config: `rc` = None (0), a number for every AD config, or {AD config: τ} (`RC_TAU`)."""
    if rc is None:
        return 0.0
    return float(rc.get(str(config), 0.0)) if isinstance(rc, dict) else float(rc)


def rc_kernel(g, tau):
    # scope input RC on the fine lag grid: causal exp(−t / τ) integrated per cell (sums to 1); δ for τ = 0
    k = np.zeros(g.n)
    if tau <= 0:
        k[0] = 1
        return k
    e = np.exp(-np.arange(g.n + 1) * g.h / tau)
    return (e[:-1] - e[1:]) / (1 - e[-1])


def scope_kernel(g, form=AD_SCOPE_FORM, sig=None):
    """Scope band limit on the fine lag grid (sums to 1): 'gauss' = G(`SIG_SCOPE`) (−3 dB at `F_SCOPE`, no ringing);
    'butter4' / 'butter8' = causal analog Butterworth at `F_SCOPE`; 'brick' = zero-phase brick-wall at `F_SCOPE` (sinc ringing)."""
    if form == 'gauss':
        return g.gauss(SIG_SCOPE if sig is None else sig)
    f = np.fft.fftfreq(g.n, g.h)
    if form == 'brick':
        H = (np.abs(f) <= F_SCOPE).astype(complex)
    else:
        from scipy.signal import butter, freqs
        bb, aa = butter(int(form[6:]), 2 * np.pi * F_SCOPE, analog=True)
        H = freqs(bb, aa, 2 * np.pi * f)[1]
    k = np.real(np.fft.ifft(H))
    return k / k.sum()


def tap_lags(D, keys, tap_min=TAP_MIN, tap_end=TAP_END):
    P0 = np.median([D[k]['grid'].P for k in keys])
    return np.arange(tap_min, P0 - tap_end, BIN_NS)


def _resp_parts(g, taus, rlags, scope=AD_SCOPE_FORM, echoes=ECHOES, bl=BL, sig_scope=None, tau_rc=0.0):
    """FFTs on the fine grid: B = cable (`bl`) ⊗ scope (σ `sig_scope`, default `SIG_SCOPE`) ⊗ input RC (`tau_rc`, `rc_kernel`) ⊗ echoes,
    SH = tap shifts (lags `taus`), SR = reflection shifts (lags `rlags`)."""
    sk = g.conv(scope_kernel(g, scope, sig_scope), rc_kernel(g, tau_rc))
    base = g.conv(g.cable(bl), sk) if echoes is None else g.conv(g.cable(bl), sk, echo_kernel(g, echoes))
    return dict(B=np.fft.fft(base), SH=np.fft.fft(np.array([g.shift(t) for t in taus]), axis=1),
                SR=np.fft.fft(np.array([g.shift(t) for t in rlags]), axis=1))


def _refl_sum(pr, m, r):
    """FFT of Σ_{j=1..m} ρ^{⊗j} (j round trips; ρ = Σ r_l δ(t − rlag_l))."""
    rf = r @ pr['SR']
    return sum(rf ** j for j in range(1, m + 1)) if m else np.zeros(pr['B'].shape, complex)


def _resp_fft(pr, m, c, r, refl=True):
    """FFT of R_AD for round-trip group m: B · (δ + taps c) · (δ + Σ_{j=1..m} ρ^{⊗j})."""
    return pr['B'] * (1 + c @ pr['SH']) * (1 + (_refl_sum(pr, m, r) if refl else 0))


def ad_groups(d, K=None, config=None, groups=None):
    """Segment `d` and its R_AD groups `K` [group, fine lag] (None: none) restricted to one AD config `period` and round-trip `groups` (None = all)."""
    m = np.ones(len(d['grp_m']), bool)
    if config is not None:
        m &= d['grp_config'] == config
    if groups is not None:
        m &= np.isin(d['grp_m'], groups)
    return dict(d, **{n: v[m] for n, v in d.items() if n.startswith('grp_')}), (None if K is None else K[m])


def ad_kernels_configs(D, keys, resps, echoes=ECHOES, bl=BL, sig_scope=None, rc=RC_TAU, h1m=None):
    """R_AD per segment and group {key: [group, fine lag]} = cable ⊗ scope ⊗ input RC ⊗ echoes ⊗ (δ + taps), taps from `resps` {AD config: `fit_ad_response` set}
    (each AD config its own set; no round-trip reflection: for head groups); `echoes` (tuple or {AD config: tuple}, `echoes_of`), `bl`, `sig_scope`,
    `rc` (τ or {AD config: τ}, `rc_of`): hardware variations.
    - `h1m` {AD config: (base AD config, H parameters)}: that AD config's R_AD = R_AD(base) ⊗ H (`h1m_response`; H carries the input RC τ_in), e.g. the 1 MΩ
      AD config 2.1 from the 50 Ω AD config 2.2 (`AD_H1M`, `fit_ad_h1m`)"""
    out = {}
    for k in keys:
        d = D[k]
        rows = []
        for pn in d['grp_config']:
            pn = str(pn)
            base, ph = h1m[pn] if h1m and pn in h1m else (pn, None)
            r = resps[base]
            pr = _resp_parts(d['grid'], r['taus'], r['rlags'], r['scope'], echoes_of(echoes, base), bl, sig_scope, rc_of(rc, base))
            F = _resp_fft(pr, 0, r['C'][base], r['rho'][base], refl=False)
            if ph is not None:
                F = F * h1m_response(np.fft.fftfreq(d['grid'].n, d['grid'].h), ph)
            rows.append(np.real(np.fft.ifft(F)))
        out[k] = np.array(rows)
    return out


def fit_ad_taps(D, keys, tap_min=TAP_MIN, tap_end=TAP_END, lam=TAP_LAM, mu=TAP_MU, n_iter=RESP_ITER, scope=AD_SCOPE_FORM, echoes=ECHOES, rc=RC_TAU,
                share=AD_TAPS_SHARE, sig_scope=None, proc=None):
    """R_AD correction taps per taps group (`share`: AD config → group, default `AD_TAPS_SHARE`; None or unlisted AD config = its own group):
    - per group: taps c on lags `tap_min` … P − `tap_end` shared by all groups of all `keys` segments in the group's AD configs (smoothness `lam`, ridge `mu`);
      b = agg2 per segment (σL, σR, β; βL = βR when `AD_B_TIE`, else βL, βR free); per group x0, amplitude; no floor / flat term, no round-trip
      reflection (head groups); full bunch period
    - alternating: taps given b (linear), b + x0 / amplitude given taps; errors scaled so the fit without taps has χ²/n = 1 per group 
    - a group with one segment: taps and that segment's b tails are not separable (taps fitted on its own data)
    - `echoes`: one (amplitude, delay) tuple for every AD config, or {AD config: tuple} (`echoes_of`); `rc`: input-RC τ or {AD config: τ} (`rc_of`);
      `sig_scope`: scope σ for every AD config, or {AD config: σ} (others `SIG_SCOPE`)
    - returns dict(taus, rlags, scope, C {AD config} (the group's taps), rho {AD config} (zeros), chi {AD config: χ²/n end of its group}, segs {AD config: its
      segments}, group {AD config: group}, chi2 {group: χ² end (scaled errors)}, nbins {group}) for `ad_kernels_configs`
    - `proc` (default `AD_TAPS_PROC`): 'joint' = as above; '3b' = `_fit_ad_taps_3b` (b from the window only), started from the joint taps
    """
    if (AD_TAPS_PROC if proc is None else proc) == '3b':
        return _fit_ad_taps_3b(D, keys, fit_ad_taps(D, keys, tap_min, tap_end, lam, mu, n_iter, scope, echoes, rc, share, sig_scope, proc='joint'),
                               lam, mu, scope, echoes, rc, share, sig_scope)
    share = share or {}
    taus = tap_lags(D, keys, tap_min, tap_end)
    P0 = np.median([D[k]['grid'].P for k in keys])
    rlags = np.arange(-P0 / 2, P0 / 2, BIN_NS)
    nT, shp = len(taus), _ad_shp()
    nb_ = len(shp)
    sg = (lambda pn: sig_scope.get(str(pn))) if isinstance(sig_scope, dict) else (lambda pn: sig_scope)
    PR = {k: _resp_parts(D[k]['grid'], taus, rlags, scope, echoes_of(echoes, D[k]['grp_config'][0]), sig_scope=sg(D[k]['grp_config'][0]),
                         tau_rc=rc_of(rc, D[k]['grp_config'][0])) for k in keys}     # one period per segment (`ad_groups`)
    groups = {}
    for k in keys:
        for i, pn in enumerate(D[k]['grp_config']):
            groups.setdefault(share.get(str(pn), str(pn)), []).append((k, i))
    out = dict(taus=taus, rlags=rlags, scope=scope, C={}, rho={}, chi={}, segs={}, group={}, chi2={}, nbins={})

    def model(k, p, xa, aa, c):
        g = D[k]['grid']
        return aa * g.nb * g.rebin(np.real(np.fft.ifft(np.fft.fft(bunch_shape(g, xa, p, 'agg2')) * PR[k]['B'] * (1 + c @ PR[k]['SH']))))

    blo, bhi = [AD_PAR[n][1] for n in shp], [AD_PAR[n][2] for n in shp]
    for pn, bl in groups.items():
        segs = sorted({k for k, _ in bl})
        c, q0, w2 = np.zeros(nT), {}, {}
        Dt = np.diff(np.eye(nT), 2, axis=0)
        for k, i in bl:          # start per group, no taps
            Y, E = D[k]['grp_Y'][i], D[k]['grp_E'][i]
            f = least_squares(lambda q: (Y - model(k, b_from_fit(q), q[nb_], q[nb_ + 1], c)) / E, [AD_PAR[n][0] for n in shp] + [0, Y.max() / 4],
                              bounds=(blo + [PAR['xa'][1], 0], bhi + [PAR['xa'][2], PAR['aa'][2]]), x_scale='jac')
            q0[(k, i)], w2[(k, i)] = f.x, 1 / (E * np.sqrt((f.fun ** 2).mean())) ** 2
        ns = nb_ * len(segs)

        def unpack(q):
            return {k: b_from_fit(q[nb_ * j:nb_ * j + nb_]) for j, k in enumerate(segs)}, q[ns:].reshape(len(bl), 2)

        def resid(q, c):
            sp, bp = unpack(q)
            return np.concatenate([(D[k]['grp_Y'][i] - model(k, sp[k], *bp[j], c)) * np.sqrt(w2[(k, i)]) for j, (k, i) in enumerate(bl)])

        lo = np.r_[blo * len(segs), [PAR['xa'][1], 0] * len(bl)]
        hi = np.r_[bhi * len(segs), [PAR['xa'][2], PAR['aa'][2]] * len(bl)]
        q = np.r_[np.concatenate([np.median([q0[b][:nb_] for b in bl if b[0] == k], 0) for k in segs]), np.concatenate([q0[b][nb_:] for b in bl])]
        for _ in range(n_iter):
            sp, bp = unpack(q)
            AtA, Atb = lam * Dt.T @ Dt + mu * np.eye(nT), np.zeros(nT)
            for j, (k, i) in enumerate(bl):
                g = D[k]['grid']
                xa, aa = bp[j]
                X = np.fft.fft(bunch_shape(g, xa, sp[k], 'agg2')) * PR[k]['B']
                A = aa * g.nb * np.real(np.fft.ifft(X[None, :] * PR[k]['SH'], axis=1)).reshape(nT, g.nb, SUB).sum(2).T
                AtA += (A * w2[(k, i)][:, None]).T @ A
                Atb += (A * w2[(k, i)][:, None]).T @ (D[k]['grp_Y'][i] - aa * g.nb * g.rebin(np.real(np.fft.ifft(X))))
            c_new = np.linalg.solve(AtA, Atb)
            q = least_squares(resid, q, args=(c_new,), bounds=(lo, hi), x_scale='jac').x
            done = np.abs(c_new - c).max() < 1e-6
            c = c_new
            if done:
                break
        r = resid(q, c)
        out['chi2'][pn], out['nbins'][pn] = float((r ** 2).sum()), len(r)
        for p_ in sorted({str(D[k]['grp_config'][i]) for k, i in bl}):
            out['C'][p_], out['rho'][p_], out['group'][p_] = c, np.zeros(len(rlags)), pn
            out['segs'][p_] = sorted({k for k, i in bl if str(D[k]['grp_config'][i]) == p_})
            out['chi'][p_] = out['chi2'][pn] / out['nbins'][pn]
    return out


def _fit_ad_taps_3b(D, keys, rt, lam, mu, scope, echoes, rc, share, sig_scope, tol=AD_TAPS_3B_TOL, n_max=AD_TAPS_3B_MAX, b_fit=None):
    """3b taps (`fit_ad_taps` proc '3b'): per taps group, outer loop from the joint taps `rt` (same lags / groups): (1) b per segment = `ad_fit_fwd` on the
    segment's groups at ±`AD_FWD_N` σ_AD with the current taps; (2) taps on the full bunch period given those b (linear, x0 / amplitude per group refit,
    smoothness `lam`, ridge `mu`, errors scaled as `fit_ad_taps`); until max |Δ tap| < `tol` → `fit_ad_taps` output (+ n_outer {group}); `b_fit(d, K, n)` (dict with 'p') replaces `ad_fit_fwd` in (1)."""
    share = share or {}
    taus, rlags = rt['taus'], rt['rlags']
    nT = len(taus)
    sg = (lambda pn: sig_scope.get(str(pn))) if isinstance(sig_scope, dict) else (lambda pn: sig_scope)
    PR = {k: _resp_parts(D[k]['grid'], taus, rlags, scope, echoes_of(echoes, D[k]['grp_config'][0]), sig_scope=sg(D[k]['grp_config'][0]),
                         tau_rc=rc_of(rc, D[k]['grp_config'][0])) for k in keys}
    groups = {}
    for k in keys:
        for i, pn in enumerate(D[k]['grp_config']):
            groups.setdefault(share.get(str(pn), str(pn)), []).append((k, i))
    out = dict(rt, C=dict(rt['C']), chi={}, chi2={}, nbins={}, n_outer={})
    Dt = np.diff(np.eye(nT), 2, axis=0)

    def model(k, p, xa, aa, c):
        g = D[k]['grid']
        return aa * g.nb * g.rebin(np.real(np.fft.ifft(np.fft.fft(bunch_shape(g, xa, p, 'agg2')) * PR[k]['B'] * (1 + c @ PR[k]['SH']))))

    for grp, bl in groups.items():
        segs = sorted({k for k, _ in bl})
        pn0 = str(D[bl[0][0]]['grp_config'][bl[0][1]])
        c = rt['C'][pn0].copy()
        w2, xa = {}, {}
        for k, i in bl:          # error scale: χ²/n = 1 per group without taps (b, x0, amplitude free, full bunch period), as `fit_ad_taps`
            f0 = ad_fit_fwd(dict(D[k], **{n: v[i:i + 1] for n, v in D[k].items() if n.startswith('grp_')}), np.real(np.fft.ifft(PR[k]['B']))[None], n=99)
            w2[(k, i)] = 1 / (D[k]['grp_E'][i] * np.sqrt(f0['chi2'] / (f0['ndof'] + f0['npar']))) ** 2
            xa[(k, i)] = tuple(f0['grp'][0])
        for it in range(n_max):
            c_in = c.copy()
            bs = {}
            for k in segs:
                idx = [i for k_, i in bl if k_ == k]
                dk = dict(D[k], **{n: v[idx] for n, v in D[k].items() if n.startswith('grp_')})
                bs[k] = (b_fit or ad_fit_fwd)(dk, np.real(np.fft.ifft(PR[k]['B'] * (1 + c @ PR[k]['SH'])))[None].repeat(len(idx), 0), AD_FWD_N)['p']
            for _ in range(n_max):
                AtA, Atb = lam * Dt.T @ Dt + mu * np.eye(nT), np.zeros(nT)
                for k, i in bl:
                    g, Y = D[k]['grid'], D[k]['grp_Y'][i]
                    xa[(k, i)] = tuple(least_squares(lambda q: (Y - model(k, bs[k], q[0], q[1], c)) * np.sqrt(w2[(k, i)]), xa[(k, i)], x_scale='jac').x)
                    x0, a0 = xa[(k, i)]
                    X = np.fft.fft(bunch_shape(g, x0, bs[k], 'agg2')) * PR[k]['B']
                    A = a0 * g.nb * np.real(np.fft.ifft(X[None, :] * PR[k]['SH'], axis=1)).reshape(nT, g.nb, SUB).sum(2).T
                    AtA += (A * w2[(k, i)][:, None]).T @ A
                    Atb += (A * w2[(k, i)][:, None]).T @ (Y - a0 * g.nb * g.rebin(np.real(np.fft.ifft(X))))
                cn = np.linalg.solve(AtA, Atb)
                done = np.abs(cn - c).max() < 1e-6
                c = cn
                if done:
                    break
            if np.abs(c - c_in).max() < tol:
                break
        r = np.concatenate([(D[k]['grp_Y'][i] - model(k, bs[k], *xa[(k, i)], c)) * np.sqrt(w2[(k, i)]) for k, i in bl])
        out['chi2'][grp], out['nbins'][grp], out['n_outer'][grp] = float((r ** 2).sum()), len(r), it + 1
        for p_ in sorted({str(D[k]['grp_config'][i]) for k, i in bl}):
            out['C'][p_] = c
            out['chi'][p_] = out['chi2'][grp] / out['nbins'][grp]
    return out


PAR = dict(sl=(1.1, .2, 4), sr=(1.3, .2, 4), fh=(.05, 0, .9), sh=(3, 1, 9), sd=(1.2, 0, 4), xs=(0, -3, 3), fb=(.05, 0, 1),
           xa=(0, -3, 3), aa=(.2, 0, 10), ca=(0, -1, 1), be=(2.0, 1.0, 6.0), bl=(2.0, 1.0, 6.0), br=(2.0, 1.0, 6.0), fe=(.05, 0, .5), te=(2.0, .2, 10),
           tl=(2.0, 0.3, 8.0), tr=(2.0, 0.3, 8.0), kl=(B_KNEE, 0.5, 5.0), kr=(B_KNEE, 0.5, 5.0))
AD_PAR = dict(PAR, bl=(PAR['bl'][0], AD_B_MIN, PAR['bl'][2]), br=(PAR['br'][0], AD_B_MIN, PAR['br'][2]))     # AD b fits (`ad_fit_fwd`, `fit_ad_taps`): β ≥ `AD_B_MIN`


def bunch_shape(g, x0, p, form='asym'):
    """True bucket b on the fine grid of `g` (sums to 1), mode at x0.
    - 'asym': split Gaussian σL / σR; 'agg': split generalised Gaussian exp(−|t/σ|^β / 2); 'agg2': same, β per side (βL `bl`, βR `br`)
    - 'aggs': per side exp(−E / 2), E = z^β (1 + (z / k)^s)^((βt − β) / s), z = |t| / σ_side, β = `bl` / `br`, tail βt = `tl` / `tr`, knee k = `kl` / `kr`
      (default `B_KNEE`), s = `B_SHARP`: z ≪ k → z^β (core), z ≫ k → ∝ z^βt (tail), smooth through k (no kink); βt = 1 exponential tail, βt = β = agg2
    - 'exp': (1 − f_e) split Gaussian + f_e G(σR) ⊗ late exponential (decay τ_e = `te`)
    - 'two': split Gaussian + second Gaussian (f2, offset d2, σ s2)
    - 'tmpl': tabulated density p['ff'] at p['tt'] (mode frame, tails included), periodic in P
    """
    if form == 'tmpl':
        f = np.interp(g.tf - x0, p['tt'], p['ff'], period=g.P)
        return f / f.sum()
    if form == 'agg':
        f = 0
        for kk in range(-2, 3):
            t = g.tf - x0 + kk * g.P
            f = f + np.exp(-0.5 * np.abs(t / np.where(t < 0, p['sl'], p['sr'])) ** p['be'])
        return f / f.sum()
    if form == 'agg2':
        f = 0
        for kk in range(-2, 3):
            t = g.tf - x0 + kk * g.P
            f = f + np.exp(-0.5 * np.abs(t / np.where(t < 0, p['sl'], p['sr'])) ** np.where(t < 0, p['bl'], p['br']))
        return f / f.sum()
    if form == 'aggs':
        f = 0
        for kk in range(-2, 3):
            t = g.tf - x0 + kk * g.P
            L = t < 0
            z = np.abs(t / np.where(L, p['sl'], p['sr']))
            be, bt = np.where(L, p['bl'], p['br']), np.where(L, p['tl'], p['tr'])
            k = np.where(L, p.get('kl', B_KNEE), p.get('kr', B_KNEE))
            f = f + np.exp(-0.5 * z ** be * (1 + (z / k) ** B_SHARP) ** ((bt - be) / B_SHARP))
        return f / f.sum()
    if form == 'exp':
        k = np.exp(-np.arange(g.n) * g.h / p['te'])
        return (1 - p['fe']) * g.bunch(x0, p['sl'], p['sr']) + p['fe'] * g.conv(g.bunch(x0, p['sr'], p['sr']), k / k.sum())
    if form == 'two':
        return (1 - p['f2']) * g.bunch(x0, p['sl'], p['sr']) + p['f2'] * g.bunch(x0 + p['d2'], p['s2'], p['s2'])
    return g.bunch(x0, p['sl'], p['sr'])


def bunch_rms(p):
    # RMS of the split generalised Gaussian exp(−|t/σ|^β / 2) (`bunch_shape` 'agg' β = p['be'], 'agg2' βL / βR = p['bl'] / p['br']; none = 2 = `asym_rms`);
    # 'tmpl' (p['tt'], p['ff']): RMS of the tabulated density
    if 'ff' in p:
        w = p['ff'] / p['ff'].sum()
        return np.sqrt(w @ (p['tt'] - w @ p['tt']) ** 2)
    be = p.get('be', 2.0)
    bl, br, sl, sr = p.get('bl', be), p.get('br', be), p['sl'], p['sr']
    c = lambda n, b: 2 ** ((n + 1) / b) * gamma((n + 1) / b) / b
    z = c(0, bl) * sl + c(0, br) * sr
    return np.sqrt((c(2, bl) * sl ** 3 + c(2, br) * sr ** 3) / z - ((c(1, br) * sr ** 2 - c(1, bl) * sl ** 2) / z) ** 2)


def bunch_side_rms(p):
    """Per-side RMS about the mode of the split generalised Gaussian exp(−|t/σ|^β / 2) ('agg2': σL, βL / σR, βR) → (left, right)."""
    c = lambda n, b: 2 ** ((n + 1) / b) * gamma((n + 1) / b) / b
    return tuple(s_ * np.sqrt(c(2, b) / c(0, b)) for s_, b in ((p['sl'], p.get('bl', 2.0)), (p['sr'], p.get('br', 2.0))))


def ad_window(d, n=AD_CORE_N):
    """AD fit window (−n · σ_AD, n · σ_AD) [ns] around the bucket centre; σ_AD = segment median AD trace σ (`ad_sigma`, `segments.segment_inputs`)."""
    w = n * np.median(d['ad_sigma'])
    return (-w, w)


def _ad_shp():
    # free b parameters of the AD agg2 fits: σL, σR, β (`AD_B_TIE`: β = βL = βR) or σL, σR, βL, βR
    return ['sl', 'sr', 'bl'] if AD_B_TIE else list(B_PARS)


def b_from_fit(x):
    # b dict (`B_PARS`) from the leading fit entries x (`_ad_shp` order; tied: βR = βL); AD fits, `sbnd_agg2_fit`, error propagation in the nbs
    return dict(sl=x[0], sr=x[1], bl=x[2], br=x[2] if AD_B_TIE else x[3])


def _ad_q_in(q):
    # q in the 4-parameter convention (σL, σR, βL, βR, group pars) → fit vector (tied: β = mean(βL, βR))
    q = np.asarray(q, float)
    return np.r_[q[:2], q[2:4].mean(), q[4:]] if AD_B_TIE else q


def _ad_q_out(x):
    # fit vector (or its errors) → the 4-parameter convention (tied: βR = βL), so callers keep q[:4] = σL, σR, βL, βR
    x = np.asarray(x, float)
    return np.r_[x[:3], x[2], x[3:]] if AD_B_TIE else x


def _resampler(ch, level, rng):
    # index draws for `ad_bootstrap` / `ad_bootstrap_joint`: chunks with replacement (traces grouped by chunk), or traces one by one
    uc = np.unique(ch)
    if level == 'trace' or (level == 'auto' and len(uc) < AD_BOOT_MIN_CHUNKS):
        return lambda: rng.integers(0, len(ch), len(ch))
    pos = {u: np.where(ch == u)[0] for u in uc}
    return lambda: np.concatenate([pos[u] for u in rng.choice(uc, len(uc))])


def h1m_response(f, p):
    """1 MΩ kernel H(f) (`AD_H1M`), f [1/ns]: RC(τ_in) · [(1 − ε) + ε / (1 + iωτ_s)] · (1 + Σ_j r_j e^{−iωT_j}); p = dict(tin, eps, ts, r0, T0, r1, T1, …)."""
    w = 2j * np.pi * f
    h = 1 / (1 + w * p['tin']) * ((1 - p['eps']) + p['eps'] / (1 + w * p['ts']))
    return h * (1 + sum(p[f'r{j}'] * np.exp(-w * p[f'T{j}']) for j in range(sum(1 for x in p if x[0] == 'r' and x[1:].isdigit()))))


_H1M_LO = dict(a=0, s=-4, b2=-0.5, tin=0, eps=0, ts=0.5, k3=-5)
_H1M_HI = dict(a=10, s=4, b2=3, tin=3, eps=0.5, ts=30, k3=5)


def fit_ad_h1m(d21, d22, tau_in=None, n_echo=AD_H1M['n_echo'], start=None, half=AD_H1M['half'], t_echo=AD_H1M['t_echo'], skew=False):
    """1 MΩ kernel H from the raw Normal handle (b-free in the instrument part): Y(2.1) = a · Y(2.2) ⊗ shift s ⊗ G(beam² b2) ⊗ H, H = `h1m_response`
    - `d21`, `d22`: single-group segments (`ad_groups`, head group) of the two AD configs; diagonal χ² (E21² + a² E22²) on |x − peak| ≤ `half`
    - τ_in fixed (default `RC_TAU[AD_H1M['config']]`); a, s, b2 = nuisances (not part of H: the beam change and alignment between the AD configs)
    - no `start`: base fit, then `n_echo` echo differences added greedily (T scan on `t_echo`, only the new echo free) and refitted jointly;
      `start` (a previous p): warm joint refit (bootstrap, τ_in variants); fewer echoes in `start` → the rest added greedily, more → the largest |r| kept
    - `skew`: + nuisance k3 [ns³] (asymmetric beam change, exp(i k3 (2πf)³ / 6); not part of H, dropped by `h1m_instrument`; 'H skew' variant of the errors)
    - returns dict(p, e (errors scaled by √(χ²/ndof) if > 1), chi2, ndof, M (model on the bins), m (window mask))"""
    tau_in = RC_TAU[AD_H1M['config']] if tau_in is None else tau_in
    g = d21['grid']
    Y1, E1, Y2, E2 = d21['grp_Y'][0], d21['grp_E'][0], d22['grp_Y'][0], d22['grp_E'][0]
    f = np.fft.fftfreq(g.nb, BIN_NS)
    F2 = np.fft.fft(Y2)
    m = np.abs(g.x - g.x[np.argmax(Y1)]) <= half

    def model(p):
        sk = np.exp(1j * p.get('k3', 0.0) * (2 * np.pi * f) ** 3 / 6) if skew else 1.0
        return p['a'] * np.real(np.fft.ifft(F2 * np.exp(-2j * np.pi * f * p['s']) * np.exp(-2 * (np.pi * f) ** 2 * p['b2']) * sk * h1m_response(f, p)))

    def fit(fix, ne, st):
        st = dict(st, k3=st.get('k3', 0.0)) if skew else st
        names = [x for x in ['a', 's', 'b2', 'eps', 'ts'] + (['k3'] if skew else []) + sum([[f'r{j}', f'T{j}'] for j in range(ne)], []) if x not in fix]
        lo = np.array([_H1M_LO.get(x, -0.3 if x[0] == 'r' else t_echo[0]) for x in names])
        hi = np.array([_H1M_HI.get(x, 0.3 if x[0] == 'r' else t_echo[1]) for x in names])

        def res(q):
            p = {**fix, **dict(zip(names, q))}
            return ((Y1 - model(p)) / np.sqrt(E1 ** 2 + (p['a'] * E2) ** 2))[m]
        r = least_squares(res, np.clip([st[x] for x in names], lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale='jac')
        nd = int(m.sum() - len(names))
        e = np.sqrt(np.diag(np.linalg.pinv(r.jac.T @ r.jac)) * max(2 * r.cost / max(nd, 1), 1))
        return {**fix, **dict(zip(names, r.x))}, dict(zip(names, e)), 2 * r.cost, nd

    fix = dict(tin=tau_in)
    if start is None:
        best = None
        for ts0 in (3.0, 10.0):
            c = fit(fix, 0, dict(a=Y1.max() / Y2.max(), s=-0.6, b2=0.2, eps=0.03, ts=ts0))
            best = c if best is None or c[2] < best[2] else best
        k0 = 0
    else:
        st = {x: v for x, v in start.items() if not (x[0] in 'rT' and x[1:].isdigit())}
        ech = sorted([(start[f'r{j}'], start[f'T{j}']) for j in range(sum(1 for x in start if x[0] == 'r' and x[1:].isdigit()))], key=lambda e_: -abs(e_[0]))
        ech = ech[:n_echo]
        st.update({**{f'r{j}': r_ for j, (r_, _) in enumerate(ech)}, **{f'T{j}': t_ for j, (_, t_) in enumerate(ech)}})
        best = fit(fix, len(ech), st)
        k0 = len(ech)
    for j in range(k0, n_echo):
        cand = None
        for T0 in np.arange(t_echo[0] + 0.1, t_echo[1], 0.32):
            for r0 in (-0.03, 0.03):
                fx = {x: v for x, v in best[0].items()}
                c = fit(fx, j + 1, {**fx, f'r{j}': r0, f'T{j}': T0})
                cand = c if cand is None or c[2] < cand[2] else cand
        best = fit(fix, j + 1, cand[0])
    p, e, chi, nd = best
    return dict(p=p, e=e, chi2=chi, ndof=nd, M=model(p), m=m)


def h1m_instrument(p):
    """The instrument part of a `fit_ad_h1m` p (τ_in, ε, τ_s, echoes) for `h1m_response` / `ad_kernels_configs(h1m=…)`: nuisances a, s, b2, k3 dropped."""
    return {x: v for x, v in p.items() if x not in ('a', 's', 'b2', 'k3')}


def ad_fit_fwd(d, kAD, n=AD_FWD_N, q0=None):
    """Forward fit of agg2 b convolved with R_AD to the AD groups: Y_i = A_i · rebin(b(t − x0_i) ⊗ R_AD,i), diagonal χ² (raw bins independent)
    over ±n σ_AD; b shared (σL, σR, β; βL = βR when `AD_B_TIE`, else βL, βR free), x0 / A per group; no floor / flat term; start values `PAR` (or `q0`)
    - returns dict(p, e, q, err (q / err: σL, σR, βL, βR, group pars), npar (free), chi2, ndof, grp [group, (x0, A)], M (model per group), wA)"""
    g = d['grid']
    Y, E = d['grp_Y'], d['grp_E']
    nbk = len(Y)
    wA = ad_window(d, n)
    m = (g.x >= wA[0]) & (g.x <= wA[1])
    FK = np.fft.fft(kAD, axis=1)
    shp = _ad_shp()
    nb_ = len(shp)

    def model(i, p, xa, aa):
        return aa * g.nb * g.rebin(np.real(np.fft.ifft(np.fft.fft(bunch_shape(g, xa, p, 'agg2')) * FK[i])))

    def resid(q):
        p, bp = b_from_fit(q), q[nb_:].reshape(nbk, 2)
        return np.concatenate([((Y[i] - model(i, p, *bp[i])) / E[i])[m] for i in range(nbk)])

    q0 = np.r_[[AD_PAR[s][0] for s in shp], np.ravel([[0.0, y.max() / (g.nb * 0.3)] for y in Y])] if q0 is None else _ad_q_in(q0)
    lo = np.r_[[AD_PAR[s][1] for s in shp], [PAR['xa'][1], 0.0] * nbk]
    hi = np.r_[[AD_PAR[s][2] for s in shp], [PAR['xa'][2], PAR['aa'][2]] * nbk]
    r = least_squares(resid, np.clip(q0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), x_scale='jac')
    try:
        err = np.sqrt(np.diag(np.linalg.inv(r.jac.T @ r.jac)))
    except np.linalg.LinAlgError:
        err = np.full(r.x.size, np.nan)
    p, bp = b_from_fit(r.x), r.x[nb_:].reshape(nbk, 2)
    q, err = _ad_q_out(r.x), _ad_q_out(err)
    return dict(p=p, e=dict(zip(B_PARS, err[:4])), q=q, err=err, npar=r.x.size, chi2=(r.fun ** 2).sum(), ndof=r.fun.size - r.x.size, grp=bp,
                M=np.array([model(i, p, *bp[i]) for i in range(nbk)]), wA=wA, kind='forward')


TERM_N_RT = 16                     # `ad_term_products`: source bunches per trace (last − 16 … last − 3) for the end-of-spill round trip
TERM_U = np.arange(-9.0, 9.0 + 1e-9, 0.04)   # round-trip profile grid [ns] around each bunch


def _term_trace(samples, trailer, p_nom=18.935):
    # one trace → (baseline PSD [counts²], V/count, S(u), R(u) per source bunch, P) or None (clipped / weak / short spill)
    v_per, off, dt = trailer
    dt_ns = dt * 1e9 if 1e-11 < dt < 1e-9 else BIN_NS
    if (samples >= at.ADC_MAX).any():
        return None
    y = samples.astype(float)
    nb = at.BASELINE_SAMPLES
    b = y[:nb]
    s = y - np.median(b)
    if s.max() < at.MIN_AMP_COUNTS:
        return None
    hw = np.hanning(nb)
    psd = np.abs(np.fft.rfft((b - b.mean()) * hw)) ** 2 / (hw ** 2).sum()
    t = np.arange(len(s)) * dt_ns
    w = np.clip(s, 0, None) ** 2
    phase = np.angle(np.sum(w * np.exp(2j * np.pi * t / p_nom))) * p_nom / (2 * np.pi)
    k = np.rint((t - phase) / p_nom).astype(int)
    full = (k > k.min()) & (k < k.max())
    k0 = k[full].min()
    A = np.bincount(k[full] - k0, weights=s[full])
    idx = np.where(A > 0.5 * np.median(A[A > 0.3 * A.max()]))[0]
    if len(idx) < 60:
        return None
    kk, tc = [], []
    for i in idx:
        mm = np.abs(t - (phase + (i + k0) * p_nom)) <= p_nom / 3
        ww = np.clip(s[mm], 0, None)
        if ww.sum() > 0:
            kk.append(i)
            tc.append((t[mm] * ww).sum() / ww.sum())
    P, tz = np.polyfit(kk, tc, 1)
    D = 21 * P - 1.5                                  # round trip (2 × `CABLE_DELAY`) lands 21 buckets later
    tj = tz + np.arange(idx[-1] - TERM_N_RT, idx[-1] - 2) * P
    tj = tj[(tj + D + TERM_U[-1]) <= t[-1] - 1]
    if len(tj) < 8:
        return None
    S = sum(np.interp(x + TERM_U, t, s) for x in tj) / len(tj)
    R = sum(np.interp(x + D + TERM_U, t, s) for x in tj) / len(tj)
    return psd, v_per, S, R, P


def ad_term_products(path):
    """b-free AD termination probes per AD config (`AD_CONFIGS`) from every stored raw trace (Run 1 + 2 stores, ~4 min, 0.5 GB) → `path`:
    - baseline noise PSD: first `at.BASELINE_SAMPLES` samples (before the spill), mean removed, Hann window, |FFT|² [counts²] + V/count per trace
    - end-of-spill round trip: bunch comb fitted per trace (bucket centroids → t0, P); S(u) = mean source bunch (last − `TERM_N_RT` … last − 3),
      R(u) = the same bunches 21 P − 1.5 ns later (after the end of the spill: no direct bunch there), u = `TERM_U` (linear interpolation)
    - S and R hold the same bunches → R / S = round-trip transfer of the scope input + splitter + Heliax + RWM end, independent of b"""
    acc = {}
    for store in (paths.STORE[1], paths.STORE[2]):
        for fn in sorted(glob.glob(f'{store}/run_*.npz')):
            z = np.load(fn)
            pns = ad_config(pd.to_datetime(z['clock_ms'].astype(np.int64), unit='ms'))      # record time
            for i in np.where(z['status'] == 1)[0]:
                r = _term_trace(z['samples'][i], z['trailer'][i]) if pns[i] else None
                pn = pns[i]
                if r is not None:
                    for name, v in zip(('psd', 'vper', 'S', 'R', 'P'), r):
                        acc.setdefault((pn, name), []).append(v)
    np.savez_compressed(path, U=TERM_U, f=np.fft.rfftfreq(at.BASELINE_SAMPLES, BIN_NS),
                        **{f'{pn}|{name}': np.array(v, np.float32 if name in ('psd', 'S', 'R') else float) for (pn, name), v in acc.items()})


def load_ad_term(path):
    z = np.load(path)
    out = {'U': z['U'], 'f': z['f']}
    for key in z.files:
        if '|' in key:
            pn, name = key.split('|')
            out.setdefault(pn, {})[name] = z[key].astype(float)
    return out


B_PARS = ('sl', 'sr', 'bl', 'br')     # true bucket shape parameters (`bucket_half`, `bucket_draw`): asym generalised Gaussian, β per side
