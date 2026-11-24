"""Tech note figures of the extraction steps (Sec. "Extraction Steps" and the appendix "True Bucket Extraction per Segment"), from the outputs of `true_bucket.ipynb`.

- Dict `I` (built in `figures.ipynb`): DATA (`segment_inputs`), TERM (`load_ad_term`), TAILS (`load_tails` + `yc` = corrected tail),
  RMOD (`reflection_models`), W / WB (per-bucket histograms before / after Step 2), CTX (`tb_context`, fitted), FIT (`tb_fits` per segment), PKEYS (segments per AD config)
- Every function draws one figure, saves it as `<out>/<name>` and prints the numbers quoted in the tech note
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import ad_trace as at
from ad_response import (SEG_NAMES, AD_TERMINATION, BIN_NS, N_BUCKETS, BL, ECHOES, Grid, scope_kernel, echo_kernel, h1m_response,
                         bunch_shape, bunch_rms)
from true_bucket import TB_END_X, round_trip_fir, group_mask
from plot_helpers import SEG_COLORS, FULL

TN_STYLE = {'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 7, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7}
FIT_COLOR = 'tab:orange'
XLIM = (-10, 10)
TN_NAMES = {**SEG_NAMES, 'during': 'Rotation'}     # segment names as in the tech note
title = lambda k: f'Run {k[0]} {TN_NAMES[k[1]]}'
OMEGA = '$\\Omega$'
cfg_label = lambda pn: f"AD config {pn} ({AD_TERMINATION[pn].replace('Ω', OMEGA)})"
GRP = [('Head', 'k')] + [(f'Group {m}', plt.cm.viridis(c)) for m, c in zip((1, 2, 3), (0.25, 0.55, 0.8))]
C_Y, C_SHORT, C_RT, C_FIT = '0.6', 'tab:red', 'tab:blue', 'k'
C_PART = {'cable': 'tab:brown', 'scope': 'tab:purple', 'echoes': 'tab:blue', 'taps': 'tab:red', 'H': 'tab:green'}
K_EX, RMS_WIN = 30, 6.0                       # Step 3: example bucket (Group 1), |r| window of the shape RMS


def finish(fig, out, name):
    for ax in fig.axes:
        ax.grid(alpha=0.3, lw=0.5)
    fig.savefig(f'{out}/{name}', dpi=200)
    plt.close(fig)
    print('saved', name)


def bucket_profiles(w, g):
    """Per trace × bucket mean of the per-bucket histograms `w` on `g.x` (NaN where under half the bins have samples)."""
    out = np.full((len(w['period']), N_BUCKETS, g.nb), np.nan)
    for i, P in enumerate(w['period']):
        edges = np.arange(-P / 2, P / 2 + BIN_NS, BIN_NS)
        nb = len(edges) - 1
        xc = (edges[:-1] + edges[1:]) / 2
        cnt = w['N'][i, :, :nb].astype(float)
        y = w['S'][i, :, :nb] / np.maximum(cnt, 1)
        for k in range(N_BUCKETS):
            ok = cnt[k] > 0
            if ok.sum() > nb // 2:
                out[i, k] = np.interp(g.x, xc[ok], y[k][ok])
    return out


def _fine_x(P):
    xs = np.arange(-P / 2 + 0.08, P / 2, 0.16)
    return xs, type('G', (), dict(x=xs, nb=len(xs)))


def bucket_shapes(w, P):
    """Mean bucket shape per bucket (peak 1) on a 0.16 ns grid → (x, shapes [bucket, x], head mean)."""
    xs, gx = _fine_x(P)
    sp = np.nanmean(bucket_profiles(w, gx) / w['amp_med'][:, None, None], 0)
    sp = sp / np.nanmax(sp, 1, keepdims=True)
    return xs, sp, np.nanmean(sp[group_mask(P, 0)], 0)


def shape_rms(w, P):
    """RMS over |r| < `RMS_WIN` of (bucket shape − head shape) per bucket."""
    xs, sp, ref = bucket_shapes(w, P)
    return np.sqrt(np.nanmean((sp - ref)[:, np.abs(xs) < RMS_WIN] ** 2, 1))


def tap_kernel(g, taus, c):
    e = np.zeros(g.n)
    e[0] = 1
    return e + c @ np.array([g.shift(t) for t in taus])


def lagsort(g, kk, lo=-2):
    lag = (np.arange(g.n) * g.h - lo) % g.P + lo
    o = np.argsort(lag)
    return lag[o], kk[o]


def response_parts(I, pn):
    """R_AD of AD config `pn` part by part on the grid of its first segment → (grid, taps set, base AD config, parts, cumulative products);
    checked against the kernel of the extraction (`CTX['KF']`)."""
    CTX = I['CTX']
    k = I['PKEYS'][pn][0]
    g = CTX['DSF'][k]['grid']
    base, ph = CTX['H1MF'].get(pn, (pn, None))
    r = CTX['RRF'][base]
    parts = dict(cable=g.cable(BL), scope=scope_kernel(g, r['scope']), echoes=echo_kernel(g, ECHOES), taps=tap_kernel(g, r['taus'], r['C'][base]))
    if ph is not None:
        parts['H'] = np.real(np.fft.ifft(h1m_response(np.fft.fftfreq(g.n, g.h), ph)))
    cum, acc = [], None
    for n, kk in parts.items():
        acc = kk if acc is None else g.conv(acc, kk)
        cum.append((n, acc))
    assert np.allclose(acc, CTX['KF'][k][0], atol=1e-9 * np.abs(acc).max()), pn
    return g, r, base, parts, cum


def fig_step1(rows, store_dir, out, name='ad_step1_trace_fit.png', color=SEG_COLORS['before_a']):
    """Step 1 on the median-amplitude trace of `rows` (trace fits of one segment), read from the store `store_dir` and refitted with `fit_trace`."""
    from ad_trace import gaussian
    rows = rows.sort_values('amp')
    r1 = rows.iloc[len(rows) // 2]
    z = np.load(at.store_path(store_dir, r1['run']))
    i1 = np.flatnonzero((z['event'] == r1['event']) & (z['ts_ns'] == r1['ts'].value))[0]
    raw, trl = z['samples'][i1], z['trailer'][i1]
    row, pb = at.fit_trace(raw, trl, 'sym')
    print(f"Step 1 trace: run {int(r1['run'])} event {int(r1['event'])} {r1['ts']}, sigma {row['sigma']:.4f} (fits {r1['sigma']:.4f}), P {row['period']:.4f}")
    t = np.arange(len(raw)) * trl[-1] * 1e9
    y = raw.astype(float) - np.median(raw[:at.BASELINE_SAMPLES])
    pos, perr, amp, sig = pb['peak_positions'], pb['peak_pos_err'], pb['bucket_amp'], pb['bucket_sigma']
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.04), constrained_layout=True, width_ratios=[1.6, 1, 1])
    ax = axes[0]
    m = (t > pos[0] - 12) & (t < pos[4] + 10)
    ax.plot(t[m], y[m], '.', ms=2, color='0.5', label='Samples')
    for k in range(5):
        ax.axvspan(pos[k] - at.BUCKET_WINDOW_NS, pos[k] + at.BUCKET_WINDOW_NS, color=color, alpha=0.06 if k % 2 else 0.12, lw=0)
        xx = np.linspace(pos[k] - 3 * sig[k], pos[k] + 3 * sig[k], 100)
        ax.plot(xx, gaussian(xx, 0.0, amp[k], pos[k], sig[k]), color=FIT_COLOR, lw=1.3, label='Fit within $\\pm 3\\sigma_k$' if k == 0 else None)
        ax.axvline(pos[k], color='k', ls=':', lw=0.8)
        ax.text(pos[k], amp[k] * 1.04, f'$k$ = {k}', ha='center', va='bottom')
    ax.set(xlabel='$t$ [ns]', ylabel='$Y$ [ADC counts]', ylim=(-15, amp[:5].max() * 1.75), title='Items 2-3: first 5 buckets')
    ax.legend(loc='upper center', ncol=2)
    ax = axes[1]
    ax.errorbar(np.arange(len(pos)), pb['period_resid'], perr, fmt='.', ms=3, color='k')
    ax.axhline(0, color='tab:red', ls='--', lw=0.8)
    ax.set(xlabel='Bucket $k$', ylabel='$x_{0,k}$ $-$ linear fit [ns]', title=f"Item 4: $P$ = {row['period']:.3f} ns")
    ax = axes[2]
    ox, oy, _ = at.stacked_profile(t, y, pos, row['period'], raw < at.ADC_MAX)
    ax.plot(ox, oy, '.', ms=2, color='0.5', label='All 81 buckets')
    xx = np.linspace(row['stacked_center'] - 3 * row['sigma'], row['stacked_center'] + 3 * row['sigma'], 200)
    ax.plot(xx, gaussian(xx, row['stacked_H'], row['stacked_amp'], row['stacked_center'], row['sigma']), color=FIT_COLOR, lw=1.3,
            label=f"Fit: $\\sigma$ = {row['sigma']:.3f} ns")
    ax.set(xlabel='$t - x_{0,k}$ [ns]', ylabel='$Y$ [ADC counts]', xlim=(-9.5, 9.5), ylim=(-10, oy.max() * 1.45), title='Item 5: stacking')
    ax.legend(loc='upper right')
    finish(fig, out, name)


def fig_step2(I, configs, examples, out, name, top_ylim=(-0.44, 0.19), b80=((1.5, 0.17), (5, 0.14)), short_txt=(45, 0.09), rt_txt=(45, -0.30),
              mid_ylim=(-0.36, 0.2)):
    """Step 2, one column per AD config: both reflections around the spill end, round trip and its FIR fit h, mean trace after bucket 80
    before / after the correction, stacked buckets of the example segment before / after; label settings: one value, or a list with one per column."""
    cols = lambda v, col: v[col] if isinstance(v, list) else v
    TERM, TAILS, RMOD, W, WB = I['TERM'], I['TAILS'], I['RMOD'], I['W'], I['WB']
    fig, axes = plt.subplots(4, 2, figsize=(FULL, 7.94), constrained_layout=True)
    for col, pn in enumerate(configs):
        top_ylim_, b80_, short_txt_, rt_txt_, mid_ylim_ = (cols(v, col) for v in (top_ylim, b80, short_txt, rt_txt, mid_ylim))
        t_, sh = TAILS[pn], RMOD[pn]['short']
        P, T, rho = t_['P'], sh['T'], sh['rho']
        short = rho * np.interp(TB_END_X - T, TB_END_X, t_['yc'], left=np.nan)
        rt = t_['y'] - t_['yc'] - short
        ax = axes[0, col]
        m = TB_END_X >= -20
        ax.plot(TB_END_X[m], t_['y'][m], color=C_Y, lw=1, label='$Y$')
        ax.plot(TB_END_X[m], short[m], color=C_SHORT, lw=1.3, label='Short reflection')
        ax.plot(TB_END_X[m], rt[m], color=C_RT, lw=1.3, label='Round trip')
        ax.set(xlim=(-20, 100), ylim=top_ylim_, xlabel='$t - x_{0,80}$ [ns]', title=f'{cfg_label(pn)}\nBoth reflections around the spill end')
        ax.annotate('Bucket 80', xy=b80_[0], xytext=b80_[1], fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.8))
        i79, i80 = np.argmin(np.abs(TB_END_X - (T - P))), np.argmin(np.abs(TB_END_X - T))
        kw = dict(arrowstyle='->', lw=0.8, color=C_SHORT)
        ax.annotate('Short reflections\nof buckets 79, 80', xy=(T, short[i80]), xytext=short_txt_, fontsize=7, color=C_SHORT, va='center', arrowprops=kw)
        ax.annotate('', xy=(T - P, short[i79]), xytext=short_txt_, arrowprops=kw)
        w = (TB_END_X > P - 1.5) & (TB_END_X < 2 * P - 1.5)
        j = np.flatnonzero(w)[np.argmin(rt[w])]
        ax.annotate('Round trips of\nbuckets 60, 61, ...', xy=(TB_END_X[j], rt[j]), xytext=rt_txt_, fontsize=7, color=C_RT, va='center',
                    arrowprops=dict(arrowstyle='->', lw=0.8, color=C_RT))
        h, Rm = round_trip_fir(TERM, pn)
        S_, R_ = TERM[pn]['S'].mean(0), TERM[pn]['R'].mean(0)
        ax = axes[1, col]
        ax.plot(TERM['U'], R_ / S_.max(), color=C_RT, lw=1.3, label=f"Mean of {len(TERM[pn]['R'])} traces")
        ax.plot(TERM['U'], Rm / S_.max(), color=FIT_COLOR, ls='--', lw=1.0, label='FIR fit $h$')
        ax.set(xlabel='$t - x_{0,k} - (21P - 1.5$ ns$)$ [ns]', title='Round trip of bucket $k$ and its fit')
        print(f'Step 2 AD config {pn}: T {T:.2f} ns, rho {rho:+.3f}, round trip min / bucket peak {np.min(R_) / S_.max():+.4f}')
        ax = axes[2, col]
        ax.plot(TB_END_X, t_['y'], color=C_Y, lw=1, label='$Y$')
        ax.plot(TB_END_X, t_['yc'], color=C_FIT, lw=1.2, label='$D$ (corrected)')
        ax.axvline(T, color=C_SHORT, ls=':', lw=1.2, label=f"Short reflection: $T$ = {T:.2f} ns, $\\rho$ = {rho:+.3f}")
        ax.set(xlim=(-20, 100), ylim=mid_ylim_, xlabel='$t - x_{0,80}$ [ns]', title='After bucket 80')
        k = examples[col]
        G_ = Grid(float(np.median(W[k]['period'])))
        ax = axes[3, col]
        for w_, c_, lab in ((W[k], C_Y, 'Before: $Y$ centred on $x_{0,k}$ fitted on $Y$'), (WB[k], C_FIT, 'After: $D$ centred on $x_{0,k}$ refitted on $D$')):
            y = np.nanmean(bucket_profiles(w_, G_) / w_['amp_med'][:, None, None], (0, 1))
            ax.plot(G_.x, y, color=c_, lw=1.0 if c_ == C_FIT else 1.6, label=lab)
        ax.set(xlim=(-9.5, 9.5), ylim=(-0.05, 1.6), xlabel='$r = t - x_{0,k}$ [ns]', title=f'{title(k)}: stacked buckets')
        for ax, loc in zip(axes[:, col], ('lower left', 'lower left', 'lower right', 'upper left')):
            ax.axhline(0, color='0.5', lw=0.6)
            ax.legend(loc=loc)
    for r_ in range(4):
        axes[r_, 0].set_ylabel('Amplitude [a.u.]')
    finish(fig, out, name)


def fig_step3(I, configs, examples, out, name):
    """Step 3 items 5-6, one column per AD config: shape RMS of bucket `K_EX` vs the head (example segment), shape RMS of each bucket
    (all segments of the AD config, before / after Step 2), the 4 stacked groups D_m of the example segment."""
    DATA, W, WB, CTX = I['DATA'], I['W'], I['WB'], I['CTX']
    fig = plt.figure(figsize=(FULL, 8.75), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[1.9, 0.9, 2.1, 1.9, 0.9])
    for col, (pn, k) in enumerate(zip(configs, examples)):
        P = DATA[k]['P']
        xs, sp, ref = bucket_shapes(WB[k], P)
        win = np.abs(xs) < RMS_WIN
        rms = np.sqrt(np.nanmean((sp[K_EX] - ref)[win] ** 2))
        print(f'Step 3 {title(k)}: shape RMS of bucket {K_EX} {rms:.3f}')
        ax = fig.add_subplot(gs[0, col])
        ax.plot(xs, ref, color='k', lw=1.0, label='Head (mean of buckets 5–19)')
        ax.plot(xs, sp[K_EX], color=SEG_COLORS[k[1]], lw=1.0, label=f'Bucket {K_EX}')
        ax.set(xlim=(-9.5, 9.5), ylim=(-0.05, 1.35), ylabel='Amplitude [a.u.]' if col == 0 else None,
               title=f'{cfg_label(pn)}\n{title(k)}: shape RMS of bucket {K_EX}')
        ax.tick_params(labelbottom=False)
        ax.legend(loc='upper left')
        ax2 = fig.add_subplot(gs[1, col], sharex=ax)
        ax2.axvspan(-RMS_WIN, RMS_WIN, color='0.9', lw=0)
        ax2.plot(xs, sp[K_EX] - ref, color=SEG_COLORS[k[1]], lw=1.0, label=f'RMS over $|r| < {RMS_WIN:.0f}$ ns: {rms:.3f}')
        ax2.axhline(0, color='0.5', lw=0.6)
        ax2.set(ylim=(-0.12, 0.18), xlabel='$r = t - x_{0,k}$ [ns]', ylabel='Bucket $-$ head' if col == 0 else None)
        ax2.legend(loc='upper left')
        ax = fig.add_subplot(gs[2, col])
        for kk in I['PKEYS'][pn]:
            ax.plot(np.arange(N_BUCKETS), shape_rms(W[kk], P), color=SEG_COLORS[kk[1]], ls='--', lw=0.8)
            ax.plot(np.arange(N_BUCKETS), shape_rms(WB[kk], P), color=SEG_COLORS[kk[1]], lw=1.2, label=title(kk))
        for b_ in (20.5, 41.5, 62.5):
            ax.axvline(b_, color='0.4', ls=':', lw=1)
        for m, (lab, _) in enumerate(GRP):
            kr = np.flatnonzero(group_mask(P, m))
            ax.text((kr[0] + kr[-1]) / 2, 0.142, f'{lab}\n{kr[0]}–{kr[-1]}', ha='center', va='center', fontsize=7)
        hs, ls_ = ax.get_legend_handles_labels()
        hs += [Line2D([], [], color='0.4', ls='--', lw=0.8), Line2D([], [], color='0.4', lw=1.2)]
        ls_ += ['Before Step 2', 'After Step 2']
        ax.legend(hs, ls_, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        ax.set(ylim=(0, 0.16), xlabel='Bucket $k$', ylabel='Bucket shape RMS vs head' if col == 0 else None, title='Shape RMS of each bucket')
        d = CTX['DSF'][k]
        Yn = d['grp_Y'] / np.nanmax(d['grp_Y'], 1, keepdims=True)
        ax = fig.add_subplot(gs[3, col])
        ax2 = fig.add_subplot(gs[4, col], sharex=ax)
        for m, (lab, c_) in enumerate(GRP):
            kr = np.flatnonzero(group_mask(P, m))
            ax.plot(d['grid'].x, Yn[m], color=c_, lw=1.0, label=f'{lab} ({kr[0]}–{kr[-1]})')
            if m:
                ax2.plot(d['grid'].x, Yn[m] - Yn[0], color=c_, lw=1.0)
        ax2.axhline(0, color='0.5', lw=0.6)
        ax.set(xlim=(-9.5, 9.5), ylim=(-0.05, 1.75), ylabel='Amplitude [a.u.]' if col == 0 else None, title=f'{title(k)}: stacked groups $D_m$')
        ax.tick_params(labelbottom=False)
        ax.legend(loc='upper left')
        ax2.set(ylim=(-0.12, 0.18), xlabel='$r = t - x_{0,k}$ [ns]', ylabel='Group $-$ head' if col == 0 else None)
    finish(fig, out, name)


def _part_ax(fig, pos, ttl):
    ax = fig.add_subplot(pos)
    ax.axhline(0, color='0.5', lw=0.6)
    ax.set(xlabel='Lag [ns]', title=ttl)
    return ax


def fig_step4_parts(I, out, name='ad_step4_parts.png', c50='1.1', c1m='2.1'):
    """Step 4 items 1-5: cable, scope, echoes, taps (AD config `c50`) and H (AD config `c1m`)."""
    B50, B1m = response_parts(I, c50), response_parts(I, c1m)
    print('taps shared (50 Ω):', np.allclose(B50[1]['C'][c50], B1m[1]['C'][B1m[2]]))
    fig = plt.figure(figsize=(FULL, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    g, r, base, parts, _ = B50
    ax = _part_ax(fig, gs[0, 0:2], 'Item 1: cable')
    x, y = lagsort(g, parts['cable'], -1)
    ax.plot(x, y / y.max(), color=C_PART['cable'], lw=1.0)
    ax.set(xlim=(-0.05, 0.5), ylim=(-0.1, 1.1), ylabel='Amplitude [a.u.]')
    ax = _part_ax(fig, gs[0, 2:4], 'Item 2: scope')
    x, y = lagsort(g, parts['scope'], -1)
    ax.plot(x, y / y.max(), color=C_PART['scope'], lw=1.0)
    ax.set(xlim=(-0.5, 6), ylim=(-0.35, 1.1), ylabel='Amplitude [a.u.]')
    ax = _part_ax(fig, gs[0, 4:6], 'Item 3: echoes')
    ax.vlines(ECHOES[1::2], 0, ECHOES[0::2], color=C_PART['echoes'], lw=1.5)
    ax.plot(ECHOES[1::2], ECHOES[0::2], 'o', ms=3, color=C_PART['echoes'])
    ax.set(xlim=(0, g.P), ylim=(-0.045, 0.075), ylabel='Amplitude / direct pulse')
    ax = _part_ax(fig, gs[1, 1:3], 'Item 4: taps')
    ax.plot(r['taus'], r['C'][base], color=C_PART['taps'], lw=1.0)
    ax.set(xlim=(0, g.P), ylim=(-0.045, 0.075), ylabel='Amplitude / direct pulse')
    x, y = lagsort(B1m[0], B1m[3]['H'], -1)
    ax = _part_ax(fig, gs[1, 3:5], f'Item 5: $H$ (AD config {c1m})')
    ax.plot(x, y / y.max(), color=C_PART['H'], lw=1.0)
    ax.set(xlim=(-0.5, 17), ylim=(-0.045, 0.075), ylabel='Amplitude [a.u.] (clipped)')
    finish(fig, out, name)


LAB = {'cable': '$\\otimes$ cable', 'scope': '$\\otimes$ scope', 'echoes': '$\\otimes$ echoes', 'taps': '$\\otimes$ ($\\delta$ + taps)', 'H': '$\\otimes$ $H$'}
LO, HI, OFF = -0.35, 1.1, 1.4


def fig_step4_buildup(I, configs, examples, out, name):
    """Step 4: R_AD of each AD config built up part by part (colour = part added last) and b of the example segment ⊗ it (grey); peak 1, offset."""
    B = {pn: response_parts(I, pn) for pn in configs}
    YR = {pn: (LO, len(B[pn][4]) * OFF + HI + 0.05) for pn in configs}
    fig = plt.figure(figsize=(FULL, 7.6))
    top, bot = 0.95, 0.06
    span_max = max(b_ - a_ for a_, b_ in YR.values())

    def periodic(g, kk):
        x, y = lagsort(g, kk, -6)
        return np.r_[x, x + g.P], np.r_[y, y] / y.max()
    for col, (pn, kx) in enumerate(zip(configs, examples)):
        g, r, base, parts, cum = B[pn]
        mx = I['FIT'][kx]['mix']
        bb = np.interp(np.arange(g.n) * g.h, mx['tt'] - mx['tt'][np.argmax(mx['ff'])], mx['ff'], period=len(mx['tt']) * (mx['tt'][1] - mx['tt'][0]))
        h = (top - bot) * (YR[pn][1] - YR[pn][0]) / span_max
        ax = fig.add_axes([0.06 + col * 0.48, top - h, 0.44, h])
        nc = len(cum)
        for i_, (n, kk) in enumerate([('b', None)] + cum):
            o = (nc - i_) * OFF
            ax.axhline(o, color='0.8', lw=0.5)
            ax.plot(*(a_ + d_ for a_, d_ in zip(periodic(g, bb if kk is None else g.conv(bb, kk)), (0, o))), color='0.6', lw=1.2)
            if kk is None:
                ax.text(13.8, o + 0.3, 'True bucket $b$', color='0.4', ha='right', va='bottom', fontsize=7)
                continue
            x, y = periodic(g, kk)
            ax.plot(x, y + o, color=C_PART[n], lw=1.0)
            ax.text(13.8, o + 0.3, LAB[n] + (' = $R_\\mathrm{AD}$' if i_ == nc else ''), color=C_PART[n], ha='right', va='bottom', fontsize=7,
                    bbox=dict(fc='w', ec='none', pad=0.5))
        ax.set(xlim=(-6, 14), ylim=YR[pn], yticks=[], xlabel='Lag [ns]', title=f'{cfg_label(pn)}: {title(kx)}',
               ylabel='Amplitude [a.u.] (offset)' if col == 0 else None)
    finish(fig, out, name)


def summed_groups(I, k):
    """4 stacked groups D_m and models M_m of segment k summed with the Step 3 weights, each on its own x0 → (x, D, M, error)."""
    d, f, wt = I['CTX']['DSF'][k], I['FIT'][k], I['CTX']['WTG'][k]
    x = d['grid'].x
    Y, M, E2 = 0.0, 0.0, 0.0
    for m in range(len(wt)):
        x0, a = f['fg'][m]['grp'][0]
        c = wt[m] / a
        Y = Y + c * np.interp(x, x - x0, d['grp_Y'][m])
        M = M + c * np.interp(x, x - x0, f['fg'][m]['M'][0])
        E2 = E2 + (c * np.interp(x, x - x0, d['grp_E'][m])) ** 2
    return x, Y, M, np.sqrt(E2)


def fig_step5(I, keys, out, name):
    """Step 5 item 2: forward fit per segment (summed groups), ratio and pull; fit window shaded."""
    nb = (len(keys) + 1) // 2
    fig, axes = plt.subplots(3 * nb, 2, figsize=(FULL, 3.6 * nb), height_ratios=[2, 1, 1] * nb, sharex=True, constrained_layout=True)
    for j_, k in enumerate(keys):
        r0, j = 3 * (j_ // 2), j_ % 2
        f = I['FIT'][k]
        x, Y, M, E = summed_groups(I, k)
        pk = M.max()
        lo = max(g['wA'][0] - g['grp'][0][0] for g in f['fg'])
        hi = min(g['wA'][1] - g['grp'][0][0] for g in f['fg'])
        chi, ndof = sum(g['chi2'] for g in f['fg']), sum(g['ndof'] for g in f['fg'])
        inw = (x >= lo) & (x <= hi) & (M > 0.1 * pk)
        print(f'Step 5 {title(k)}: chi2/ndof {chi:.0f}/{ndof:.0f}, stacked/model in window (model > 10 % of peak) '
              f'{np.min(Y[inw] / M[inw]):.3f}-{np.max(Y[inw] / M[inw]):.3f}, median |pull| in window {np.median(np.abs((Y - M) / E)[inw]):.1f}')
        ax = axes[r0, j]
        ax.errorbar(x, Y / pk, E / pk, fmt='.', ms=2, color='k', label='Stacked groups $D_m$')
        ax.plot(x, M / pk, color=FIT_COLOR, lw=1.2, label=f'Model $M_m$: $\\chi^2$/ndof = {chi:.0f}/{ndof:.0f}')
        ax.set(ylim=(-0.1, 1.6), title=title(k))
        ax.legend(loc='upper left')
        axes[r0 + 1, j].errorbar(x, Y / M, E / np.abs(M), fmt='.', ms=2, color='k')
        axes[r0 + 1, j].axhline(1, color='0.5', lw=0.6)
        axes[r0 + 1, j].set_ylim(0.75, 1.25)
        axes[r0 + 2, j].plot(x, (Y - M) / E, '.', ms=2, color='k')
        axes[r0 + 2, j].axhline(0, color='0.5', lw=0.6)
        axes[r0 + 2, j].set(ylim=(-25, 25), xlabel='$r - x_{0,m}$ [ns]')
        for ax in axes[r0:r0 + 3, j]:
            ax.axvspan(lo, hi, color='0.5', alpha=0.12)
            ax.set_xlim(*XLIM)
    for r0 in range(0, 3 * nb, 3):
        axes[r0, 0].set_ylabel('Amplitude [a.u.]')
        axes[r0 + 1, 0].set_ylabel('Data / model')
        axes[r0 + 2, 0].set_ylabel('Pull')
    finish(fig, out, name)


def fig_step5_illustration(I, k, out, name='ad_step5_illustration.png', m0=0):
    """Step 5 item 3: group m0 of segment k: b_m ⊗ R_AD = M_m, fitted to D_m."""
    CTX = I['CTX']
    d, g = CTX['DSF'][k], CTX['DSF'][k]['grid']
    ff = I['FIT'][k]['fg'][m0]
    x0f = ff['grp'][0][0]
    bm = bunch_shape(g, 0.0, ff['p'], 'agg2')
    lagR, R = lagsort(g, CTX['KF'][k][m0], -6)
    bR = g.conv(bm, CTX['KF'][k][m0])
    pk = lambda y: y / np.nanmax(y)
    fig = plt.figure(figsize=(FULL, 1.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 0.18, 1, 0.18, 1, 0.18, 1])
    PAN = [('$b_m$', 'b'), ('$R_\\mathrm{AD}$', 'R'), ('$M_m = b_m \\otimes R_\\mathrm{AD}$', 'M'), ('Fit to $D_m$', 'F')]
    for c, (tl, key) in enumerate(PAN):
        ax = fig.add_subplot(gs[0, 2 * c])
        ax.axhline(0, color='0.5', lw=0.6)
        if key == 'b':
            ax.plot(g.tf, pk(bm), color=SEG_COLORS[k[1]], lw=1.2)
        elif key == 'R':
            ax.plot(lagR, pk(R), color='k', lw=1.0)
        elif key == 'M':
            ax.plot(g.tf, pk(bR), color=FIT_COLOR, lw=1.2)
        else:
            ax.axvspan(ff['wA'][0] - x0f, ff['wA'][1] - x0f, color='0.5', alpha=0.12, lw=0)
            s_ = np.nanmax(ff['M'][0])
            ax.plot(g.x - x0f, d['grp_Y'][m0] / s_, '.', ms=2, color='k')
            ax.plot(g.x - x0f, ff['M'][0] / s_, color=FIT_COLOR, lw=1.0)
        ax.set(xlim=(-6, 9), ylim=(-0.25, 1.25), yticks=[0, 1], title=tl, xlabel='Lag [ns]' if key == 'R' else '$r - x_{0,m}$ [ns]')
        if c == 0:
            ax.set_ylabel('Amplitude [a.u.]')
        else:
            ax.tick_params(labelleft=False)
        ax.grid(alpha=0.3, lw=0.5)
    for c, op in enumerate(['$\\otimes$', '$=$', '$\\rightarrow$']):
        ax = fig.add_subplot(gs[0, 2 * c + 1])
        ax.axis('off')
        ax.text(0.5, 0.5, op, ha='center', va='center', fontsize=12, transform=ax.transAxes)
    fig.savefig(f'{out}/{name}', dpi=200)
    plt.close(fig)
    print('saved', name)


def fig_step6(I, keys, out, name):
    """Step 6: b per segment and the 4 weighted group b_m it is made of, on t − mode of b; s² > 0: grey = sum before the G(s²) width."""
    nr = (len(keys) + 1) // 2
    fig, axes = plt.subplots(nr, 2, figsize=(FULL, 2.4 * nr), constrained_layout=True, squeeze=False)
    for j, k in enumerate(keys):
        f, wt = I['FIT'][k], np.asarray(I['CTX']['WTG'][k], float)
        G = Grid(I['CTX']['DSF'][k]['grid'].P)
        x0s = np.array([fg['grp'][0][0] for fg in f['fg']])
        xb = (wt * x0s).sum() / wt.sum()
        comp = [w * (lambda q: q / q.sum())(bunch_shape(G, x0 - xb, fg['p'], 'agg2')) for w, x0, fg in zip(wt, x0s, f['fg'])]
        xm = G.tf - G.tf[np.argmax(f['mix_ad']['ff'])]
        fw = f['mix']
        pk_ = fw['ff'].max()
        ax = axes.flat[j]
        for m, ((lab, c_), y) in enumerate(zip(GRP, comp)):
            ax.plot(xm, y / pk_, color=c_, lw=1.0, label=f'{lab}: $w_m$ = {wt[m]:.2f}')
        if f['s2']:
            ax.plot(xm, f['mix_ad']['ff'] / pk_, color='0.6', lw=1.2, label='Sum before 1 M$\\Omega$ width')
        ax.plot(fw['tt'], fw['ff'] / pk_, color=SEG_COLORS[k[1]], lw=1.6, label=f"$b$: RMS {fw['rms']:.3f} ns")
        print(f"Step 6 {title(k)}: w = {np.round(wt, 2)}, group RMS = {[round(float(bunch_rms(fg['p'])), 3) for fg in f['fg']]}, "
              f"b RMS {fw['rms']:.3f}, s2 {f['s2']}")
        ax.set(ylim=(0, 1.25), xlim=(-5, 12), title=title(k), xlabel='$t$ $-$ mode of $b$ [ns]')
        ax.legend(loc='upper right')
    for r_ in range(nr):
        axes[r_, 0].set_ylabel('Amplitude [a.u.]')
    finish(fig, out, name)
