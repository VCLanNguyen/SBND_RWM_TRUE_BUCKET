# SBND RWM True Bucket

True bucket b per segment from the raw RWM traces: the proton time distribution of one BNB bucket, before any detector response. The raw traces of the accelerator RWM scope (IFBeam device `E:NBMI12WV[]`) are cleaned of cable reflections, aligned into 4 bucket groups, and b is found by a forward fit through the AD hardware response R_AD. MC draws its bucket times from b.

Method and results: SBND timing tech note (SBN-doc-XXXXX, TODO), Sec. "True Bucket Extraction from Raw RWM Traces".

## Layout

```
true_bucket/
  fetch_traces.ipynb   optional: rebuild the raw-trace store from IFBeam (Fermilab network)
  trace_fits.ipynb     Step 1: fit every trace (81 buckets, positions, period, σ, QC)
  true_bucket.ipynb    Steps 2-6: reflections removed, 4 groups, R_AD, forward fit → b per segment
  errors.ipynb         Step 6 item 3: total errors of b + cross-segment covariance
  results.ipynb        results table and figure (tech note Tab. / Fig. "Extraction Results")
  figures.ipynb        tech note figures of Steps 1-6 and of the appendix "True Bucket Extraction per Segment"
lib/
  ad_trace.py          IFBeam queries, raw-trace store, Step 1 trace fit
  segments.py          time chunks and segments from the calibration CSV, traces per segment
  ad_response.py       R_AD (cable, scope, echoes, taps, 1 MΩ H), true-bucket shape, forward fit
  true_bucket.py       Steps 2-6 over the whole spill, errors, products on disk
  plot_helpers.py      figures: segment colours, style
  step_figures.py      tech note figures of the extraction steps (`figures.ipynb`)
  paths.py             input / output paths (`set_paths`), CSV writer
inputs/
  traces_run{1,2}.csv  every stored trace key (run, event, ts_ns, status); `fit_row` ≥ 0 = trace of the extraction, in fit order
  s2_tune_inputs.npz   frozen SBND / MC inputs of the s² tune (Step 6 item 2)
```

## Inputs

| Input | What | Set in |
|---|---|---|
| Raw-trace store, Run 1 / Run 2 | one `run_<DAQ run>.npz` per DAQ run: 12000 samples + trailer per IFBeam record | `STORE` in the Setup cell |
| Calibration CSV, Run 1 / Run 2 | `sbnd_fine_calibration.csv` from [SBND_RWM_CALIBRATION](https://github.com/VCLanNguyen/SBND_RWM_CALIBRATION): time chunks, periods, segments | `CALIB_CSV` in the Setup cell |
| `inputs/traces_run{1,2}.csv` | which traces the extraction fits, and in which order (the order sets the bootstrap) | shipped |
| `inputs/s2_tune_inputs.npz` | SBND AD-matched stack moments per segment, MC D_ν, AD-stat weights; only the s² retune of the error variants reads it | shipped |

- Any readable path works (local or Fermilab); the inputs are never written to (`fetch_traces` writes only missing keys into `STORE`)
- The only SBND inputs: the β range 1.8-2.4 and s² = 0.175 ns² (constants `AD_B_COND`, `AD_B_S2` in `lib/ad_response.py`), and `s2_tune_inputs.npz`

## Data on Fermilab

`/pnfs/sbnd/persistent/users/lnguyen/RWM_Calibration/`:

| File | What | md5 |
|---|---|---|
| `run1_rwm_traces.tar` | Run 1 raw-trace store (122 files, 17 MB); `tar -xf` → `run1_rwm_traces/` | `2d9e7ea3e666b0c9f032e629f15a4005` |
| `run2_rwm_traces.tar` | Run 2 raw-trace store (532 files, 88 MB); `tar -xf` → `run2_rwm_traces/` | `89a42ff9b8e71f43f2d966c4cd594a8a` |
| `run1_calibration.csv`, `run2_calibration.csv` | calibration CSVs (SBND_RWM_CALIBRATION README) | see there |

## Environment

Python 3.10: `pip install -r requirements.txt`

## Run

First set the paths at the top of each notebook's Setup cell ("Set these paths"):

| Notebook | Paths |
|---|---|
| `fetch_traces` | `STORE` |
| `trace_fits` | `STORE`, `OUT_DIR` |
| `true_bucket`, `errors` | `STORE`, `CALIB_CSV`, `OUT_DIR` |
| `results` | `CALIB_CSV`, `OUT_DIR` |
| `figures` | `STORE`, `CALIB_CSV`, `OUT_DIR` |

- `STORE` and `CALIB_CSV` are placeholders (`/path/to/…`) and must be set; `OUT_DIR` defaults to `<repo>/output` and must be the same in all notebooks
- Then:

```bash
cd true_bucket
for nb in trace_fits true_bucket errors results figures; do
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb
done
```

| Notebook | Time | Peak RAM | Writes |
|---|---|---|---|
| `fetch_traces` | not timed (~35 k IFBeam queries at ~40 / s for an empty store) | not measured | `STORE` (missing keys only) |
| `trace_fits` | ~3.5 min | 1.4 GB | `OUT_DIR/run{1,2}/trace_fits.csv`, `trace_buckets.npz` |
| `true_bucket` | ~10 min | 2.1 GB | `OUT_DIR/true_bucket_density.npz` + caches (`ad_term.npz`, `end_of_spill.npz`, `corrected_histograms.npz`, `true_bucket_fits.pkl`) |
| `errors` | ~45 min | 0.8 GB | `OUT_DIR/true_bucket.csv`, `true_bucket_cov.npz`, `errors_progress.log` |
| `results` | ~5 s | 0.3 GB | `OUT_DIR/figures/true_bucket_results.png` |
| `figures` | ~2 min | 2.3 GB | `OUT_DIR/figures/ad_*.png` (needs `trace_fits`, `true_bucket`) |

Run order as listed; each notebook reads the outputs of the ones before it.

## Outputs

- `true_bucket_density.npz`: per segment (`<run>|<seg>|tt`, `ff`, `P`): b on a fine grid (`tt` [ns] from the mode, `ff` normalised to 1), bunch period P [ns]
- `true_bucket.csv`: one row per segment:

| Column | Meaning |
|---|---|
| `run`, `seg`, `ad_config`, `head_first` | segment, its AD config (1.1, 1.3, 2.1, 2.2), first bucket of the head group (1-based: 6 = bucket k = 5) |
| `sl`, `sr`, `bl`, `br` | shape of b: σL, σR [ns], βL, βR (agg2 fitted to the mixture) |
| `rms`, `rms_left`, `rms_right` | RMS of b, and per side [ns] |
| `side_L`, `side_R`, `LmR` | side RMS about the mean [ns], 2 × left − 2 × right semi-variance [ns²] |
| `<value>_stat` | bootstrap error |
| `<value>_<group>` | error per extraction choice: `window`, `weights`, `taps`, `taps procedure`, `hardware`, `echo model`, `scope`, `alignment`, `H(2.1)`, `H skew`, `s2`, `reflection`, `ripple` (RMS only), `time` |
| `<value>_window_lo`, `_window_hi` | window error per side; `_window` = the larger |
| `<value>_tot` (`_tot_lo`, `_tot_hi`) | total error, all of the above in quadrature (window per side) |
| `time_flag`, `pins_flag`, `start_flag`, `s2_fail` | time scatter > statistics; variants with a b condition at a different bound; variants whose two fit starts disagree; variants whose s² retune failed |
| `s2`, `rms_ad`, `rms_agg2`, `rms_head` | s² [ns²]; RMS before the s² smear; RMS of the agg2 fit; RMS of the head-group b |
| `w_g<m>`, `<shape>_g<m>`, `chi2ndof_g<m>` | per group m = 0-3 (head, Groups 1-3): weight, b shape, forward-fit χ²/ndof |
- `true_bucket_cov.npz`: `keys` (run|seg), `<value>` = cross-segment covariance, `<value>|rho|<group>` = correlation per error group

## Changing the extraction

- Segments and time chunks: in SBND_RWM_CALIBRATION (then rerun from `true_bucket`); segment number → name: `lib/segments.py` `SEGMENT_NAMES`
- AD configs (dates, scope input): `AD_CONFIGS`, `AD_TERMINATION` in `lib/ad_response.py`
- Fit window, b conditions, s²: `AD_FWD_N`, `AD_B_COND`, `AD_B_S2` in `lib/ad_response.py`; error variants: `TB_*` constants in `lib/true_bucket.py`, Setup cell of `errors.ipynb`
