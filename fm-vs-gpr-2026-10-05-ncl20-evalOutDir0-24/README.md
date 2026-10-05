# FM vs. GPR — separate GPR training / evaluation samples (ncl ≥ 20)

Same comparison as [`../fm-vs-gpr-2026-10-02-ncl20`](../fm-vs-gpr-2026-10-02-ncl20/README.md)
(ensemble FM vs. GPR dE/dx, both with ≥ 20 clusters per track). What changed: the GPR
**training** and **evaluation** samples are now configured separately.

| | Sample | Seeds |
|---|---|---|
| GPR training (bands + priors) — unchanged | `calotrkana-1M-ncl.root` = `condor_pid_x10/OutDir0–999` | 4,783,606 |
| GPR evaluation (compared with FM) — **new** | `calotrkana-eval-OutDir0-24-ncl.root` = `condor_pid_x10/OutDir0–24` | 119,324 |
| FM | ensemble per-point CSV (unchanged) | — |

The evaluation file is built with the same skim as the training file
(`../fm-vs-gpr/make_ncl_root.py --list dedx-eval-OutDir0-24.lst`); it matches a
direct read of the 25 job files exactly. The same selection and cluster cut are
applied to both samples.

**Overlap:** OutDir0–24 is also part of the training sample (2.5% of it). That
was kept on purpose, since the training was left unchanged. The GPR model is a smooth
1-D band per species plus per-bin class fractions, so the effect is expected to be
negligible; the evaluation AUCs below agree with the full-sample ones to ≤ 0.006.

## Code change (`../fm-vs-gpr/compare_pid_roc.py`)

- `--root-file` = GPR training sample; new `--gpr-eval-file` = GPR evaluation sample
  (defaults to `--root-file`, i.e. the previous behavior).
- GPR is split into `load_gpr_sample` → `train_gpr` → `score_gpr`. The fitted
  bands/priors are cached as `gpr_model_<train>_<fit range>_ncl<N>.csv`, so a new
  evaluation sample is scored without the ~15-min refit. Scores for a different eval
  sample are cached as `gpr_scores_..._eval-<eval file>.csv`.

## Validation

1. **Train = eval** (`--root-file` and `--gpr-eval-file` both the 1M file, ncl ≥ 20,
   GPR refit from scratch): reproduces `../fm-vs-gpr-2026-10-02-ncl20` exactly —
   pixel-identical PNGs, identical AUCs and counts in all three bins, and the 3.8M
   per-track GPR scores are bit-identical.
2. **Model-cache path**: rescoring from the cached model is bit-identical to scoring
   right after the fit. (This required reading the cache with
   `float_precision="round_trip"`; pandas' default parser drops the last bit, which
   changed scores by ≤ 4e-13.)

## Results (AUC, ncl ≥ 20)

| bin [GeV/c] | species | GPR eval = 1M (prev.) | **GPR eval = OutDir0–24** | FM ensemble |
|---|---|---|---|---|
| [0.8, 1.2) | π | 0.854 | **0.853** | 0.821 |
|            | K | 0.701 | **0.695** | 0.749 |
|            | p | 0.998 | **0.999** | 0.999 |
| [1.0, 2.0) | π | 0.722 | **0.716** | 0.731 |
|            | K | 0.649 | **0.643** | 0.700 |
|            | p | 0.918 | **0.921** | 0.934 |
| [0.0, 1.0) | π | 0.650* | **0.650*** | 0.991 |
|            | K | 0.636* | **0.632*** | 0.988 |
|            | p | 0.909* | **0.909*** | 1.000 |

\* GPR band extrapolated below 0.5 GeV/c (fit range 0.5–2.0); not a fair low-p comparison.

GPR performance on the smaller evaluation sample matches the full sample (|ΔAUC| ≤ 0.006),
and its statistics are now comparable to FM (e.g. [0.8,1.2): 13,509 GPR vs 7,815 FM
tracks). The conclusions are unchanged: in [1,2) FM leads for all three species; in
[0.8,1.2) FM leads for kaons, ties for protons, and GPR leads for pions.

Caveat: the FM ensemble CSV and OutDir0–24 are not known to be the *same* events, only
statistically equivalent samples.

## Reproduce

```bash
source fm-vs-gpr/env.sh
D=/sphenix/user/hwyu/calotrack_tree/macro/dedx-v2
# (once) eval sample: OutDir0-24, same skim as the training file
$PY fm-vs-gpr/make_ncl_root.py --list $D/dedx-eval-OutDir0-24.lst \
    --output $D/calotrkana-eval-OutDir0-24-ncl.root
NEW=/sphenix/tg/tg01/commissioning/CaloCalibWG/sli/fm4npp_eval/ensemble/PID_ensemble_per_point_data.csv
for b in "0.8 1.2" "0.0 1.0" "1.0 2.0"; do set -- $b
  $PY fm-vs-gpr/compare_pid_roc.py --root-file calotrkana-1M-ncl.root \
      --gpr-eval-file calotrkana-eval-OutDir0-24-ncl.root --ncl-min 20 --fm-csv $NEW \
      --outdir fm-vs-gpr-2026-10-05-ncl20-evalOutDir0-24 --bin-lo $1 --bin-hi $2
done
```

## Files

- `pid_roc_comparison_{0.8_1.2,0.0_1.0,1.0_2.0}.{png,pdf}`, `auc_summary_*.csv`
- caches (CSV, not for git): `gpr_model_calotrkana-1M-ncl_0.5_2.0_ncl20.csv`,
  `gpr_scores_0.5_2.0_ncl20_eval-calotrkana-eval-OutDir0-24-ncl.csv`, `fm_track_scores.csv`
