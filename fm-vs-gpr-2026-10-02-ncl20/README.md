# FM vs. GPR — ensemble FM with a cluster-count cut (ncl ≥ 20)

Same comparison as [`../fm-vs-gpr-2026-10-02`](../fm-vs-gpr-2026-10-02/README.md)
(ensemble FM vs. GPR dE/dx), with one change: **both methods keep only tracks
with at least 20 clusters**. This addresses the concern that GPR only classifies
tracks the traditional reconstruction found, which may be easier, longer tracks.

## Cut definition

| Method | Cluster count used | Cut applied to |
|---|---|---|
| GPR | `tpc_seeds_nclusters` (clusters on the reconstructed TPC seed) | band fit, priors, and evaluation |
| FM  | number of truth clusters (points) of the truth track in the per-point CSV | track population before the ROC |

The FM definition matches the "at least 20 truth clusters" requirement used for
track finding in the paper.

The original combined GPR file (`calotrkana-1M.root`) didn't keep the cluster
count, so `../fm-vs-gpr/make_ncl_root.py` re-skims the same 1000 source files
into `calotrkana-1M-ncl.root`, adding `tpc_seeds_nclusters`. The three existing
branches are identical to the original (same 4,783,606 seeds; dedx including its
537 NaNs).

## Effect of the cut

| | GPR seeds removed | FM tracks removed |
|---|---|---|
| p ∈ [0.8, 1.2) | 1.1% (540,811 → 534,671) | 21% (9,848 → 7,815) |
| all momenta    | —                        | 16% (54,373 → 45,595) |

Reconstructed seeds almost always have ≥ 20 clusters already, so the cut mostly
removes short truth tracks from the FM sample.

## Results (AUC)

| bin [GeV/c] | species | GPR (no cut → ncl≥20) | FM ensemble (no cut → ncl≥20) |
|---|---|---|---|
| [0.8, 1.2) | π | 0.855 → 0.854 | 0.808 → **0.821** |
|            | K | 0.704 → 0.701 | 0.735 → **0.749** |
|            | p | 0.997 → 0.998 | 0.997 → **0.999** |
| [1.0, 2.0) | π | 0.723 → 0.722 | 0.704 → **0.731** |
|            | K | 0.651 → 0.649 | 0.678 → **0.700** |
|            | p | 0.916 → 0.918 | 0.900 → **0.934** |
| [0.0, 1.0) | π | 0.646 → 0.650* | 0.988 → **0.991** |
|            | K | 0.633 → 0.636* | 0.984 → **0.988** |
|            | p | 0.898 → 0.909* | 1.000 → **1.000** |

\* GPR band extrapolated below 0.5 GeV/c (fit range 0.5–2.0); not a fair low-p comparison.

With the cut, GPR is essentially unchanged while FM improves everywhere. In
[1,2) the FM now beats GPR for all three species; in [0.8,1.2) the FM leads for
kaons and is tied for protons, while GPR still leads for pions (0.854 vs 0.821).

## Reproduce

```bash
source fm-vs-gpr/env.sh
NEW=/sphenix/tg/tg01/commissioning/CaloCalibWG/sli/fm4npp_eval/ensemble/PID_ensemble_per_point_data.csv
# (once) $PY fm-vs-gpr/make_ncl_root.py   # builds calotrkana-1M-ncl.root
for b in "0.8 1.2" "0.0 1.0" "1.0 2.0"; do set -- $b
  $PY fm-vs-gpr/compare_pid_roc.py --ncl-min 20 --fm-csv $NEW \
      --outdir fm-vs-gpr-2026-10-02-ncl20 --bin-lo $1 --bin-hi $2
done
```

`--ncl-min 20` is now the script default; `--ncl-min 0` disables the cut.

## Validation

- With `--ncl-min 0`, the modified script on the new ROOT file reproduces
  `../fm-vs-gpr-2026-10-02` exactly (pixel-identical PNGs and identical AUCs and
  counts in all three bins), so the code change only adds the cut.
- The underlying workflow was validated earlier by rebuilding
  `../fm-vs-gpr/pid_roc_comparison_1.0_2.0.png` from scratch (pixel-identical).

## Files

- `pid_roc_comparison_{0.8_1.2,0.0_1.0,1.0_2.0}.{png,pdf}`, `auc_summary_*.csv`
- `fm_track_scores.csv` (now includes `n_clusters`), `gpr_scores_0.5_2.0_ncl20.csv` — caches (large; not for git)
