# FM vs. GPR — updated ensemble FM (2026-10-02)

Same workflow as [`../fm-vs-gpr`](../fm-vs-gpr/README.md) (unchanged
`fm-vs-gpr/compare_pid_roc.py`), with the FM input replaced by the ensemble PID
results:

```
/sphenix/tg/tg01/commissioning/CaloCalibWG/sli/fm4npp_eval/ensemble/PID_ensemble_per_point_data.csv
```

The GPR side is identical to `../fm-vs-gpr` (bands + priors fit over 0.5–2.0 GeV/c).

## Reproduce

```bash
source fm-vs-gpr/env.sh
NEW=/sphenix/tg/tg01/commissioning/CaloCalibWG/sli/fm4npp_eval/ensemble/PID_ensemble_per_point_data.csv
for b in "0.8 1.2" "0.0 1.0" "1.0 2.0"; do set -- $b
  $PY fm-vs-gpr/compare_pid_roc.py --fm-csv $NEW --outdir fm-vs-gpr-2026-10-02 --bin-lo $1 --bin-hi $2
done
```

Add `--force` to the first call to rebuild the GPR and FM caches from scratch.

## Workflow validation

Before running on the new input, the full workflow was rerun from scratch
(`--force`: GPR refit + FM aggregation of the old CSV) in a temporary directory and
compared with `../fm-vs-gpr/pid_roc_comparison_1.0_2.0.png`:

- PNG pixel-identical; all AUCs and track counts equal (|ΔAUC| ≤ 2e-16)
- FM track scores bit-identical; GPR scores equal to within 2e-8 (floating-point noise in the threaded GP fit)

The GPR cache from that validated rerun is what's used here.

## Results (AUC)

| bin [GeV/c] | species | GPR | FM old | **FM ensemble** |
|---|---|---|---|---|
| [0.8, 1.2) | π | 0.855 | 0.795 | **0.808** |
|            | K | 0.704 | 0.690 | **0.735** |
|            | p | 0.997 | 0.996 | **0.997** |
| [0.0, 1.0) | π | 0.646* | 0.985 | **0.988** |
|            | K | 0.633* | 0.976 | **0.984** |
|            | p | 0.898* | 1.000 | **1.000** |
| [1.0, 2.0) | π | 0.723 | 0.667 | **0.704** |
|            | K | 0.651 | 0.632 | **0.678** |
|            | p | 0.916 | 0.884 | **0.900** |

\* GPR band extrapolated below 0.5 GeV/c (fit range 0.5–2.0); not a fair low-p comparison.

The ensemble FM improves on the old FM in every bin and species. It now beats
GPR for kaons in [0.8,1.2) and [1,2), and is within ~0.02 of GPR for pions and
protons in [1,2).

## Note on statistics

The ensemble CSV covers 6,943 events (5.79M points) vs 7,625 events (6.34M
points) in the old FM file, so there are ~9% fewer FM tracks (54,373 vs 59,762
π/K/p tracks); the species fractions are unchanged.

## Files

- `pid_roc_comparison_{0.8_1.2,0.0_1.0,1.0_2.0}.{png,pdf}` — plots
- `auc_summary_*.csv` — AUC tables
- `fm_track_scores.csv`, `gpr_scores_0.5_2.0.csv` — caches (large; not for git)
