# FM vs. GPR on the same reconstructed seeds (ncl ≥ 20)

Both methods are evaluated on the **identical list of reconstructed TPC seeds**,
with the same truth label (`|tpc_seeds_maxparticle_pid|`) and truth momentum
(`tpc_seeds_maxparticle_p`). Previous comparisons used GPR on reco seeds but FM
on truth tracks; this removes that difference.

## Inputs

- **Merged files:** `/sphenix/user/hwyu/calotrack_tree/macro/dedx-v2/fm-merged-run21-25seg/OutDir{0..24}_calotrkana_fm.root`,
  written by `../fm-vs-gpr/merge_fm_to_root.py`. These hold the traditional seeds of
  `condor_pid_x10/OutDir0–24`, with Gabor's per-cluster FM scores
  (`/sphenix/u/ggalgoczi/FM-PID_ensemble_combiner_run21_25seg.csv.gz`) attached.
- **GPR model:** `../fm-vs-gpr-2026-10-05-ncl20-evalOutDir0-24/gpr_model_calotrkana-1M-ncl_0.5_2.0_ncl20.csv`
  (unchanged training: OutDir0–999, ncl ≥ 20, fit 0.5–2.0 GeV/c).

## FM ↔ trad matching (merge step)

- **Event:** CSV `(segment, entry)` = `OutDir<segment>`, tree entry `<entry>`.
  (`evtnumber` is 0 for every entry, so it can't be used.)
- **Cluster:** each FM point → nearest TPC reco cluster in (x, y, z).
  All 16,979,350 FM points matched (max distance 5.05e-5 cm, no duplicates). Energy
  and track-id cross-checks agree 100% (`E == reco_cluster_E`,
  `seg_target == reco_cluster_g4hit_trkid`).
- **Seed:** `tpc_seeds_clusters[start_idx : start_idx + nclusters]` → `reco_cluster_id`
  (100% found). The seed's FM score is the mean of the per-cluster FM probabilities over
  the seed clusters that have one. 84.4% of all seed clusters have an FM score. FM scores
  98.9% of primary-particle TPC clusters but only 86.1% of secondary-particle ones, so 96%
  of the unscored clusters are from secondaries (negative G4 track ids; an earlier version
  of this note wrongly called them noise). See `../fm-vs-gpr-2026-10-05-seed-purity`.
- **FM coverage:** 19,725 of the 25,000 events. Missing are mostly tiny events
  (< 50 TPC clusters) plus the largest ones; this looks like a selection in the FM data
  preparation, not confirmed.

## Seed selection

| step | seeds |
|---|---|
| all seeds, OutDir0–24 | 119,324 |
| GPR base selection (finite, dE/dx < 1000, p > 0, π/K/p, seed nclusters ≥ 20) | 94,957 |
| + event is in the FM sample | 89,391 |
| + ≥ 1 seed cluster with an FM score (**used**) | **89,390** |

## Scores

- **GPR:** `score_c = L_c π_c(p) / Σ_k L_k π_k(p)` from the trained model; identical to
  `compare_pid_roc.py`.
- **FM:** the seed's mean per-cluster probabilities, renormalized over π/K/p.

## Results (AUC, same seeds)

| bin [GeV/c] | species | N_sig / N_tot | GPR | FM (seeds) | FM (truth tracks, prev.) |
|---|---|---|---|---|---|
| [0.8, 1.2) | π | 9,390 / 12,731 | 0.852 | **0.875** | 0.821 |
|            | K | 1,260 / 12,731 | 0.695 | **0.713** | 0.749 |
|            | p | 2,081 / 12,731 | 0.999 | 0.998 | 0.999 |
| [1.0, 2.0) | π | 6,583 / 9,056 | 0.718 | **0.730** | 0.731 |
|            | K | 1,106 / 9,056 | **0.649** | 0.605 | 0.700 |
|            | p | 1,367 / 9,056 | 0.921 | **0.940** | 0.934 |
| [0.0, 1.0) | π | 67,666 / 79,555 | 0.652* | 0.994 | 0.991 |
|            | K | 4,125 / 79,555 | 0.633* | 0.984 | 0.988 |
|            | p | 7,764 / 79,555 | 0.910* | 1.000 | 1.000 |

\* GPR band extrapolated below 0.5 GeV/c (fit range 0.5–2.0); not a fair low-p comparison.

On the same seeds, FM is better than or equal to GPR in 5 of the 6 species × bin
combinations within the GPR fit range. The exception is kaons in [1,2), where GPR leads
(0.649 vs 0.605). In [0.8,1.2) the pion result flips relative to the truth-track
comparison (FM now 0.875 vs GPR 0.852).

FM's kaon AUC is lower on seeds than on truth tracks (0.713 vs 0.749 and 0.605 vs
0.700). The seed-purity study (`../fm-vs-gpr-2026-10-05-seed-purity`) shows this is
**not** due to impure seeds: seeds are very pure (median 0.93). Instead, 33% of seeds
have a secondary as their top truth track, with a very different species mix, while
the truth-track FM numbers only include primary tracks (`seg_target > 0`). On primary
seeds, GPR still leads for kaons in [1,2) (0.631 vs 0.587).

## Validation

- **Merge:** every FM point matched with E and track-id cross-checks; for segment 0,
  copied branches are identical to the original, per-truth-track FM means rebuilt from
  the ROOT file equal the CSV (≤ 1.2e-7), per-seed means match an independent loop
  (≤ 2.4e-7), and the file works as a ROOT friend tree.
- **Same-seed GPR:** with `--no-require-fm` (all seeds), GPR AUCs and counts are identical
  to `../fm-vs-gpr-2026-10-05-ncl20-evalOutDir0-24` in all three bins.
- **Refactor:** `compare_pid_roc.py` (plotting and selection moved into `plot_comparison`
  / `gpr_base_mask`) reproduces the OutDir0–24 plots pixel-identically.

## Reproduce

```bash
source fm-vs-gpr/env.sh
# (once) merge FM per-cluster scores into the trad files
$PY fm-vs-gpr/merge_fm_to_root.py --segments 0-24 \
    --outdir /sphenix/user/hwyu/calotrack_tree/macro/dedx-v2/fm-merged-run21-25seg
$PY fm-vs-gpr/compare_same_seeds.py --outdir fm-vs-gpr-2026-10-05-same-seeds-ncl20
```

## Files

- `pid_roc_comparison_{0.8_1.2,0.0_1.0,1.0_2.0}.{png,pdf}`, `auc_summary_*.csv`, `run.log`
- `same_seed_scores.csv` — per-seed GPR and FM scores (large-ish; not for git)
