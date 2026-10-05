# Seed-purity study — same-seed FM vs. GPR (ncl ≥ 20)

Follow-up to [`../fm-vs-gpr-2026-10-05-same-seeds-ncl20`](../fm-vs-gpr-2026-10-05-same-seeds-ncl20/README.md).
Question: is FM's lower kaon AUC on reco seeds (vs. on truth tracks) caused by impure seeds?
**Answer: no.** Seeds are very pure. The difference comes mostly from the seed population,
which includes many secondaries, and a real FM kaon deficit at 1–2 GeV/c remains on primary seeds.

Same seeds, selection and scores as the same-seed comparison (89,390 seeds).

## Definitions

- **Truth track of a cluster:** `reco_cluster_g4hit_trkid` from the original trees.
  **Negative ids are G4 secondaries** (all present in the particle table); 0 = no truth.
- **Seed purity:** (seed clusters from the seed's most common truth track) / (seed `nclusters`).
- **Origin:** primary if that top truth track has id > 0, secondary if id < 0.

Sanity check: the top truth track's PID equals `tpc_seeds_maxparticle_pid` for 99.96% of seeds.

## 1. Seeds are pure — purity doesn't explain the kaon result

- Purity quantiles (5/25/50/75/95%): 0.82 / 0.89 / 0.93 / 0.96 / 1.00 (`seed_purity_distribution.png`).
- Purity ≥ 0.5 keeps 99.9% of seeds; purity ≥ 0.9 keeps 68%.
- Within the populated purity bins the FM–GPR ordering doesn't change (`auc_vs_purity_*.png`).
  E.g. kaons in [1,2): FM 0.600 / 0.608 vs GPR 0.658 / 0.653 for purity [0.80,0.95) / [0.95,1].
- Restricting to purity ≥ 0.9 (`purity_ge_0.9/`) leaves the conclusions unchanged:

| bin | species | GPR | FM |
|---|---|---|---|
| [0.8,1.2) | π | 0.865 | **0.886** |
|           | K | 0.691 | **0.714** |
|           | p | 0.999 | 0.999 |
| [1,2)     | π | 0.724 | **0.741** |
|           | K | **0.655** | 0.611 |
|           | p | 0.919 | **0.942** |

## 2. One third of seeds are secondaries, with a very different species mix

33% of seeds have a secondary as the top truth track. Species mix (p ∈ [0.5, 2) GeV/c):

| origin | π | K | p |
|---|---|---|---|
| primary   | 84.2% | 10.8% | 5.0% |
| secondary | 46.8% |  2.0% | 51.2% |

AUC by origin (± Hanley–McNeil standard error):

| bin | origin | species | N_sig | GPR | FM |
|---|---|---|---|---|---|
| [0.8,1.2) | primary   | π | 8,258 | 0.774 ± 0.005 | 0.792 ± 0.005 |
|           |           | K | 1,193 | 0.676 ± 0.009 | 0.684 ± 0.009 |
|           |           | p |   692 | 0.999 ± 0.001 | 0.998 ± 0.001 |
|           | secondary | π | 1,132 | 0.947 ± 0.005 | **0.982 ± 0.003** |
|           |           | K |    67 | 0.761 ± 0.034 | 0.799 ± 0.033 |
|           |           | p | 1,389 | 0.998 ± 0.001 | 0.998 ± 0.001 |
| [1,2)     | primary   | π | 5,980 | 0.655 ± 0.007 | 0.644 ± 0.007 |
|           |           | K | 1,070 | **0.631 ± 0.010** | 0.587 ± 0.010 |
|           |           | p |   622 | 0.909 ± 0.008 | 0.927 ± 0.007 |
|           | secondary | π |   603 | 0.860 ± 0.011 | **0.913 ± 0.008** |
|           |           | K |    36 | 0.752 ± 0.047 | 0.647 ± 0.050 |
|           |           | p |   745 | 0.929 ± 0.007 | 0.939 ± 0.007 |

Takeaways:
- The all-seed AUCs mix two populations with very different class compositions, so
  part of FM's overall pion advantage comes from secondary seeds (FM 0.98 vs GPR 0.95 in [0.8,1.2)).
- **On primary seeds**, FM and GPR are within ~1–2σ in [0.8,1.2) for all species.
  In [1,2), GPR is clearly better for kaons (0.631 vs 0.587, ~3σ).
- The earlier truth-track FM comparisons kept only `seg_target > 0`, i.e. **primary truth
  tracks only**, so they are not directly comparable to the all-seed numbers.

## 3. Which TPC clusters does FM not score? (correction)

| truth origin | TPC clusters (FM events) | scored by FM | share of the unscored |
|---|---|---|---|
| primary (id > 0)   |  6,625,589 | 98.9% |  4.1% |
| secondary (id < 0) | 12,110,320 | 86.1% | 95.9% |

Earlier notes called the unscored clusters "noise (g4hit_trkid ≤ 0)". That was wrong:
they are almost all **secondary-particle** clusters.

## Open question

FM kaons on primary seeds in [1,2) (0.587) are still well below FM kaons on primary
truth tracks (0.700). Two differences remain:
- the seed score averages FM over the seed's clusters only, while the truth-track
  score averages over all of the track's points;
- the populations differ (seeds found by the traditional reco vs. all truth tracks).

Scoring the primary seeds with their truth track's full-track FM score would separate the two.

## Reproduce

```bash
source fm-vs-gpr/env.sh
$PY fm-vs-gpr/seed_purity_study.py --outdir fm-vs-gpr-2026-10-05-seed-purity
```

## Files

- `seed_purity_distribution.png`, `auc_vs_purity_{0.8_1.2,1.0_2.0}.png`, `auc_vs_purity.csv`
- `auc_by_origin.csv`, `species_mix_by_origin.csv`, `fm_skipped_clusters.csv`
- `purity_ge_{0.5,0.9}/` — same-seed ROC plots and AUC tables after the purity cut
- `origin_{primary,secondary}/` — same-seed ROC plots and AUC tables per origin of the top
  truth track; `origin_primary/pid_roc_comparison_0.8_1.2.pdf` is the paper's PID figure
  (`paper-draft/figures/pid_roc_0.8_1.2.pdf`)
- `seed_purity_scores.csv` — per-seed scores + purity (large-ish; not for git)
