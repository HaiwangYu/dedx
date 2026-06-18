#!/usr/bin/env python3
"""Apple-to-apple PID ROC comparison: Foundation Model (FM) vs. GPR dE/dx.

Builds one figure with three panels (pi / K / p), each overlaying the
one-vs-rest ROC curve of the two methods in a single momentum bin
(default 0.8-1.2 GeV/c).

Design decisions (see README.md in this directory for rationale):

  * Charge folding: both methods use momentum magnitude and treat a species
    charge-inclusively (|pid| for GPR, gt_pid_class for FM).
  * Population / "rest": both methods classify among pi/K/p ONLY.  The
    one-vs-rest negative set is therefore the other two species.  FM
    per-class probabilities are renormalised over classes {pi,K,p} to mirror
    the GPR likelihood fraction (score_frac), which is normalised over the
    same three hypotheses.
  * Momentum variable: each method uses its own truth momentum.  For the GPR
    ROOT file that is `tpc_seeds_maxparticle_p` (the truth momentum of the
    dominant contributing particle); for FM it is |(px,py,pz)|.
  * GPR scoring matches the deployed configuration: likelihood = 1/|dedx-mean|
    with sigma forced to 1, multiplied by a per-momentum-bin class prior, then
    normalised across the three species (identical to dedx_analysis with
    --prior-distribution-dir and the default --force-sigma-one).

The ROC curve itself is computed with a single shared numpy routine for both
methods so the only differences are the scores being thresholded.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- locate the GPR package (dedx_analysis) -------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dedx_analysis.analysis import load_root_branches, _fit_gaussian_process  # noqa: E402
from dedx_analysis.pipeline import (  # noqa: E402
    _BandModel,
    _PriorDistribution,
    _compute_score_frac_matrix,
)

# pi / K / p only -- the shared 3-class hypothesis set
SPECIES = [211, 321, 2212]
SPECIES_NAME = {211: "$\\pi$", 321: "$K$", 2212: "$p$"}
# FM gt_pid_class -> PDG code (train/downstream get_pidlabel convention)
FM_CLASS_TO_PDG = {1: 211, 2: 321, 3: 2212}


# ---------------------------------------------------------------------------
# Shared ROC routine -- used identically for both methods.
# ---------------------------------------------------------------------------
def roc_curve_np(y_true, y_score):
    """One-vs-rest ROC. Returns (fpr, tpr, auc) with fpr/tpr starting at 0."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]

    P = int(y_true.sum())
    N = int(y_true.size - P)
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    s = y_score[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    # keep the last index of each group of equal scores
    distinct = np.where(np.diff(s) != 0)[0]
    idx = np.r_[distinct, s.size - 1]
    tpr = np.r_[0.0, tps[idx] / P]
    fpr = np.r_[0.0, fps[idx] / N]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


# ---------------------------------------------------------------------------
# GPR side
# ---------------------------------------------------------------------------
def build_gpr_scores(
    root_file,
    fit_lo,
    fit_hi,
    momentum_bins=200,
    dedx_max=1000.0,
    sample_size=5000,
    seed=42,
):
    """Fit folded GPR bands + priors over [fit_lo, fit_hi] and return a track
    table with score_frac per species. Charge-folded (|pid|), momentum = p."""
    arrays = load_root_branches(
        input_file=root_file,
        tree_name="T",
        dedx_branch="tpc_seeds_dedx",
        momentum_branch="tpc_seeds_maxparticle_p",
        pid_branch="tpc_seeds_maxparticle_pid",
    )
    dedx = np.asarray(arrays["dedx"], dtype=float)
    p = np.asarray(arrays["momentum"], dtype=float)
    apid = np.abs(np.asarray(arrays["pid"], dtype=float))

    base = (
        np.isfinite(dedx)
        & np.isfinite(p)
        & np.isfinite(apid)
        & (dedx < dedx_max)
        & (p > 0)
        & np.isin(apid.astype(np.int64), SPECIES)
    )
    fit_window = base & (p >= fit_lo) & (p <= fit_hi)

    dedx_f = dedx[fit_window]
    p_f = p[fit_window]
    apid_f = apid[fit_window].astype(np.int64)

    # --- bands: reuse the exact GPR fitting code, folded over |p| -----------
    eval_grid = np.linspace(fit_lo, fit_hi, momentum_bins)
    band_models = {}
    for sp in SPECIES:
        m = apid_f == sp
        mean, sigma = _fit_gaussian_process(
            p_f[m],
            dedx_f[m],
            eval_grid,
            sample_size=sample_size,
            random_seed=seed,
            momentum_gap=(-0.2, 0.2),  # all data is positive |p|, no gap effect
            analysis_range=(fit_lo, fit_hi),
        )
        band_models[sp] = _BandModel(eval_grid.copy(), mean, sigma)

    # --- priors: per |p|-bin class fraction over the 3-species population ----
    bin_edges = np.linspace(fit_lo, fit_hi, momentum_bins + 1)
    total_counts, _ = np.histogram(p_f, bins=bin_edges)
    prior_models = {}
    for sp in SPECIES:
        counts, _ = np.histogram(p_f[apid_f == sp], bins=bin_edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.divide(
                counts, total_counts,
                out=np.zeros(counts.shape, dtype=float),
                where=total_counts > 0,
            )
        prior_models[sp] = _PriorDistribution(bin_edges, probs)

    # --- score the full 3-species population (deployed scoring) -------------
    p_e = p[base]
    dedx_e = dedx[base]
    apid_e = apid[base].astype(np.int64)
    score_frac = _compute_score_frac_matrix(
        p_e, dedx_e, SPECIES, band_models, prior_models, force_sigma_one=True
    )  # shape (3, n)

    df = pd.DataFrame({"p": p_e, "apid": apid_e})
    for i, sp in enumerate(SPECIES):
        df[f"score_{sp}"] = score_frac[i]
    return df


# ---------------------------------------------------------------------------
# FM side
# ---------------------------------------------------------------------------
def build_fm_scores(csv_path, chunksize=2_000_000):
    """Aggregate per-point FM CSV to track level (mean prob per class over the
    track's points), restrict to pi/K/p tracks, renormalise probs over the
    three classes. Returns a track table with score per species + momentum.

    Vectorised: each chunk is reduced with a groupby to per-track partial
    sums/counts; partials are concatenated and reduced once at the end."""
    event_cols = ["batch_idx", "sample_idx_in_batch"]
    track_col = "seg_target"
    label_col = "gt_pid_class"
    prob_cols = ["pid_prob_class_1", "pid_prob_class_2", "pid_prob_class_3"]
    kin_cols = ["px", "py", "pz"]
    usecols = event_cols + [track_col, label_col] + prob_cols + kin_cols
    key_cols = event_cols + [track_col]

    partials = []
    n_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        n_rows += len(chunk)
        chunk = chunk.dropna(subset=key_cols + [label_col])
        chunk = chunk[chunk[track_col] > 0]
        if len(chunk) == 0:
            continue
        # unassigned points write -1.0 in prob cols -> treat as missing
        for c in prob_cols:
            chunk.loc[chunk[c] < 0, c] = np.nan

        agg = {label_col: "first"}
        for c in kin_cols:
            agg[c] = "sum"
        for c in prob_cols:
            agg[c] = "sum"
        g = chunk.groupby(key_cols, sort=False)
        part = g.agg(agg)
        part["kin_cnt"] = g["px"].count()
        for c in prob_cols:
            part[f"{c}_cnt"] = g[c].count()
        partials.append(part.reset_index())
        print(f"  FM chunk: rows={n_rows:,} partial-tracks={len(part):,}", flush=True)

    allp = pd.concat(partials, ignore_index=True)
    sum_cols = kin_cols + prob_cols + ["kin_cnt"] + [f"{c}_cnt" for c in prob_cols]
    final = allp.groupby(key_cols, sort=False).agg(
        {label_col: "first", **{c: "sum" for c in sum_cols}}
    )

    kin_cnt = final["kin_cnt"].to_numpy()
    px = final["px"].to_numpy() / np.maximum(kin_cnt, 1)
    py = final["py"].to_numpy() / np.maximum(kin_cnt, 1)
    pz = final["pz"].to_numpy() / np.maximum(kin_cnt, 1)
    out = pd.DataFrame({
        "gt_pid_class": final[label_col].to_numpy().astype(int),
        "p": np.sqrt(px * px + py * py + pz * pz),
    })
    for i, c in enumerate(prob_cols, start=1):
        cnt = final[f"{c}_cnt"].to_numpy()
        out[f"prob_{i}"] = np.where(cnt > 0, final[c].to_numpy() / np.maximum(cnt, 1), np.nan)
    out = out[kin_cnt > 0]

    # restrict to pi/K/p tracks and renormalise probs over the 3 classes
    out = out[out["gt_pid_class"].isin([1, 2, 3])].copy()
    denom = out[["prob_1", "prob_2", "prob_3"]].sum(axis=1, min_count=1)
    for i, sp in zip([1, 2, 3], SPECIES):
        out[f"score_{sp}"] = out[f"prob_{i}"] / denom
    return out


# ---------------------------------------------------------------------------
def roc_for_bin(df, score_col, truth_col, truth_val, lo, hi):
    m = (df["p"] >= lo) & (df["p"] < hi)
    sub = df.loc[m]
    y_true = (sub[truth_col].to_numpy() == truth_val).astype(int)
    y_score = sub[score_col].to_numpy()
    fpr, tpr, auc = roc_curve_np(y_true, y_score)
    return fpr, tpr, auc, int(y_true.sum()), int(y_true.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root-file", default=os.path.join(REPO_ROOT, "calotrkana-1M.root"))
    ap.add_argument("--fm-csv", default=os.path.join(
        REPO_ROOT, "fm4npp_eval/PID",
        "d9_m5_k30_p20_joint_lora_focal_joint_pretrain_d70000_0308_seed42"
        "_eval_per_point_pid_from_prototype.csv"))
    ap.add_argument("--fit-lo", type=float, default=0.5)
    ap.add_argument("--fit-hi", type=float, default=2.0)
    ap.add_argument("--bin-lo", type=float, default=0.8)
    ap.add_argument("--bin-hi", type=float, default=1.2)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--force", action="store_true", help="ignore caches and recompute")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gpr_cache = os.path.join(args.outdir, f"gpr_scores_{args.fit_lo}_{args.fit_hi}.csv")
    fm_cache = os.path.join(args.outdir, "fm_track_scores.csv")

    if args.force or not os.path.exists(gpr_cache):
        print(f"[GPR] fitting bands+priors over [{args.fit_lo}, {args.fit_hi}] ...")
        gpr_df = build_gpr_scores(args.root_file, args.fit_lo, args.fit_hi)
        gpr_df.to_csv(gpr_cache, index=False)
        print(f"[GPR] cached -> {gpr_cache}")
    else:
        print(f"[GPR] using cache {gpr_cache}")
        gpr_df = pd.read_csv(gpr_cache)

    if args.force or not os.path.exists(fm_cache):
        print("[FM] aggregating per-point CSV to track level ...")
        fm_df = build_fm_scores(args.fm_csv)
        fm_df.to_csv(fm_cache, index=False)
        print(f"[FM] cached -> {fm_cache}")
    else:
        print(f"[FM] using cache {fm_cache}")
        fm_df = pd.read_csv(fm_cache)

    # --- plot ---------------------------------------------------------------
    # Paper-sized fonts: ~ body-text size relative to the figure.
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 19,
    })
    lo, hi = args.bin_lo, args.bin_hi
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), dpi=150)
    summary = []
    for ax, sp in zip(axes, SPECIES):
        fpr_g, tpr_g, auc_g, npos_g, ntot_g = roc_for_bin(
            gpr_df, f"score_{sp}", "apid", sp, lo, hi)
        cls = [k for k, v in FM_CLASS_TO_PDG.items() if v == sp][0]
        fpr_f, tpr_f, auc_f, npos_f, ntot_f = roc_for_bin(
            fm_df, f"score_{sp}", "gt_pid_class", cls, lo, hi)

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random")
        ax.plot(fpr_f, tpr_f, color="tab:blue", lw=2.6,
                label=f"FM  AUC={auc_f:.3f}")
        ax.plot(fpr_g, tpr_g, color="tab:red", lw=2.6,
                label=f"GPR AUC={auc_g:.3f}")
        ax.set_title(f"{SPECIES_NAME[sp]} one-vs-rest")
        ax.set_xlabel("False Positive Rate")
        if ax is axes[0]:
            ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=14, handlelength=1.4,
                  borderpad=0.3, labelspacing=0.3, framealpha=0.85)
        summary += [
            dict(species=sp, method="FM", auc=auc_f, n_sig=npos_f, n_tot=ntot_f),
            dict(species=sp, method="GPR", auc=auc_g, n_sig=npos_g, n_tot=ntot_g),
        ]

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png = os.path.join(args.outdir, f"pid_roc_comparison_{lo}_{hi}.png")
    pdf = os.path.join(args.outdir, f"pid_roc_comparison_{lo}_{hi}.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(args.outdir, f"auc_summary_{lo}_{hi}.csv"), index=False)
    print("\n=== AUC summary (p in [%.2f, %.2f)) ===" % (lo, hi))
    print(sdf.to_string(index=False))
    print(f"\nSaved: {png}\n       {pdf}")


if __name__ == "__main__":
    main()
