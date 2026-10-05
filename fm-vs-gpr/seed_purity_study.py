#!/usr/bin/env python3
"""Seed-purity study for the same-seed FM vs. GPR comparison.

Seed purity = (number of the seed's clusters whose truth track is the seed's most
common truth track) / (seed nclusters). The truth track of a cluster is
reco_cluster_g4hit_trkid from the original trad trees (negative ids are G4
secondaries and count as real particles; 0 = no truth).

Uses the seed selection and both scores from compare_same_seeds.py, then:
  * purity distribution per species,
  * AUC vs purity bin for both methods (Hanley-McNeil standard errors),
  * same-seed ROC plots restricted to purity >= each --purity-cuts value,
  * AUC split by the origin of the seed's top truth track (primary id>0 vs
    secondary id<0), the species mix of each, and same-seed ROC plots per origin,
  * a breakdown of which TPC clusters FM did not score.
"""

import argparse
import os
import sys

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pid_roc import SPECIES, SPECIES_NAME, plot_comparison, roc_curve_np  # noqa: E402
from compare_same_seeds import GPR_MODEL, MERGED_DIR, build_same_seed_table  # noqa: E402

ROOT_TEMPLATE = "/sphenix/user/shuhangli/calotrack_tree/macro/condor_pid_x10/OutDir{N}/calotrkana.root"


def seed_purity(segments):
    """Per-seed purity for every seed of the given segments."""
    br = ["reco_cluster_id", "reco_cluster_g4hit_trkid", "tpc_seeds_nclusters",
          "tpc_seeds_clusters", "tpc_seeds_maxparticle_pid", "particle_track_id", "particle_pid"]
    rows = []
    for N in segments:
        a = uproot.open(ROOT_TEMPLATE.format(N=N))["T"].arrays(br, library="np")
        for e in range(len(a["reco_cluster_id"])):
            ids = a["reco_cluster_id"][e]
            order = np.argsort(ids)
            keys = a["tpc_seeds_clusters"][e]
            trk = a["reco_cluster_g4hit_trkid"][e][order[np.searchsorted(ids[order], keys)]]
            pid_of = dict(zip(a["particle_track_id"][e].tolist(), a["particle_pid"][e].tolist()))
            start = 0
            for s, n in enumerate(a["tpc_seeds_nclusters"][e].astype(int)):
                t = trk[start:start + n]
                start += n
                real = t[t != 0]
                if real.size:
                    vals, cnt = np.unique(real, return_counts=True)
                    top, ntop = int(vals[cnt.argmax()]), int(cnt.max())
                else:
                    top, ntop = 0, 0
                rows.append((N, e, s, n, ntop, top, pid_of.get(top, 0),
                             int(a["tpc_seeds_maxparticle_pid"][e][s])))
    df = pd.DataFrame(rows, columns=["segment", "entry", "seed_idx", "ncl", "n_top", "top_trkid",
                                     "top_pid", "maxparticle_pid"])
    df["purity"] = df["n_top"] / df["ncl"]
    return df


def auc_se(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) standard error of the AUC."""
    if not (n_pos and n_neg) or not np.isfinite(auc):
        return np.nan
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    return float(np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                          + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)))


def plot_subset(sub, outdir, bins):
    """Same-seed ROC plots (plot_comparison) for a subset of the seed table."""
    os.makedirs(outdir, exist_ok=True)
    gpr_df = pd.DataFrame({"p": sub.p, "apid": sub.apid,
                           **{f"score_{sp}": sub[f"gpr_score_{sp}"] for sp in SPECIES}})
    fm_df = pd.DataFrame({"p": sub.p, "gt_pid_class": sub.apid.map({211: 1, 321: 2, 2212: 3}),
                          **{f"score_{sp}": sub[f"fm_score_{sp}"] for sp in SPECIES}})
    for b in bins.split(";"):
        lo, hi = (float(x) for x in b.split(","))
        plot_comparison(gpr_df, fm_df, lo, hi, outdir)


def fm_skipped_breakdown(merged_dir, segments):
    rows = []
    for N in segments:
        o = uproot.open(ROOT_TEMPLATE.format(N=N))["T"].arrays(
            ["reco_cluster_detid", "reco_cluster_g4hit_trkid"], library="ak")
        m = uproot.open(os.path.join(merged_dir, f"OutDir{N}_calotrkana_fm.root"))["T"].arrays(
            ["fm_has_event", "reco_cluster_fm_matched"], library="ak")
        ev = m["fm_has_event"] == 1
        tpc = o["reco_cluster_detid"][ev] == 2
        trk = ak.to_numpy(ak.flatten(o["reco_cluster_g4hit_trkid"][ev][tpc]))
        mat = ak.to_numpy(ak.flatten(m["reco_cluster_fm_matched"][ev][tpc]))
        rows.append(pd.DataFrame({"trk": trk, "matched": mat}))
    d = pd.concat(rows)
    d["category"] = np.select([d.trk > 0, d.trk < 0], ["primary (id>0)", "secondary (id<0)"], "no truth (id=0)")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged-dir", default=MERGED_DIR)
    ap.add_argument("--gpr-model", default=GPR_MODEL)
    ap.add_argument("--ncl-min", type=int, default=20)
    ap.add_argument("--bins", default="0.8,1.2;1.0,2.0", help="momentum bins 'lo,hi;...'")
    ap.add_argument("--purity-bins", default="0,0.5,0.8,0.95,1.0001")
    ap.add_argument("--purity-cuts", default="0.5,0.9")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    seeds, _, _ = build_same_seed_table(args.merged_dir, args.gpr_model, ncl_min=args.ncl_min)
    segs = sorted(seeds["segment"].unique())
    print(f"[purity] computing seed purity for segments {segs[0]}..{segs[-1]} ...")
    pur = seed_purity(segs)
    seeds = seeds.merge(pur, on=["segment", "entry", "seed_idx"], how="left", validate="1:1")
    assert seeds["purity"].notna().all() and (seeds["ncl"] == seeds["nclusters"]).all()
    seeds.to_csv(os.path.join(args.outdir, "seed_purity_scores.csv"), index=False)

    label_ok = (np.abs(seeds.top_pid) == seeds.apid).mean()
    print(f"[purity] seeds {len(seeds):,}; top-track PID == maxparticle PID for {label_ok:.4f}; "
          f"top track is a secondary (id<0) for {(seeds.top_trkid < 0).mean():.3f}")
    print("[purity] quantiles (5/25/50/75/95%):",
          np.round(np.percentile(seeds.purity, [5, 25, 50, 75, 95]), 3).tolist())

    # --- purity distribution ---------------------------------------------------
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for sp, c in zip(SPECIES, ["tab:blue", "tab:orange", "tab:green"]):
        x = seeds.loc[seeds.apid == sp, "purity"]
        ax.hist(x, bins=np.linspace(0, 1, 41), histtype="step", lw=2, density=True, color=c,
                label=f"{SPECIES_NAME[sp]} (N={len(x):,}, median {x.median():.2f})")
    ax.set_xlabel("Seed purity"); ax.set_ylabel("Density"); ax.set_yscale("log")
    ax.legend(fontsize=12, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "seed_purity_distribution.png")); plt.close(fig)

    # --- AUC vs purity -----------------------------------------------------------
    pedges = [float(x) for x in args.purity_bins.split(",")]
    rows = []
    for b in args.bins.split(";"):
        lo, hi = (float(x) for x in b.split(","))
        inb = (seeds.p >= lo) & (seeds.p < hi)
        for k in range(len(pedges) - 1):
            m = inb & (seeds.purity >= pedges[k]) & (seeds.purity < pedges[k + 1])
            for sp in SPECIES:
                y = (seeds.apid[m] == sp).to_numpy().astype(int)
                for meth in ["gpr", "fm"]:
                    _, _, auc = roc_curve_np(y, seeds.loc[m, f"{meth}_score_{sp}"].to_numpy())
                    npos, nneg = int(y.sum()), int(len(y) - y.sum())
                    rows.append(dict(bin_lo=lo, bin_hi=hi, purity_lo=pedges[k], purity_hi=min(pedges[k + 1], 1.0),
                                     species=sp, method=meth.upper(), auc=auc,
                                     auc_se=auc_se(auc, npos, nneg), n_sig=npos, n_tot=len(y)))
    av = pd.DataFrame(rows)
    av.to_csv(os.path.join(args.outdir, "auc_vs_purity.csv"), index=False)

    for (lo, hi), g in av.groupby(["bin_lo", "bin_hi"]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
        for ax, sp in zip(axes, SPECIES):
            for meth, c, dx in [("FM", "tab:blue", -0.12), ("GPR", "tab:red", 0.12)]:
                q = g[(g.species == sp) & (g.method == meth)]
                x = np.arange(len(q)) + dx
                ax.errorbar(x, q.auc, yerr=q.auc_se, fmt="o", color=c, capsize=4, ms=7, label=meth)
            q = g[(g.species == sp) & (g.method == "FM")]
            ax.set_xticks(np.arange(len(q)))
            ax.set_xticklabels([f"[{a:.2f},{b:.2f}]\nN$_{{sig}}$={n}" for a, b, n in
                                zip(q.purity_lo, q.purity_hi, q.n_sig)], fontsize=11)
            ax.set_title(f"{SPECIES_NAME[sp]} one-vs-rest")
            ax.set_xlabel("Seed purity bin"); ax.grid(alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel("AUC")
            ax.legend(fontsize=12, loc="lower right")
        fig.suptitle(f"p $\\in$ [{lo}, {hi}) GeV/c", fontsize=15)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"auc_vs_purity_{lo}_{hi}.png")); plt.close(fig)

    with pd.option_context("display.width", 200):
        print(av.pivot_table(index=["bin_lo", "bin_hi", "purity_lo", "species"], columns="method",
                             values=["auc", "n_sig"]).round(3).to_string())

    # --- same-seed ROC restricted to purity >= cut -----------------------------
    for cut in [float(x) for x in args.purity_cuts.split(",")]:
        sub = seeds[seeds.purity >= cut]
        print(f"\n##### purity >= {cut}: {len(sub):,}/{len(seeds):,} seeds")
        plot_subset(sub, os.path.join(args.outdir, f"purity_ge_{cut}"), args.bins)

    # --- AUC by origin of the top truth track (primary vs secondary) ------------
    seeds["origin"] = np.where(seeds.top_trkid > 0, "primary", "secondary")
    rows = []
    for b in args.bins.split(";"):
        lo, hi = (float(x) for x in b.split(","))
        for org in ["primary", "secondary", "all"]:
            m = (seeds.p >= lo) & (seeds.p < hi)
            if org != "all":
                m &= seeds.origin == org
            for sp in SPECIES:
                y = (seeds.apid[m] == sp).to_numpy().astype(int)
                r = dict(bin_lo=lo, bin_hi=hi, origin=org, species=sp, n_sig=int(y.sum()), n_tot=len(y))
                for meth in ["gpr", "fm"]:
                    a = roc_curve_np(y, seeds.loc[m, f"{meth}_score_{sp}"].to_numpy())[2]
                    r[meth.upper()] = a
                    r[f"{meth.upper()}_se"] = auc_se(a, int(y.sum()), int(len(y) - y.sum()))
                rows.append(r)
    ao = pd.DataFrame(rows)
    ao.to_csv(os.path.join(args.outdir, "auc_by_origin.csv"), index=False)
    print("\n[AUC by origin of top truth track]\n" + ao.round(3).to_string(index=False))
    m = (seeds.p >= 0.5) & (seeds.p < 2.0)
    mix = pd.crosstab(seeds.origin[m], seeds.apid[m], normalize="index")
    mix.to_csv(os.path.join(args.outdir, "species_mix_by_origin.csv"))
    print("\n[species mix by origin, p in [0.5, 2)]\n" + mix.round(3).to_string())
    for org in ["primary", "secondary"]:
        sub = seeds[seeds.origin == org]
        print(f"\n##### origin = {org}: {len(sub):,}/{len(seeds):,} seeds")
        plot_subset(sub, os.path.join(args.outdir, f"origin_{org}"), args.bins)

    # --- which TPC clusters does FM not score? -----------------------------------
    sk = fm_skipped_breakdown(args.merged_dir, segs)
    tab = sk.groupby("category").agg(clusters=("matched", "size"), fm_scored=("matched", "mean"))
    tab["share_of_skipped"] = sk[sk.matched == 0].groupby("category").size() / (sk.matched == 0).sum()
    tab.to_csv(os.path.join(args.outdir, "fm_skipped_clusters.csv"))
    print("\n[FM-skipped TPC clusters, FM events]\n" + tab.round(3).to_string())


if __name__ == "__main__":
    main()
