#!/usr/bin/env python3
"""FM vs. GPR PID on the *same* reconstructed TPC seeds.

Reads the merged files written by merge_fm_to_root.py (traditional seeds with
per-seed FM scores attached) and scores every seed with both methods:

  * GPR: the trained band + prior model (gpr_model_*.csv from compare_pid_roc.py)
         applied to the seed's dE/dx -- identical scoring to compare_pid_roc.py.
  * FM:  the seed's FM probabilities = mean of the per-cluster FM probabilities
         over the seed's clusters that have an FM score, renormalised over
         pi/K/p (same convention as compare_pid_roc.py).

Both methods see the identical seed list, truth label (|tpc_seeds_maxparticle_pid|)
and truth momentum (tpc_seeds_maxparticle_p). Seed selection: the GPR base
selection (gpr_base_mask: finite, dE/dx < 1000, p > 0, pi/K/p, seed nclusters
>= --ncl-min), the event is in the FM sample (fm_has_event), and the seed has
at least --fm-min-scored clusters with an FM score.
"""

import argparse
import glob
import os
import sys

import awkward as ak
import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pid_roc import (  # noqa: E402
    FM_CLASS_TO_PDG,
    SPECIES,
    gpr_base_mask,
    models_from_table,
    plot_comparison,
    score_gpr,
)

MERGED_DIR = "/sphenix/user/hwyu/calotrack_tree/macro/dedx-v2/fm-merged-run21-25seg"
GPR_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fm-vs-gpr-2026-10-05-ncl20-evalOutDir0-24",
    "gpr_model_calotrkana-1M-ncl_0.5_2.0_ncl20.csv")
FM_CLASSES = [1, 2, 3]  # pi, K, p in the FM class convention


def load_merged_seeds(files):
    seed_br = ["tpc_seeds_dedx", "tpc_seeds_maxparticle_p", "tpc_seeds_maxparticle_pid",
               "tpc_seeds_nclusters", "tpc_seeds_fm_nclusters_scored"]
    seed_br += [f"tpc_seeds_fm_pid_prob_{c}" for c in FM_CLASSES]
    parts = []
    for f in files:
        a = uproot.open(f)["T"].arrays(seed_br + ["fm_has_event"], library="ak")
        n = ak.num(a["tpc_seeds_dedx"])
        cols = {b: ak.to_numpy(ak.flatten(a[b])) for b in seed_br}
        cols["fm_has_event"] = np.repeat(ak.to_numpy(a["fm_has_event"]), ak.to_numpy(n))
        cols["file"] = os.path.basename(f)
        parts.append(pd.DataFrame(cols))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged-dir", default=MERGED_DIR)
    ap.add_argument("--gpr-model", default=GPR_MODEL,
                    help="trained GPR model table (gpr_model_*.csv)")
    ap.add_argument("--fit-lo", type=float, default=0.5)
    ap.add_argument("--fit-hi", type=float, default=2.0)
    ap.add_argument("--ncl-min", type=int, default=20, help="min seed clusters (0 = no cut)")
    ap.add_argument("--fm-min-scored", type=int, default=1,
                    help="min seed clusters with an FM score")
    ap.add_argument("--no-require-fm", action="store_true",
                    help="keep all seeds (GPR only; for cross-checks)")
    ap.add_argument("--bins", default="0.8,1.2;0.0,1.0;1.0,2.0",
                    help="momentum bins 'lo,hi;lo,hi;...'")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.merged_dir, "OutDir*_calotrkana_fm.root")),
                   key=lambda f: int(os.path.basename(f)[6:].split("_")[0]))
    print(f"[seeds] {len(files)} merged files from {args.merged_dir}")
    d = load_merged_seeds(files)
    p = d["tpc_seeds_maxparticle_p"].to_numpy(float)
    dedx = d["tpc_seeds_dedx"].to_numpy(float)
    apid = np.abs(d["tpc_seeds_maxparticle_pid"].to_numpy(float))
    ncl = d["tpc_seeds_nclusters"].to_numpy(np.int64)

    base = gpr_base_mask(p, dedx, apid, ncl, args.ncl_min)
    in_fm = d["fm_has_event"].to_numpy() == 1
    scored = d["tpc_seeds_fm_nclusters_scored"].to_numpy() >= args.fm_min_scored
    sel = base if args.no_require_fm else base & in_fm & scored
    print(f"[seeds] all {len(d):,} | GPR base selection (ncl >= {args.ncl_min}) {base.sum():,} | "
          f"+ event in FM {(base & in_fm).sum():,} | + >= {args.fm_min_scored} FM-scored "
          f"cluster(s) {(base & in_fm & scored).sum():,} | used {sel.sum():,}")

    # --- GPR scores with the trained model --------------------------------------
    table = pd.read_csv(args.gpr_model, float_precision="round_trip")
    band_models, prior_models = models_from_table(table, args.fit_lo, args.fit_hi)
    apid_i = apid[sel].astype(np.int64)
    gpr_df = score_gpr(p[sel], dedx[sel], apid_i, band_models, prior_models)

    # --- FM scores of the same seeds ---------------------------------------------
    pdg_to_fm = {v: k for k, v in FM_CLASS_TO_PDG.items()}
    probs = np.stack([d[f"tpc_seeds_fm_pid_prob_{c}"].to_numpy(float)[sel] for c in FM_CLASSES])
    probs = np.where(probs >= 0, probs, np.nan)  # -1 = no FM score
    denom = np.nansum(probs, axis=0)
    fm_df = pd.DataFrame({"p": p[sel], "gt_pid_class": [pdg_to_fm[a] for a in apid_i]})
    for i, sp in enumerate(SPECIES):
        fm_df[f"score_{sp}"] = probs[i] / np.where(denom > 0, denom, np.nan)

    seeds = pd.DataFrame({"file": d["file"].to_numpy()[sel], "p": p[sel], "apid": apid_i,
                          "dedx": dedx[sel], "nclusters": ncl[sel],
                          "fm_nclusters_scored": d["tpc_seeds_fm_nclusters_scored"].to_numpy()[sel]})
    for sp in SPECIES:
        seeds[f"gpr_score_{sp}"] = gpr_df[f"score_{sp}"].to_numpy()
        seeds[f"fm_score_{sp}"] = fm_df[f"score_{sp}"].to_numpy()
    seeds.to_csv(os.path.join(args.outdir, "same_seed_scores.csv"), index=False)

    for b in args.bins.split(";"):
        lo, hi = (float(x) for x in b.split(","))
        plot_comparison(gpr_df, fm_df, lo, hi, args.outdir)


if __name__ == "__main__":
    main()
