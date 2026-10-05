#!/usr/bin/env python3
"""Merge per-cluster FM PID scores into the traditional-reco ROOT files.

Input
  * FM per-point CSV (e.g. FM-PID_ensemble_combiner_run21_25seg.csv.gz), with
    `segment` and `entry` columns identifying the event.
  * Traditional reco/truth trees `.../OutDir<N>/calotrkana.root` (tree "T").

Matching
  * Event:   CSV (segment, entry) == file OutDir<segment>, tree entry <entry>.
             (`evtnumber` is 0 for every entry, so it cannot be used.)
  * Cluster: each FM point is matched to the nearest TPC reco cluster
             (reco_cluster_detid == 2) in (x, y, z). A match is accepted if the
             distance is < --tol cm and no two points share a cluster.
             Cross-checks recorded per event: E == reco_cluster_E and
             seg_target == reco_cluster_g4hit_trkid.
  * Seed:    a seed's clusters are tpc_seeds_clusters[start_idx:start_idx+ncl]
             (cluster keys), looked up in reco_cluster_id.

Output: one file per segment, <outdir>/OutDir<N>_calotrkana_fm.root, tree "T"
with the same entries (in the same order) as the input tree, so it can be used
on its own or as a friend tree (T->AddFriend("T", "...fm.root")). Branches:

  copied:   runnumber, evtnumber, reco_cluster_{id,detid},
            tpc_seeds_{id,nclusters,start_idx,dedx,maxparticle_pid,maxparticle_p},
            tpc_seeds_clusters
  event:    fm_has_event       1 if the event is in the FM CSV, else 0
  cluster:  reco_cluster_fm_matched           1 if an FM point matched this cluster
            reco_cluster_fm_pid_prob_<c>      FM per-point probability (-1 if none)
            reco_cluster_fm_seg_target        FM truth track id (0 if none)
            reco_cluster_fm_pred_assignment   FM predicted track id (-1 if none)
  seed:     tpc_seeds_fm_nclusters_scored     seed clusters with an FM score
            tpc_seeds_fm_pid_prob_<c>         mean FM probability over those
                                              clusters (-1 if none)

Counter branches written by uproot: nreco_cluster, ntpc_seeds, ntpc_seeds_clusters.
"""

import argparse
import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from scipy.spatial import cKDTree

ROOT_TEMPLATE = "/sphenix/user/shuhangli/calotrack_tree/macro/condor_pid_x10/OutDir{N}/calotrkana.root"
FM_CSV = "/sphenix/u/ggalgoczi/FM-PID_ensemble_combiner_run21_25seg.csv.gz"
TPC_DETID = 2

SEED_BRANCHES = [
    "tpc_seeds_id", "tpc_seeds_nclusters", "tpc_seeds_start_idx", "tpc_seeds_dedx",
    "tpc_seeds_maxparticle_pid", "tpc_seeds_maxparticle_p",
]
ROOT_BRANCHES = [
    "runnumber", "evtnumber",
    "reco_cluster_x", "reco_cluster_y", "reco_cluster_z", "reco_cluster_E",
    "reco_cluster_detid", "reco_cluster_id", "reco_cluster_g4hit_trkid",
    *SEED_BRANCHES, "tpc_seeds_clusters",
]


def parse_segments(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def load_fm(csv_path, segments):
    header = pd.read_csv(csv_path, nrows=0).columns
    prob_cols = sorted([c for c in header if c.startswith("pid_prob_class_")],
                       key=lambda c: int(c.rsplit("_", 1)[1]))
    cols = ["segment", "entry", "x", "y", "z", "E", "seg_target", "pred_assignment"] + prob_cols
    dtypes = {c: np.float64 for c in ["x", "y", "z", "E"]}
    dtypes.update({c: np.float32 for c in prob_cols})
    dtypes.update({"segment": np.int32, "entry": np.int32,
                   "seg_target": np.int64, "pred_assignment": np.int64})
    parts = []
    for chunk in pd.read_csv(csv_path, usecols=cols, dtype=dtypes, chunksize=2_000_000):
        parts.append(chunk[chunk["segment"].isin(segments)])
        print(f"  read FM rows: {sum(len(p) for p in parts):,}", flush=True)
    fm = pd.concat(parts, ignore_index=True)
    return fm, prob_cols


def merge_event(r, pts, prob_cols, tol):
    """Return per-cluster and per-seed FM arrays for one event, plus stats."""
    n_cl = len(r["reco_cluster_id"])
    n_prob = len(prob_cols)
    matched = np.zeros(n_cl, np.int32)
    probs = np.full((n_cl, n_prob), -1.0, np.float32)
    seg = np.zeros(n_cl, np.int64)
    pred = np.full(n_cl, -1, np.int64)
    st = dict(fm_points=0, matched=0, bad_dist=0, dup=0, e_mismatch=0, trk_mismatch=0, max_dist=0.0)

    if pts is not None and len(pts):
        st["fm_points"] = len(pts)
        tpc = np.flatnonzero(r["reco_cluster_detid"] == TPC_DETID)
        R = np.c_[r["reco_cluster_x"][tpc], r["reco_cluster_y"][tpc], r["reco_cluster_z"][tpc]]
        P = pts[["x", "y", "z"]].to_numpy()
        dist, j = cKDTree(R).query(P)
        idx = tpc[j]
        ok = dist < tol
        st["bad_dist"] = int((~ok).sum())
        st["max_dist"] = float(dist.max())
        # a cluster may be claimed by only one point; drop all claims on a duplicate
        uniq, cnt = np.unique(idx[ok], return_counts=True)
        dup_clusters = uniq[cnt > 1]
        if dup_clusters.size:
            bad = np.isin(idx, dup_clusters)
            st["dup"] = int((bad & ok).sum())
            ok &= ~bad
        sel = idx[ok]
        matched[sel] = 1
        probs[sel] = pts[prob_cols].to_numpy()[ok]
        seg[sel] = pts["seg_target"].to_numpy()[ok]
        pred[sel] = pts["pred_assignment"].to_numpy()[ok]
        st["matched"] = int(ok.sum())
        st["e_mismatch"] = int((np.round(r["reco_cluster_E"][sel]) != pts["E"].to_numpy()[ok]).sum())
        st["trk_mismatch"] = int((r["reco_cluster_g4hit_trkid"][sel] != seg[sel]).sum())

    # --- per seed: mean FM probability over the seed's scored clusters --------
    ids = r["reco_cluster_id"]
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if len(np.unique(sorted_ids)) != len(sorted_ids):
        raise ValueError("reco_cluster_id not unique within event")
    keys = r["tpc_seeds_clusters"]
    cl_of_key = np.full(len(keys), -1, np.int64)
    if len(keys) and len(sorted_ids):
        pos = np.clip(np.searchsorted(sorted_ids, keys), 0, len(sorted_ids) - 1)
        found = sorted_ids[pos] == keys
        cl_of_key[found] = order[pos[found]]
    st["seed_keys"] = int(len(keys))
    st["seed_keys_missing"] = int((cl_of_key < 0).sum())

    n_seed = len(r["tpc_seeds_nclusters"])
    seed_of_key = np.repeat(np.arange(n_seed), r["tpc_seeds_nclusters"].astype(np.int64))
    key_scored = (cl_of_key >= 0) & (matched[np.maximum(cl_of_key, 0)] == 1)
    n_scored = np.bincount(seed_of_key[key_scored], minlength=n_seed).astype(np.int32)
    seed_probs = np.full((n_seed, n_prob), -1.0, np.float32)
    for c in range(n_prob):
        s = np.bincount(seed_of_key[key_scored], weights=probs[cl_of_key[key_scored], c],
                        minlength=n_seed)
        seed_probs[:, c] = np.where(n_scored > 0, s / np.maximum(n_scored, 1), -1.0)
    st["seed_keys_scored"] = int(key_scored.sum())
    return matched, probs, seg, pred, n_scored, seed_probs, st


def process_segment(N, fm_seg, prob_cols, args):
    src = args.root_template.format(N=N)
    r_all = uproot.open(src)["T"].arrays(ROOT_BRANCHES, library="np")
    n_ev = len(r_all["runnumber"])

    # consistency of the seed index branches with the cluster-key array
    groups = {e: g for e, g in fm_seg.groupby("entry")} if fm_seg is not None else {}
    if groups and max(groups) >= n_ev:
        raise ValueError(f"segment {N}: FM entry {max(groups)} >= {n_ev} tree entries")

    out_cl = {k: [] for k in ["matched", "probs", "seg", "pred"]}
    out_seed = {k: [] for k in ["n_scored", "probs"]}
    has_event = np.zeros(n_ev, np.int32)
    stats = []
    for e in range(n_ev):
        r = {k: v[e] for k, v in r_all.items()}
        ncl = r["tpc_seeds_nclusters"].astype(np.int64)
        if len(ncl) and not (np.array_equal(r["tpc_seeds_start_idx"], np.r_[0, np.cumsum(ncl)[:-1]])
                             and ncl.sum() == len(r["tpc_seeds_clusters"])):
            raise ValueError(f"segment {N} entry {e}: inconsistent seed indices")
        pts = groups.get(e)
        has_event[e] = int(pts is not None)
        m, p, s, pr, ns, sp, st = merge_event(r, pts, prob_cols, args.tol)
        out_cl["matched"].append(m); out_cl["probs"].append(p)
        out_cl["seg"].append(s); out_cl["pred"].append(pr)
        out_seed["n_scored"].append(ns); out_seed["probs"].append(sp)
        stats.append(dict(segment=N, entry=e, in_fm=int(pts is not None),
                          n_reco_clusters=len(r["reco_cluster_id"]), **st))

    def jag(lst):
        return ak.unflatten(np.concatenate(lst), np.array([len(x) for x in lst], np.int64))
    reco = {
        "id": jag(list(r_all["reco_cluster_id"])),
        "detid": jag(list(r_all["reco_cluster_detid"])),
        "fm_matched": jag(out_cl["matched"]),
        "fm_seg_target": jag(out_cl["seg"]),
        "fm_pred_assignment": jag(out_cl["pred"]),
    }
    for c, col in enumerate(prob_cols):
        reco[f"fm_pid_prob_{col.rsplit('_', 1)[1]}"] = jag([p[:, c] for p in out_cl["probs"]])
    seeds = {b[len("tpc_seeds_"):]: jag(list(r_all[b])) for b in SEED_BRANCHES}
    seeds["fm_nclusters_scored"] = jag(out_seed["n_scored"])
    for c, col in enumerate(prob_cols):
        seeds[f"fm_pid_prob_{col.rsplit('_', 1)[1]}"] = jag([p[:, c] for p in out_seed["probs"]])

    out_path = os.path.join(args.outdir, f"OutDir{N}_calotrkana_fm.root")
    data = {
        "runnumber": r_all["runnumber"].astype(np.int32),
        "evtnumber": r_all["evtnumber"].astype(np.int32),
        "fm_has_event": has_event,
        "reco_cluster": ak.zip(reco),
        "tpc_seeds": ak.zip(seeds),
        "tpc_seeds_clusters": jag(list(r_all["tpc_seeds_clusters"])),
    }
    # mktree writes a classic TTree (plain assignment can produce an RNTuple),
    # so the output works with TTree::AddFriend; fields are named <outer>_<inner>.
    with uproot.recreate(out_path) as fout:
        fout.mktree("T", {k: (v.dtype if isinstance(v, np.ndarray) else v.type.content)
                          for k, v in data.items()})
        fout["T"].extend(data)
    return out_path, pd.DataFrame(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fm-csv", default=FM_CSV)
    ap.add_argument("--root-template", default=ROOT_TEMPLATE,
                    help="input file pattern with {N} for the segment")
    ap.add_argument("--segments", default="0-24", help="e.g. 0-24 or 0,3,5-7")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tol", type=float, default=1e-3, help="max match distance [cm]")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    segments = parse_segments(args.segments)
    print(f"[FM] loading {args.fm_csv} (segments {segments[0]}..{segments[-1]}) ...")
    fm, prob_cols = load_fm(args.fm_csv, segments)
    by_seg = {s: g for s, g in fm.groupby("segment")}

    all_stats = []
    for N in segments:
        path, st = process_segment(N, by_seg.get(N), prob_cols, args)
        all_stats.append(st)
        t = st.sum(numeric_only=True)
        print(f"[seg {N:2d}] events in FM {int(t.in_fm)}/{len(st)} | points matched "
              f"{int(t.matched):,}/{int(t.fm_points):,} (bad dist {int(t.bad_dist)}, dup {int(t.dup)}, "
              f"E mismatch {int(t.e_mismatch)}, trk mismatch {int(t.trk_mismatch)}) | seed clusters "
              f"scored {int(t.seed_keys_scored):,}/{int(t.seed_keys):,} -> {path}", flush=True)

    stats = pd.concat(all_stats, ignore_index=True)
    stats_path = os.path.join(args.outdir, "merge_stats.csv")
    stats.to_csv(stats_path, index=False)
    t = stats.sum(numeric_only=True)
    print("\n=== total ===")
    print(f"events in FM: {int(t.in_fm):,}/{len(stats):,}")
    print(f"FM points matched: {int(t.matched):,}/{int(t.fm_points):,}; bad distance {int(t.bad_dist)}, "
          f"duplicates {int(t.dup)}, E mismatch {int(t.e_mismatch)}, track-id mismatch {int(t.trk_mismatch)}, "
          f"max distance {stats.max_dist.max():.2e} cm")
    print(f"seed cluster keys missing from reco_cluster_id: {int(t.seed_keys_missing)}")
    print(f"seed clusters with an FM score: {int(t.seed_keys_scored):,}/{int(t.seed_keys):,} "
          f"({t.seed_keys_scored / max(t.seed_keys, 1):.3f})")
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
