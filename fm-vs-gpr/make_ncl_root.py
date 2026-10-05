#!/usr/bin/env python3
"""Rebuild the combined GPR input with the seed cluster count.

`calotrkana-1M.root` was skimmed by `calotrack_tree/macro/dedx-v2/combine.C`,
which kept only dedx / maxparticle_pid / maxparticle_p. This script redoes that
skim over the same file list and additionally keeps `tpc_seeds_nclusters`, so
the comparison can apply a cluster-count cut. The output is a flat tree "T"
(one entry per seed) in the same order as the flattened original.
"""

import argparse

import awkward as ak
import numpy as np
import uproot

BRANCHES = [
    "tpc_seeds_dedx",
    "tpc_seeds_maxparticle_pid",
    "tpc_seeds_maxparticle_p",
    "tpc_seeds_nclusters",
]
DTYPES = {
    "tpc_seeds_dedx": np.float32,
    "tpc_seeds_maxparticle_pid": np.int32,
    "tpc_seeds_maxparticle_p": np.float32,
    "tpc_seeds_nclusters": np.int32,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", default="/sphenix/user/hwyu/calotrack_tree/macro/dedx-v2/dedx-1M.lst")
    ap.add_argument("--output", default="/sphenix/user/hwyu/calotrack_tree/macro/dedx-v2/calotrkana-1M-ncl.root")
    args = ap.parse_args()

    files = []
    with open(args.list) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                files.append(line)

    parts = {b: [] for b in BRANCHES}
    for i, f in enumerate(files):
        arr = uproot.open(f)["T"].arrays(BRANCHES + ["nTPCSeeds"], library="ak")
        n = ak.to_numpy(arr["nTPCSeeds"])
        for b in BRANCHES:
            # keep the first nTPCSeeds entries per event, as combine.C does
            jag = arr[b][ak.local_index(arr[b]) < n[:, None]]
            parts[b].append(ak.to_numpy(ak.flatten(jag)).astype(DTYPES[b]))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} files", flush=True)

    out = {b: np.concatenate(parts[b]) for b in BRANCHES}
    with uproot.recreate(args.output) as fout:
        fout["T"] = out
    print(f"wrote {len(out[BRANCHES[0]]):,} seeds from {len(files)} files -> {args.output}")


if __name__ == "__main__":
    main()
