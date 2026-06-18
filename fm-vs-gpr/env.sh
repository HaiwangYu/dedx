#!/bin/bash
# Source this to get the python environment used for the FM-vs-GPR comparison.
#   source fm-vs-gpr/env.sh
# Provides numpy/scipy/sklearn/pandas/matplotlib/uproot/awkward + dedx_analysis.
source /cvmfs/sphenix.sdcc.bnl.gov/alma9.2-gcc-14.2.0/opt/sphenix/core/bin/sphenix_setup.sh -n >/dev/null 2>&1
export PY=/sphenix/user/hwyu/calotrack_tree/scripts/venv-dev/bin/python3.13
echo "python: $PY"
