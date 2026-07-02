#!/bin/bash
# ============================================================
# good-detail — run the smoke test locally (macOS / Linux, CPU)
#
#   bash run_local.sh
#
# Creates a .venv, installs deps, generates the synthetic corpus, and runs
# tiny DPO + KTO fine-tunes. Uses the Apple Silicon GPU (MPS) automatically
# if available, else CPU. First run ~10-15 min (downloads torch + the 0.5B
# model once); later runs are fast.
#
# If an MPS op ever errors, force CPU: add --cpu to the two python calls below.
#
# Needs Python 3.10 or 3.11. If your default python3 is older, install a
# newer one (Homebrew: `brew install python@3.11`) and run:
#   PYTHON=python3.11 bash run_local.sh
#
# Success = both loops run and loss is finite / generally falling. The
# model has NOT learned "good detail" at this scale -- ignore its outputs;
# this only proves the machinery and data plumbing.
# ============================================================
set -e
cd "$(dirname "$0")"

# let unsupported MPS ops fall back to CPU instead of crashing
export PYTORCH_ENABLE_MPS_FALLBACK=1

PY=${PYTHON:-python3}
echo "Using $($PY --version 2>&1)"

if [ ! -d .venv ]; then
  echo "Creating virtual environment (.venv) ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip -q
echo "Installing dependencies (first run downloads torch ~ a few hundred MB) ..."
pip install -r requirements-local.txt -q

echo -e "\n[1/3] Generating synthetic corpus ..."
python generate_synthetic_corpus.py

echo -e "\n[2/3] DPO smoke test (20 steps; MPS if available) ..."
python dpo_kto_train.py --method dpo --max_steps 20

echo -e "\n[3/3] KTO smoke test (20 steps; MPS if available) ..."
python dpo_kto_train.py --method kto --max_steps 20

echo -e "\nDone. Adapters: smoke-dpo-adapter/ , smoke-kto-adapter/"
echo "If both ran with finite, generally-falling loss, the loop is good to build on."
