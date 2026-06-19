#!/bin/bash
# ============================================================
# "Good detail" — the WHOLE pipeline, bringing every stage together.
#
# TIER A (no GPU, no torch — seconds): data + serialise + ingest + all reward
#         arms. This is the part that validates the pipeline.
# TIER B (needs torch on MPS/CUDA, ~minutes each): v2 LLM rendering, SFT on the
#         verified targets, then GRPO from the SFT baseline for each reward arm.
#
#   bash run_all.sh            # Tier A only (safe everywhere)
#   bash run_all.sh --train    # Tier A + Tier B (needs torch; MPS/CUDA)
#
# For an LLM backend (v2 rendering / judge) pick one: ollama (local) or gemini.
# ============================================================
set -e
cd "$(dirname "$0")"
# Prefer 'python' (usually the conda/venv interpreter that has torch+datasets);
# fall back to python3. Override with PYTHON=... if needed.
PY=${PYTHON:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
BACKEND=${BACKEND:-ollama}          # ollama (local) | gemini | uva
MODEL=${MODEL:-llama3.2}
V2_LIMIT=${V2_LIMIT:-2}             # cases for v2 LLM rendering (keep small — it's slow);
                                    # SFT --source llm uses these + templated for the rest

step () { printf '\n========== %s ==========\n' "$1"; }

# Capture everything to a timestamped log AND show it on screen.
mkdir -p logs
LOG="logs/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "Logging to:  $LOG"
echo "Interpreter: $($PY -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PY")"
echo "Started:     $(date)"

# ---------- TIER A : data + rewards (no torch) ----------
step "A1  pipeline: generate -> TEI/IOB -> ingest -> oracle vs linguistic"
"$PY" run_pipeline.py
step "A2  linguistic reward (per-feature blindness)"; "$PY" linguistic_reward.py
step "A3  composite reward (faithfulness-gated; good>fab 16/16)"; "$PY" composite_reward.py
step "A4  synthetic human arm + persona pluralism"; "$PY" human_reward.py

if [ "$1" != "--train" ]; then
  printf '\nTier A complete.  Add --train for the torch stages (SFT -> GRPO).\n'
  echo "Log saved to: $LOG"
  exit 0
fi

# ---------- TIER B : the model-training chain (needs torch) ----------
if ! "$PY" -c "import torch, datasets, transformers, trl, peft" 2>/dev/null; then
  echo "ERROR: the training stack (torch/datasets/transformers/trl/peft) is not importable with:"
  echo "  $($PY -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PY")"
  echo "Use the interpreter that has it (the one you ran DPO/GRPO with), e.g.:"
  echo "  PYTHON=python bash run_all.sh --train"
  echo "or install it:  $PY -m pip install -r requirements-local.txt"
  exit 1
fi

step "B1  v2 rendering: LLM writes narratives, faithfulness verifies grounding"
"$PY" llm_render.py --backend "$BACKEND" --model "$MODEL" --limit "$V2_LIMIT" --max-tries 2 || \
  echo "(v2 render skipped/failed — SFT will fall back to templated targets)"

step "B2  SFT baseline on the v2 (grounding-verified) good narratives"
"$PY" sft_narrative.py --source llm --epochs 3

step "B3  GRPO from the SFT baseline — composite (faithfulness-gated) reward"
"$PY" grpo_train.py --reward composite --init_adapter sft-narrative-adapter --max_steps 20

step "B4  GRPO from the SFT baseline — human arm (grounding_first persona)"
"$PY" grpo_train.py --reward human --persona grounding_first --init_adapter sft-narrative-adapter --max_steps 20

printf '\nFull pipeline complete.  SFT -> GRPO ran for two reward arms from the same baseline.\n'
printf 'Compare arms by training a third (--reward linguistic) and inspecting generations.\n'
echo "Finished:   $(date)"
echo "Log saved to: $LOG"
