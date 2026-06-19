#!/usr/bin/env bash
# ============================================================================
# run_overnight.sh — one-shot overnight run, all outputs collected for analysis.
#
#   1. EXTRACT a fact base from raw EHRI (DSPy / uva)
#   2. VALIDATE the extractor vs gold
#   3. (optional) TRAIN GRPO on REAL prompts (composite + linguistic arms)
#   4. METRICS sweep — every arm x source x relations-ablation, aggregate (--summary)
#   5. HISTORIAN EXAMPLES — full-prose best-of-N + trained-arm samples (small N)
#   6. SUMMARY.md — index + all the MEAN rows in one place
#
# Everything lands under  results/overnight_<timestamp>/  for easy analysis.
# DIAGNOSTIC: all narratives are machine-generated, redacted, UNVERIFIED. The
# bulk sweep reports metrics only; only a few example narratives are kept.
#
# Scale / toggles via env (defaults = a sensible overnight run):
#   EXTRACT_LIMIT=400  GEN_LIMIT=150  EXAMPLE_LIMIT=5  K=5
#   TRAIN=1  TRAIN_PROMPTS=100  TRAIN_STEPS=300  NUMGEN=8
#   BACKEND=uva  GENBACKEND=uva  SKIP_EXTRACT=0  SFT=sft-narrative-adapter
#   ADAPTERS="grpo-composite-adapter grpo-linguistic-adapter grpo-human-adapter"
#
#   bash run_overnight.sh                          # full overnight run
#   TRAIN=0 EXTRACT_LIMIT=200 bash run_overnight.sh # no training, smaller
# Keep BACKEND/GENBACKEND=uva so API work and the GPU don't collide.
# ============================================================================
set -uo pipefail

PY=${PYTHON:-python}
BACKEND=${BACKEND:-uva}             # extraction LM
GENBACKEND=${GENBACKEND:-uva}       # best-of-N generation LM (pluggable: any current model)
EXTRACT_LIMIT=${EXTRACT_LIMIT:-400}
GEN_LIMIT=${GEN_LIMIT:-150}
EXAMPLE_LIMIT=${EXAMPLE_LIMIT:-5}
K=${K:-5}
TRAIN=${TRAIN:-1}
TRAIN_PROMPTS=${TRAIN_PROMPTS:-100}
TRAIN_STEPS=${TRAIN_STEPS:-300}
NUMGEN=${NUMGEN:-8}
SFT=${SFT:-sft-narrative-adapter}
SKIP_EXTRACT=${SKIP_EXTRACT:-0}
ADAPTERS=${ADAPTERS:-grpo-composite-adapter grpo-linguistic-adapter grpo-human-adapter}

TS=$(date +%Y%m%d_%H%M)
R=results/overnight_$TS
mkdir -p "$R"

{
echo "############################################################"
echo "# OVERNIGHT RUN $TS"
echo "# backend=$BACKEND gen=$GENBACKEND extract=$EXTRACT_LIMIT gen_limit=$GEN_LIMIT"
echo "# train=$TRAIN steps=$TRAIN_STEPS prompts=$TRAIN_PROMPTS  | $(date)"
echo "# DIAGNOSTIC — machine-generated, UNVERIFIED outputs."
echo "############################################################"

echo; echo "===== 0. ENV ====="
$PY -c "import torch;print('cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 || true
$PY -m ingest EHRI --source xml --list 2>&1 || true

# ---- 1. EXTRACT --------------------------------------------------------------
echo; echo "===== 1. EXTRACT fact base ($BACKEND, limit=$EXTRACT_LIMIT) ====="
if [ "$SKIP_EXTRACT" != "1" ]; then
  $PY dspy_extract.py --corpus EHRI --source xml --backend "$BACKEND" --limit "$EXTRACT_LIMIT" \
    2>&1 | tee "$R/01_extract.txt" || echo "(extract failed)"
else
  echo "skipped (SKIP_EXTRACT=1)"
fi
echo "extracted records: $(wc -l < data/EHRI/iob/extracted.jsonl 2>/dev/null || echo 0)"

# ---- 2. VALIDATE -------------------------------------------------------------
echo; echo "===== 2. VALIDATE extraction vs gold ====="
$PY validate_extraction.py --backend "$BACKEND" --limit 10 2>&1 | tee "$R/02_validation.txt" \
  || echo "(validation skipped/failed)"

# ---- 3. TRAIN on real prompts (optional) ------------------------------------
REAL_ARMS=""
if [ "$TRAIN" = "1" ]; then
  for Rw in composite linguistic; do
    echo; echo "===== 3. TRAIN GRPO on REAL prompts — $Rw ====="
    $PY grpo_train.py --reward "$Rw" --prompts corpus:EHRI/extracted \
        --prompt_limit "$TRAIN_PROMPTS" --init_adapter "$SFT" \
        --max_steps "$TRAIN_STEPS" --num_generations "$NUMGEN" \
        2>&1 | tee "$R/03_train_${Rw}_real.log" || echo "(train failed: $Rw)"
    [ -d "grpo-${Rw}-real-adapter" ] && REAL_ARMS="$REAL_ARMS grpo-${Rw}-real-adapter"
  done
fi

# ---- 4. METRICS sweep --------------------------------------------------------
echo; echo "===== 4. METRICS sweep (aggregate) ====="
{
  for A in $ADAPTERS $REAL_ARMS; do
    for SRC in xml extracted; do
      echo; echo "----- arm=$A source=$SRC -----"
      $PY realdata_generate.py --corpus EHRI --source "$SRC" --sft_adapter "$SFT" \
          --adapter "$A" --limit "$GEN_LIMIT" --summary || echo "(fail $A/$SRC)"
    done
    echo; echo "----- arm=$A source=extracted --no_relations -----"
    $PY realdata_generate.py --corpus EHRI --source extracted --no_relations --sft_adapter "$SFT" \
        --adapter "$A" --limit "$GEN_LIMIT" --summary || echo "(fail $A/norel)"
  done
} 2>&1 | tee "$R/04_metrics.txt"

# ---- 5. HISTORIAN examples (full prose, small N) ----------------------------
echo; echo "===== 5. HISTORIAN examples ====="
echo "--- best-of-N (pluggable LLM = $GENBACKEND), the reward-as-selector demo ---"
$PY bestofn_demo.py --corpus EHRI --source extracted --gen_backend "$GENBACKEND" \
    --k "$K" --limit "$EXAMPLE_LIMIT" --judge 2>&1 | tee "$R/05_examples_bestofn.txt" \
  || echo "(bestofn failed)"
echo "--- trained-arm sample narratives (composite vs linguistic) ---"
for A in grpo-composite-adapter grpo-linguistic-adapter $REAL_ARMS; do
  echo; echo "##### $A #####"
  $PY realdata_generate.py --corpus EHRI --source extracted --sft_adapter "$SFT" \
      --adapter "$A" --limit "$EXAMPLE_LIMIT" || echo "(fail $A)"
done 2>&1 | tee "$R/05_examples_arms.txt"

# ---- 6. SUMMARY --------------------------------------------------------------
echo; echo "===== 6. SUMMARY ====="
{
  echo "# Overnight run $TS — summary"
  echo
  echo "Config: backend=$BACKEND gen=$GENBACKEND extract=$EXTRACT_LIMIT gen_limit=$GEN_LIMIT"
  echo "        train=$TRAIN steps=$TRAIN_STEPS prompts=$TRAIN_PROMPTS"
  echo "Extracted records: $(wc -l < data/EHRI/iob/extracted.jsonl 2>/dev/null || echo 0)"
  echo
  echo "## Files"
  echo "- 01_extract.txt           DSPy extraction log"
  echo "- 02_validation.txt        extractor vs gold (entity/relation F1)"
  echo "- 03_train_*_real.log      GRPO-on-real-prompts training logs (if TRAIN=1)"
  echo "- 04_metrics.txt           the aggregate sweep (MEAN rows below)"
  echo "- 05_examples_bestofn.txt  reward-as-selector demo (full prose, for historians)"
  echo "- 05_examples_arms.txt     trained-arm sample narratives"
  echo "- run.log                  everything"
  echo
  echo "## Aggregate metrics (every MEAN row)"
  echo '```'
  grep -h "=== MEAN" "$R/04_metrics.txt" 2>/dev/null || echo "(none — check 04_metrics.txt)"
  echo '```'
  echo
  echo "## Best-of-N divergence (the historian headline)"
  grep -h "Rewards diverged" "$R/05_examples_bestofn.txt" 2>/dev/null || echo "(see 05_examples_bestofn.txt)"
  echo
  echo "All narratives are machine-generated, redacted, and UNVERIFIED — a methods"
  echo "diagnostic, not historical sources."
} | tee "$R/SUMMARY.md"

echo; echo "############################################################"
echo "# DONE. Everything under: $R"
echo "############################################################"
} 2>&1 | tee "$R/run.log"

echo
echo "Overnight run complete. Collected in: $R"
echo "Start with $R/SUMMARY.md"
