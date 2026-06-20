#!/usr/bin/env bash
# ============================================================================
# run_ehri.sh — full EHRI real-data sweep, end to end, in one go.
#   raw testimony -> DSPy fact base -> validate -> grounded generate + score
#   -> aggregate metrics (per arm x source x relations-ablation).
#
# DIAGNOSTIC ONLY. Every narrative is machine-generated and UNVERIFIED; this
# script reports AGGREGATE metrics, not the prose (it runs realdata_generate in
# --summary mode), keeping the footprint on real victim testimony minimal.
#
# Scale via env vars (defaults = a modest "larger than 3" run):
#   EXTRACT_LIMIT  chunks to extract into the fact base        (default 200)
#   GEN_LIMIT      records scored per arm/source               (default 100)
#   ADAPTERS       arms to run                                 (default the 3)
#   BACKEND        extraction LM backend (uva = no GPU/quota)  (default uva)
#   SKIP_EXTRACT=1 reuse an existing data/EHRI/iob/extracted.jsonl
#   SFT            SFT adapter                                 (default sft-narrative-adapter)
#
# Examples:
#   bash run_ehri.sh                                   # modest sweep
#   EXTRACT_LIMIT=905 GEN_LIMIT=400 bash run_ehri.sh   # whole corpus
#   SKIP_EXTRACT=1 ADAPTERS=grpo-composite-adapter bash run_ehri.sh   # reuse fact base, one arm
#
# Note: keep BACKEND=uva so extraction (API) and generation (GPU) don't fight
# over the A10. If you use vllm for extraction, stop it before the generate step.
# ============================================================================
set -uo pipefail

PY=${PYTHON:-python}
BACKEND=${BACKEND:-uva}
EXTRACT_LIMIT=${EXTRACT_LIMIT:-200}
GEN_LIMIT=${GEN_LIMIT:-100}
SFT=${SFT:-sft-narrative-adapter}
ADAPTERS=${ADAPTERS:-grpo-composite-adapter grpo-linguistic-adapter grpo-human-adapter}
SKIP_EXTRACT=${SKIP_EXTRACT:-0}

mkdir -p logs
OUT=logs/ehri_sweep_$(date +%Y%m%d_%H%M).txt

{
  echo "# EHRI sweep | backend=$BACKEND extract=$EXTRACT_LIMIT gen=$GEN_LIMIT | $(date)"
  echo "# DIAGNOSTIC — machine-generated, UNVERIFIED. Aggregate metrics only."

  if [ "$SKIP_EXTRACT" != "1" ]; then
    echo; echo "===== 1. EXTRACT fact base (DSPy / $BACKEND) ====="
    $PY dspy_extract.py --corpus EHRI --source xml --backend "$BACKEND" --limit "$EXTRACT_LIMIT" \
      || echo "(extraction failed — check backend/key)"
  else
    echo; echo "===== 1. EXTRACT skipped (SKIP_EXTRACT=1) ====="
  fi
  echo "extracted records: $(wc -l < data/EHRI/iob/extracted.jsonl 2>/dev/null || echo 0)"

  echo; echo "===== 2. VALIDATE extraction vs gold ====="
  $PY validate_extraction.py --backend "$BACKEND" --limit 10 || echo "(validation skipped/failed)"

  echo; echo "===== 3. GENERATE + SCORE (aggregate; --summary) ====="
  for A in $ADAPTERS; do
    for SRC in xml extracted; do
      echo; echo "----- arm=$A source=$SRC -----"
      $PY realdata_generate.py --corpus EHRI --source "$SRC" --sft_adapter "$SFT" \
          --adapter "$A" --limit "$GEN_LIMIT" --summary \
        || echo "(run failed: $A/$SRC)"
    done
    echo; echo "----- arm=$A source=extracted --relations (analysis: relations ON) -----"
    $PY realdata_generate.py --corpus EHRI --source extracted --relations --sft_adapter "$SFT" \
        --adapter "$A" --limit "$GEN_LIMIT" --summary \
      || echo "(run failed: $A/extracted/no_rel)"
  done
} 2>&1 | tee "$OUT"

echo
echo "================  SWEEP COMPLETE  ================"
echo "full log: $OUT"
echo
echo "Aggregate results (the comparison you want):"
grep -h "=== MEAN" "$OUT" || echo "(no MEAN rows — check the log for errors)"
