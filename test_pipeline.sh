#!/bin/bash
# ============================================================
# Test the "good detail" pipeline end to end — Tier A only.
# No GPU, no torch: pure stdlib, runs in seconds. This validates the
# DATA + REWARD pipeline (not the training, which needs torch — see below).
#
#   bash test_pipeline.sh
#
# For the training smoke tests (need torch on MPS/CUDA):
#   python dpo_kto_train.py --method dpo --max_steps 20
#   python grpo_train.py --reward composite --max_steps 20
#   python grpo_train.py --reward human --persona grounding_first --max_steps 20
# ============================================================
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

step () { printf '\n\033[1m========== %s ==========\033[0m\n' "$1"; }
run_tier_a_reward () {
  local module="$1"
  "$PY" - <<PY
import config
config.FAITHFULNESS_METHOD = "lexical"
import ${module}
${module}.main()
PY
}

step "1/4  GENERATE + SERIALISE + INGEST + dual-reward (run_pipeline.py)"
"$PY" run_pipeline.py

step "2/4  LINGUISTIC reward — per-feature blindness check"
"$PY" linguistic_reward.py

step "3/4  COMPOSITE reward — faithfulness-gated (Tier A smoke uses lexical faithfulness)"
run_tier_a_reward composite_reward

step "4/4  SYNTHETIC HUMAN arm + pluralism (two-arm comparison)"
run_tier_a_reward human_reward

printf '\n\033[1mTier A complete.\033[0m Data + all four reward arms validated on synthetic data.\n'
printf 'Next (needs torch): the training smoke tests listed at the top of this script.\n'
