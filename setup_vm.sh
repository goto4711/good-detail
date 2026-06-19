#!/bin/bash
# ============================================================
# Set up the PIPELINE environment on a CUDA VM (e.g. SURF Research Cloud,
# 2x A10). This is the training + client env. vLLM runs in a SEPARATE env
# (see SURF_RESEARCH_CLOUD.md) so its pinned torch/transformers don't clash
# with TRL.
#
#   bash setup_vm.sh
#   source .venv/bin/activate
# ============================================================
set -e
cd "$(dirname "$0")"
PYBIN=${PYBIN:-python3}
echo "Python: $($PYBIN --version 2>&1)"

$PYBIN -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CUDA torch for the A10s (Ampere). torch >= 2.6 is REQUIRED: current TRL's GRPO
# trainer imports FSDPModule (FSDP2), which older torch doesn't export. cu124
# wheels; bump to a newer cuXXX only if the driver needs it.
pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6,<2.9"
pip install -r requirements-local.txt

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  GPU", i, torch.cuda.get_device_name(i))
EOF

echo
echo "Pipeline env ready.  Activate with:  source .venv/bin/activate"
echo "Serve the judge/v2 model with vLLM (separate env) — see SURF_RESEARCH_CLOUD.md"
