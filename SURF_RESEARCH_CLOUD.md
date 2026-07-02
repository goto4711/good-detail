# Running on SURF Research Cloud (VM + vLLM, 2× A10)

A full VM (not batch/SLURM) with two A10 GPUs (24 GB each). Two pieces:

1. **Pipeline env** — training (SFT/DPO/GRPO) + the HTTP clients. `setup_vm.sh`.
2. **vLLM server** — serves the judge / v2-rendering model on the OpenAI API.
   Runs in its **own** environment, because vLLM pins specific torch/transformers
   versions that would conflict with the TRL training stack. The pipeline only
   talks to it over HTTP (`requests`), so it never imports vLLM.

## 1. Pipeline env

```bash
bash setup_vm.sh                 # clean .venv, CUDA torch, requirements
source .venv/bin/activate
```

## 2. Serve a model with vLLM (separate env)

```bash
python -m venv ~/vllm-env && source ~/vllm-env/bin/activate
pip install vllm
# one A10 (leaves the other free for training):
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
#   both A10s for a bigger judge:  --tensor-parallel-size 2   (no GPU left to train)
```

Run it in `tmux`/`screen` so it stays up. Health check:
`curl http://localhost:8000/v1/models`.

**Port 8000 taken?** On a Jupyter-based VM, the Jupyter server usually already
listens on `127.0.0.1:8000` (don't kill it — it's your access). Serve vLLM on a
free port and point the pipeline at it with `VLLM_URL`:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8002
export VLLM_URL=http://localhost:8002        # the judge / v2 render read this
```

## 3. Point the pipeline at vLLM

```bash
source .venv/bin/activate
# validate the judge agrees with the oracle (do this first):
python llm_judge_reward.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --validate
# label a preference set / v2-render with the served model:
python llm_judge_reward.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --label --limit 8
python llm_render.py       --backend vllm --model Qwen/Qwen2.5-7B-Instruct --limit 8
# whole chain (v2 -> SFT -> GRPO) with vLLM for rendering:
BACKEND=vllm MODEL=Qwen/Qwen2.5-7B-Instruct V2_LIMIT=8 bash run_all.sh --train
```

`--model` must be the **exact** name you passed to `vllm serve`. If you started
vLLM with `--api-key`, put it in `.env` as `VLLM_API_KEY` (otherwise no key needed).

## GPU allocation with only 2× A10

The LLM judge is used **offline** (to label preferences / v2-render), not during
GRPO. So two clean options:

- **Sequential:** serve vLLM on both A10s → label/render → stop vLLM → train
  (SFT/GRPO) on the GPU(s). Simplest, no contention.
- **Concurrent:** vLLM on GPU 0 (`CUDA_VISIBLE_DEVICES=0`, `--tensor-parallel-size 1`),
  training on GPU 1 (`CUDA_VISIBLE_DEVICES=1 python grpo_train.py ...`). A
  0.5–3B model + LoRA fits comfortably on one A10.

For the real GRPO run, see the **real-run profile** in `PROJECT_STATUS_2026-06-17.md`
(model size, steps, num_generations, etc.). One A10 is plenty for a 1.5–3B LoRA run.

## Do you need 2× A10? (credits)

**No, for this project 1× A10 is enough.** Training is ≤3B + LoRA (fits 24 GB), the
vLLM judge is ≤7B (~14 GB, fits 24 GB), and the judge runs *offline* — never at the
same time as GRPO. Run sequentially on one GPU: serve vLLM → label/render → stop →
train. Reserve **A10 – 1 GPU** and save credits.

Reserve **2× A10** only if you want to (a) serve the judge *and* train concurrently,
(b) use a judge larger than ~13B (`--tensor-parallel-size 2`), or (c) ~2× labeling
throughput. Avoid RTX2080 — Turing, no bf16, 8–11 GB: too small for a 7B judge or
comfortable GRPO generation.
