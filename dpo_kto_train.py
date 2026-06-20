#!/usr/bin/env python3
"""
dpo_kto_train.py
================================================================
Preference-optimisation training (DPO / KTO) for the "good detail" pipeline.

Consumes a preference_pairs.jsonl of {prompt, chosen, rejected} — the synthetic
good-vs-flattened/fabricated set, or a real one built by build_training_data.py
(reward- / human- / authored-labelled) — and runs a DPO or KTO fine-tune of a
small Qwen via the TRL >= 1.0 API.

Defaults are deliberately small (a handful of steps, a small model) so you can
first confirm the loop runs and the data maps cleanly onto TRL's schema; scale up
via --max_steps / --model for a real experiment.

Two modes
---------
  --dry_run   : stdlib only. Loads the pairs, builds the exact DPO and KTO
                schemas, validates them, prints stats. Runs anywhere, no torch.
                USE THIS to verify the data plumbing before touching a GPU.

  (default)   : real run. Needs the pinned stack (requirements-posttrain.txt)
                and downloads the base model. Run on Snellius / a GPU box.

Examples
--------
  python dpo_kto_train.py --dry_run --method dpo
  python dpo_kto_train.py --dry_run --method kto
  python dpo_kto_train.py --method dpo --max_steps 20      # on GPU
  python dpo_kto_train.py --method kto --max_steps 20 --model Qwen/Qwen2.5-0.5B-Instruct

Note on TRL versions: the >=1.0 API uses per-trainer config classes
(DPOConfig / KTOConfig) and `processing_class=` for the tokenizer. Exact
argument names can still shift point-release to point-release -- if an arg is
rejected, check the docs for the version you pinned.
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_PAIRS = HERE / "synthetic_corpus" / "preference_pairs.jsonl"


# ----------------------------------------------------------------------
# Data loading + schema construction (stdlib only)
# ----------------------------------------------------------------------

def load_pairs(path: Path):
    pairs = []
    with path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for k in ("prompt", "chosen", "rejected"):
                if k not in d or not str(d[k]).strip():
                    raise ValueError(f"line {ln}: missing/empty '{k}'")
            pairs.append(d)
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def build_dpo(pairs):
    """DPO schema: one row per pair -> {prompt, chosen, rejected}."""
    return [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
            for p in pairs]


def build_kto(pairs):
    """KTO schema: {prompt, completion, label(bool)}.
    Each 'good' (chosen) becomes a positive once per prompt; every 'rejected'
    becomes a negative. Dedup positives so a good answer isn't double-counted
    when it appears in both the good>flattened and good>fabricated pairs."""
    rows, seen_pos = [], set()
    for p in pairs:
        key = (p["prompt"], p["chosen"])
        if key not in seen_pos:
            seen_pos.add(key)
            rows.append({"prompt": p["prompt"], "completion": p["chosen"], "label": True})
        rows.append({"prompt": p["prompt"], "completion": p["rejected"], "label": False})
    return rows


# ----------------------------------------------------------------------
# Dry run: validate plumbing without torch/trl
# ----------------------------------------------------------------------

def _lens(strings):
    ls = [len(s) for s in strings]
    return f"min {min(ls)}, median {int(statistics.median(ls))}, max {max(ls)} chars"


def dry_run(pairs, method):
    print(f"Loaded {len(pairs)} preference pairs from {DEFAULT_PAIRS.name}")
    by_kind = Counter(p.get("rejected_kind", "?") for p in pairs)
    print(f"  rejected_kind breakdown: {dict(by_kind)}")

    if method == "dpo":
        ds = build_dpo(pairs)
        print(f"\nDPO dataset: {len(ds)} rows, columns = {list(ds[0].keys())}")
        print(f"  prompt   : {_lens([r['prompt'] for r in ds])}")
        print(f"  chosen   : {_lens([r['chosen'] for r in ds])}")
        print(f"  rejected : {_lens([r['rejected'] for r in ds])}")
        # sanity: chosen != rejected everywhere
        bad = sum(1 for r in ds if r["chosen"].strip() == r["rejected"].strip())
        print(f"  rows where chosen == rejected (should be 0): {bad}")
        ex = ds[0]
        print("\n--- example DPO row ---")
        print("PROMPT:  ", ex["prompt"].replace("\n", " ")[:160], "...")
        print("CHOSEN:  ", ex["chosen"][:160], "...")
        print("REJECTED:", ex["rejected"][:160], "...")
    else:
        ds = build_kto(pairs)
        labels = Counter(r["label"] for r in ds)
        print(f"\nKTO dataset: {len(ds)} rows, columns = {list(ds[0].keys())}")
        print(f"  label balance: True(desirable)={labels[True]}  False(undesirable)={labels[False]}")
        print(f"  prompt    : {_lens([r['prompt'] for r in ds])}")
        print(f"  completion: {_lens([r['completion'] for r in ds])}")
        pos = next(r for r in ds if r["label"])
        neg = next(r for r in ds if not r["label"])
        print("\n--- example KTO rows ---")
        print(f"[+] {pos['completion'][:150]} ...")
        print(f"[-] {neg['completion'][:150]} ...")

    print("\nDRY RUN OK -- data maps cleanly onto the TRL", method.upper(), "schema.")
    print("Run without --dry_run on a GPU box (pinned stack) to train.")


# ----------------------------------------------------------------------
# Real run: tiny DPO / KTO training via TRL >= 1.0
# ----------------------------------------------------------------------

def real_run(pairs, args):
    # let unsupported MPS ops fall back to CPU instead of crashing (Apple Silicon)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from peft import LoraConfig
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall the stack first, e.g. "
                 f"pip install -r requirements-local.txt")

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    use_mps = (not use_cuda) and (not args.cpu) and \
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    use_cpu = not (use_cuda or use_mps)
    device_msg = "cuda" if use_cuda else ("mps (Apple Silicon GPU)" if use_mps else "cpu")
    print(f"Device: {device_msg}  |  model: {args.model}  |  method: {args.method}  |  steps: {args.max_steps}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
    )

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])

    # KTO requires an ACTUAL per-device batch size > 1 (its KL term pairs
    # mismatched prompts/completions within the batch). DPO is fine with 1.
    batch_size = 2 if args.method == "kto" else 1
    common = dict(
        output_dir=str(HERE / f"{args.method}-adapter"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        max_steps=args.max_steps,
        learning_rate=5e-6,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=use_cuda,          # bf16 only on CUDA; MPS/CPU use fp32
        use_cpu=use_cpu,        # False -> Trainer auto-picks CUDA or MPS
        seed=args.seed,
    )

    # NOTE: we pass only stable args (beta + standard TrainingArguments).
    # Length caps like max_length / max_prompt_length have moved/been renamed
    # across TRL versions; our sequences are short, so we rely on defaults.
    if args.method == "dpo":
        from trl import DPOConfig, DPOTrainer
        ds = Dataset.from_list(build_dpo(pairs))
        cfg = DPOConfig(beta=0.1, **common)
        trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    else:
        from trl import KTOConfig, KTOTrainer
        ds = Dataset.from_list(build_kto(pairs))
        cfg = KTOConfig(beta=0.1, **common)
        trainer = KTOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)

    print(f"Dataset rows: {len(ds)}  | starting {args.method.upper()} ...")
    result = trainer.train()
    print("\nTraining metrics:", result.metrics)
    out = HERE / f"{args.method}-adapter"
    trainer.save_model(str(out))
    print(f"Saved adapter to: {out}")
    print(f"{args.method.upper()} training complete. (Loss should be finite and generally trending down.)")


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--method", choices=["dpo", "kto"], default="dpo")
    from config import DEFAULT_MODEL
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    ap.add_argument("--dry_run", action="store_true", help="validate data plumbing only (no torch)")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs)
    if args.dry_run:
        dry_run(pairs, args.method)
    else:
        real_run(pairs, args)


if __name__ == "__main__":
    main()
