#!/usr/bin/env python3
"""
sft_narrative.py
================================================================
SFT BASELINE — train a model to WRITE the micro-narratives (Stage 0).

Until now nothing was trained to write narratives: the grid text is templated,
and GRPO ran on the raw instruct model. This SFTs the base model on
(record-prompt -> grounded micro-narrative) pairs, giving a sane starting
policy that DPO/KTO/GRPO then refine.

SFT data is already in the pipeline: the **"good" renderings** ARE the targets,
paired with the same open-book record prompt GRPO uses (imported from
grpo_train, so the SFT model expects the exact prompt format GRPO feeds).

Honest caveats (this is a *baseline*, not the finished model):
  - Targets are TEMPLATED "good" renderings -> the SFT model will inherit their
    somewhat stilted, repetitive style. Fine as a starting policy; the rewards
    refine from there. For a real run, use varied / historian-written / LLM-
    rewritten-and-verified targets.
  - Only 16 examples (8 cases x 2 registers) -> tiny. Expand the cases for real.

Usage:
  python sft_narrative.py --dry_run            # data plumbing, no torch
  python sft_narrative.py --epochs 3           # real run (MPS/CUDA/CPU)
Then start GRPO/DPO from the adapter:  ./sft-narrative-adapter
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

try:
    from generate_synthetic_corpus import CASES, REGISTERS, render
    from grpo_train import INSTRUCTION, _record_block, chat_prompt
except ImportError as e:
    sys.exit(f"Run from the project folder (needs generate_synthetic_corpus.py, grpo_train.py): {e}")

HERE = Path(__file__).parent
OUT = HERE / "sft-narrative-adapter"


def build_sft_dataset(source="template", tok=None):
    """(record prompt -> good micro-narrative) pairs, prompt format == GRPO's.
    `tok` applies the model's chat template (native EOS); pass it for the real
    run so the SFT prompts match what GRPO/generate feed. source='llm' uses the
    v2 grounding-verified renderings (llm_render.py), falling back to templated
    'good' for any cell not present."""
    llm = {}
    if source == "llm":
        path = HERE / "synthetic_corpus" / "llm_renderings.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    if d.get("style") == "good" and d.get("accepted"):
                        llm[(d["case_id"], d["register"])] = d["text"]
            print(f"Using {len(llm)} LLM-verified 'good' targets (rest templated).")
        else:
            print("(no llm_renderings.jsonl — run llm_render.py first; using templated targets)")
    rows = []
    for case in CASES:
        for register in REGISTERS:
            instr = INSTRUCTION.format(register=register.replace("_", "-"),
                                       unit=case["entities"][case["focal"]]["name"],
                                       record=_record_block(case))
            comp = llm.get((case["case_id"], register)) or render(case, register, "good")
            rows.append({"prompt": chat_prompt(tok, instr), "completion": " " + comp})
    return rows


def dry_run(args):
    ds = build_sft_dataset(args.source)
    print(f"SFT dataset ({args.source} targets): {len(ds)} (record -> good narrative) pairs.")
    print(f"  prompt chars   : min {min(len(r['prompt']) for r in ds)}, "
          f"median {int(statistics.median(len(r['prompt']) for r in ds))}, "
          f"max {max(len(r['prompt']) for r in ds)}")
    print(f"  completion chars: min {min(len(r['completion']) for r in ds)}, "
          f"median {int(statistics.median(len(r['completion']) for r in ds))}, "
          f"max {max(len(r['completion']) for r in ds)}")
    ex = ds[0]
    print("\n--- example pair ---")
    print("PROMPT (tail):", ex["prompt"][-200:].replace("\n", " "))
    print("COMPLETION   :", ex["completion"].strip()[:200], "...")
    print("\nDRY RUN OK — pairs build and use GRPO's prompt format. Train without --dry_run.")


def real_run(args):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall: pip install -r requirements-local.txt")

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    use_mps = (not use_cuda) and (not args.cpu) and \
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    use_cpu = not (use_cuda or use_mps)
    print(f"Device: {'cuda' if use_cuda else ('mps' if use_mps else 'cpu')}  |  model: {args.model}  "
          f"|  epochs: {args.epochs}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if use_cuda else torch.float32)

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    # End every target with EOS so the policy LEARNS TO STOP. Without this the
    # model never emits an end token in this prompt format and rambles to the
    # length cap (clipped_ratio=1), which swamps the GRPO reward with the length
    # penalty and prevents any arm from showing grounded behaviour.
    rows = build_sft_dataset(args.source, tok)
    for r in rows:
        if not r["completion"].rstrip().endswith(tok.eos_token):
            r["completion"] = r["completion"].rstrip() + tok.eos_token
    ds = Dataset.from_list(rows)

    cfg = SFTConfig(
        output_dir=str(OUT),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=use_cuda,
        use_cpu=use_cpu,
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    print(f"Pairs: {len(ds)} | starting SFT ...")
    result = trainer.train()
    print("\nMetrics:", result.metrics, "(loss should fall as it learns the format)")
    trainer.save_model(str(OUT))
    print(f"Saved SFT adapter -> {OUT}")
    print("Use it as the starting policy for GRPO/DPO (load base + this adapter).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from config import DEFAULT_MODEL
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--source", choices=["template", "llm"], default="template",
                    help="SFT targets: templated 'good', or v2 grounding-verified (llm_render.py)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    dry_run(args) if args.dry_run else real_run(args)


if __name__ == "__main__":
    main()
