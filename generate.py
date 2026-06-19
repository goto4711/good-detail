#!/usr/bin/env python3
"""
generate.py
================================================================
Qualitative half of the comparison: see (and score) what a trained model
actually WRITES for a record. Generates a micro-narrative per case/register
using the SAME open-book prompt the training used, then prints it with its
reward scores so you can compare arms side by side.

Loading the right stack (important for GRPO-from-SFT):
  - SFT baseline only:        --adapter sft-narrative-adapter
  - GRPO trained from SFT:    --sft_adapter sft-narrative-adapter --adapter grpo-composite-adapter
    (the SFT LoRA is merged into the base first, then the GRPO LoRA is applied —
     matching how grpo_train.py --init_adapter built the model)
  - raw base (no training):   (omit both)

Examples:
  python generate.py --adapter sft-narrative-adapter --limit 2
  python generate.py --sft_adapter sft-narrative-adapter --adapter grpo-composite-adapter --limit 2
  python generate.py --sft_adapter sft-narrative-adapter --adapter grpo-human-adapter --limit 2
"""

import argparse
import os
import sys

try:
    from generate_synthetic_corpus import CASES, REGISTERS, render
    from grpo_train import INSTRUCTION, _record_block, chat_prompt
    from composite_reward import composite_reward, faithfulness
    from linguistic_reward import linguistic_reward
    from human_reward import human_reward
except ImportError as e:
    sys.exit(f"Run from the project folder: {e}")


def _scores(text, case, judge_fn=None):
    F, nu = faithfulness(text, case)
    s = (f"F={F:.2f} unsup={nu}  composite={composite_reward(text, case):+.2f}  "
         f"linguistic={linguistic_reward(text):.2f}  human={human_reward(text, case):+.2f}")
    if judge_fn is not None:
        try:
            s += f"  judge={judge_fn(text, case):+.2f}"
        except Exception:
            s += "  judge=ERR"   # backend down / parse fail — keep the rest of the row
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from config import DEFAULT_MODEL
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sft_adapter", default=None, help="SFT adapter to MERGE first (GRPO-from-SFT)")
    ap.add_argument("--adapter", default=None, help="adapter to APPLY (the trained arm)")
    ap.add_argument("--limit", type=int, default=2, help="number of cases")
    ap.add_argument("--register", choices=["testimony", "finding_aid", "both"], default="testimony")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--judge", action="store_true",
                    help="also score each output with the LLM judge (RLAIF column; needs a backend up)")
    ap.add_argument("--judge_backend", default="vllm", help="judge backend (vllm/ollama/gemini/uva)")
    ap.add_argument("--judge_model", default=None, help="judge model (default: backend default)")
    args = ap.parse_args()

    judge_fn = None
    if args.judge:
        import llm_judge_reward as J
        J._BACKEND = args.judge_backend
        jmodel = args.judge_model or J.BACKENDS[args.judge_backend]["default_model"]
        judge_fn = lambda text, case: J.llm_judge_reward(text, case, model=jmodel)
        print(f"LLM judge: backend={args.judge_backend} model={jmodel}")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall: pip install -r requirements-local.txt")

    use_cuda = torch.cuda.is_available() and not args.cpu
    use_mps = (not use_cuda) and (not args.cpu) and \
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if use_cuda else torch.float32)

    label = "base"
    if args.sft_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()
        label = f"sft({args.sft_adapter})"
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        label = f"{label}+{args.adapter}" if args.sft_adapter else args.adapter
    model.to(device).eval()
    print(f"Device: {device} | model: {label}\n")

    registers = REGISTERS if args.register == "both" else [args.register]
    for case in CASES[:args.limit]:
        for reg in registers:
            instr = INSTRUCTION.format(register=reg.replace("_", "-"),
                                       unit=case["entities"][case["focal"]]["name"],
                                       record=_record_block(case))
            prompt = chat_prompt(tok, instr)   # templated -> model emits <|im_end|> and stops
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=(args.temperature > 0), temperature=args.temperature or None,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"=== {case['case_id']} / {reg} — {case['entities'][case['focal']]['name']} ===")
            print(text)
            print(f"    [{_scores(text, case, judge_fn)}]\n")


if __name__ == "__main__":
    main()
