#!/usr/bin/env python3
"""
grpo_train.py
================================================================
GRPO smoke test for the "good detail" pipeline — the stage that trains the
model directly against a REWARD FUNCTION (not preference pairs).

Why this should hurt less than last time
----------------------------------------
1. RIGHT PARADIGM. GRPO consumes PROMPTS + reward_funcs. It does NOT use
   (chosen, rejected) pairs — that's DPO. The old run_grpo_paired.py mixed
   the two, which is a big reason it "ran but didn't train". Here the dataset
   is prompt-only and the signal comes entirely from the reward function.
2. A REAL REWARD THAT VARIES. The classic GRPO failure is a reward with no
   variance across the sampled group -> zero advantage -> no gradient -> "it
   doesn't train". `linguistic_reward` is continuous and varies completion to
   completion, so there's always a learning signal. (--dry_run proves this
   below before you spend a GPU-hour.)
3. KNOWN-GOOD STACK. Same TRL>=1.0 env the DPO/KTO smoke test already ran on.

Reward signature (TRL>=1.0): def reward(completions, **kwargs) -> list[float].
You may pass several; the total is their (weighted) sum. Here we use the
surface linguistic reward — it's computable from the generated text alone,
which is exactly what a GRPO reward must be. (The grounding/oracle component
needs the source facts; that's the next step, wired via a dataset column.)

Usage
-----
  python grpo_train.py --dry_run          # reward plumbing, no torch
  python grpo_train.py --max_steps 20     # real run (GPU/MPS)
"""

import argparse
import os
import re
import statistics
import sys

try:
    from generate_synthetic_corpus import CASES, REGISTERS, PROFILES, render
    from linguistic_reward import linguistic_reward
    from composite_reward import composite_reward_by_id
    from human_reward import human_reward_by_id, PERSONAS
except ImportError:
    sys.exit("Run from the project folder (needs generate_synthetic_corpus.py, "
             "linguistic_reward.py, composite_reward.py, human_reward.py).")

# the open-book writing instruction + base model live in config.py
from config import INSTRUCTION, DEFAULT_MODEL


def _record_block(case):
    """Compact text rendering of the fact base, so the model has something to be
    faithful TO. Without this the task is closed-book and faithfulness is
    unachievable (every generation must hallucinate)."""
    lines = []
    focal = case["entities"][case["focal"]]
    lines.append(f"Focal: {focal['name']} ({focal['type']})")
    lines.append("Entities:")
    for ent in case["entities"].values():
        attrs = "; ".join(f"{a['key']}: {a['value']}" for a in ent["attributes"])
        lines.append(f"- {ent['name']}" + (f" ({attrs})" if attrs else ""))
    lines.append("Events:")
    for ev in case["events"]:
        srcs = ",".join(ev["grounded_by"])
        lines.append(f"- {ev['summary']} [{ev['date']['value']}, {ev['date']['certainty']}; src {srcs}]")
    lines.append("Sources: " + "; ".join(
        f"{sid}={s['kind']} {s['archive_ref']}" for sid, s in case["sources"].items()))
    return "\n".join(lines)


def chat_prompt(tok, user_text):
    """Wrap the instruction in the model's OWN chat template so the policy can
    use its native turn-end token (<|im_end|>) to STOP — the root fix for the
    non-terminating, rambling completions. With tok=None (dry run) returns the
    raw text. Imported by sft_narrative.py and generate.py so all three stages
    feed the model the identical templated prompt."""
    if tok is None:
        return user_text
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True)


# Real-data prompts: GRPO needs only PROMPTS + a reward (not gold targets like SFT),
# so it can train on real records straight from the ingest layer. _REAL_RECORDS maps
# a prompt's id -> its Record, so the reward fns can score against the real source.
_REAL_RECORDS = {}


def build_prompts(tok=None, source=None, limit=None):
    """source=None -> synthetic CASES (the controlled study). source='corpus:NAME/SRC'
    -> real records via ingest.load_corpus (e.g. 'corpus:EHRI/extracted'), prompt =
    record block + a focal-relevant source excerpt, the SAME open-book format."""
    if source:
        from ingest import load_corpus, record_block
        from realdata_generate import retrieve
        spec = source.split(":", 1)[1] if ":" in source else source
        name, _, src = spec.partition("/")
        rows = []
        for rec in load_corpus(name, src or None, limit=limit):
            _REAL_RECORDS[rec.id] = rec
            focal = rec.unit or rec.title
            excerpt = " ".join(retrieve(rec.source_text, focal, 8))
            block = record_block(rec) + ("\n\nSOURCE EXCERPT:\n" + excerpt if excerpt else "")
            instr = INSTRUCTION.format(register=rec.register.replace("_", "-"),
                                       unit=focal, record=block)
            rows.append({"prompt": chat_prompt(tok, instr), "case_id": rec.id})
        return rows
    rows = []
    for case in CASES:
        for register in REGISTERS:
            instr = INSTRUCTION.format(register=register.replace("_", "-"),
                                       unit=case["entities"][case["focal"]]["name"],
                                       record=_record_block(case))
            rows.append({"prompt": chat_prompt(tok, instr), "case_id": case["case_id"]})
    return rows


def _real_composite(text, rec):
    """Composite reward on a REAL record: NLI faithfulness vs retrieved source +
    attested-entity check, gated into the same F^gamma(Q+..) formula."""
    from realdata_generate import retrieve, attested_index, _composite
    from faithfulness import nli_faithfulness
    att_t, att_y = attested_index(rec)
    prem = retrieve(rec.source_text, (rec.unit or "") + " " + text, 20)
    F, unsup = nli_faithfulness(text, prem, subject=rec.unit,
                                grounded_toks=att_t, grounded_years=att_y)
    return _composite(text, F, unsup)


def _texts(completions):
    return [c if isinstance(c, str) else c[-1]["content"] for c in completions]


def reward_linguistic(completions, **kwargs):
    """Surface-only reward. completions -> one float each."""
    return [float(linguistic_reward(t)) for t in _texts(completions)]


def reward_composite(completions, case_id=None, **kwargs):
    """Faithfulness-gated reward. Needs each completion's source case, which
    GRPO forwards from the dataset's `case_id` column (batch-aligned)."""
    texts = _texts(completions)
    if case_id is None:                       # fallback (e.g. dry run without dataset)
        return [0.0 for _ in texts]
    return [(_real_composite(t, _REAL_RECORDS[cid]) if cid in _REAL_RECORDS
             else float(composite_reward_by_id(t, cid)))
            for t, cid in zip(texts, case_id)]


_ACTIVE_PERSONA = "balanced"   # set from --persona in main()


def reward_human(completions, case_id=None, **kwargs):
    """Synthetic-HUMAN reward (oracle dimensions, persona-weighted). The human
    arm of the comparison; persona is the pluralism knob (--persona)."""
    texts = _texts(completions)
    if case_id is None:
        return [0.0 for _ in texts]
    return [float(human_reward_by_id(t, cid, persona=_ACTIVE_PERSONA))
            for t, cid in zip(texts, case_id)]


# --- LLM-as-judge reward (the RLAIF arm) ----------------------------------
_JUDGE_BACKEND = "vllm"   # set from --judge_backend in main()
_JUDGE_MODEL = None       # set from --judge_model in main() (else backend default)
_judge_fn = None


def reward_judge(completions, case_id=None, **kwargs):
    """LLM-as-judge reward — the RLAIF arm: each completion is scored by the
    rubric judge over the chosen backend. EXPENSIVE (one LLM call per
    completion), so use few steps and a fast local backend (vllm). Backend must
    be up; on any error a completion scores 0.0 so training does not crash."""
    global _judge_fn, _JUDGE_MODEL
    texts = _texts(completions)
    if case_id is None:
        return [0.0 for _ in texts]
    if _judge_fn is None:
        import llm_judge_reward as J
        J._BACKEND = _JUDGE_BACKEND
        _JUDGE_MODEL = _JUDGE_MODEL or J.BACKENDS[_JUDGE_BACKEND]["default_model"]
        _judge_fn = J.llm_judge_reward_by_id
    out = []
    for t, cid in zip(texts, case_id):
        try:
            out.append(float(_judge_fn(t, cid, model=_JUDGE_MODEL)))
        except Exception:
            out.append(0.0)
    return out


REWARDS = {"linguistic": reward_linguistic, "composite": reward_composite,
           "human": reward_human, "judge": reward_judge}

# --- length / termination guard -------------------------------------------
# The reward pays for detail, so GRPO tends to ramble to the cap and never emit
# EOS (observed: completions/clipped_ratio=1.0, mean_terminated_length=0). This
# soft guard penalises over-length completions, nudging toward concise,
# terminating micro-narratives. It's a SEPARATE reward function that GRPO sums
# with the main one, so its effect is logged separately as
# rewards/length_guard/mean — and you can disable it with --no_length_guard.
from config import TARGET_WORDS, W_LEN   # length-guard dials in config.py


def length_guard(completions, **kwargs):
    out = []
    for t in _texts(completions):
        n = len(t.split())
        out.append(-W_LEN * max(0.0, (n - TARGET_WORDS) / TARGET_WORDS))
    return out


# --- anti-copy guard ------------------------------------------------------
# With the record in the prompt, a lazy optimum is to COPY the record verbatim:
# high faithfulness, but not a narrative. This penalises excessive verbatim
# overlap (fraction of the completion's 5-grams that also appear in the record),
# above a tolerance that leaves normal factual reuse of names/phrases unpenalised.
# Needs each completion's record, rebuilt from case_id. Separate reward stream
# (logged as rewards/anti_copy_guard/mean); disable with --no_anti_copy.
CASE_BY_ID = {c["case_id"]: c for c in CASES}
from config import ANTICOPY_N, ANTICOPY_TOL, W_COPY   # anti-copy dials in config.py


def _ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def anti_copy_guard(completions, case_id=None, **kwargs):
    texts = _texts(completions)
    if case_id is None:
        return [0.0 for _ in texts]
    out = []
    for t, cid in zip(texts, case_id):
        comp = _ngrams(t.lower().split(), ANTICOPY_N)
        if not comp:
            out.append(0.0)
            continue
        if cid in _REAL_RECORDS:
            from ingest import record_block as _rb
            r = _REAL_RECORDS[cid]
            rec_text = _rb(r) + " " + r.source_text
        else:
            rec_text = _record_block(CASE_BY_ID[cid])
        rec = _ngrams(rec_text.lower().split(), ANTICOPY_N)
        overlap = len(comp & rec) / len(comp)
        out.append(-W_COPY * max(0.0, overlap - ANTICOPY_TOL))
    return out


# --- format guard ----------------------------------------------------------
# The anti-copy guard catches VERBATIM reuse, but the linguistic arm found a
# subtler exploit: REFORMAT the record into a list ("Focal: … / Entities: / - … /
# Events: - … [1938, uncertain; src S2]"). That games BOTH the surface reward
# (dense names/dates) AND faithfulness (a copied record is trivially "grounded")
# while not being a narrative. This penalises the record's scaffolding markers
# and non-prose list structure, enforcing "write prose, not a reformatted record".
# General (not synthetic-specific): the same dump-the-record temptation is even
# stronger on real EHRI records. Logged as rewards/format_guard/mean; disable
# with --no_format_guard.
from config import W_FORMAT
_SCAFFOLD = re.compile(
    r"(?im)^\s*(focal|entities|events|sources|notes|record)\s*:"   # section headers
    r"|\bsrc\s+s?\d"                                              # "src S1" provenance
    r"|\[\s*\d{3,4}\s*,"                                          # "[1938, …]" date-source brackets
    r"|^\s*[-*•]\s+",                                             # bullet list lines
    re.M)


def format_guard(completions, **kwargs):
    out = []
    for t in _texts(completions):
        hits = len(_SCAFFOLD.findall(t))
        extra_newlines = max(0, t.count("\n") - 1)   # a micro-narrative is ~1 paragraph
        out.append(-W_FORMAT * (hits + 0.5 * extra_newlines))
    return out


# ----------------------------------------------------------------------

def dry_run(args):
    rows = build_prompts(source=args.prompts, limit=args.prompt_limit)
    print(f"Reward: {args.reward}")
    src = args.prompts or "synthetic CASES"
    print(f"Prompt dataset: {len(rows)} prompts from {src}, each with a case_id.")
    print(f"Example prompt:\n  {rows[0]['prompt'][:400]}\n")
    if args.prompts:
        print("REAL-PROMPT MODE — reward is scored against each record's real source "
              "(composite = NLI grounding; linguistic = surface). Train without --dry_run.")
        return

    # The crucial check: does the reward VARY across candidate completions?
    # If it didn't, GRPO would get zero advantage and never train.
    case = CASES[0]
    cands = [render(case, "testimony", p) for p in PROFILES]
    if args.reward == "composite":
        rewards = [composite_reward_by_id(t, case["case_id"]) for t in cands]
    elif args.reward == "human":
        rewards = [human_reward_by_id(t, case["case_id"], persona=args.persona) for t in cands]
    elif args.reward == "judge":
        rewards = reward_judge(cands, case_id=[case["case_id"]] * len(cands))
    else:
        rewards = reward_linguistic(cands)
    print("Reward across candidate completions (must vary; ideally good > fabricated):")
    for p, r in zip(PROFILES, rewards):
        print(f"  {p:<12} reward = {r:.3f}")
    spread = max(rewards) - min(rewards)
    good_beats_fab = rewards[PROFILES.index("good")] > rewards[PROFILES.index("fabricated")]
    print(f"  spread = {spread:.3f}  ->  {'OK (non-zero signal)' if spread > 1e-6 else 'FLAT — GRPO cannot learn'}")
    note = {"composite": "composite gates fabrication",
            "human": f"human reward, persona={args.persona}",
            "judge": f"LLM judge, backend={args.judge_backend}",
            "linguistic": "surface reward can be fooled"}[args.reward]
    print(f"  good > fabricated? {good_beats_fab}  ({note})")
    print("\nDRY RUN OK — prompts build and the reward returns a varying list[float].")
    print("Run without --dry_run on GPU/MPS to train.")


def real_run(args):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall the stack: pip install -r requirements-local.txt")

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    use_mps = (not use_cuda) and (not args.cpu) and \
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    use_cpu = not (use_cuda or use_mps)
    print(f"Device: {'cuda' if use_cuda else ('mps' if use_mps else 'cpu')}  |  model: {args.model}  "
          f"|  steps: {args.max_steps}  |  num_generations: {args.num_generations}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if use_cuda else torch.float32)

    # Start from the SFT baseline: merge its LoRA into the base, then train a
    # fresh LoRA on top. This is the SFT -> RL hand-off.
    if args.init_adapter:
        from peft import PeftModel
        print(f"Starting from SFT adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    ds = Dataset.from_list(build_prompts(tok, args.prompts, args.prompt_limit))  # synthetic or real corpus
    # distinct adapter name for real-prompt runs so they don't clobber the synthetic arms.
    arm_tag = args.reward + ("-real" if args.prompts else "")

    # per_device batch must be a multiple of num_generations (GRPO groups).
    cfg = GRPOConfig(
        output_dir=f"grpo-{arm_tag}-adapter",
        per_device_train_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=use_cuda,
        use_cpu=use_cpu,
        seed=args.seed,
    )
    reward_funcs = [REWARDS[args.reward]]
    if args.length_guard:
        reward_funcs.append(length_guard)
    if args.anti_copy:
        reward_funcs.append(anti_copy_guard)
    if args.format_guard:
        reward_funcs.append(format_guard)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
    )
    guards = "".join(s for s, on in [(" + length_guard", args.length_guard),
                                     (" + anti_copy", args.anti_copy),
                                     (" + format_guard", args.format_guard)] if on)
    rname = f"{args.reward}({args.persona})" if args.reward == "human" else args.reward
    print(f"Reward: {rname}{guards} | max_completion_length: {args.max_completion_length} "
          f"| Prompts: {len(ds)} | starting GRPO ...")
    result = trainer.train()
    print("\nMetrics:", result.metrics)
    print("Watch 'reward' (mean) and 'reward_std' in the logs: reward should drift")
    print("up and reward_std should stay > 0. If reward_std hits 0, the group")
    print("collapsed (no signal) — raise num_generations or completion length.")
    out_dir = f"grpo-{arm_tag}-adapter"
    trainer.save_model(out_dir)
    print(f"Saved adapter to: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--reward", choices=["linguistic", "composite", "human", "judge"],
                    default="linguistic")
    ap.add_argument("--persona", choices=list(PERSONAS), default="balanced",
                    help="persona for --reward human (the pluralism knob)")
    ap.add_argument("--judge_backend", default="vllm",
                    help="backend for --reward judge (vllm/ollama/gemini/uva)")
    ap.add_argument("--judge_model", default=None,
                    help="model for --reward judge (default: the backend's default)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts", default=None,
                    help="prompt source: synthetic CASES (default) or 'corpus:NAME/SRC' "
                         "to train on REAL records, e.g. corpus:EHRI/extracted")
    ap.add_argument("--prompt_limit", type=int, default=100,
                    help="max real prompts to load (with --prompts)")
    ap.add_argument("--init_adapter", default=None,
                    help="start GRPO from an SFT adapter (e.g. sft-narrative-adapter)")
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_completion_length", type=int, default=128)
    ap.add_argument("--no_length_guard", dest="length_guard", action="store_false",
                    help="disable the over-length penalty")
    ap.add_argument("--no_format_guard", dest="format_guard", action="store_false",
                    help="disable the record-scaffold / list-format penalty")
    ap.add_argument("--no_anti_copy", dest="anti_copy", action="store_false",
                    help="disable the verbatim-copy penalty")
    ap.set_defaults(length_guard=True, anti_copy=True, format_guard=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.prompts and args.reward in ("human", "judge"):
        sys.exit(f"--reward {args.reward} is synthetic-only (needs the case schema / is costly). "
                 "With real --prompts use --reward composite or linguistic.")
    global _ACTIVE_PERSONA, _JUDGE_BACKEND, _JUDGE_MODEL
    _ACTIVE_PERSONA = args.persona
    _JUDGE_BACKEND, _JUDGE_MODEL = args.judge_backend, args.judge_model
    dry_run(args) if args.dry_run else real_run(args)


if __name__ == "__main__":
    main()
