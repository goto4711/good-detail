#!/usr/bin/env python3
"""
realdata_generate.py
================================================================
Corpus-agnostic real-data shakedown: run the trained arms on a few REAL records
(via the `ingest` layer) and score them, to SEE THE DIFFERENCE from synthetic.
A diagnostic, NOT the research. Works on any corpus with a data/<NAME>/corpus.json.

Per record it:
  1. loads it through `ingest.load_corpus(corpus, source)` — entities/relations +
     source_text, in the canonical Record shape;
  2. builds the SAME open-book prompt the arms were trained on (config.INSTRUCTION
     + chat template), with the record = record_block (entities + relations) + a
     focal-relevant source excerpt, unit = the record's focal subject;
  3. generates a micro-narrative with the chosen arm;
  4. scores it with the transferable rewards:
        linguistic — surface (unchanged);
        F / unsup  — NLI faithfulness against RETRIEVED source sentences (premise
                     = the real source) + an attested-entity check (a name absent
                     from record+source is a candidate fabrication);
        composite  — the same F-gated formula, recomputed with the real F;
        judge      — optional LLM judge (--judge).

SAFEGUARDS (real survivor testimony): tiny N, the focal name is REDACTED in
printed output, every narrative is stamped unverified. Never a historical source.

  python realdata_generate.py --corpus EHRI --source xml --sft_adapter sft-narrative-adapter --adapter grpo-composite-adapter --limit 3
  python realdata_generate.py --corpus EHRI --source iob --adapter sft-narrative-adapter --limit 3 --judge --judge_backend gemini
"""

import argparse
import os
import re
import sys
import warnings

try:
    from ingest import load_corpus, record_block
    from grpo_train import INSTRUCTION, chat_prompt
    from linguistic_reward import features, linguistic_reward, WORD_RE
    from composite_reward import sensationalism
    from faithfulness import is_unscoreable_status, nli_faithfulness
    from config import GAMMA, W_C, W_FAB, W_SENS
except ImportError as e:
    sys.exit(f"Run from the project folder: {e}")

Q_KEYS = ("proper_noun_density", "number_date_density", "concreteness", "lexical_density")
_BANNER = "⚠ machine-generated, UNVERIFIED — not a historical source"
_ADMIN = re.compile(r"master\s*-?\s*index|index number|number of pages|title of document"
                    r"|author or source|^\s*languages|^\s*\d+\.\s|\bEW\s*\d|P\s*III"
                    r"|scheme|signed\b|block\s?elder|prisoner number|coll\.\s*\d", re.I)


def _toks(s):
    return set(w.lower() for w in WORD_RE.findall(s))


def _sentences(source_text):
    chunks = re.split(r"(?<=[.!?])\s+|\n+", source_text)
    return [c.strip() for c in chunks if len(c.split()) >= 5 and not _ADMIN.search(c)]


def retrieve(source_text, query, k):
    sents = _sentences(source_text)
    from config import RETRIEVAL_METHOD
    if RETRIEVAL_METHOD == "embed":
        try:
            from retrieval import embed_retrieve
            hit = embed_retrieve(sents, query, k)
            if hit is not None:
                return hit
        except Exception:
            pass                      # fall back to lexical on any failure
    q = _toks(query)
    return sorted(sents, key=lambda s: len(q & _toks(s)), reverse=True)[:k]


def attested_index(rec, with_relations=True):
    blob = record_block(rec, with_relations=with_relations) + " " + rec.source_text
    return _toks(blob), set(re.findall(r"\b\d{4}\b", blob))


def _redact(text, focal):
    parts = [p for p in re.split(r"[,\s]+", focal) if len(p) > 2]
    out = text
    for p in sorted(set(parts), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(p)}\b", "[unit]", out, flags=re.I)
    return re.sub(r"\[unit\](?:[,\s]+\[unit\])+", "[unit]", out)


def _composite(text, F, unsup):
    f = features(text)
    Q = sum(f[k] for k in Q_KEYS) / len(Q_KEYS)
    return round((F ** GAMMA) * (Q + W_C * f["calibration"]) - W_FAB * unsup
                 - W_SENS * sensationalism(text), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from config import DEFAULT_MODEL
    ap.add_argument("--corpus", default="EHRI")
    ap.add_argument("--source", default=None, help="source key in corpus.json (else default)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sft_adapter", default=None, help="SFT adapter to MERGE first")
    ap.add_argument("--adapter", default=None, help="arm adapter to APPLY")
    ap.add_argument("--limit", type=int, default=3, help="number of records (keep small)")
    ap.add_argument("--excerpt_sents", type=int, default=8, help="source sentences in the prompt")
    ap.add_argument("--premise_sents", type=int, default=20, help="source sentences as NLI premise")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--relations", action="store_true",
                    help="include extracted relations in the record block "
                         "(OFF by default — ablation showed they add nothing to grounding)")
    ap.add_argument("--no_redact", action="store_true", help="show the focal name (off by default)")
    ap.add_argument("--summary", action="store_true",
                    help="aggregate mode: suppress narratives, print per-record scores + a final MEAN row")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge_backend", default="gemini")
    ap.add_argument("--judge_model", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    judge = None
    if args.judge:
        import llm_judge_reward as J
        J._BACKEND = args.judge_backend
        jmodel = args.judge_model or J.BACKENDS[args.judge_backend]["default_model"]
        judge = lambda record, narr: J.judge(record, narr, model=jmodel)[0]

    recs = load_corpus(args.corpus, args.source, limit=args.limit)
    if not recs:
        sys.exit(f"No records for corpus={args.corpus} source={args.source}")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall: pip install -r requirements-local.txt")

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
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

    wr = args.relations
    print(f"Device: {device} | model: {label} | corpus: {args.corpus}/{args.source or 'default'} "
          f"| records: {len(recs)} | relations: {'on (analysis)' if wr else 'off (default)'}")
    print(f"{_BANNER}\n")

    agg = {"F": [], "unsup": [], "composite": [], "linguistic": [], "judge": []}
    for rec in recs:
        focal = rec.unit or rec.title
        excerpt = " ".join(retrieve(rec.source_text, focal, args.excerpt_sents))
        record = record_block(rec, with_relations=wr) + ("\n\nSOURCE EXCERPT:\n" + excerpt if excerpt else "")
        prompt = chat_prompt(tok, INSTRUCTION.format(
            register=rec.register.replace("_", "-"), unit=focal, record=record))
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=(args.temperature > 0),
                                 temperature=args.temperature or None,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        att_t, att_y = attested_index(rec, with_relations=wr)
        premises = retrieve(rec.source_text, focal + " " + text, args.premise_sents)
        F, unsup, faith_status = nli_faithfulness(
            text, premises, subject=focal, grounded_toks=att_t, grounded_years=att_y,
            return_status=True)
        if is_unscoreable_status(faith_status):
            warnings.warn(f"{rec.id}: faithfulness unscoreable ({faith_status})", RuntimeWarning, stacklevel=1)
        ling = linguistic_reward(text)
        comp = _composite(text, F, unsup)
        jval = judge(record, text) if judge else None
        js = f"  judge={jval:+.2f}" if jval is not None else ""

        for k, v in (("F", F), ("unsup", unsup), ("composite", comp), ("linguistic", ling)):
            agg[k].append(v)
        if jval is not None:
            agg["judge"].append(jval)

        fdisp = faith_status.upper() if faith_status != "scored" else f"{F:.2f}"
        score_line = f"    [F={fdisp} unsup={unsup}  composite={comp:+.2f}  linguistic={ling:.2f}{js}]"
        if args.summary:
            print(f"  {rec.id}: F={fdisp} unsup={unsup} comp={comp:+.2f} ling={ling:.2f}{js}")
        else:
            unit_disp = focal if args.no_redact else "[unit]"
            print(f"=== {rec.id}  —  {unit_disp}  ({rec.title[:55]}…) ===")
            print(text if args.no_redact else _redact(text, focal))
            print(score_line)
            print(f"    [{_BANNER}]\n")

    # final aggregate row (always printed; grep-friendly for the orchestrator)
    n = len(agg["F"])
    if n:
        def m(k):
            return sum(agg[k]) / len(agg[k]) if agg[k] else float("nan")
        jm = f" judge={m('judge'):+.2f}" if agg["judge"] else ""
        print(f"\n=== MEAN  arm={label}  source={args.corpus}/{args.source or 'default'}  "
              f"relations={'on' if wr else 'off'}  N={n}:  "
              f"F={m('F'):.3f}  unsup={m('unsup'):.2f}  composite={m('composite'):+.3f}  "
              f"linguistic={m('linguistic'):.3f}{jm} ===")


if __name__ == "__main__":
    main()
