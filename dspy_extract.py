#!/usr/bin/env python3
"""
dspy_extract.py
================================================================
The EXTRACTION FRONT-END (NEXT_STEPS §3): raw text -> typed entities + relations,
via DSPy. Produces a structured fact base (the merged_intermediate_data schema)
that the `iob_jsonl` ingest adapter reads directly — so the *situated* reward
runs on EVERY record, not only the pre-extracted chunks. This is the connective
tissue between the earlier EHRI NER/RE work and the good-detail rewards.

Why this matters (shown empirically): on the XML path the record is an entity
*list* and even the grounded arm fabricates (unsup ~6-8); with a real fact base of
relations the composite arm hit F=1.00. The front-end turns extraction into that
fact base for all records.

LM: reuses config.BACKENDS (gemini / vllm / uva / ollama), all OpenAI-compatible.

  # 1. extract a fact base from the EHRI testimony bodies (chunked):
  python dspy_extract.py --corpus EHRI --source xml --backend gemini \
         --limit 20 --out data/EHRI/iob/extracted.jsonl
  # 2. ground against it (corpus.json already has an "extracted" source):
  python realdata_generate.py --corpus EHRI --source extracted \
         --sft_adapter sft-narrative-adapter --adapter grpo-composite-adapter --limit 3
  # optional: optimise the extractor's prompt against the gold data:
  python dspy_extract.py --compile --backend gemini --gold data/EHRI/iob/merged_intermediate_data.jsonl
"""

import argparse
import json
import os
import re
import sys

ENTITY_TYPES = "PERSON, LOCATION, ORGANIZATION, DATE, EVENT, MISCELLANEOUS"


# ---------------------------------------------------------------- LM wiring
def make_lm(backend, model=None):
    """Build a dspy.LM from config.BACKENDS (every backend is an OpenAI-compatible
    chat endpoint, so litellm's openai provider handles all of them uniformly)."""
    import dspy
    from config import BACKENDS
    cfg = BACKENDS[backend]
    chat_url = cfg["chat_url"]
    if backend == "vllm" and os.getenv("VLLM_URL"):
        chat_url = os.getenv("VLLM_URL").rstrip("/") + "/v1/chat/completions"
    api_base = chat_url.rsplit("/chat/completions", 1)[0]          # -> .../v1
    key = (os.getenv(cfg["key_env"]) if cfg.get("key_env") else None) or "EMPTY"
    model = model or cfg["default_model"]
    lm = dspy.LM(f"openai/{model}", api_base=api_base, api_key=key)
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------- the program
def build_extractor(relations=False):
    """Entity-only by default — the objective is text summaries, and the overnight
    ablation showed extracted relations add nothing to grounding (and are ~0.40 F1).
    `relations=True` restores entity+relation extraction (analysis only / heavier)."""
    import dspy

    if relations:
        out_desc = ("JSON object with keys 'entities' and 'relationships'. "
                    "entities: list of {entity_text, entity_type}. "
                    "relationships: list of {relation_type, head_entity_text, tail_entity_text}. "
                    "relation_type is a descriptive verb phrase (e.g. located_in, imprisoned_at).")
    else:
        out_desc = ("JSON object with key 'entities': a list of "
                    "{entity_text, entity_type}. Entities only — no relationships.")

    class Extraction(dspy.Signature):
        """Extract EVERY entity from a historical document (testimony, record, news
        report). Use only the given entity types. Return strict JSON. Do not invent
        anything not in the text."""
        document_text: str = dspy.InputField(desc="the source passage")
        entity_types: str = dspy.InputField(desc="comma-separated valid entity types")
        extracted = dspy.OutputField(desc=out_desc)

    class Extractor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.ChainOfThought(Extraction)

        def forward(self, document_text, entity_types=ENTITY_TYPES):
            r = self.predict(document_text=document_text, entity_types=entity_types)
            return parse_extracted(r.extracted)

    return Extractor()


def parse_extracted(x):
    """Tolerant parse -> {'entities': [...], 'relationships': [...]}."""
    if isinstance(x, dict):
        d = x
    else:
        m = re.search(r"\{.*\}", str(x), re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            d = {}
    ents = d.get("entities") or d.get("new_entities") or []
    rels = d.get("relationships") or []
    ents = [{"entity_text": e.get("entity_text", ""), "entity_type": e.get("entity_type", "MISCELLANEOUS")}
            for e in ents if e.get("entity_text")]
    rels = [{"relation_type": r.get("relation_type", "related_to"),
             "head_entity_text": r.get("head_entity_text", ""),
             "tail_entity_text": r.get("tail_entity_text", "")}
            for r in rels if r.get("head_entity_text") and r.get("tail_entity_text")]
    return {"entities": ents, "relationships": rels}


def _chunks(text, chunk_words):
    words = text.split()
    for i in range(0, len(words), chunk_words):
        yield " ".join(words[i:i + chunk_words])


# ---------------------------------------------------------------- optimisation
def entity_f1(example, pred, trace=None):
    """Default compile metric: F1 of predicted vs gold entity_texts (case-insensitive)."""
    gold = {e["entity_text"].lower() for e in (example.get("entities") or [])}
    got = {e["entity_text"].lower() for e in (pred.get("entities") or [])}
    if not gold and not got:
        return 1.0
    if not got or not gold:
        return 0.0
    tp = len(gold & got)
    p = tp / len(got)
    r = tp / len(gold)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def downstream_metric(example, pred, trace=None):
    """EXPERIMENTAL — the methods contribution: score an extraction by whether the
    fact base it yields lets the grounded reward do its job, not by NER overlap.
    Concretely: a claim copied from the source should be judged grounded (high F)
    and an off-record claim should not. Returns the grounding separation in [0,1].
    Requires the NLI model; falls back to entity_f1 if unavailable."""
    try:
        from faithfulness import nli_faithfulness
    except Exception:
        return entity_f1(example, pred, trace)
    premises = [example["document_text"]]
    grounded = example["document_text"].split(".")[0][:200]              # a real source clause
    fake = "On 14 March 1939 Colonel Brandt arrived at the Westmark depot."  # off-record
    fg, _ = nli_faithfulness(grounded, premises)
    ff, _ = nli_faithfulness(fake, premises)
    return max(0.0, fg - ff)            # high when the extraction supports separation


def compile_extractor(backend, gold_path, model=None, metric_name="entity_f1", max_demos=8):
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    make_lm(backend, model)
    metric = {"entity_f1": entity_f1, "downstream": downstream_metric}[metric_name]
    trainset = []
    for line in open(gold_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if not d.get("document_text"):
            continue
        ents = d.get("base_entities", []) + d.get("new_entities", [])
        ex = dspy.Example(document_text=d["document_text"], entity_types=ENTITY_TYPES,
                          entities=ents, relationships=d.get("relationships", [])
                          ).with_inputs("document_text", "entity_types")
        trainset.append(ex)
    print(f"Compiling on {len(trainset)} gold examples (metric={metric_name}) …")
    tele = BootstrapFewShot(metric=metric, max_bootstrapped_demos=max_demos)
    compiled = tele.compile(build_extractor(relations=False), trainset=trainset)
    out = "extractor_compiled.json"
    compiled.save(out)
    print(f"Saved compiled extractor -> {out}")
    return compiled


# ---------------------------------------------------------------- run
def run(args):
    try:
        import dspy  # noqa: F401
    except ImportError:
        sys.exit("DSPy not installed. On the VM:  pip install dspy-ai")
    from ingest import load_corpus

    make_lm(args.backend, args.model)
    extractor = build_extractor(relations=args.relations)
    if args.load and os.path.exists(args.load):
        extractor.load(args.load)
        print(f"Loaded compiled extractor: {args.load}")

    recs = load_corpus(args.corpus, args.source, limit=args.doc_limit)
    written = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in recs:
            for ci, chunk in enumerate(_chunks(rec.source_text, args.chunk_words)):
                if len(chunk.split()) < 20:
                    continue
                try:
                    ex = extractor(document_text=chunk)
                except Exception as e:
                    print(f"  extract fail {rec.id}#{ci}: {e}")
                    continue
                f.write(json.dumps({
                    "id": f"{rec.id}_x{ci}", "document_text": chunk,
                    "base_entities": [], "new_entities": ex["entities"],
                    "relationships": ex["relationships"]}, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                print(f"  [{written}/{args.limit}] {rec.id}#{ci}: "
                      f"{len(ex['entities'])} entities, {len(ex['relationships'])} relations", flush=True)
                if args.limit and written >= args.limit:
                    break
            if args.limit and written >= args.limit:
                break
    print(f"Wrote {written} extracted fact-base records -> {args.out}")
    print(f"Ground against them:  python realdata_generate.py --corpus {args.corpus} "
          f"--source extracted --adapter grpo-composite-adapter --limit 3")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="EHRI")
    ap.add_argument("--source", default="xml", help="ingest source to extract FROM")
    ap.add_argument("--backend", default="gemini", help="LM backend (config.BACKENDS)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="data/EHRI/iob/extracted.jsonl")
    ap.add_argument("--chunk_words", type=int, default=150)
    ap.add_argument("--doc_limit", type=int, default=None, help="max source records to read")
    ap.add_argument("--limit", type=int, default=20, help="max chunks to extract (cost guard)")
    ap.add_argument("--load", default=None, help="a compiled extractor json to load")
    ap.add_argument("--relations", action="store_true",
                    help="also extract relations (OFF by default — entity-only; relations "
                         "added nothing to grounding in the ablation)")
    # compile mode
    ap.add_argument("--compile", action="store_true", help="optimise the extractor on gold data")
    ap.add_argument("--gold", default="data/EHRI/iob/merged_intermediate_data.jsonl")
    ap.add_argument("--metric", default="entity_f1", choices=["entity_f1", "downstream"])
    args = ap.parse_args()

    if args.compile:
        compile_extractor(args.backend, args.gold, args.model, args.metric)
    else:
        run(args)


if __name__ == "__main__":
    main()
