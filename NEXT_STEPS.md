# Next steps — roadmap (captured while the 3B run is training)

A running list of the agreed forward work, so nothing is lost between sessions.
`PROJECT_STATUS_2026-06-17.md` remains the source of truth for what is *built*;
this file is what's *next*.

## 1. Finish and read the current 4-arm run
The first run with everything correct at once: Qwen2.5-3B-Instruct, chat-template
prompts (native EOS → terminating completions), NLI faithfulness, and the four
reward arms (composite / human / linguistic / judge). When the zip lands, read
the GRPO logs (expect `clipped_ratio < 1`, `terminated_length > 0` at last) and
the per-arm generations, and fill the results table in `experiments.md`. Watch
the off-diagonal: linguistic chasing surface specificity into fabrication while
composite/human stay grounded.

## 2. EHRI real-data shakedown (generate + score, tiny + marked)
Run the trained arms on a handful of REAL EHRI records to see the difference vs
synthetic — a diagnostic, not the research. Data is in `prepare_data/`:
- **`ehri_xml_subdocuments/`** — each document = a `_header.xml` + many
  `_split_NNN.xml` chunks.
- The **header is a ready-made fact base**: `<person>/<persName>`,
  `<place>/<placeName>/<location>/<geo>`, `<org>/<orgName>`, and `<linkGrp>/<link>`
  relations — essentially our synthetic record schema, already extracted.
- The **split chunks are the source testimony** (`<div><p>` with inline
  `<term ref="…ehri_terms…">` keyword annotations).

Plan: per record, **record block** (entities + relations → prompt context +
structured grounding) and **source passages → NLI premise** (claim-level
faithfulness against the real source) — so *both* grounding checks run on real
data. Generate a micro-narrative per arm, score with the transferable rewards
(linguistic, NLI-F, composite, judge), compare composite/human vs linguistic.

**BUILT (2026-06-19) — the corpus-agnostic ingest backbone:**
- `data/<CORPUS>/corpus.json` manifest + `data/EHRI/{xml,iob}` payloads;
- `ingest/` package: one canonical `Record`, pluggable adapters `tei_ehri`
  (header+splits → entities + source), `iob_jsonl` (the relation-extraction
  data → **typed entities + relations** + source), `synthetic`; one entry point
  `load_corpus(name, source)`;
- `realdata_generate.py --corpus EHRI --source xml|iob` — generate + score any
  corpus through the same code; `faithfulness.nli_faithfulness()` grades against
  retrieved real source sentences; focal name redacted, outputs stamped unverified.

Run it (on the VM, after uploading `ingest/`, `data/EHRI/`, `realdata_generate.py`):
```bash
python -m ingest EHRI --source iob --list                 # sanity
for A in grpo-composite-adapter grpo-linguistic-adapter grpo-human-adapter; do
  python realdata_generate.py --corpus EHRI --source xml --sft_adapter sft-narrative-adapter --adapter $A --limit 3
done
```
Still **to do:** the IOB focal pick is crude (prefers PERSON, else first entity);
lexical retrieval could be upgraded to embeddings; validate on messier documents.
Safeguards (non-negotiable): tiny N (3–5), outputs stamped unverified, focal
anonymised, grounded arms shown alongside linguistic so fabrication reads AS the
finding.

## 3. Start the pipeline from RAW TEI/IOB — DSPy extraction front-end
The pipeline should *begin* at the raw EHRI TEI + IOB data and build the fact
base itself, rather than assuming one. This is the connective tissue between the
earlier EHRI NER/relation-extraction project and the good-detail rewards.
- **Use DSPy** (worked well last time): write extraction as declarative
  signatures/modules (TEI passage + IOB tags → entities → relations → fact base)
  and let the optimiser compile prompts/few-shot examples against a metric.
- **Synergy to exploit:** make the extraction metric *downstream-aware* — judge an
  extracted fact base by whether it lets the composite reward correctly separate
  grounded from fabricated narratives, not just by span-level NER F1. The front-end
  is then optimised for the grounding the rest of the pipeline needs.
- **Reuse, don't restart:** `prepare_data/` already has `prepare_tei_data_dspy.py`,
  `prepare_iob_data_dspy.py`, `dspy_finetune_prep.py`, the `qwen_*`/`*_tei_*` jsonl
  datasets, and a trained `simple-re-model/` to modernise from.
- Output of this stage = the structured fact base + source passages that stages
  2 (real-data test) and the full reward/training pipeline consume.

**BUILT (2026-06-19) — `dspy_extract.py`:** DSPy `Signature` + `ChainOfThought`
extractor (text → typed entities + relations, the merged_intermediate_data
schema); LM from `config.BACKENDS` (gemini/vllm/uva/ollama, uniform OpenAI-compat);
chunks long testimony; writes `data/EHRI/iob/extracted.jsonl`, wired as the
**`extracted`** source in `corpus.json` so the rest of the pipeline grounds against
it unchanged. Optional `--compile` (BootstrapFewShot) with two metrics:
`entity_f1` (default) and the experimental **`downstream`** metric (scores an
extraction by whether its fact base lets the grounded reward separate
grounded-from-fabricated — the methods contribution, to refine).
```bash
pip install dspy-ai
python dspy_extract.py --corpus EHRI --source xml --backend gemini --limit 20   # build fact base
python realdata_generate.py --corpus EHRI --source extracted \
       --sft_adapter sft-narrative-adapter --adapter grpo-composite-adapter --limit 3
python dspy_extract.py --compile --backend gemini --metric downstream            # optimise (optional)
```
**Still to do:** validate extraction quality vs the gold `merged_intermediate_data.jsonl`;
flesh out the `downstream` metric; per-document (not per-chunk) fact-base merge so a
record's whole fact base is one unit; tie `tei_xxx_split` ids back to source docs.

## 3b. Inference-time track — reward-as-selector (BUILT 2026-06-19)
`bestofn_demo.py`: a strong, **pluggable** LLM (`--gen_backend uva|vllm|gemini|ollama`,
`--gen_model <anything>`) writes K candidate narratives per real record; each reward
then *picks* its favourite (best-of-N). Where the picks diverge (composite picks the
faithful one, linguistic the vivid-but-fabricated one, judge the fluent one) is the
thesis, shown with a current model and **no training**. The historian-facing artifact;
see `methods_rewards.md §7` (training vs. selection). Pairs with `realdata_generate.py`
(the trained-arm controlled study).

## On "more prompts" (the deferred GRPO-scale question — largely superseded)
The original worry was that GRPO on **16 synthetic prompts** under-trained the arms.
How the pivot resolves it:
- the **reward-level findings** (the spine) never depended on prompt count — they're
  properties of the reward functions on the grid;
- **statistical power now comes from real data**, not more synthetic prompts:
  `run_ehri.sh` scores hundreds of real records (N=100–400), turning N=3 into real
  numbers without touching synthetic;
- the **inference-time track** sidesteps GRPO scaling entirely;
- and if a *trained-on-real* result is wanted, **GRPO needs only prompts + a reward,
  not gold targets** — so it can train on hundreds of real EHRI records directly
  (prompts from the `ingest` layer, reward = composite), starting from the synthetic
  SFT adapter for format. That is the right way to "add prompts" — with *real* ones.
  *(Small build: let `grpo_train.build_prompts()` pull from `load_corpus()`.)*
- Expanding the synthetic `CASES` (8 → 40+) remains a cheap option **only** if a
  cleaner *synthetic* controlled-study divergence is wanted; otherwise skip it.

## 4. Smaller open items (from PROJECT_STATUS)
- Real **concreteness norms** (`concreteness_norms.csv`) to replace the stub.
- **Validate the LLM judge against real historians** (it's currently an
  operationalisation, not an oracle — see `methods_rewards.md` §4 and §6).
- The historian **workshop** that makes the "human" arm genuinely independent of
  the automated rewards.

## 5. Forward roadmap — how to get better results (2026-06-20)

What the overnight run proved are **dead ends at this scale** (don't repeat):
more GRPO training (synthetic *and* real-prompt both came up near-null), and more
synthetic prompts (the reward-level findings never needed them). "Better results"
is not a compute problem. Priorities, highest-leverage first:

**P1 — the historian workshop (the unlock; not technical).** Every situated/human
signal is still synthetic, so "whose good detail?" is asserted, not shown. ~30
records × 3–4 candidate narratives (strong model) × 4–5 historians rank/rate them.
One dataset yields: (a) ground truth to validate **which automated reward best
predicts human judgment** (the headline), (b) a real human arm, (c) measured
pluralism (do historians disagree?), (d) preference data for real DPO if wanted.

**P2 — cheap technical wins (do now).**
- *Faithfulness-gated best-of-N* — reject candidates with `unsup>0`, then optimise
  richness among survivors, so the output shown is always grounded. Deployable use
  of the reward; no training. (`bestofn_demo.py --gate`.)
- *Embedding retrieval for the NLI premise* — the grounding gain came from focused
  source passages; lexical retrieval is crude. (`retrieval.py`,
  `config.RETRIEVAL_METHOD="embed"`.)

**P3 — strengthen the measurement.** Validate the NLI faithfulness estimator
against the P1 annotations; add a second estimator (LLM claim-verifier) so
faithfulness isn't single-method.

**P4 — make inference-time selection the centre of gravity.** Trained policies
converge, so the contribution is the reward-as-lens over current LLMs: bigger K,
strongest models, evaluate selector quality vs human ranking (P1), report
cross-reward disagreement at scale.

**P5 — extraction/relations: DECIDED → drop (2026-06-20).** The objective is text
summaries, not a knowledge graph; the ablation was null and relation-F1 only 0.40.
So relations are **off by default** everywhere now: `record_block(with_relations=False)`,
`realdata_generate`/`bestofn_demo` need `--relations` to opt in, and `dspy_extract`
is **entity-only** by default (`--relations` to restore). The "fact base" is now
honestly *source passages + entities*. Relations remain available only as an
optional analysis artifact (and `validate_extraction` still scores them, since
that 0.40 is the evidence that justified dropping them).

**P6 — rigour for the writeup.** CIs / multiple seeds on metrics; the
RLAIF-pluralism sub-study (judge with 2–3 different models); write the **negative
results** up plainly (policies converge; relations don't help; LLM judge biased).

Suggested order: **P2 → P1 → P4/P6**, P5 a quick decision, P3 folded into P1.
Through-line: the strength is **measurement/operationalisation of a contested
construct, validated against real community judgment** — not training a model to
write well.
