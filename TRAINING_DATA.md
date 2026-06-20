# Creating training data for "good detail"

How to turn real archival records into data that trains a model to write **good
detail** — grounded, faithful micro-narratives — rather than vivid-but-fabricated
ones. This is the data-centric half of the project: for a contested generative
target, scaling a culture-blind objective does not help; the *right* data does.

There is no public corpus of archival "good detail" summaries (see
[What already exists](#what-already-exists)), so this pipeline manufactures it,
three ways, from cheapest to highest-quality.

---

## The shape of one training example

Every route emits the same two artifacts, in the format the trainers already read:

- **SFT target** — `{prompt, completion}`: the record prompt (exactly the chat
  instruction GRPO/`generate` feed) → a *good* micro-narrative. Teaches the model
  to **write**. Consumed by `sft_narrative.py --source jsonl --data …`.
- **Preference pair** — `{prompt, chosen, rejected, rejected_kind}`: a good
  narrative vs. a worse one for the same record. Teaches the model to **prefer**
  good detail. Consumed by `dpo_kto_train.py --pairs …`.

The prompt is rebuilt identically across all routes (from the stored `prompt_user`,
or from the record block) **and chat-wrapped with the model's own template** (the same
`apply_chat_template` SFT/GRPO/`generate` use), so the training prompt is byte-identical
to what the model sees at inference — and TRL gets a clean token prefix (no
"tokenized prompt mismatch" warning). This needs `transformers` installed (it is, on the
GPU box); without it the harness warns and falls back to raw prompts. `--no_chat_wrap`
opts out deliberately; `--model` picks the tokenizer (default `config.DEFAULT_MODEL`).

---

## The three routes

| route | who labels "good" | needs | cost | quality | command |
|---|---|---|---|---|---|
| **reward** | the grounded composite/NLI reward | nothing (bootstrap) | free | only as good as the reward | `--label reward` |
| **human** | historians rate machine candidates | a workshop | medium | gold ranking | `--label human` |
| **authored** | historians write the summary | a workshop | higher | gold *writing* | `--label authored` |

All three run through one script, `build_training_data.py`. Pick by `--label`.

### Route 1 — reward (bootstrap, no humans)

Generate candidates, let the grounded reward pick `chosen` (faithful AND rich,
past the faithfulness gate) and a vivid-but-unfaithful candidate as `rejected`.

```bash
python bestofn_demo.py --corpus EHRI --source extracted \
  --gen_backend uva --k 6 --limit 20 --judge --save workshop_candidates.json

python build_training_data.py workshop_candidates.json --label reward --dry_run   # inspect first
python build_training_data.py workshop_candidates.json --label reward --gate_unsup 2
```

Scales immediately. The catch: on real archival data, genuinely grounded
candidates are scarce, so a strict gate yields few targets. The run prints a
**gate-sensitivity curve** so you can see the trade — e.g. on the simulated set:

```
unsup ≤ 0:  0/5 records have a groundable target
unsup ≤ 1:  1/5 records
unsup ≤ 2:  4/5 records
```

That scarcity is not a bug; it *is* the thesis. Loosening `--gate_unsup` trades
faithfulness for volume — a dial you set deliberately, not a default to ignore.

### Route 2 — human (rate machine candidates)

Run the workshop apparatus: build a blind sheet, historians rate, you analyse.

```bash
python make_annotation_sheet.py workshop_candidates.json -o annotation.html
# historians return responses_*.json into a folder, then:
python analyze_annotations.py workshop_candidates.json responses/      # validates the rewards
python build_training_data.py workshop_candidates.json \
  --label human --responses responses/ --min_gap 1.0
```

`chosen` = top-rated candidate, `rejected` = bottom-rated — but **only** when the
mean-rating gap clears `--min_gap`, so you never train on noise where the
historians had no real preference. Add `--gate_human_chosen` to also drop a
liked-but-ungrounded target.

### Route 3 — authored (write the gold)

The highest-value route. Historians read a record and **write** the good summary
themselves; each `(record → human summary)` pair is a top-quality SFT target.

```bash
# build the authoring sheet from the live corpus (also writes records.json with prompts)
python make_authoring_sheet.py --corpus EHRI --source extracted --limit 20 -o authoring.html
# historians write, return authored_*.json into a folder, then:
python build_training_data.py records.json --label authored --authored authored/
```

Every authored summary becomes SFT gold. If you pass a `bestofn --save` file as the
records argument instead of `records.json`, you **also** get `human_vs_machine`
preference pairs (human-written = `chosen`, least-faithful machine candidate =
`rejected`) — the strongest DPO signal available.

**Multiple historians, on purpose.** `--authored` takes a *folder*: drop every
`authored_*.json` in it and the harness reads them all. When several historians
write for the same record, **each summary is kept as its own gold target** — nothing
is deduplicated or averaged. This is deliberate: disagreement about what counts as
good detail becomes *several* gold targets per record, encoding the pluralism into
the training set instead of collapsing it to one "right" answer. Every row carries
its `author` (and `confidence`) in the provenance, so you can later weight, filter,
or train per-historian. Smoke-tested: 2 historians × overlapping records → 5 distinct
SFT rows, all retained.

---

## End-to-end

```
ingest (real records)
        │
        ├─ bestofn_demo.py --save ─► candidates.json ──┐
        │                                              │
        │   make_annotation_sheet.py → historians rate │  route: human
        │   make_authoring_sheet.py  → historians write│  route: authored
        │                                              │
        └──────────────────────────────► build_training_data.py --label {reward|human|authored}
                                                       │
                                  ┌────────────────────┴───────────────────┐
                                  ▼                                        ▼
                        sft_<label>.jsonl                      preference_pairs_<label>.jsonl
                                  │                                        │
                  sft_narrative.py --source jsonl            dpo_kto_train.py --method dpo --pairs
                                  │                                        │
                                  └──────────────► trained adapter ◄───────┘
                                                          │
                                       generate.py / bestofn_demo.py (eyeball, then re-rate)
```

The loop closes: a trained model's outputs can be fed back through best-of-N and
the workshop, refining the data again.

Proven end-to-end on real EHRI data: `--label human` on the historian-rated set
yields preference pairs whose `chosen` is the grounded human-preferred summary and
whose `rejected` is a fluent fabrication (e.g. one asserting an unsupported death
date), and both `sft_narrative.py` and `dpo_kto_train.py` train on the output
(DPO loop runs green, adapter saved).

---

## Keeping ratings and candidates in sync (important)

The workshop ratings join to candidates by **`(record_id, candidate_i)`** — so the
ratings belong to the *exact* `workshop_candidates.json` the annotation sheet was
built from. If you regenerate candidates after the workshop (any new `bestofn` run
mints new record IDs and re-indexes candidates, and empty generations get dropped so
a record can end up with fewer than `k`), the ratings silently stop lining up: shared
records survive but the rated candidate slots don't, and pairs collapse to nothing.

`build_training_data.py` now reports the join before doing anything:

- `(5/5 records rated; 30/30 rated candidate-slots present in this file)` — good.
- `⚠ ID MISMATCH` — no rated record is in this file (wrong file entirely).
- `⚠ CANDIDATE-SET MISMATCH` — record IDs overlap but the candidate sets differ
  (the file is a different `bestofn` run). Fix: point at the file the sheet was built
  from.

Practice: **freeze the candidates file** the historians rate against (e.g. name it
`workshop_candidates_<date>.json`) so a later generation can't overwrite it.

---

## Training on the output

```bash
python sft_narrative.py --source jsonl --data data/EHRI/train/sft_<label>.jsonl --epochs 3
python dpo_kto_train.py --method dpo --pairs data/EHRI/train/preference_pairs_<label>.jsonl
```

Run each with `--dry_run` first — it loads the data, checks the schema and prints
length stats without touching the GPU. A typical recipe is **SFT on authored gold,
then DPO on the preference pairs** (SFT teaches the register; DPO sharpens the
good-vs-fabricated distinction).

---

## Dials

| flag | route | meaning |
|---|---|---|
| `--gate_unsup N` | reward, human | max unsupported specifics allowed in a "good" target (the faithfulness bar) |
| `--fab_unsup N` | reward | unsupported specifics that mark a candidate fabricated (→ `rejected`) |
| `--fab_F X` | reward | faithfulness at/below which a candidate counts as unfaithful |
| `--min_gap X` | human | min top-vs-bottom mean-rating gap (1–5) to emit a pair |
| `--gate_human_chosen` | human | also require the top-rated target to pass the faithfulness gate |
| `--dry_run` | all | summarise + gate curve, write nothing |
| `--out_dir` | all | where the jsonl files land (default `data/EHRI/train`) |

`make_authoring_sheet.py`: `--corpus/--source/--limit` (live ingest) or `--from
records.json`; `-o` the HTML, `--records_out` the record list.

---

## Safeguards (read before a real run)

- **Real names, real testimony.** Authored and non-redacted candidate texts carry
  real names. The output folder `data/EHRI/train/`, plus `records.json`,
  `authoring.html`, `authored_*.json`, `*candidates*.json`, `sft_*.jsonl` and
  `preference_pairs_*.jsonl`, are **gitignored**. Keep them local; do not commit.
- **Every row is stamped** `UNVERIFIED — machine-generated training target, not a
  historical source`, with provenance (corpus, source, gen model, record id,
  label source, author/confidence). The data is a methods artifact, not evidence.
- **Faithfulness first.** The gate exists so a fabrication never becomes a "good"
  target. Prefer a smaller, grounded set over a larger, hallucinated one — that is
  the whole argument.

---

## What already exists

No ready-made corpus of archival "good detail" summaries exists. The nearest
resources, and how they relate:

- **WikiBio** — 728k `infobox → biography` pairs ([Grangier et al.](https://github.com/DavidGrangier/wikipedia-biography-dataset)).
  The same task shape (structured record → short grounded prose about a person)
  and the standard data-to-text SFT corpus. Its known failure is *ours*: ~62% of
  biographies state facts absent from the infobox, which spawned a line of
  faithful-decoding work ([Tian et al., "Sticking to the Facts"](https://arxiv.org/pdf/1910.08684)).
  Use it as a **warm-start / faithfulness benchmark**, not as gold.
- **EHRI-NER** — multilingual Holocaust NER ([HTRes 2024](https://aclanthology.org/2024.htres-1.3.pdf)).
  Same corpus family, but entity recognition — feeds the *extraction* front-end
  (`dspy_extract.py`), not the summaries.
- **HTRes** — Holocaust Testimonies as Language Resources
  ([LREC 2024](https://aclanthology.org/volumes/2024.htres-1/),
  [HTRes-2026](https://www.clarin.eu/HTRes2026)). The community/venue; no
  off-the-shelf testimony→summary gold, which is precisely the gap this pipeline
  fills.

The absence of a gold set is the point: historian-authored "good detail" is the
missing data, and Route 3 is how it gets made.
