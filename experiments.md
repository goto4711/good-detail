# The experiments — three (+1) ways to define "good detail"

Every experiment optimises **the same base model**, from **the same SFT baseline**,
toward "good detail". They differ in **one thing only: where the notion of *good*
comes from.** That single-variable design is the whole point — read the result off
how the trained models *diverge*.

```
fact base → SFT baseline → GRPO (or DPO) under reward X → compare
                                         ▲
                            X = the only thing that changes
```

## The conditions

| # | Experiment | *Whose* "good"? | Reward source | Run it |
|---|---|---|---|---|
| 1 | **RLHF** (human feedback) | a community's values | `human_reward.py` (persona-weighted), or DPO on human/oracle pairs | `--reward human --persona grounding_first` |
| 2 | **RLAIF** (AI feedback) | a large model's values | LLM judge labels preferences → DPO | label with `llm_judge_reward.py --label`, then `dpo_kto_train.py --pairs …_llm.jsonl` |
| 3 | **Linguistic automation** | *no one* — surface stats | `linguistic_reward.py` (computed text features) | `--reward linguistic` |
| (R) | **Grounded automation** *(reference)* | the record itself | `composite_reward.py` (faithfulness-gated) | `--reward composite` |

Notes that protect the headline:

- **#1 is *simulated* RLHF for now** — the "human" is the oracle's value dimensions,
  persona-weighted (`human_reward.py`). It becomes *real* RLHF when the workshop /
  annotation supplies actual historian preferences. Same machinery, real labels.
- **#3 (linguistic) and (R) (grounded) are both automation but behave oppositely.**
  The surface reward is *blind to fabrication*; the grounded reward *catches* it.
  Keep them distinct — conflating them erases the key finding. Treat (R) as the
  reference / closest computable proxy to ground truth.
- **#2 (RLAIF) is a family, not a point.** Different AI judges disagree (Qwen vs.
  Gemini on the bland-vs-fabricated ordering), so `RLAIF-Qwen` and `RLAIF-Gemini`
  are distinct conditions — a sub-experiment about pluralism *between* AI judges.

## The exact commands (from the SFT baseline)

```bash
# shared baseline (once):
python sft_narrative.py --source llm --epochs 3      # writes sft-narrative-adapter

# 1 — RLHF (simulated human), pick the persona = whose values:
python grpo_train.py --reward human --persona grounding_first \
       --init_adapter sft-narrative-adapter --max_steps 400 --num_generations 8

# 3 — linguistic automation (culture-blind):
python grpo_train.py --reward linguistic \
       --init_adapter sft-narrative-adapter --max_steps 400 --num_generations 8

# R — grounded automation (reference):
python grpo_train.py --reward composite \
       --init_adapter sft-narrative-adapter --max_steps 400 --num_generations 8

# 2 — RLAIF (serve an LLM judge first, label, then DPO):
python llm_judge_reward.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --label --limit 8
python dpo_kto_train.py --pairs synthetic_corpus/preference_pairs_llm.jsonl --method dpo --max_steps 400
```

## What you read off the comparison

Two levels:

**Reward-level (no training, instant)** — `python run_pipeline.py`, `composite_reward.py`,
`human_reward.py` print how each reward ranks the same good/flattened/fabricated
texts. This already shows the core result: the surface reward rates the fabrication
as high as the grounded text (≈4/16 good>fab), while the grounded/human rewards get
it 16/16. *Where the rewards disagree is the experiment in miniature.*

**Model-level (after training)** — generate from each trained arm on the same record
and compare prose + scores:

```bash
python generate.py --sft_adapter sft-narrative-adapter --adapter grpo-composite-adapter   --limit 2
python generate.py --sft_adapter sft-narrative-adapter --adapter grpo-human-adapter        --limit 2
python generate.py --sft_adapter sft-narrative-adapter --adapter grpo-linguistic-adapter   --limit 2
```

Watch especially the **linguistic** arm for *fluent-but-fabricated* drift (invented
names/dates, `unsup` > 0) — the failure the surface reward cannot see, and the
clearest demonstration of what a "culture-blind" reward does to the writing.

## Results (synthetic; run of 2026-06-19)

Read at two levels. **The reward level is the spine of the result** — it is clean,
cheap, and does not depend on training. The model level is an
infrastructure-complete but **scale-limited demonstration** (16 prompts, 300 GRPO
steps, one 24 GB A10). Per the validity argument in `methods_rewards.md §6`, the
findings live in the *off-diagonal* (what an arm costs on the axes it was *not*
trained on) and in the agreement structure — never in the tautological diagonal.

### Reward level (fixed good/flattened/fabricated text — the spine)

- **Discrimination.** The composite (faithfulness-gated) reward ranks *good* above
  *fabricated* in **16/16** grid cells; the surface **linguistic** reward only
  **4/16** — it rates fluent fabrication as highly as grounded text.
- **NLI faithfulness separates fabrication from blandness** (mean F / unsupported):
  good **0.83 / 0.56**, flattened **0.88 / 0.12**, fabricated **0.49 / 3.00**.
  Bland-but-honest is *not* punished as false — the distinction the lexical and
  naïve versions both missed.
- **Orthogonality (answers "is grounding just the surface reward?").**
  `corr(linguistic_reward, NLI-faithfulness) = −0.34` across the grid. Negative:
  optimising surface specificity *trades against* grounding. They measure different
  constructs — the quantitative core of the culture-blind-vs-situated claim.
- **The LLM judge (RLAIF) is vividness-biased and inconsistent** (16 cells, Gemini
  2.5-flash, 1–10): it gives a **perfect 10** to the linguistic arm's least-grounded
  output (F=0.00, composite −1.50), and assigns **9 vs 2** to two near-identical
  texts (both F=0.75). It does not track grounding and is high-variance.

### Model level (3B, terminating, NLI grounding — scale-limited demonstration)

Infrastructure now correct: Qwen2.5-3B-Instruct, chat-template prompts → the policy
**terminates** (`mean_terminated_length ≈ 100`, `clipped_ratio` 0.25–0.5, down from a
pegged 1.0), NLI faithfulness, four named arms, three guards. But the 3B SFT
baseline is already strong, so 300 GRPO steps on 16 prompts moved the policy little
— several arms are byte-identical to baseline in most cells. Means over the 4-cell grid:

| arm (reward) | mean F | unsup | composite | linguistic | judge |
|---|---|---|---|---|---|
| baseline (SFT) | 0.62 | 2.0 | −0.70 | 2.63 | 7.0 |
| **composite** (grounded ref) | 0.65 | 2.25 | −0.71 | 2.76 | **3.75** |
| **human** (RLHF, balanced) | 0.63 | 2.0 | −0.69 | 2.62 | 7.25 |
| **linguistic** (culture-blind) | 0.56 | 1.75 | **−0.41** | **3.39** | 7.0 |

What moved is instructive. The **linguistic arm** is the one arm that visibly trained
— straight into surface **reward-hacking**: ALL-CAPS names, a fabricated person
("*Wilhelm Lederer*", invented by expanding the source code "WL/SYN/301"), and a
**record-dump** that reformats the fact base into a list, gaming *both* the surface
reward (highest linguistic, 3.39) *and* faithfulness (copying = trivially grounded).
The judge **rewards this arm (7.0) over the grounded composite arm (3.75)** — so
RLAIF with this judge would push *away* from grounding. The record-dump exploit is
now blocked by a `format_guard` (added 2026-06-19) for future runs.

### The story
A surface measure cannot define good archival detail (it rewards fabrication and is
gameable); a grounded one can; and the two *automatable alternatives a practitioner
would reach for* — surface statistics and an LLM judge — **fail differently**: the
surface reward is blind to fabrication, the LLM judge is biased toward fluent
fabrication and internally inconsistent. *Whose* grounded values to use (human
persona, or which AI judge) is itself contested — the thesis, as an experiment.

**Honest limits:** synthetic data; 16 cells; one judge model; GRPO under-powered at
this scale (the policy-level separation needs more prompts/steps or a real-data
setting — see `NEXT_STEPS.md`). The reward-level findings do not depend on any of
that.

## Real-data shakedown — EHRI (run of 2026-06-19)

A first contact with real material: the trained arms run on real EHRI early
Holocaust testimony via the `ingest` layer (`realdata_generate.py --corpus EHRI`),
scored with NLI faithfulness against the **real source text** + an attested-entity
check. Outputs are machine-generated, redacted and stamped UNVERIFIED — a methods
diagnostic, never a historical source. Three documents, two grounding sources.

**The synthetic off-diagonal reproduces on real testimony — more starkly.** Means
over the 3 XML records:

| arm | mean F | mean unsup | mean composite | mean linguistic |
|---|---|---|---|---|
| **composite** (grounded) | **0.59** | **6.3** | **−2.92** | 2.51 |
| human | 0.48 | 6.7 | −3.14 | 2.56 |
| **linguistic** (culture-blind) | 0.47 | **8.0** | **−3.81** | **3.01** |

The linguistic arm is again the most fabricated *and* highest on its own surface
score. Clearest on the Carlebach/Buchenwald record: composite → linguistic, `unsup`
**doubles 5 → 10** and `F` **halves 0.67 → 0.33** while linguistic *rises* 2.31 →
3.16. Concretely, the linguistic arm embellished "Ernst Heilmann, *former President
of the Prussian Landtag*", "*Kurt Eisner Jr.*", and an SS "*Dr. List*" — confident,
specific-sounding invention with no anchor in the record. The thesis, on real data.

**Structured grounding (the IOB source) is the lever.** All XML-source arms
fabricate far more on real data than synthetic (unsup ~6–8 vs ~1–3), because the
TEI *header* is an entity **list**, not an event fact base — the model fills gaps
with plausible invention. But with the relation-extraction source (`--source iob`,
typed entities + relations), the composite arm produced its first **fully faithful**
real-data output — `iob_chunk_2`: **F=1.00, unsup=0, composite +0.62** (the Rabbi of
Kittsee and innkeeper Sieber). No XML-source output came close. Direct evidence that
a real *fact base of relations* lifts grounding — and the motivation for the DSPy
extraction front-end (`NEXT_STEPS §3`).

**Honest limits:** N=3 (directional, not significant); the per-record signal is
carried mostly by Carlebach (records 2–3 are noisier — on the Berlin record
linguistic's F edged composite's); the arms were never trained on real EHRI; lexical
retrieval for the premise is crude. What holds is the *direction* and the IOB result,
both of which point at the fact-base front-end as the next real lever.

### From raw text — the DSPy extraction front-end (run of 2026-06-19)

Closing the loop: raw testimony → `dspy_extract.py` (DSPy `ChainOfThought`, LM via
`config.BACKENDS` — uva proxy or local vLLM) → a relation-level fact base
(`data/EHRI/iob/extracted.jsonl`, wired as the `extracted` corpus source) → grounded
generation. So the pipeline now *begins* at raw text and builds its own fact base.

Grounding against that extracted fact base roughly **halves fabrication** versus the
entity-list XML path:

| grounding source | arm | mean F | mean unsup |
|---|---|---|---|
| XML (entity *list*) | composite | 0.59 | 6.3 |
| XML | linguistic | 0.47 | 8.0 |
| **extracted (relations)** | composite | 0.59 | **3.7** |
| **extracted** | linguistic | 0.56 | **4.0** |

The `iob_chunk_2`-style win reproduced: a concrete, fact-dense chunk (`x2`: rations,
work hours, escape attempts) scored **F=1.00, unsup=0** for composite and F=0.92 for
linguistic. And a second, subtler observation: on the extracted path the **arm
difference collapses** (composite ≈ linguistic, near-identical generations) — when
the grounding context is tight, the *fact base* dominates and the reward arm matters
less. Culture-blind vs situated rewards diverge most when the record is loose.

**Two confounds — not yet a clean A/B.** The `--limit 3` extracted run scored three
*chunks of the Carlebach document*, while the XML run scored three *different
documents* — not the same text. And the extracted path feeds short (~150-word)
self-contained chunks as the premise vs the whole 27k-char document, so part of the
`unsup` drop is chunk size, not relations. The isolating experiment — same chunks,
relations on vs off (`realdata_generate.py --no_relations`) — is wired and pending. If
relations-on beats relations-off on identical text, the fact base earns its keep free
of the chunk-size confound. Provisional reading: **the extraction front-end lifts
grounding on real data, and the fact base — not the reward — is the dominant lever.**

### Inference-time track — reward-as-selector (the historian demo)

Separate from the trained-arm study above: `bestofn_demo.py` has a **strong, pluggable
LLM** write K candidates per record, and each reward *picks* its favourite. No training,
no GPU for generation, any current model (`--gen_backend`/`--gen_model`). Where composite,
linguistic, and judge select *different* candidates is the thesis made legible with a
recognisable model — and it scales the comparison to "the newest LLM" without GRPO. See
`methods_rewards.md §7` for the training-vs-selection framing. Run at scale with
`run_ehri.sh` (trained arms, aggregate metrics); demo with `bestofn_demo.py`.

### At scale (overnight run, N=150) — what held and what didn't

The 2026-06-19 overnight sweep (extract 400 chunks, score 150/arm, GRPO-on-real
arms, best-of-N) revises the N=3 reading honestly:

1. **Source focus is the real lever.** Extracted/chunked source roughly halves
   fabrication vs the XML entity-list path — unsup 6.7 → 4.2, F 0.43 → 0.59 —
   robustly across every arm at N=150.
2. **The relations ablation is NULL.** Relations on ≈ off (composite F 0.590 vs
   0.600, unsup 4.19 vs 4.19; same for all arms). The gain is the *focused source
   passages* (a better NLI premise), **not** the relation triples — consistent with
   the extractor's relation-link F1 of only 0.40 (entities 0.78), i.e. the relations
   are both noisy and unhelpful here. The earlier "the fact base is the lever" claim,
   from N=3, was an overclaim the ablation caught.
3. **The reward arms converge on real trained-policy output.** composite ≈ linguistic
   ≈ human (extracted: F 0.590 / 0.591 / 0.589; unsup 4.19 / 4.43 / 4.17) — predicted
   direction, negligible magnitude. **GRPO-on-real-prompts did not help** (composite
   reward drifted −2.38 → −2.58 over 300 steps; the real-trained arm matches/under-
   performs the synthetic one).

**Conclusion.** The contribution lives at the **reward level** (discrimination on
fixed text, validated on the synthetic grid) and in **inference-time selection**
(best-of-N) — *not* in trained-policy differences, which do not transfer to real
testimony at this scale. The trained-arm comparison is, at N=150, a near-null.
