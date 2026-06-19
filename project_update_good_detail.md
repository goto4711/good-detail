# From Profile to Narrative: Reward Models for "Good Detail" in Historical Archives

> **Status (17 June 2026):** much of the workflow proposed below is now prototyped on synthetic data — TEI/IOB serialisers + ingestion adapters, a decomposable linguistic reward, a faithfulness-gated composite reward, and a guarded GRPO loop (open-book prompts, length + anti-copy guards) that runs end to end on a MacBook. For the live build status, the latest results, the reward-functions reference, and the real-run profile, see **`PROJECT_STATUS_2026-06-17.md`** — that file is the source of truth; this document is the conceptual rationale.

**Project update / research design — June 2026**
**Corpus:** EHRI early Holocaust testimonies (Wiener Library) and other historical/archival sites
**Codebase:** `For-Chloe/` (this folder)

---

## 1. The pivot in one paragraph

The project is no longer about Named Entity Recognition and relation extraction as an end in itself. The new core contribution is a **reward model for "good detail"**: a co-designed signal, built with historians and Holocaust-studies experts, that uses preference-based post-training to nudge a model toward **context-rich, archivally grounded micro-narratives** — of resistance, counter-narratives, the unexpected — and *away* from generic, flattened, profile-like summaries. The earlier idea of using optimisation "not to remove but to detect the unexpected" now operates at the level of **narrative paths through collections**, rather than entity spans. NER/RE does not disappear; it is repurposed as the *verifiable grounding component* of the new reward.

---

## 2. The key conceptual shift: from a verifiable to a non-verifiable reward

The existing GRPO experiment (`grpo-experimental/run_grpo_paired.py`) rewards **verifiable** properties: JSON well-formedness, schema consistency, and entity F1 against ground truth. This is exactly why the open-r1-text2graph recipe works — structured extraction has a deterministic checker, just like maths or code. Reinforcement learning with verifiable rewards (RLVR) is a good fit there.

"Good detail" is **not verifiable** in this sense. There is no ground-truth function that returns whether a micro-narrative is detailed, contextual and ethically attentive. This is the heart of the new research problem, and it means the recipe cannot be copied across unchanged. The contribution is precisely *how you turn "good detail" into a trainable signal at all.*

This also explains a confusion currently sitting in the code. `generate_paired_dataset.py` produces `(prompt, chosen, rejected)` triples — the data shape for **DPO / preference optimisation** — but `run_grpo_paired.py` then hands `reward_funcs` to `GRPOTrainer`, which does not consume chosen/rejected pairs at all; GRPO samples completions per prompt and scores them with the reward function. The two halves belong to two different paradigms. The pivot is the moment to pick one deliberately (see §4).

---

## 3. Decomposing the reward — the central design move

Do not treat "good detail" as a single opaque score. The three criteria from the brief sit at very different levels of difficulty, and separating them is what makes the project tractable *and* what makes the existing pipeline reusable.

**(a) Grounding / specificity — partly verifiable. This is where the old pipeline returns.**
Actors, dates, places, institutions and material contexts named in a narrative can be checked against the source record. Does this entity/date actually appear in, or follow from, the document? The NER/RE machinery in `prepare_tei_data.py` and `prepare_iob_data.py` becomes a **faithfulness reward**: reward narratives whose specifics are anchored in the record, penalise invented ones. A natural-language-inference / entailment check (narrative claim ⊨ source text) gives a softer, graded version of the same idea. *This component must dominate the reward* — see §6 on hallucinated detail.

**(b) Source awareness / calibrated uncertainty — detectable, fuzzier.**
Reward explicit reference to archival sources (document IDs, provenance) and appropriately hedged language; penalise confident claims the record does not support. Partly rule-based (does it cite a source?), partly judged.

**(c) Anti-sensationalism / anti-denial-and-relativisation — hardest, and should not live in a scalar reward alone.**
A single reward number invites gaming on exactly the dimension where gaming is most harmful. Handle this with expert-curated negative examples and a *separate guardrail classifier*, not only as a term inside the optimisation target.

The reward is therefore a **composite**: deterministic grounding checks (reusing NER) + entailment-based faithfulness + a rubric scored by an LLM-judge for the soft qualities + a guardrail. Transparency of the rubric is itself a research and ethics asset — historians can read, contest and revise it.

---

## 4. Recommended training path (pragmatic, given a small team)

Classic RLHF — train a separate Bradley-Terry reward model, then PPO/GRPO against it — is the most data-hungry and least stable route, and is plausibly why the current GRPO stalls. Suggested staging instead:

**Stage 0 — New SFT baseline (narrative, not entity-JSON).**
Replace the entity-JSON target in the SFT data with a **micro-narrative target** (3–5 sentences, or a small summary graph) about a historical unit: what it is, where it sits, who is involved, when and why it matters. Bootstrap targets from a strong model *constrained to real record fields*, plus a small historian-written gold set. This reuses the chunking and instruction-templating already in `config.json`, `prepare_tei_data.py`, `prepare_iob_data.py` and `merge_and_split_dataset.py` — only the assistant target changes.

**Stage 1 — Preference optimisation with DPO or KTO before any RL.**
These are far more stable and need no reward-model loop or online sampling. **KTO is especially well suited here**: it needs only *binary* good/bad labels, not ranked pairs — historians can mark a narrative "good" or "bad," a much lighter annotation ask than ranking. This reuses the `(prompt, chosen, rejected)` plumbing in `generate_paired_dataset.py`, but with one critical change: **the preference must be over detail quality, not "gold vs. the model's own sample."** Currently `rejected` is just a temperature-0.7 generation from the SFT model and `chosen` is the gold target — that teaches the model to imitate gold, not to prefer good detail. Replace it with genuine human (or validated AI) judgements of detail.

**Stage 2 — GRPO with the composite reward, once the reward is trustworthy.**
Keep the `GRPOTrainer` scaffold in `run_grpo_paired.py`, but swap the extraction reward functions for the composite reward of §3, and feed it *prompts* (not chosen/rejected pairs). **Validate the LLM-judge against a small set of real historian ratings — measure inter-rater agreement — before you trust it to train anything.** A reward you have not audited will be optimised against faithfully, mistakes and all.

---

## 5. Data you now need

The bottleneck moves from entity annotation to **preference data over narratives**:

1. A pool of micro-narratives per record, generated at varied temperature so they differ in detail and grounding.
2. Judgements from domain experts — ideally binary good/bad (for KTO) plus a short reason, which doubles as rubric-development material and as audit trail.
3. A held-out, expert-rated evaluation set never used in training.
4. A small adversarial probe set: superficially "detailed" but fabricated narratives, denial/relativisation patterns, and genuinely surprising-but-grounded details, to test that the reward rewards the right thing.

A lightweight annotation interface for historians (record on one side, candidate narratives + good/bad + reason on the other) is now a first-class deliverable, not an afterthought.

---

## 6. Two tensions to state explicitly in the project (not bury)

**Hallucinated detail.** A reward for "detail" is a reward for producing specifics — and the cheapest way to score well is to *invent* plausible specifics. On Holocaust testimony this is not a minor failure mode; it is an ethical line. Mitigation: grounding/faithfulness (§3a) must outweigh every other term, and **uncertainty must be rewarded, not penalised**.

**Homogenisation vs. "detecting the unexpected."** RL optimises toward whatever the reward favours, which tends to *narrow* outputs onto a single high-scoring template — the opposite of surfacing the unexpected. There is a real tension between optimising a reward for "good detail" and preserving the irregular, surprising detail the project most values (reward hacking via surface markers: cram in dates and names). Mitigations worth building in and studying: a novelty/diversity term in the reward, strong KL regularisation to keep the base model's range, and preference data that explicitly rewards *surprising-but-grounded* detail. **This tension is the intellectually interesting core of the pivot and should be framed as a research question, not a footnote.**

Co-design with historians, archivists and (where appropriate) survivor communities is therefore both an ethical commitment and a methodological necessity: the reward *is* the theory of "good detail," and it should not be authored by engineers alone.

---

## 7. Concrete changes to this repository

| Current file | Change |
|---|---|
| `prepare_tei_data.py`, `prepare_iob_data.py`, `config.json` | Keep chunking + Gemini enrichment, but add a **micro-narrative target** alongside the entity JSON. Entities become *grounding metadata* for the reward, not the output. |
| `trl_sft_pipeline.py` | Repoint SFT from entity-JSON to narrative targets. Note this script *bypasses TRL entirely* — it uses a plain `transformers.Trainer` with manual tokenisation/collation, forces CPU/float32, and trains 1 epoch — a workaround for the old TRL compatibility pain. On current TRL (§8) you can drop the workaround and use `SFTTrainer` + `SFTConfig` cleanly. Plan the real run on SURF/Snellius GPU with a larger Qwen. |
| `merge_and_split_dataset.py` | Reuse as-is for the narrative SFT split; add a separate split for the **preference** dataset. |
| `grpo-experimental/generate_paired_dataset.py` | Repurpose to emit candidate narratives for *human judging*; stop using "gold vs. self-sample" as the preference signal. |
| `grpo-experimental/run_grpo_paired.py` | Decide the paradigm: **DPO/KTO** (consume chosen/rejected/binary) *or* **GRPO** (prompts + composite reward). Replace the extraction `reward_funcs` with the §3 composite. |
| `evaluate.py` | Retire entity-F1-only framing into a **faithfulness/grounding metric** + expert-rated narrative quality + the adversarial probe set. |
| *(new)* `reward/` | Composite reward: grounding check (NER-based), entailment/faithfulness, rubric LLM-judge, guardrail classifier. |
| *(new)* `annotation/` | Historian-facing good/bad + reason interface; exports KTO/DPO data. |

---

## 8. The TRL rewrite — assume the framework code is stale

The early-2025 TRL these scripts were written against has been superseded. **TRL reached v1.0 (April 2026), a unified post-training stack covering SFT, reward modelling, DPO and GRPO.** The churn you fought is real, but the upside is that the modern API is more consistent and stable. Treat the framework-facing code as needing a rewrite, not a patch — and isolate yourself from future churn:

**Pin the stack.** Add a `requirements.txt` (or lock file) pinning exact versions of `trl`, `transformers`, `peft`, `accelerate`, `datasets`, `bitsandbytes`. Most of your past pain came from these drifting independently. Pin once, upgrade deliberately. Pin the *same* versions locally and on Snellius so "works on my laptop, breaks on the GPU node" stops happening.

**Start from the official example for the installed version, not from old code.** For each stage, copy the current TRL example script for *your pinned version* and port your data/reward into it, rather than migrating the 2025 scripts line by line. The examples track the API; your old scripts do not.

**Concrete API changes to expect when rewriting:**

- **Per-trainer config classes.** Each trainer now has its own config — `SFTConfig`, `DPOConfig`, `KTOConfig`, `GRPOConfig` — all inheriting from `transformers.TrainingArguments`. Trainer-specific knobs (`max_seq_length`, `packing`, `dataset_text_field`, `completion_only_loss`, etc.) live on these configs, not on a bare `TrainingArguments`.
- **`tokenizer=` → `processing_class=`.** The `SFTTrainer`/`GRPOTrainer` argument for the tokenizer/processor was renamed; old `tokenizer=...` calls will warn or break.
- **SFT data handling is simpler.** `SFTConfig` defaults to `completion_only_loss=True` for prompt–completion datasets and handles chat templates (it can even patch a template lacking generation markers). Much of the manual masking/formatting in the old scripts is no longer needed — a `formatting_func` or a conversational dataset is usually enough.
- **GRPO reward-function signature is fixed:** `def reward_func(completions, **kwargs) -> list[float]`. You may pass a *list* of reward functions; the total reward is their sum, or a weighted sum if you set `reward_weights` in `GRPOConfig`. This maps cleanly onto the composite reward of §3 — each component (grounding, faithfulness, rubric, guardrail) becomes one function in the list, with `reward_weights` making faithfulness dominant.
- **GRPO consumes prompts, not pairs.** Reconfirming §2/§4: feed `GRPOTrainer` a prompt-only dataset plus `reward_funcs`. The `(chosen, rejected)` columns from `generate_paired_dataset.py` are for `DPOTrainer`/`KTOTrainer` instead.

**Practical sequence:** pin versions → reproduce a trivial `SFTTrainer` run on 10 examples to confirm the environment → port the narrative SFT → add `KTOTrainer`/`DPOTrainer` → only then rebuild GRPO with the composite reward. Get each stage green on the pinned stack before adding the next.

---

## 9. Suggested next steps

1. **Pin the TRL stack** (§8) and get a trivial `SFTTrainer` run green on both laptop and Snellius — fix the environment before touching the science.
2. Write the micro-narrative SFT target spec with one historian; hand-author ~50 gold narratives.
3. Stand up the grounding-faithfulness reward first (it reuses existing NER and is the safety-critical term).
4. Collect a few hundred binary good/bad judgements; run **KTO** as the first real alignment experiment.
5. Validate an LLM-judge against those human ratings; only then attempt GRPO with the composite reward.
6. Keep the adversarial probe set running throughout as a reward-hacking tripwire.

---

*Open question for the next working session: pull the current literature on rubric/AI-feedback rewards, long-form faithfulness scoring, and DPO/KTO so the design above can be properly cited.*
