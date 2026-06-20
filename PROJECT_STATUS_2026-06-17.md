# Good Detail — project status & handover

> ⚠️ **Dated snapshot (17 June 2026) — superseded.** The project has moved well beyond
> this note: NLI faithfulness, the corpus-agnostic ingest layer, real EHRI data end to
> end, the DSPy extraction front-end, at-scale results (N=150), inference-time best-of-N
> selection, and the historian-workshop apparatus. **For the current state read
> `experiments.md` (results), `FRAMING.md` (framing), and `NEXT_STEPS.md` (roadmap).**
> Kept for history. (The folder is now `good-detail/`, not `For-Chloe/`.)

**Date:** 17 June 2026
**Folder:** `For-Chloe/` (this folder)
**Purpose of this note:** remind future-me, in ~2 weeks, where the project stands and what to do next.

---

## What this project is now (the pivot)

It started as a TRL fine-tuning experiment for NER + relation extraction on EHRI Holocaust testimony. We **repivoted** it into a methods contribution to *culturally aware deep learning in DH*:

> A reusable **workflow** for aligning a generative model to a contested, community-defined notion of quality — here **"good detail"** in archival micro-narratives — and **comparing two ways of operationalising it**: a **human/community reward** vs. a **linguistic (surface) reward**. The headline question is whether a humanistic quality can be captured by computable text measures, or whether it needs situated human judgement.

Generality is claimed **across corpora** (registers first); EHRI is just one future instantiation. The NER work isn't discarded — it becomes the *grounding/faithfulness component* of the reward.

Deeper rationale is in **`project_update_good_detail.md`**. The experimental substrate design is in **`synthetic_factbase_spec.md`**. Read those two first if the framing has gone fuzzy.

---

## What we achieved today

1. **Reframed the project** and wrote it up (`project_update_good_detail.md`), including the TRL-rewrite plan (TRL hit v1.0 in April 2026; old scripts target a dead API).
2. **Designed the synthetic fact-base** (`synthetic_factbase_spec.md`): a two-layer model (register-invariant facts → rendered text), five control tags (specificity, grounding, certainty, salience, sensitivity), and a 2×3 rendering grid (testimony / finding-aid × good / flattened / fabricated).
3. **Built and ran the generator** (`generate_synthetic_corpus.py`): 8 fictional cases → 48 renderings + 32 preference pairs, with an **oracle** judge (sees ground truth) and a **linguistic** judge (surface only).
4. **Got the key result** (see below).
5. **Built the preference-optimisation smoke test** (`dpo_kto_train.py`) and **ran it green on the MacBook (MPS)** — DPO ✓ and KTO ✓.
6. Produced **SURF batch scripts** (`surf/`) and a **local runner** (`run_local.sh`) for when the cluster is back.

## The key result (the empirical seed)

On the synthetic grid:

| profile | ORACLE (ground-truth) | LINGUISTIC (surface) |
|---|---|---|
| good | 0.85 | 1.131 |
| flattened | 0.075 | 0.235 |
| fabricated | **−0.275** | **1.232** |

The **oracle** cleanly separates good ≫ flattened ≫ fabricated. The **linguistic** metric is *blind to grounding*: it rates fabricated **higher** than good on average (and in 10/16 cells). → A plausible surface specificity metric **cannot tell grounded detail from fabricated detail**, while a judge with access to grounding can. That's the existence proof underpinning the whole human-vs-linguistic argument.

---

## Implementation status — built vs. not built

**Read this if nothing else.** Today was mostly *scaffolding + one static result*. We proved the machinery runs; we have **not** yet trained any model that produces or improves "good detail," and nothing touches real corpora or real people yet.

Legend: ✅ done · 🟡 partial / proof-of-concept only · ⬜ not started.

**Data & substrate**
- ✅ Synthetic fact base + generator (8 fictional cases)
- ✅ Deterministic templated renderer → 2×3 grid (testimony/finding-aid × good/flattened/fabricated)
- 🟡 LLM rendering with grounding-verification (the "v2" path, `llm_render.py`) — LLM writes the narrative (reusing the judge backends), then `faithfulness()` verifies: **good accepted only with 0 unsupported specifics** (else regenerate, then fall back to templated), fabricated accepted only if it hallucinates. Validated with real ollama llama3.2 (good 4/4 accepted at F=1.00, natural prose, e.g. "I remember the day I left Brünn…"). Output `llm_renderings.jsonl` → cleaner, less-stilted SFT targets than the templated ones. *Caveat:* F-verification is entity-level — it catches invented names/dates, NOT a fabricated **frame** (e.g. first-person "I remember", which the record doesn't establish). Claim-level NLI would; eyeball good outputs for framing liberties. (Nice asymmetry: llama3.2 is too weak as a *judge* but fine as a *generator*.)
- ✅ TEI + IOB serialisers with gold annotations (`emit_tei_iob.py`) — round-trip lossless on synthetic data (TEI 16/16, IOB 16/16)
- 🟡 Ingestion adapters (TEI/IOB → record units, `ingest_adapters.py`) — tested reference parsers, **but only on clean synthetic files**; real messy EHRI files still to harden against. Also demonstrated the **tag-loss boundary**: the linguistic reward survives ingestion, the oracle/human reward does not (TEI/IOB carry text + entities, not the salience/grounding tags).

**Rewards / judges**
- ✅ Oracle judge (ground-truth, from the tags) — but weights are placeholders, to be set with historians
- 🟡 Linguistic reward (`linguistic_reward.py`) — now **decomposable**: proper-noun density, number/date density, concreteness, lexical density, calibration; length-normalised; per-feature validation on the grid (shows which features are blind to fabrication vs. which catch it). **Requires an external concreteness lexicon** (see below). Surprisal feature (needs an LM) and NLI-grounding: ⬜
- 🟡 Human / community reward — **synthetic** version built (`human_reward.py`): the oracle's value dimensions scored from text, weighted by a **persona** (the pluralism knob), with optional noise. Wired into GRPO via `--reward human --persona ...`. Real human annotations still ⬜ (this synthesises the human arm so the two-arm experiment runs now).
- 🟡 **LLM-as-judge** (#4, `llm_judge_reward.py`) — RLAIF, **backend-swappable** via `--backend`: `uva` (proxy), `ollama` (local, no key/network — easiest), `gemini` (free OpenAI endpoint). The *realistic* synthetic human. Used offline: `--validate` checks agreement with the oracle (do this before trusting it), `--label` writes an LLM-judged preference set for DPO/KTO. `--mock` runs the harness offline. Not a live GRPO reward (API rate limits). Validation against *real* historians still ⬜.
- 🟡 Composite reward (`composite_reward.py`) — built + validated on the grid (good>fabricated **16/16** vs. **4/16** surface-only; faithfulness gate now active, fabricated F≈0.64); wired into GRPO via `--reward composite`.

**Model training**
- 🟡 SFT narrative baseline (`sft_narrative.py`) — built: SFTs Qwen2.5-0.5B + LoRA on (record → **good** narrative) pairs (the "good" renderings are the targets; same open-book prompt format GRPO uses). Smoke-test stage. Caveats: targets are *templated* (stilted style the model will inherit) and only 16 examples — for real, use varied / historian-written / LLM-rewritten-and-verified targets + more cases. **Hand-offs now wired** (the chain is continuous): SFT can train on the v2 grounding-verified targets via `--source llm`, and GRPO starts from the SFT baseline via `--init_adapter sft-narrative-adapter`. The whole chain runs with `bash run_all.sh --train`.
- ✅ Synthetic preference pairs (32: good>flattened, good>fabricated)
- ✅ DPO/KTO training **loop mechanics** — smoke-tested green (runs end-to-end)
- ⬜ *Meaningful* preference training + evaluation of the trained model (the smoke test learned nothing; 32 toy pairs, 20 steps)
- 🟡 GRPO (`grpo_train.py`) — loop + reward-function plumbing built and **dry-run validated** (correct paradigm: prompts + `reward_funcs`, *not* chosen/rejected; reward variance confirmed so there's a learning signal). **Now completes a real 20-step run on MPS** (~8.6 s/step) with NO group collapse (`reward_std`>0, `frac_reward_zero_std`=0). Now supports **`--reward composite`** (faithfulness-gated) and a soft **over-length penalty** (`length_guard`, default on; `max_completion_length` 256) to counter the observed reward-hacking (`completions/clipped_ratio`=1.0, none terminating). The length guard is a separate reward function summed in, logged as `rewards/length_guard/mean`. Prompts are now **open-book** — each carries a compact rendering of the source record (entities, events, dates, source refs) and says "use ONLY facts from the record." A first composite run with *closed-book* prompts (just the focal name) made the reward uniformly negative with no gradient toward good, because the model had nothing to be faithful to; open-book fixes that. An **anti-copy guard** (separate reward stream, default on) penalises excessive verbatim overlap with the record — 5-gram overlap above a 0.30 tolerance — so the model can't game faithfulness by regurgitating the record (validated: verbatim copy −1.4, real narrative 0). GRPO reward streams are now: composite + length_guard + anti_copy_guard. Latest 20-step MPS run (open-book, all three guards): no collapse, `anti_copy`=0 (model not copying), composite floor lifted vs. closed-book, but reward flat and `clipped_ratio`=1.0 — all expected at 20 steps. A real *learning* run needs the profile in **Reward functions § GRPO runs & real-run profile** (⬜).

**The actual experiment (comparison / audit)**
- ✅ Static "blindness" demo: oracle vs linguistic scoring of the renderings (the result in the table above)
- ⬜ Train a human-aligned model **and** a linguistic-aligned model, cross-evaluate, measure the divergence — *this is the study, and it is not started*

**Infrastructure**
- ✅ Local runner (MacBook/MPS) + requirements; ran green today
- ✅ SURF batch bundle (`surf/`) — written, not yet run (cluster maintenance)
- 🟡 Pinned lock file — do `pip freeze > requirements-local.lock.txt`

**Bottom line:** the *pipeline plumbing* exists and runs; the *science* (a real narrative model, real human reward, rich linguistic reward, the composite reward, GRPO, and the two-model comparison) is still ahead, plus everything that needs the workshop and real corpora.

**External data resources required (NOT pip packages — must be obtained separately):**
- **Concreteness lexicon** for the linguistic reward — the Brysbaert et al. (2014) concreteness norms (~40k words). Drop it beside the code as `concreteness_norms.csv` (columns: `word,rating`) and `linguistic_reward.py` loads it automatically; otherwise it falls back to a small built-in **stub** whose numbers are illustrative only, not research-grade.
- (later) a small generic **language model** (e.g. GPT-2/distilGPT-2) for the surprisal / "detect the unexpected" feature.
- (later) an **NLI model** for the grounding/faithfulness reward (narrative claim ⊨ source).

---

## Reward functions (the heart of the project)

The reward is the *research object*, not a training detail — it's where "good" gets
defined, and the study compares two ways of defining it: a **situated/grounded**
notion vs. a **culture-blind surface** notion.

| Reward | Reads | Role | Status |
|---|---|---|---|
| **Oracle** | hidden fact-base tags (salience, grounding, certainty, sensitivity) | ground-truth "good detail" judge; stand-in for historians in the synthetic phase | ✅ (weights provisional) |
| **Linguistic** (`linguistic_reward.py`) | generated text only (surface) | the **culture-blind** arm — decomposable, length-normalised; mostly blind to fabrication | 🟡 (needs real concreteness lexicon; surprisal ⬜) |
| **Faithfulness / grounding (F)** | generation vs. source record | situated anchor: are the specifics actually supported? | 🟡 exact lookup built (`composite_reward.py`); NLI/LLM-judge for real ⬜ |
| **Composite, faithfulness-gated** | all of the above | the deployable GRPO training reward; text-computable analogue of the oracle | 🟡 built + validated (good>fab **16/16** vs **4/16** surface); gate active; wired into GRPO |
| **Synthetic human** (`human_reward.py`) | generation + record | the **human arm**: oracle dimensions, persona-weighted, optional noise — stands in for historians so the two-arm experiment runs now | 🟡 built + validated; wired into GRPO (`--reward human --persona`) |
| **LLM-as-judge** (`llm_judge_reward.py`) | generation + record, via UvA proxy | realistic synthetic human (RLAIF) + judge-vs-oracle validator + offline preference labeler | 🟡 built; offline labeler/validator (not a live GRPO reward — API rate limits); needs `UVA_LLM_API_KEY` |

### The composite, faithfulness-gated reward

Idea in one line: **ground first, then reward detail** — richness counts only if it's faithful.

```
reward = F^γ · (w_q·Q + w_c·C)  −  w_fab·(#unsupported_specifics)  −  w_sens·Sensational
```

- **F** (faithfulness ∈ [0,1]) gates *multiplicatively*: low grounding scales the quality you can earn toward 0 (γ>1 sharpens the gate). You cannot bank detail-reward without being faithful.
- **#unsupported_specifics** is penalised, so fabrication is *negative*, not merely unrewarded — ethically required here (inventing detail about a victim should score worse than blandness).
- **Q** = surface quality (the linguistic features), **C** = calibration, **Sensational** = restraint penalty.

**Computing F:** entity grounding (a named specific absent from the record = unsupported) + claim-level **NLI entailment** (claim ⊨ record). It's an *exact, free lookup in the synthetic phase* (we own the facts), and a *noisy model estimate on real corpora* (validate the estimator against humans first).

**Plugs into GRPO** via a per-prompt `record` dataset column that TRL forwards into the reward function's `**kwargs`, so each completion is scored against its own source.

**Worked example (Marta, testimony):** *good* names only grounded specifics and hedges the uncertain ones → F≈1, no fabrications → high reward. *Fabricated* adds "Gestapo seized the workshop / March 1939 / deported to the east" (none in the record) → F low, so `F^γ·Q ≈ 0`, minus fabrication and sensationalism penalties → **strongly negative**. This flips the error the linguistic-only reward made (where fabricated outscored good). In effect the composite is the **deployable analogue of the oracle**: the bridge from the oracle's hidden-tag judgement to a signal computable from real generations.

**Gate validated (17 June).** The fabrications now inject *hallucinated specifics* (false named entities + a false precise date — the realistic hallucination mode), so the gate is properly exercised. Result on the grid:

| profile | F | unsupported | linguistic | composite |
|---|---|---|---|---|
| good | 1.00 | 0.00 | 2.79 | **0.80** |
| flattened | 1.00 | 0.00 | 1.13 | 0.36 |
| fabricated | **0.64** | 4.25 | **3.00** | **−2.02** |

good > fabricated: **composite 16/16** vs. **linguistic-only 4/16**. Note the sharpening: hallucinated specifics make the *surface* reward score fabricated even higher (3.00, the best of the three!), collapsing linguistic-only to 4/16, while the faithfulness gate (F 1.00→0.64, so `F^γ` ≈ 0.41) plus the unsupported-specifics penalty drive the composite strongly negative. **This is the project thesis in its sharpest form: the better a fabrication looks on the surface, the more the situated/grounded reward is needed — and the more the culture-blind reward fails.**

**Subtleties to build in:**
- *Hedged uncertainty ≠ fabrication* — penalise a claim only if asserted *as certain* and unsupported.
- *Guard against verbatim copying* (a model can max F by parroting the source) — add an n-gram-overlap penalty / reward synthesis.
- *The gate is only as good as the F estimator* — on real data F is an NLI/LLM-judge estimate; validate it against human ratings before trusting it.
- Keep optimizer-level defences (KL-to-base, group diversity, per-feature caps) and the fabricated renderings as a permanent reward-hacking tripwire.

**Open design question:** does **calibration** belong *inside* the linguistic (culture-blind) arm or as its own axis? The grid shows calibration partly *detects* fabrication via hedging cues, so putting it in the "culture-blind" arm makes that arm not-quite-blind. Decide deliberately — it changes what the human-vs-linguistic comparison actually measures.

### Synthetic human arm & pluralism (validated 17 June)

`human_reward.py` synthesises the **human arm** without annotators: it scores the oracle's value
dimensions (coverage, specificity, grounding, source-awareness, calibration, restraint, minus a
fabrication penalty) from text, weighted by a **persona**. Result on the grid:

| persona | good | flattened | fabricated | ordering |
|---|---|---|---|---|
| balanced | 5.78 | 4.07 | 0.49 | good > flat > fab |
| grounding_first | 6.07 | 4.53 | **−3.80** | good > flat > fab |
| ethics_first | 7.07 | 5.42 | −0.85 | good > flat > fab |
| specificity_first | 4.92 | 3.09 | **+2.30** | good > flat > fab |

- **Two-arm comparison:** human(balanced) ranks good>fabricated **16/16**; the linguistic (surface) arm only **4/16**. The situated arm sees through hallucination; the culture-blind arm doesn't. That gap *is* the study.
- **Pluralism:** every persona rejects fabrication, but they value it very differently (fabricated scores −3.80 under `grounding_first`, +2.30 under `specificity_first`), and `specificity_first` even ranks the *vivid fake above the bland-but-honest* record in 2/16 cells (others: 0/16). So **"whose good detail?" becomes measurable** — the synthetic form of value pluralism, and a result you can get with zero real annotation.
- **Noise** knob (`noise=σ`) lets you synthesise a realistic, disagreeing annotation set for data-efficiency studies.

**Convergence finding (LLM judge ↔ personas, 17 June).** Ran `llm_judge_reward.py --backend gemini` (gemini-2.5-flash) on the grid. The judge **agrees with the oracle on the core ordering** — good strictly best (2/2), fabrication rejected (good>fab 2/2): good≈9, fabricated≈1. Where it "disagrees" is illuminating: it also scores the *flattened* (bland, vague) narrative ≈1, marking it ungrounded because its generic claims ("an ordinary person caught up in the events") aren't in the record — so it does **not** reliably rank flattened above fabricated. That is exactly the **contested value the personas surfaced** (`specificity_first` ranked the vivid fake above the bland record). So two *independent* synthetic-human methods — hand-weighted personas and a real LLM judge — land on the **same contested boundary** (bland-vagueness vs. vivid-fabrication). That convergence is a genuine, citable result about where "good detail" is actually contested. (Note: gemini-2.5-flash is a *thinking* model — the judge call uses `max_tokens=800` and a fenced-JSON-tolerant parser; `--debug` shows raw output. A small model like `llama3.2` was too weak — it clustered scores ~6 and didn't penalise fabrication.)

**Judge-model-dependent ordering (18 June, on the VM via vLLM + Qwen2.5-7B).** Same core agreement (good best 2/2, good>fab 2/2), and the judge *explicitly lists* every injected hallucination in its `reason`. But Qwen scores the **vivid fabrication (6) ABOVE the bland flattened (2–5)** — i.e. detecting the fabrication did **not** make it rank it below the dull-but-honest record (a "vividness bias"). gemini-2.5-flash did the *opposite* (fabricated ≈ low). So the flattened-vs-fabricated ordering is **judge-model-dependent** — value pluralism now appears *across judge models*, not just across hand-weighted personas. (This is why `--validate` against the oracle/historians matters before trusting any one judge.)

Next: the full two-model comparison (GRPO `--reward human` vs `--reward composite`/`linguistic`). The LLM-as-judge (#4, `llm_judge_reward.py`) is built and validated against the oracle; the remaining step is validating it against *real* historians.

### GRPO runs & real-run profile

**What the smoke-test runs showed (all on MacBook MPS, Qwen2.5-0.5B, 20 steps):**

| run | setup | result |
|---|---|---|
| 1 | linguistic reward | runs, no collapse; reward bounces; `clipped_ratio`=1.0 (rambles to cap) |
| 2 | composite, **closed-book** prompt (name only) | reward uniformly very negative (−2…−15): nothing to be faithful *to* → no gradient toward good |
| 3 | composite + length + anti-copy, **open-book** prompt | no collapse; `anti_copy`=0 (not copying); floor lifted to ~−8; `clipped_ratio` still 1.0; reward flat — all expected at 20 steps |

**Conclusion:** the *guarded composite GRPO loop runs end to end and behaves sanely* — but 20 steps on a 0.5B model cannot *learn* the task. Actual learning is a separate, larger commitment. Note: the open-book run was slower (~28 min vs ~5) only because the prompts are bigger (the record block), **not** because it learned more — runtime is not a quality signal.

**Real-run profile (for a proper GPU, e.g. SURF `gpu_a100`):**

| knob | smoke value | real-run value | why |
|---|---|---|---|
| model | Qwen2.5-0.5B | **Qwen2.5-1.5B / 3B-Instruct (LoRA)** | 0.5B is too small to learn grounded narration |
| max_steps | 20 | **500–2000** (start 500) | GRPO needs many updates to move the policy |
| learning_rate | 1e-5 | 1e-5 (try 2e-5 if flat) | LoRA tolerates a bit higher |
| num_generations | 4 | **8** | larger groups = better advantage estimate |
| per_device_train_batch_size | 4 | 8–16 (multiple of num_generations) | A100 headroom |
| max_completion_length | 256 | **160** | 3–5 sentences ≈ 120 tok; shorter = faster iteration |
| data | 16 prompts | **expand cases + epochs** | more prompts stabilise GRPO |
| precision / device | fp32 / MPS | **bf16 / CUDA** | speed |

**Curves to expect in a healthy real run:** composite reward mean drifts **up** over hundreds of steps; `clipped_ratio` falls below 1.0 and `mean_terminated_length` rises (model learns to stop); `anti_copy_guard` stays ≈0 (if it dives negative, the model found the copy hack → raise `W_COPY` / lower tolerance); `length_guard` mean rises toward 0; `entropy` declines *gently* (a crash to ~0 = mode collapse → lower LR / raise KL `beta`); `reward_std`>0 throughout.

**Validation discipline:** every N steps, *generate on held-out prompts and read them*, don't trust reward alone — reward rising while text degrades is reward hacking. Keep the fabricated renderings as a permanent tripwire.

---

## File inventory

**Run it all:** `bash run_all.sh` (Tier A: data + rewards, no GPU) · `bash run_all.sh --train` (adds v2 render → SFT → GRPO from the SFT baseline). `bash test_pipeline.sh` = Tier A only.


| File | What it is |
|---|---|
| `project_update_good_detail.md` | The reframing + TRL-rewrite plan. Conceptual home base. |
| `synthetic_factbase_spec.md` | Fact-base design: schema, tags, rendering grid, oracle. |
| `generate_synthetic_corpus.py` | Authors 8 cases, renders the grid, scores both judges, writes outputs. |
| `synthetic_corpus/` | Generated: `cases.json`, `renderings.jsonl`, `preference_pairs.jsonl`, `report.md`. |
| `llm_render.py` | v2 rendering: LLM writes narratives, `faithfulness()` verifies grounding (good accepted only if fully grounded). `--mock` offline. Output: `llm_renderings.jsonl`. |
| `sft_narrative.py` | SFT baseline: trains the model to *write* micro-narratives on (record → good narrative) pairs. `--dry_run` checks data w/o torch. Output: `sft-narrative-adapter`. |
| `dpo_kto_train.py` | Tiny DPO/KTO fine-tune of Qwen2.5-0.5B + LoRA. `--dry_run` validates data w/o torch. Auto CUDA→MPS→CPU. |
| `emit_tei_iob.py` | Serialises the fact base to TEI XML + EHRI-style IOB with gold annotations; round-trip validates. Outputs to `synthetic_corpus/tei` + `/iob`. |
| `ingest_adapters.py` | Ingestion side: TEI/IOB → `Record` (text + entities). Tested end-to-end; shows the tag-loss boundary. Seed of the real adapters. |
| `linguistic_reward.py` | Decomposable surface-only reward (5 features) + per-feature validation on the grid. Needs `concreteness_norms.csv` for real concreteness. |
| `run_pipeline.py` | **One-command end-to-end run** of all synthetic stages (generate → serialise → ingest → dual-reward comparison → training pointer). |
| `grpo_train.py` | GRPO smoke test (TRL ≥1.0): prompt-only dataset, `--reward linguistic\|composite`. Trains on MPS; `--dry_run` checks reward variance without torch. |
| `composite_reward.py` | Faithfulness-gated composite reward; exact `F` from the fact base. Validation flips good>fabricated to 16/16. |
| `human_reward.py` | Synthetic human arm: oracle dimensions, persona-weighted, optional noise. Two-arm + pluralism report. Wired into GRPO. |
| `llm_judge_reward.py` | LLM-as-judge (#4), `--backend uva\|ollama\|gemini`. `--validate` (vs oracle), `--label` (preference set), `--mock` (offline). |
| `run_local.sh` + `requirements-local.txt` | One-command local run (MacBook, MPS). **← this is what worked today.** |
| `surf/` | Self-contained SURF bundle: `setup_env.sbatch`, `run_smoke.sbatch`, requirements, README. Upload to `/home/tblanke2/good-detail`. |

---

## How to re-run

**Locally (MacBook, what worked today):**
```bash
bash run_local.sh                 # full chain: corpus + DPO + KTO on MPS
# or just one leg:
source .venv/bin/activate
python dpo_kto_train.py --method kto --max_steps 20
```

**On SURF (when the cluster is out of maintenance):** upload `surf/` contents, then `sbatch setup_env.sbatch` → `sbatch run_smoke.sbatch`. (Whole cluster was in a maintenance drain on 17 June — check `scontrol show reservation` for the return window.)

---

## Gotchas already solved (so I don't re-hit them)

- **TRL version drift** is the recurring tax. Fixes already applied to `dpo_kto_train.py`:
  - dropped `max_length` / `max_prompt_length` from `DPOConfig`/`KTOConfig` (removed/renamed in installed TRL).
  - KTO needs **actual** `per_device_train_batch_size > 1` → set to 2 for KTO (1 is fine for DPO).
  - KTO now lives in `trl.experimental` — the `from trl import` shim still works (just warns).
- **DID I freeze the pins?** If not done yet: `source .venv/bin/activate && pip freeze > requirements-local.lock.txt`. This is the single most useful anti-drift move.
- KTO data is imbalanced ~1:2 desirable:undesirable — for a *real* run set `desirable_weight`/`undesirable_weight` in `KTOConfig`.

---

## Next steps (in priority order)

1. **Freeze the lock file** if not already (`requirements-local.lock.txt`).
2. **Workshop with historians/archivists** — decisions that gate the real study: pick the construct rubric for "good detail," the real corpora, and the oracle weights. These are *their* calls, not engineering defaults.
3. **Expand the synthetic corpus** from 8 → ~15–20 cases, more focal types and sensitivity levels (the generator is built to extend — just add to `CASES`).
4. **Richer linguistic reward** — move beyond proper-noun/number density to concreteness norms, NLI-based grounding, and **surprisal** (the "detect the unexpected" feature).
5. **Build the composite reward** (grounding/faithfulness + linguistic + restraint guardrail) and validate an LLM-judge against a small set of real human ratings.
6. **The real GRPO experiment** — human-reward vs linguistic-reward models, same SFT init, cross-evaluate; the divergence is the finding.
7. **Harden the ingestion adapters on REAL EHRI files** — `ingest_adapters.py` is validated only on clean synthetic TEI/IOB. Run it against actual `ehri_en.txt` and real Wiener Library TEI; expect messiness (namespaces, nesting, encoding, coreference) the synthetic data doesn't have.
8. **Literature pull** (deferred all along): rubric/AI-feedback rewards, long-form faithfulness scoring, DPO/KTO, participatory/pluralistic alignment, critical DH (data-vs-capta). Needed to pin the novelty claim.

## Still-open design decisions

- Headline framing: "two rewards → different *models*" vs. "linguistic features *predict* human judgement" (correlation study — cheaper first result).
- Whether the human–linguistic gap is **stable across corpora** or **corpus-specific** — the central research question once real corpora exist.
- Transparent hand-built linguistic features vs. surprisal/perplexity (interpretable & DH-friendly vs. elegant but black-box).

---

*Reminder of the through-line: the reward function is where "good" is actually defined. Making that definition explicit, contestable, and community-owned — and showing where a culture-blind metric fails — is the contribution.*
