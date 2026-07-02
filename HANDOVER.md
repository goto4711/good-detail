# Handover — Good Detail: culturally aware reward models for archival micro-narratives

*Handover document, 2026-07-02. Audience: an incoming researcher/engineer taking over the project. Source of truth for build status remains `PROJECT_STATUS_2026-06-17.md`; this document is the orientation layer over it.*

## 1. What this project is

Good Detail is a digital-humanities research pipeline that studies what happens when you align a language model to a **contested, community-defined cultural value**: "good detail" in micro-narratives written from historical archives, with EHRI early Holocaust testimony as the example corpus. It is the generative sequel to *"The Occasional Bitter Lesson"* (Deep Culture Team, UvA): where that paper tested Sutton's Bitter Lesson on discriminative DH tasks, this one carries the question into generation under a value that has no ground-truth label — "good detail" is grounded, source-aware, calibrated, and restrained, and *whose* definition counts is itself the research question.

The method is to operationalise "good detail" as several rival, explicit reward functions — a culture-blind surface reward, a situated faithfulness-gated reward, a persona-weighted "human" reward, and an LLM-as-judge (RLAIF) reward — then hold everything else constant (same base model, same SFT baseline) and vary only the reward. The disagreement between the rewards is the finding, not a nuisance: it makes visible that the choice of reward is a choice of values.

The headline conclusion so far: for this class of problem the Bitter Lesson **inverts**. Scale (more GRPO steps, more prompts, bigger judge models) homogenised, hallucinated, or collapsed; what moved grounding was situated signal — focused source passages, explicit theorised rewards, and (pending) real historian judgment. The rewards work best as a **lens over a strong model's output** (best-of-N selection), not as a training signal at feasible scale.

## 2. Architecture

Four layers, with one deliberate single-variable seam between them.

**Substrate (data + ingest).** Records enter either synthetically or from real archives, and everything downstream consumes one canonical `Record` dataclass (id, register, focal unit, entities, relations, source_text, provenance). The `ingest/` package holds per-format adapters (`tei_ehri`, `iob_jsonl`, `synthetic`) registered by decorator and dispatched via `data/<CORPUS>/corpus.json` manifests — this is what makes the pipeline corpus-agnostic in practice, not just in intention. The synthetic side (`generate_synthetic_corpus.py`) hand-authors structured fact bases (entities, events with dates and certainty flags, per-fact source grounding, salience, sensitivity) and renders each into three quality profiles of the same narrative: **good** (specific, grounded, hedged), **flattened** (bland but not false), **fabricated** (vivid but inventing specifics). Because the facts are owned, an exact **oracle** exists in the synthetic phase — the yardstick the rewards are measured against, never itself a trainable reward. `emit_tei_iob.py` serialises fact bases to TEI XML and IOB so the ingest path can be round-trip checked. For real raw text, `dspy_extract.py` (DSPy `ChainOfThought`, backends from `config.BACKENDS`) builds the fact base itself (entity-only by default — see §4 on the relations ablation), and `retrieval.py` selects focused source passages (lexical or embedding).

**Rewards (the intellectual core).** All dials live in `config.py`; each reward file doubles as its own validator when run directly.

- `linguistic_reward.py` — the **culture-blind surface arm**: length-normalised weighted sum of proper-noun density, number/date density, concreteness (Brysbaert norms or stub), lexical density, and calibration (hedges minus over-claims). Deliberately blind to truth; its blindness is measured by its validator, not assumed.
- `composite_reward.py` — the **situated arm**: `reward = F^γ·(Q + w_c·C) − w_fab·n_unsup − w_sens·S`. The design choice that matters: faithfulness `F` is a *multiplicative gate* (γ=2), so a vivid fabrication cannot buy back reward with more vividness; additive penalties for unsupported specifics and sensationalism push fabrication below bland-but-honest.
- `faithfulness.py` — the pluggable `F` estimator, the core methods contribution: `lexical` (entity/year bag overlap — fast but blind to recombination), `nli` (recommended: atomise the narrative into subject-restored clauses, turn the fact base / retrieved source into atomic premises, run a small local NLI model with max-over-premises, four-way verdict where **neutral-and-vague is excluded rather than punished** — blandness is not unfaithfulness), and `llm` (judge counts unsupported claims; validation oracle, too costly/fallible for dense RL). Results cached per (method, case, text-hash).
- `human_reward.py` — six value dimensions (coverage, specificity, grounding, source-attribution, calibration, restraint) minus a fabrication penalty, weighted by a **persona** vector (`balanced`, `grounding_first`, `ethics_first`, `specificity_first` in `config.PERSONAS`). Currently a *synthetic stand-in* for historian judgment — honestly labelled a scaffold until the workshop supplies real labels.
- `llm_judge_reward.py` — the RLAIF arm: rubric-prompted scoring via OpenAI-compatible backends (ollama, vLLM, Gemini, UvA proxy). Studied as another contested operationalisation, not an oracle; its model-dependence is a result.

**Optimisation.** `sft_narrative.py` SFTs the base model (Qwen2.5-3B-Instruct) on (record-prompt → good rendering) pairs to give a sane starting policy; `grpo_train.py` then runs GRPO under one chosen reward (prompts + reward function, no pairs), with three anti-reward-hacking **guards** that carry no value signal (shared module `guards.py` since 2026-07-02, also applied at best-of-N selection time): a length guard (target ~90 words), an anti-copy guard (n-gram overlap with the source), and a format guard (blocks the record-dump exploit, added 2026-06-19). `dpo_kto_train.py` handles the preference-pair route (used for RLAIF: judge labels pairs → DPO). Each arm produces a LoRA adapter.

**Evaluation and human loop.** `generate.py` / `realdata_generate.py` generate from every arm and cross-score on every metric. The validity discipline (`methods_rewards.md §6`) is enforced throughout: the diagonal ("the linguistic arm wins on the linguistic reward") is tautological and never reported; findings live in the **off-diagonal** (what an arm costs on axes it wasn't trained on), the **agreement matrix** between rewards, and the oracle. A second usage mode sits alongside training: `bestofn_demo.py` has a strong, pluggable LLM write K candidates per record and lets each reward *select* its favourite — no training, model-agnostic, and the historian-facing artifact. The workshop apparatus is built: `bestofn_demo.py --save` → `make_annotation_sheet.py` (blind browser form) → `analyze_annotations.py` (inter-annotator agreement + each reward's Spearman correlation with human ranking).

**Orchestration.** `run_pipeline.py` / `test_pipeline.sh` / `run_all.sh` run the no-GPU synthetic tier in seconds; `run_all.sh --train` runs the training chain; `run_ehri.sh` and `run_overnight.sh` drive the real-data runs; `setup_vm.sh` provisions a CUDA VM (SURF Research Cloud — see `SURF_RESEARCH_CLOUD.md`). EHRI data is **not** redistributed; bring your own under `data/EHRI/` matching `corpus.json`. API keys (`UVA_LLM_API_KEY`, `GEMINI_API_KEY`) go in `.env`.

## 3. Data flow

Synthetic tier (seconds, no torch): fact base → rendering grid (good/flattened/fabricated × registers) + preference pairs → TEI/IOB serialisation → adapter ingest (round-trip checked) → all rewards score the grid → agreement/disagreement report.

Training tier: record rendered as an open-book prompt → SFT baseline → GRPO under reward X (group sampling, within-group advantage, guards) → per-arm adapter → cross-metric generation and scoring.

Real-data tier: raw EHRI testimony → DSPy extraction (or direct TEI/IOB ingest) → `record_block()` prompt context + retrieved source passages as the NLI premise → grounded generation per arm → scoring against the real source text. Outputs are redacted and stamped UNVERIFIED — methods diagnostics, never historical sources.

Inference-time tier: real record → strong LLM writes K candidates → every reward scores and picks → divergence of picks reported; optional faithfulness gate (`--gate`, `unsup ≤ 1`) so the shown output is always grounded.

## 4. Initial results

**Reward level — the spine (synthetic grid, run of 2026-06-19; revised 2026-07-02 after the code fixes — see `FIXES_REPORT.md`).** The composite reward ranks good above fabricated in **16/16** grid cells (it dipped to 15/16 until `debug_cell.py` diagnosed the cause — the NLI model scored a hedged grounded claim, "removed … *probably* in 1940", as a c=1.00 contradiction of its definite premise, punishing exactly the hedging calibration rewards; fixed by `config.NLI_STRIP_HEDGES`); the surface linguistic reward in **0/16** after the proper-noun ALL-CAPS fix (previously reported 4/16) — it *never* prefers grounded text over fluent fabrication. NLI faithfulness separates fabrication from blandness correctly (good 0.89, flattened 0.88, fabricated 0.49 with ~3 fabrications/cell). Crucially, **corr(linguistic, NLI-faithfulness) ≈ −0.38** (−0.34 pre-fix): the surface and grounded constructs are *anti-correlated*, so optimising surface specificity mechanically trades against grounding. Known open blind spot: relational fabrication with no new entity ("last child to leave the burning schoolhouse") lands in the benign vague bucket — the gap the LLM claim-verifier (P3) is for. All docs (`README.md`, `experiments.md`, `FRAMING.md`, `methods_rewards.md`, `DIALS.md`, `PROJECT_STATUS`) were updated 2026-07-02 with the revised numbers and revision notes; the fixes *strengthen* the headline. The LLM judge is vividness-biased and inconsistent — it gave a fully ungrounded narrative (F=0.00) a perfect 10 and scored two near-identical texts 9 vs 2.

**Model level (3B, 300 GRPO steps, 16 prompts — scale-limited).** Infrastructure correct (terminating completions, NLI grounding, guards), but the SFT baseline was strong and most arms barely moved. The one arm that visibly trained was the **linguistic arm — straight into reward-hacking**: ALL-CAPS names, a fabricated person ("Wilhelm Lederer", invented by expanding the source code "WL/SYN/301"), and a record-dump exploit gaming both surface score and faithfulness (now blocked by the format guard). The judge rewarded this arm (7.0) over the grounded composite arm (3.75) — RLAIF with this judge would push *away* from grounding.

**Real EHRI shakedown (N=3, directional).** The off-diagonal reproduced more starkly on real testimony: composite → linguistic, unsupported specifics doubled 5 → 10 and F halved 0.67 → 0.33 on the Carlebach/Buchenwald record, with confident invented specifics ("Ernst Heilmann, former President of the Prussian Landtag", an SS "Dr. List"). All arms fabricated more against the XML entity-list source than synthetic; the structured IOB source produced the first fully faithful real-data output (F=1.00, unsup=0).

**At scale (overnight sweep, N=150) — the honest revision.**
1. **Source focus is the real lever:** extracted/chunked source passages roughly halve fabrication vs the XML entity-list path (unsup 6.7 → 4.2, F 0.43 → 0.59), robust across every arm.
2. **The relations ablation is NULL:** relations on ≈ off (F 0.590 vs 0.600, unsup identical); the extractor's relation F1 is only 0.40. The earlier "the fact base of relations is the lever" claim from N=3 was an overclaim the ablation caught. Relations are now off by default everywhere; the fact base is honestly *source passages + entities*.
3. **Trained-policy differences converge to a near-null:** composite ≈ linguistic ≈ human (F 0.590/0.591/0.589), and GRPO on real prompts did not help (reward drifted −2.38 → −2.58 over 300 steps).

**Best-of-N on a clean instruct model (mistral-small-3.2, 5 real chunks, K=6) — the sharpest form of the thesis.** The rewards diverged on **5/5 records**: linguistic repeatedly selected the most surface-rich candidate that NLI marks ungrounded (one at F=0.00, unsup=5), composite selected the hedged faithful one (F=0.86, unsup=1). The LLM judge saturated (~8/10 for nearly everything, rating an F=0.00 fabrication and an F=0.86 grounded narrative the same). Also: gpt-oss-120b (a reasoning model) could not reliably produce clean micro-narratives at all — the bottleneck was model *kind*, not scale. Practical default: faithfulness gate at `unsup ≤ 1` (`unsup=0` passed 0/6 real candidates).

**Where the contribution lives.** At the reward level (discrimination and anti-correlation on fixed text) and in inference-time selection — *not* in trained-policy differences, which do not transfer to real testimony at this scale. The recurring pathologies map one-to-one onto the earlier paper's taxonomy: leakage → reward-hacking, majority-class → collapse to baseline, median collapse → the saturated judge.

## 5. Open work (see `NEXT_STEPS.md` for detail)

- **P1 — run the historian workshop (the unlock).** All situated/human signal is still synthetic; ~30 records × 3–4 candidates × 4–5 historians would yield ground truth for which reward best predicts human judgment, a real human arm, measured pluralism, and DPO-grade preference data. The apparatus is fully built — this is now a recruitment/logistics task, not an engineering one.
- **P3 — validate the measurement:** NLI faithfulness and the LLM judge against the P1 annotations; add a second faithfulness estimator.
- **P4 — make inference-time selection the centre of gravity:** bigger K, strongest models, selector quality vs human ranking, cross-reward disagreement at scale.
- **P6 — writeup rigour:** CIs/seeds, the RLAIF-pluralism sub-study (2–3 judge models), and plain write-ups of the negative results (policies converge; relations don't help; the judge is biased).
- Smaller: real concreteness norms (`concreteness_norms.csv`), embedding retrieval as default, per-document fact-base merge in extraction.
- **Known dead ends — do not repeat:** more GRPO training (synthetic and real both near-null), more synthetic prompts, and relation extraction for grounding.

## 6. Quick start for the incoming person

```bash
pip install -r requirements-local.txt
bash test_pipeline.sh                 # Tier A: data + all reward arms, no GPU, seconds
bash run_all.sh --train               # Tier B: SFT → GRPO chain (torch, MPS/CUDA)
python bestofn_demo.py                # the historian demo (needs an LLM backend)
```

Read in this order: `FRAMING.md` (the argument) → `experiments.md` (conditions + results) → `methods_rewards.md` (why each reward is built as it is, incl. the circularity defence) → `DIALS.md` (every knob) → `TRAINING_DATA.md` → `PROJECT_STATUS_2026-06-17.md` (live status). `overview_for_historians.md` is the plain-language briefing for the workshop participants.
