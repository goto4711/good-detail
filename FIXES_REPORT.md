# FIXES_REPORT

## Headline changes

- **Moved:** `run_pipeline.py` Stage 4 synthetic linguistic `good vs fabricated` changed from **4/16** to **0/16**.
- **Moved:** `human_reward.py` balanced two-arm line `good > fabricated : ... linguistic` changed from **4/16** to **0/16**.
- **Moved:** `faithfulness.py --method nli --agreement` changed from **r = -0.34** to **r = -0.41**.
- **Stable:** `faithfulness.py --method nli` mean F stayed **good 0.83 / flattened 0.88 / fabricated 0.49**.
- **Baseline caveat:** in this repo state, `python composite_reward.py` already printed **composite good > fabricated = 15/16** before the fixes, not the published 16/16. After the fixes it remains **15/16**. The torch-free Tier A smoke path now forces lexical faithfulness and prints **16/16** there. **Resolved 2026-07-02 — see "Diagnosis of the 15/16 cell" below**; the `NLI_STRIP_HEDGES` fix addresses the mechanism.

## Validator summary

| Command | Before | After | Notes |
|---|---:|---:|---|
| `python linguistic_reward.py` aggregate good / flat / fabricated | 2.789 / 1.134 / 3.000 | 2.573 / 1.134 / 2.984 | proper-noun bug fix reduced the surface reward |
| `python composite_reward.py` good>fabricated cells | 15/16 | 15/16 | repo baseline already differed from the published 16/16 |
| `python composite_reward.py` linguistic-only good>fabricated cells | 4/16 | 0/16 | follows the stricter proper-noun count |
| `python human_reward.py` balanced `good > fabricated : ... linguistic` | 4/16 | 0/16 | inherited from the linguistic fix |
| `python faithfulness.py --method nli` mean F (good / flat / fabricated) | 0.83 / 0.88 / 0.49 | 0.83 / 0.88 / 0.49 | unchanged |
| `python faithfulness.py --method nli --agreement` | -0.34 | -0.41 | more anti-correlated after the proper-noun fix |
| `python run_pipeline.py` Stage 4 linguistic `good vs fabricated` | 4/16 | 0/16 | same headline move as above |

## 1. Vacuous faithfulness F=1.0

**Change.** `nli_faithfulness()` now exposes an optional `return_status=True` path and emits a warning when it hits the vacuous `claims/premises empty` branch. `bestofn_demo.py --gate` rejects unscoreable candidates with an explicit `filtered: ...` reason, and `realdata_generate.py` warns per record and marks unscoreable faithfulness in output instead of printing `F=1.0`.

**Delta.** No synthetic-grid movement: `python faithfulness.py --method nli` stayed at **0.83 / 0.88 / 0.49**.

## 2. Selection-time guards for best-of-N

**Change.** Extracted the GRPO length / anti-copy / format guards into `guards.py`. `grpo_train.py` now imports the shared functions, and `bestofn_demo.py` uses them as selection-time filters/penalties by default, with `--no_guards` restoring the old raw ranking.

**Delta.** No validator-number movement; this only affects best-of-N selection behavior. `python grpo_train.py --dry_run` still reports the same reward spread and still ends with `DRY RUN OK`.

## 3. Silent estimator fallback

**Change.** `_faith_llm()` now counts fallback events, warns with the triggering exception, prints a summary line at process exit, and obeys `config.FAITHFULNESS_STRICT = False` / `True`.

**Delta.** No validator-number movement. A forced parse-failure smoke test now prints a warning plus `1/1 faithfulness calls fell back to lexical` instead of failing silently.

## 4. Contradiction vs entailment across premises

**Change.** Added `config.NLI_CONTRADICTION_VETO = False`. When set to `True`, contradiction above threshold vetoes entailment from another premise.

**Comparison.**

| Check | `False` | `True` |
|---|---|---|
| Grid (`python faithfulness.py --method nli` equivalent) | good `(0.83, 0.56)` / flattened `(0.88, 0.12)` / fabricated `(0.49, 3)` | same |
| Real-data-style direct NLI check | `(1.0, 0, 'scored')` | `(0.0, 1, 'scored')` |

The real-data-style check used one retrieved EHRI sentence about **efficient resistance activity** plus a contradictory free-text premise in the same retrieved-premise format. Default `False` preserves the current synthetic numbers; `True` flips the mixed-evidence claim to fabricated.

## 5. Proper-noun feature counts ALL-CAPS

**Change.** `proper_noun_density()` now counts TitleCase tokens plus short all-caps acronyms (`<=4` chars), but no longer counts arbitrary long ALL-CAPS tokens.

**Delta.**

- `proper_noun_density` mean: good **0.868 → 0.651**, fabricated **1.116 → 1.101**
- `python linguistic_reward.py` aggregate: good **2.789 → 2.573**, fabricated **3.000 → 2.984**
- `python run_pipeline.py` Stage 4 linguistic `good vs fabricated`: **4/16 → 0/16**
- `python faithfulness.py --method nli --agreement`: **-0.34 → -0.41**

## 6. Sensationalism penalises attested vocabulary

**Change.** Added `config.SENSATIONAL_EXEMPT_ATTESTED = True`. `sensationalism()` now accepts `case=` / `source_text=` and ignores sensational lexicon words when the same word stem is attested in the source/premises.

**Delta.** No synthetic-grid movement on the current cases: `python composite_reward.py` sensationalism means stayed **0.000 / 0.000 / 0.235**.

## 7. Coverage substring false positives

**Change.** `human_reward.component_scores()` now uses case-insensitive word-boundary regex matching instead of raw substring inclusion.

**Delta.** No movement on the current synthetic grid: `python human_reward.py` persona means were unchanged before/after.

## 8. Faithfulness caching

**Change.** The generic faithfulness cache key now uses `case_id` when present, else a stable premise-content hash. Direct `nli_faithfulness()` calls now cache on premise hash, subject, text hash, thresholds, veto flag, and attested-token/year sets.

**Delta.** No validator-number movement. A direct-cache smoke test went `before 0 -> after first 1 -> after second 1`, confirming repeated direct NLI calls now reuse cached results.

## 9. Rewards importable without the synthetic corpus

**Change.** `linguistic_reward.py`, `composite_reward.py`, and `human_reward.py` no longer import `generate_synthetic_corpus.py` at module import time. Synthetic validator data is now loaded lazily in `main()` / helper lookups.

**Validation.** `python -c "import linguistic_reward, composite_reward, human_reward, faithfulness"` succeeded while `generate_synthetic_corpus.py` was temporarily renamed away.

## 10. Smaller items

**Change.**

- `_load_nli()` now uses a normal `import torch` and selects `device=0` for CUDA, `"mps"` on Apple Silicon when available, else CPU.
- `linguistic_reward()` and `composite_reward()` now resolve config defaults at call time instead of binding them in function signatures at import time.
- `split_claims()` no longer prepends the subject to pronoun/possessive-led fragments, and `_has_invented_specific()` ignores `<5`-word claims so malformed stubs land in `vague` instead of `INVNT`.
- `.gitignore` now ignores `requirements-local.lock`, and the local `requirements-local.lock.` file was renamed to `requirements-local.lock`.
- `git ls-files` showed **no tracked** `__pycache__/`, `*.pyc`, or `.DS_Store` entries, so there was nothing to `git rm --cached`.

**Debug output before/after (`python faithfulness.py --debug`).**

The stock `synth_case_001` example was unchanged before/after:

- good: `F=1.00 unsupported=0`
- flattened: `F=0.00 unsupported=1`
- fabricated: `F=0.50 unsupported=4`

That is expected: the hardening targets malformed split fragments and very short invented-specific stubs, which do not appear in the canned debug example.

**Tier A smoke.** `bash test_pipeline.sh` now passes with `torch` intentionally blocked by a temporary `sitecustomize.py`. The script keeps Tier A torch-free by forcing lexical faithfulness only for the synthetic composite/human smoke path; the standalone NLI validator remains separate (`python faithfulness.py --method nli --agreement`).

## Diagnosis of the 15/16 cell (2026-07-02, `debug_cell.py`)

The failing cell is **`synth_case_005/testimony`**, lost by **0.008** (good −0.194 vs
fabricated −0.186). Per-claim NLI verdicts show a *double artifact*, not environment
noise:

1. **Good over-penalised — the hedge-contradiction artifact.** "Pavel Stern was
   removed from the school register **probably** in 1940" scores **CONTR at c=1.00**
   against the definite premise "…removed from the school register in 1940"; the
   identical claim *without* "probably" (in the fabricated text) gets e=1.00 OK. The
   NLI model reads hedging a definite fact as contradiction — the estimator punished
   exactly the hedging that the calibration feature rewards. This is the
   `methods_rewards.md §2a` "hedged claim" artifact, but landing in the *penalised*
   CONTR bucket, not the benign vague one, and it is systematic: good draws unsup=1–2
   in 6/16 cells (003/testimony good F=0.33).
2. **Fabricated under-penalised — relational fabrication escapes.** "He was the last
   child to leave the burning schoolhouse in 1940" lands **vague** (e=0.00, c=0.21):
   it invents no capitalised entity and no new year, so the INVNT check cannot see it
   and the NLI model does not contradict it. Only "Major Keller … Ostheim … 27 August
   1942" is caught (INVNT). Net: fabricated keeps F=0.75/unsup=1 despite two inventions.

**Fix applied:** `config.NLI_STRIP_HEDGES = True` (fix 11) — hedge adverbs
(`config.HEDGES`) are stripped from a claim before NLI scoring, so the factual core is
what must be entailed; the original claim is kept for the INVNT check and debug
display. The cache key includes the flag. **Confirmed on MPS, 2026-07-02:** good's
claim 3 in 005 flips CONTR → OK (e=1.00), good F=1.00/unsup=0, and the grid recovers
**composite 16/16** — by mechanism, not threshold-tuning. Final post-fix(11) grid
state: mean F **good 0.89 / flattened 0.88 / fabricated 0.49** (good's unsup mean
0.56 → **0.38** — the systematic hedge penalty removed), and
corr(linguistic, NLI-F) settles at **−0.38** (path: −0.34 original → −0.41 after
fix 5 → −0.38 after fix 11; still solidly anti-correlated). Artifact 2 (relational
fabrication) remains a known blind spot; it is the residual gap between NLI-F and
the LLM claim-verifier (P3 in `NEXT_STEPS.md`).

**Why the published run showed 16/16:** cannot be fully pinned without the original
A10 environment; the verdicts here are saturated (c=1.00), so simple numeric drift is
unlikely — more probably a transformers/tokenizer version difference in the NLI stack.
With `NLI_STRIP_HEDGES` the question is moot: the mechanism is removed.

**Veto flag warning (fix 4 follow-up):** the same debug showed the *fully entailed*
claim "removed from the school register in 1940" carrying **max-contradiction c=1.00
from a different premise** (likely "enrolled … in 1938"). `NLI_CONTRADICTION_VETO=True`
would therefore mark even perfectly grounded claims as fabricated. Keep it False; a
useful veto needs per-premise verdicts (contradiction from the same best premise),
noted in `config.py`.

## Acceptance checks completed

- `bash test_pipeline.sh` passed with `torch` blocked.
- `python linguistic_reward.py` ran clean.
- `python composite_reward.py` ran clean.
- `python human_reward.py` ran clean.
- `python faithfulness.py --method nli --agreement` ran clean and stayed negative.
- `python run_pipeline.py` ran clean.
- `python grpo_train.py --dry_run` still ran clean.
