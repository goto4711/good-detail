# Task: fix known issues in the good-detail reward/selection code

You are working in the `good-detail` repo (Python, no package structure — flat modules importing each other; all experiment dials in `config.py`). Read `README.md` and `methods_rewards.md` first for context. The code implements several reward functions for "good detail" in archival micro-narratives plus a best-of-N selection demo; a review found the issues below. Fix them in the order given.

## Ground rules

- **Do not silently change published numbers.** The synthetic-grid validators (`python linguistic_reward.py`, `python composite_reward.py`, `python human_reward.py`, `python faithfulness.py --method nli`, `python run_pipeline.py`) print the headline results (composite good>fab 16/16, linguistic 4/16, NLI F ≈ 0.83/0.88/0.49). Any fix that changes these by design must be reported: run each validator before and after, and write the deltas to `FIXES_REPORT.md`. Do not edit `experiments.md` or other results docs.
- Where a fix changes scoring *semantics* (marked ⚑), put the new behaviour behind a `config.py` flag with the old behaviour available, choose the default as specified, and document the flag in the style of the existing config comments.
- Keep the `(F, n_unsup)` return signature of `faithfulness()` intact — `composite_reward.py`, `human_reward.py`, `generate.py`, `realdata_generate.py` all rely on it.
- No new hard dependencies. Match the existing code style (module docstrings, self-validating `main()` per module).
- Tier A must stay GPU-free: `bash test_pipeline.sh` must pass without torch installed.

## Fixes (priority order)

### 1. Vacuous faithfulness F=1.0 (critical)
`faithfulness.py`, `nli_faithfulness()` (~line 240): when `claims` or `premises` is empty it returns `(1.0, 0)` — "vacuously faithful". A candidate whose sentences are all filtered out by `split_claims`, or a record whose retrieval returned no premises, scores *perfect* and passes the best-of-N gate.

Fix: distinguish "unscoreable" from "faithful". Add a module-level way for callers to detect the vacuous case (e.g. an optional `return_status=True` kwarg, or a companion function — your call, but do not break the 2-tuple default). Then:
- in `bestofn_demo.py` (`--gate`): an unscoreable candidate must FAIL the gate, and the demo output must say why;
- in `realdata_generate.py`: log a loud warning per unscoreable record and mark it in the output rather than reporting F=1.0;
- emit a `warnings.warn` (or logged warning) whenever the vacuous branch is hit anywhere.

### 2. Selection-time guards for best-of-N (critical)
The anti-reward-hacking guards (length guard, anti-copy guard, format guard) currently exist only inside `grpo_train.py`. `bestofn_demo.py` ranks candidates by raw reward, so the known exploits are live at selection time: √length inflation, verbatim source-copying (maximises F with no anti-copy), hedge-stuffing.

Fix:
- Extract the three guard functions from `grpo_train.py` into a new shared module `guards.py` (pure functions: `text, source_text/record -> penalty or bool`). `grpo_train.py` imports from it; behaviour there must be byte-identical (same weights, same flags).
- In `bestofn_demo.py`, apply the guards as **pre-ranking filters**: a candidate that violates the format guard or exceeds the anti-copy tolerance is excluded from selection (reported as "filtered: <reason>"); the length guard becomes a soft penalty added to every reward's score, weight from `config.py`. Add `--no_guards` to restore old behaviour.
- Note in the docstring that guard thresholds/weights live in `config.py`.

### 3. Silent estimator fallback (critical)
`faithfulness.py`, `_faith_llm()` (~lines 276–294): any exception — backend down, JSON parse failure — silently falls back to `_faith_lexical`, so a run can mix estimators with no trace.

Fix: count fallbacks in a module-level counter, log a warning with the exception on each, print a summary line ("N/M faithfulness calls fell back to lexical") at exit or on first access from validators, and add a `config.FAITHFULNESS_STRICT = False` flag — when True, raise instead of falling back.

### 4. ⚑ Contradiction vs entailment across premises
`faithfulness.py` (~lines 247–254): `ent` and `con` are maxima over *different* premises, and entailment is checked first — a claim entailed by one retrieved sentence but contradicted by another counts as supported. Defensible for the internally-consistent synthetic fact base; questionable for real retrieved passages.

Fix: add `config.NLI_CONTRADICTION_VETO = False` (default — preserves current behaviour and published numbers). When True, `con >= NLI_CONTRADICT_THRESHOLD` marks the claim fabricated even if entailed elsewhere. Run the grid validator and one real-data-style check both ways; put the comparison table in `FIXES_REPORT.md`.

### 5. ⚑ Proper-noun feature counts ALL-CAPS (the observed reward hack)
`linguistic_reward.py`, `proper_noun_density()` (~line 94): `re.match(r"^[A-Z]", tt)` counts every token of ALL-CAPS text — exactly the hack the GRPO linguistic arm found.

Fix (default ON, this is a bug not a semantics choice, but report the grid delta): count a token only if it is TitleCase (`^[A-ZÀ-Þ][a-zà-ÿ]`) — i.e. an uppercase letter followed by a lowercase one. Short acronyms (≤4 chars, all caps, e.g. "SS", "USC") still count once; longer all-caps words do not. Keep the `i > 0` sentence-initial exclusion as is.

### 6. ⚑ Sensationalism penalises attested vocabulary
`composite_reward.py` `sensationalism()` + `config.SENSATIONAL`: words like "deported", "shot", "seized" are frequently *attested facts* in Holocaust testimony; a faithful narrative accurately reporting a deportation loses restraint score.

Fix: add `config.SENSATIONAL_EXEMPT_ATTESTED = True` (default True for real-data paths, but compute it everywhere): a lexicon word does **not** count toward S if the same word (case-insensitive, simple stem match is fine) appears in the record's source text / premise facts. `sensationalism()` gains an optional `case`/`source_text` argument; callers that have the record pass it. When no source is available, behave as before. Report grid deltas.

### 7. Coverage substring false positives
`human_reward.py`, `component_scores()` (~line 59): `s.lower() in low` — "Anna" matches "Susanna", "1939" matches inside longer numbers.

Fix: word-boundary regex match (`\b` + `re.escape(s)` + `\b`, case-insensitive). Report grid delta.

### 8. Faithfulness caching
`faithfulness.py` `_key()` (~line 57): records without `case_id` collapse to `"?"` — two different records scoring identical text share a cache entry. Separately, direct `nli_faithfulness()` calls (the real-data path) bypass the cache entirely, so best-of-N re-scores duplicate candidates at full NLI cost.

Fix: (a) key on `case_id` if present, else a stable hash of the premise content; (b) add caching to `nli_faithfulness()` keyed on (hash of premises tuple, subject, text-hash, the two thresholds).

### 9. Rewards must be importable without the synthetic corpus
`linguistic_reward.py` (lines 43–46), `composite_reward.py`, `human_reward.py`: they `sys.exit` at *import* if `generate_synthetic_corpus.py` is missing — the supposedly corpus-agnostic reward layer cannot be deployed standalone.

Fix: move the `CASES/REGISTERS/PROFILES/render` imports into the `main()` validators (and `CASE_BY_ID` construction lazy). The scoring functions themselves must not need the synthetic module. Verify: `python -c "import linguistic_reward, composite_reward, human_reward, faithfulness"` succeeds in a directory where `generate_synthetic_corpus.py` is renamed away (restore after).

### 10. Smaller items
- `faithfulness.py` `_load_nli()` (~line 199): replace `getattr(__import__("torch"), "cuda").is_available()` with a normal `import torch`; add MPS support (`device = 0 if cuda else ("mps" if torch.backends.mps.is_available() else -1)` — check the `transformers` pipeline accepts "mps" in the pinned version; if not, leave CPU and comment why).
- `linguistic_reward.py` `linguistic_reward(text, weights=WEIGHTS)`: default arg binds config at import time. Change to `weights=None` → resolve `config.WEIGHTS` at call time. Same pattern anywhere else config values are bound as defaults.
- `split_claims()`: fragments produced by splitting on " and "/commas can be malformed ("Marta Hellinger her sister moved…"). Minimal hardening: do not prepend the subject when the fragment starts with a possessive/pronoun ("her", "his", "their", "she", "he", "they"); and exclude claims of < 5 words from the INVNT (invented-specific) check so malformed stubs land in "vague", not "fabricated". Inspect `python faithfulness.py --debug` output before/after and include it in `FIXES_REPORT.md`.
- Repo hygiene: add a `.gitignore` covering `__pycache__/`, `*.pyc`, `.DS_Store`, `logs/`, `.env`; `git rm -r --cached` the tracked `__pycache__` dirs and `.DS_Store`. Leave adapters, zips, and data directories alone. Rename `requirements-local.lock.` (trailing dot) to `requirements-local.lock`.

## Acceptance

1. `bash test_pipeline.sh` passes without torch.
2. All five module validators run clean; `python faithfulness.py --method nli --agreement` still reports a negative correlation.
3. `python grpo_train.py --dry_run` unchanged in behaviour.
4. Standalone-import check from fix 9 passes.
5. `FIXES_REPORT.md` at repo root: per fix — what changed, before/after validator numbers where they moved, and the veto-flag comparison from fix 4. Flag anything that moves a headline number (16/16, 4/16, the F triple, r ≈ −0.34) prominently at the top.
6. One commit per numbered fix, message prefixed `fix(N):`.
