# Good Detail — culturally aware reward models for archival micro-narratives

A pipeline that aligns a model to a **contested, community-defined notion of "good
detail"** in historical archives, and compares two ways of defining it: a
**situated/grounded** reward vs. a **culture-blind surface** reward. EHRI Holocaust
testimony is the example corpus; the workflow is meant to generalise.

**`PROJECT_STATUS_2026-06-17.md` is the source of truth** — live build status,
results, the reward-functions reference, and the real-run profile. This README is
just the map and the commands.

**Guides:** `experiments.md` (the 3+1 reward conditions and how to run/compare them) ·
`methods_rewards.md` (why & how each reward is built — the methods section, incl. the
pluggable NLI faithfulness and the circularity/validity argument) · `DIALS.md` (every
knob you can turn — flags + the keyword lists/weights/prompts and where they live) ·
`overview_for_historians.md` (plain-language briefing for non-technical historians &
archivists).

## Run it

```bash
# Tier A — data + all reward arms (no GPU, no torch, seconds):
bash test_pipeline.sh            # or:  bash run_all.sh

# Tier B — the model-training chain (needs torch on MPS/CUDA):
bash run_all.sh --train          # v2 render -> SFT -> GRPO (writes logs/)
#   use the interpreter that has torch:  PYTHON=python bash run_all.sh --train
```

Setup: `pip install -r requirements-local.txt` (laptop) · `bash setup_vm.sh` (CUDA
VM, e.g. **SURF Research Cloud** — see `SURF_RESEARCH_CLOUD.md`). LLM backends for
the judge / v2 rendering: **ollama** (local, no key), **vllm** (local GPU server),
**gemini** (`GEMINI_API_KEY`), **uva** (`UVA_LLM_API_KEY`).

### Data and secrets

- **EHRI corpus is not redistributed in this repo.** Only the manifest
  (`data/EHRI/corpus.json`) is checked in. Bring your own EHRI testimony and
  place TEI XML under `data/EHRI/xml/` and/or IOB JSONL under `data/EHRI/iob/`,
  matching the paths declared in `corpus.json`.
- **API keys go in `.env`** (gitignored). At minimum the project expects
  `UVA_LLM_API_KEY` and/or `GEMINI_API_KEY` depending on which backend you use
  for the LLM judge and v2 rendering.

## The pieces

| stage | file |
|---|---|
| **all experiment dials (weights, keywords, personas, prompts, model)** | **`config.py`** (see `DIALS.md`) |
| synthetic data + rendering grid | `generate_synthetic_corpus.py` |
| TEI/IOB serialise + ingest (adapters) | `emit_tei_iob.py`, `ingest_adapters.py` |
| **corpus-agnostic ingest** (one `Record`, per-format adapters) | **`ingest/`** (`tei_ehri`, `iob_jsonl`, `synthetic`) + `data/<CORPUS>/corpus.json` |
| **real-data shakedown** (generate + score any corpus) | `realdata_generate.py` (`--corpus EHRI --source xml\|iob`) |
| v2: LLM rendering + grounding-verify | `llm_render.py` |
| rewards | `linguistic_reward.py`, `composite_reward.py`, `human_reward.py`, `llm_judge_reward.py`, `faithfulness.py` (lexical/nli/llm) |
| training | `sft_narrative.py` → `dpo_kto_train.py` / `grpo_train.py` |
| orchestration | `run_pipeline.py`, `test_pipeline.sh`, `run_all.sh` |

## What's done vs. not

Done (prototype/smoke level): the whole pipeline runs end to end on synthetic data —
all four reward arms + the LLM judge, SFT → DPO/KTO → GRPO with guards, the v2
grounding-verified renderer, and the two-arm + persona-pluralism comparison.

Not done (the research): GPU learning runs at scale, **real corpora** + an NLI
faithfulness estimator, **real concreteness norms** (`concreteness_norms.csv`), and
validating the LLM judge against **real historians**. See the status doc.
