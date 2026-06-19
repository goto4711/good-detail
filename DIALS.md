# DIALS — every knob you can turn

For DH researchers using the pipeline. Two kinds of dial:

- **Part 1 — command-line / environment dials.** No editing. Pass a flag or set an
  environment variable.
- **Part 2 — `config.py` dials.** One file holds every reward weight, keyword list,
  persona, prompt, and the base model. Open `config.py`, change a named value, save.

**The golden loop:** after editing `config.py`, *see the effect instantly* by
re-running the matching validator (no GPU, seconds) — it prints how your change
re-scores the good / flattened / fabricated examples:

| you changed (in `config.py`)… | run this to see the effect |
|---|---|
| composite weights (`GAMMA`,`W_FAB`…) / `SENSATIONAL` | `python composite_reward.py` |
| `FAITHFULNESS_METHOD` (lexical/nli/llm) / `NLI_MODEL` | `python faithfulness.py --method nli --agreement` |
| linguistic `WEIGHTS` / `HEDGES` / `STOPWORDS`… | `python linguistic_reward.py` |
| `PERSONAS` | `python human_reward.py` |
| `RUBRIC_*` (judge) | `python llm_judge_reward.py --backend ollama --validate` |
| `FALSE_NAMES/PLACES/DATES`, `REGISTERS`, `PROFILES` | `python generate_synthetic_corpus.py` then any of the above |

---

## Part 1 — command-line & environment dials (no editing)

**LLM backend (judge & v2 rendering)** — `--backend` + env vars:

| dial | how | notes |
|---|---|---|
| backend | `--backend ollama\|vllm\|gemini\|uva` | ollama = laptop; vllm = your GPU server |
| model | `--model <name>` | must match the served model exactly (vllm/ollama) |
| vLLM address | `export VLLM_URL=http://localhost:8002` | if 8000 is taken (e.g. Jupyter) |
| API keys | `.env`: `GEMINI_API_KEY=…`, `UVA_LLM_API_KEY=…` | ollama/vllm need no key |

**Whole pipeline** (`run_all.sh`) — environment variables:

```bash
BACKEND=vllm MODEL=Qwen/Qwen2.5-7B-Instruct V2_LIMIT=8 PYTHON=python bash run_all.sh --train
```

**Training** (`grpo_train.py`):

| flag | default | meaning |
|---|---|---|
| `--reward` | linguistic | `linguistic` / `composite` / `human` / `judge` — **which experiment** |
| `--persona` | balanced | for `--reward human` (the persona names come from `config.PERSONAS`) |
| `--judge_backend` / `--judge_model` | vllm | for `--reward judge` (RLAIF) — LLM scores every completion; use a fast local backend & few steps |
| `--prompts` | synthetic | `corpus:NAME/SRC` trains on REAL records (e.g. `corpus:EHRI/extracted`); reward scored vs the real source. `composite`/`linguistic` only |
| `--prompt_limit` | 100 | how many real prompts to load (with `--prompts`) |
| `--init_adapter` | (none) | start from an SFT adapter, e.g. `sft-narrative-adapter` |
| `--max_steps` | 20 | 300–500 on a GPU to actually see learning |
| `--num_generations` | 4 | 8 on a GPU = better signal |
| `--max_completion_length` | 256 | 160 ≈ a 3–5 sentence narrative |
| `--no_length_guard` / `--no_anti_copy` / `--no_format_guard` | (on) | disable any of the three guards |
| `--seed` | 42 | reproducibility |
| `--model` | from `config.DEFAULT_MODEL` | the model being trained |

Other scripts: `sft_narrative.py --source template|llm --epochs N`; `dpo_kto_train.py
--method dpo|kto --pairs <file>`; `llm_judge_reward.py --validate|--label|--mock --limit N
--debug`; `llm_render.py --limit N --max-tries N`; `generate.py --adapter … --sft_adapter …`.
Every script: `--help`.

---

## Part 2 — `config.py` (open it, change a named value)

Everything below is in **`config.py`**, grouped into numbered sections. Search for
the NAME.

| goal | value in `config.py` |
|---|---|
| **base model** to train (SFT/DPO/GRPO) | `DEFAULT_MODEL` (§1) |
| LLM backends / default models / retries | `BACKENDS`, `MAX_RETRIES`… (§2) |
| **composite reward**: gate sharpness, penalties | `GAMMA`, `W_C`, `W_FAB`, `W_SENS` (§3) |
| **how grounding F is estimated** (lexical/nli/llm) | `FAITHFULNESS_METHOD`, `NLI_MODEL`, `NLI_ENTAIL_THRESHOLD` (§3) |
| **linguistic reward** feature weights | `WEIGHTS` (§3) |
| **personas** — *whose "good"?* (add your own) | `PERSONAS` (§3) |
| sensational / dramatic words (restraint) | `SENSATIONAL` (§4) |
| hedging / over-claim markers | `HEDGES`, `OVERCLAIM` (§4) |
| source-attribution words; stopwords | `SOURCE_WORDS`, `STOPWORDS` (§4) |
| concreteness lexicon | `CONCRETENESS_STUB` (§4) — or drop a real `concreteness_norms.csv` (word,rating) beside `linguistic_reward.py` |
| what "fabricated" invents (hallucination pool) | `FALSE_NAMES`, `FALSE_PLACES`, `FALSE_DATES` (§5) |
| registers / quality profiles | `REGISTERS`, `PROFILES` (§5) |
| the **writing instruction** (GRPO/SFT/generate) | `INSTRUCTION` (§6) |
| the **judge rubric** | `RUBRIC_SYSTEM`, `RUBRIC_USER` (§6) |
| the **v2 generation prompts** | `RENDER_SYS`, `STYLE_PROMPT` (§6) |
| GRPO length / anti-copy / format guards | `TARGET_WORDS`, `W_LEN`, `ANTICOPY_N`, `ANTICOPY_TOL`, `W_COPY`, `W_FORMAT` (§7) |

**Not in `config.py`** (too large / it's code — edit in place):
- the **records themselves** → `CASES` in `generate_synthetic_corpus.py`
- the linguistic feature *functions* (how each feature is computed) → `linguistic_reward.py`

*After any edit, run the matching validator from the golden-loop table to watch your
change move the numbers — that is the fastest way to understand what a dial does.*
