#!/usr/bin/env python3
"""
config.py — ONE place for every experiment dial.

Edit values here; every module imports from this file, so a change here changes
the whole pipeline. After editing, re-run the matching validator to SEE the
effect on the good/flattened/fabricated examples (no GPU, seconds):
    composite weights / sensational words  -> python composite_reward.py
    linguistic weights / lexicons          -> python linguistic_reward.py
    personas                               -> python human_reward.py
    judge rubric                           -> python llm_judge_reward.py --backend ollama --validate
    cases / fabrication words              -> python generate_synthetic_corpus.py

NOT here (too large / code, edit in place): the records themselves are `CASES` in
generate_synthetic_corpus.py; the linguistic feature *functions* are in
linguistic_reward.py. Everything else you'd tune for an experiment is below.
"""

# ====================================================================== #
# 1. BASE MODEL (the model that gets trained: SFT / DPO / GRPO)
# ====================================================================== #
# 3B is the sweet spot for GRPO on one 24GB A10 (terminates cleanly, fabricates
# less than 0.5B). Drop to 1.5B for headroom; avoid 7B for GRPO on a single A10
# (8-way generation will likely OOM — 7B is fine for SFT/inference only).
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ====================================================================== #
# 2. LLM BACKENDS (for the judge #4 and v2 rendering). Pick with --backend.
# ====================================================================== #
BACKENDS = {
    "uva":    {"chat_url": "https://llmproxy.uva.nl/v1/chat/completions",
               "key_env": "UVA_LLM_API_KEY", "default_model": "gpt-oss-120b",
               "retry_models": {"gpt-oss-120b"}},
    "ollama": {"chat_url": "http://localhost:11434/v1/chat/completions",
               "key_env": None, "default_model": "llama3.2", "retry_models": set()},
    "gemini": {"chat_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "key_env": "GEMINI_API_KEY", "default_model": "gemini-2.5-flash", "retry_models": set()},
    # local vLLM (SURF Research Cloud). Key optional; override host/port with VLLM_URL.
    "vllm":   {"chat_url": "http://localhost:8000/v1/chat/completions",
               "key_env": "VLLM_API_KEY", "key_optional": True,
               "default_model": "Qwen/Qwen2.5-7B-Instruct", "retry_models": set()},
}
MAX_RETRIES, RETRY_DELAY, REQUEST_DELAY = 3, 2.0, 1.0

# ====================================================================== #
# 3. REWARD WEIGHTS — what counts as "good detail"
# ====================================================================== #
# Composite (faithfulness-gated) reward:
GAMMA = 2.0                       # how hard grounding F gates quality (>1 = sharper)
W_C, W_FAB, W_SENS = 0.3, 0.5, 1.0  # calibration weight, fabrication penalty, sensationalism penalty

# HOW faithfulness F is estimated (the grounding signal inside the composite reward):
#   "lexical" surface entity/year overlap — fast, offline, but BLIND to recombination
#             (shares the linguistic reward's surface machinery → risks construct collapse)
#   "nli"     claim-level entailment vs the fact base — orthogonal to surface form (recommended)
#   "llm"     ask the LLM judge to count unsupported claims — strongest, slowest, fallible
# Validate / compare with:  python faithfulness.py --method nli --agreement
FAITHFULNESS_METHOD = "nli"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # small, local, runs on the A10
NLI_ENTAIL_THRESHOLD = 0.5        # min entailment prob to count a claim as supported
NLI_CONTRADICT_THRESHOLD = 0.5    # min contradiction prob to count a claim as fabricated
NLI_MAX_PREMISE_CHARS = 2000      # truncate the record-as-premise to this many chars

# Source-passage RETRIEVAL (the NLI premise / prompt excerpt). The overnight run
# showed grounding hinges on focused source passages, so retrieval quality matters.
#   "lexical" token overlap (default, no deps) · "embed" sentence-embedding cosine
RETRIEVAL_METHOD = "lexical"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast; used iff "embed"

# Linguistic (surface) reward — per-feature weights:
WEIGHTS = {
    "proper_noun_density": 1.0,
    "number_date_density": 1.0,
    "concreteness": 1.0,
    "lexical_density": 0.5,
    "calibration": 0.5,
}

# Human reward PERSONAS — the pluralism knob (whose "good"?). Add your own here;
# it auto-appears in `--persona`. Each is a weighting of the value dimensions.
PERSONAS = {
    "balanced":          dict(coverage=1.0, specificity=1.0, grounding=1.5, source=1.0, calibration=1.0, restraint=1.0, fab=1.0),
    "grounding_first":   dict(coverage=1.0, specificity=0.5, grounding=3.0, source=1.0, calibration=0.5, restraint=0.5, fab=2.0),
    "ethics_first":      dict(coverage=0.5, specificity=0.3, grounding=1.5, source=1.5, calibration=1.5, restraint=2.5, fab=1.5),
    "specificity_first": dict(coverage=1.5, specificity=2.0, grounding=1.0, source=0.5, calibration=0.3, restraint=0.3, fab=0.5),
}

# ====================================================================== #
# 4. KEYWORD LEXICONS (edit the word lists)
# ====================================================================== #
SENSATIONAL = {   # dramatic/gratuitous words -> restraint penalty (composite)
    "dragged", "dawn", "firebombed", "stormtroopers", "executed", "betrayal",
    "smuggled", "shot", "burned", "burning", "blazing", "mob", "gunpoint",
    "seized", "vanished", "slaughtered", "brutally", "savagely", "deported",
}
HEDGES = {"probably", "perhaps", "seems", "seem", "believed", "believe",
          "reportedly", "apparently", "likely", "may", "might", "uncertain",
          "unconfirmed", "possibly", "around", "approximately", "thought",
          "appears", "appear", "suggests", "undated", "presumably"}
OVERCLAIM = {"confirmed", "certainly", "definitely", "clearly", "undoubtedly",
             "proven", "exactly", "always", "never", "indisputably",
             "obviously", "undeniably", "precisely"}
SOURCE_WORDS = {"registry", "testimony", "letter", "catalogue", "caption",
                "according", "records", "recorded", "per", "src", "source"}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by",
    "for", "with", "as", "was", "were", "is", "are", "be", "been", "being",
    "her", "his", "she", "he", "it", "they", "them", "that", "this", "which",
    "who", "whom", "had", "has", "have", "not", "no", "so", "such", "than",
    "then", "there", "here", "from", "into", "out", "up", "down", "over",
    "about", "after", "before", "during", "while", "like", "many", "much",
    "some", "any", "all", "one", "later", "still", "ever", "yet", "i", "you",
    "we", "my", "our", "its", "their", "would", "could", "did", "do", "does",
}
# Concreteness STUB (1=abstract..5=concrete). Auto-replaced by a real
# `concreteness_norms.csv` (word,rating) placed beside linguistic_reward.py.
CONCRETENESS_STUB = {
    "workshop": 4.8, "hat": 5.0, "registry": 4.2, "milliner": 4.4, "café": 4.9,
    "coffee": 5.0, "street": 4.9, "house": 4.9, "home": 4.4, "school": 4.7,
    "hospital": 4.8, "nurse": 4.7, "synagogue": 4.8, "suitcase": 5.0,
    "leather": 4.9, "papers": 4.3, "document": 4.2, "documents": 4.0,
    "committee": 3.4, "office": 4.5, "round": 3.5, "gate": 4.7, "child": 4.6,
    "brother": 4.4, "family": 4.0, "town": 4.6, "city": 4.6, "border": 4.2,
    "person": 3.6, "people": 3.7, "events": 1.8, "event": 2.2, "period": 2.4,
    "hardship": 1.7, "displacement": 1.9, "life": 2.6, "experience": 2.0,
    "significance": 1.6, "ordinary": 2.3, "affected": 2.0, "fate": 1.9,
    "wartime": 2.6, "emigration": 2.7, "trade": 3.0, "worker": 3.8,
    "owner": 3.5, "proprietor": 3.6, "meeting": 3.2, "inquiries": 2.1, "war": 3.0,
}

# ====================================================================== #
# 5. SYNTHETIC DATA dials (the records are CASES in generate_synthetic_corpus.py)
# ====================================================================== #
REGISTERS = ["testimony", "finding_aid"]
PROFILES = ["good", "flattened", "fabricated"]
# What the 'fabricated' profile invents (the hallucination pool):
FALSE_NAMES = ["Colonel Brandt", "Inspector Hoffmann", "Major Keller",
               "Officer Reinhardt", "Captain Vogel"]
FALSE_PLACES = ["the Sankt-Möring transit office", "the Westmark depot",
                "the Drautal holding station", "the Ostheim registry"]
FALSE_DATES = ["14 March 1939", "the night of 3 February 1940",
               "late autumn 1941", "27 August 1942"]

# ====================================================================== #
# 6. PROMPTS & the JUDGE RUBRIC (what the model and judges are told)
# ====================================================================== #
# The open-book writing instruction (used by GRPO, SFT, generate):
INSTRUCTION = (
    "You are given an archival record. Write a short (3-5 sentence), archivally grounded "
    "micro-narrative about {unit}, in the style of {register}. Use ONLY facts from the record; "
    "include specific, sourced detail; acknowledge uncertainty; do NOT invent names, places, or "
    "dates.\n\nRECORD:\n{record}\n\nMICRO-NARRATIVE:")

# The LLM judge rubric:
RUBRIC_SYSTEM = (
    "You are a meticulous archival historian. You evaluate a short micro-narrative written "
    "FROM a source record. You judge the quality of its DETAIL: rich, specific detail that is "
    "GROUNDED in the record, with calibrated uncertainty, source-awareness, and restraint. "
    "You punish invented names, places, dates or claims (hallucination) and sensationalism, "
    "even when they read vividly. Reply with ONLY a JSON object."
)
RUBRIC_USER = """RECORD:
{record}

MICRO-NARRATIVE:
{narrative}

Score the narrative's "good detail" from 0 to 10, where:
- 8-10: specific, fully grounded in the record, uncertainty hedged, restrained.
- 4-7 : grounded but generic/flattened, or thin.
- 0-3 : contains ANY invented specific (a name, place, date or claim not in the record) or sensationalism.

Use the FULL range; do NOT cluster scores in the middle. If a narrative contains
even one specific that is not in the record, it MUST score 0-3 regardless of how
fluent it reads. If unsure whether a detail is in the record, treat it as invented.

Return ONLY this JSON:
{{"score": <0-10 number>, "grounded": <true|false>, "hallucinations": [<quoted invented bits>], "reason": "<one sentence>"}}"""

# v2 rendering prompts (LLM writes the narratives):
RENDER_SYS = ("You write short archival micro-narratives. Output ONLY the narrative prose — "
              "no preamble, no headings, no JSON.")
STYLE_PROMPT = {
    "good": ("Write a 3-5 sentence micro-narrative about {unit} in the style of {register}, "
             "using ONLY facts present in the record. Be specific (names, places, dates), "
             "attribute claims to their source, and hedge anything the record marks uncertain. "
             "Invent NOTHING that is not in the record."),
    "flattened": ("Write a vague, generic 2-3 sentence summary about {unit}. Do NOT use any "
                  "specific names, dates or places; keep it bland and non-committal."),
    "fabricated": ("Write a vivid 3-5 sentence narrative about {unit}. Make it engaging by adding "
                   "specific dramatic details — invented names, dates, or events that are NOT in "
                   "the record. (This is synthetic test data; keep it non-graphic.)"),
}

# ====================================================================== #
# 7. GRPO GUARDS
# ====================================================================== #
TARGET_WORDS = 90          # length guard: penalise completions longer than this
W_LEN = 1.0                # weight of the over-length penalty
ANTICOPY_N = 5             # n-gram size for the verbatim-copy check
ANTICOPY_TOL = 0.30        # allowed overlap fraction before penalising
W_COPY = 2.0               # weight of the anti-copy penalty
W_FORMAT = 1.0             # format guard: penalty per record-scaffold marker /
#                            non-prose list line (kills the "reformat the record
#                            into a list" exploit; 0 disables the term)
