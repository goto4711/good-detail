#!/usr/bin/env python3
"""
faithfulness.py
================================================================
PLUGGABLE grounding estimator — the heart of the "situated" reward.

    faithfulness(text, case, method=None) -> (F in [0,1], n_unsupported)

Three interchangeable backends, selected by `config.FAITHFULNESS_METHOD`
(or the `method=` argument / `--method` flag):

  "lexical"  surface entity+year overlap against the record (the ORIGINAL).
             Fast, offline, deterministic. BLIND to recombination: it scores
             "moved to Rotterdam in 1939" as supported because those tokens
             appear in the record, even when the *claim* is invented. This is
             the same surface machinery the linguistic reward uses — which is
             exactly why a composite reward built on it can collapse into the
             linguistic reward.

  "nli"      claim-level ENTAILMENT against the fact base (the real signal).
             Splits the narrative into claims; each claim is supported only if
             the record ENTAILS it (a small NLI model). Measures propositions,
             not surface form, so it is orthogonal to the linguistic reward and
             catches recombination / relational fabrication ("helped establish
             the workshop", "died in 1968", "member of the resistance").

  "llm"      ask an LLM judge to list unsupported claims (strongest, slowest,
             and itself fallible). Good as a validation oracle; not ideal as a
             dense RL reward (token cost + its own hallucination).

All three share the (F, n_unsup) return signature, so composite_reward.py and
generate.py are unchanged. Results are cached by (method, case_id, text-hash)
so the NLI/LLM cost is paid once per unique completion inside the RL loop.

  python faithfulness.py                 # validate lexical on the grid (offline)
  python faithfulness.py --method nli    # load the NLI model, validate + agree-matrix
  python faithfulness.py --agreement     # how much does each method track linguistic?
"""

import hashlib
import re
import sys
import warnings
import atexit

from config import (FAITHFULNESS_METHOD, FAITHFULNESS_STRICT, NLI_MODEL, NLI_ENTAIL_THRESHOLD,
                    NLI_CONTRADICT_THRESHOLD, NLI_MAX_PREMISE_CHARS)

try:
    from linguistic_reward import STOPWORDS, WORD_RE
except ImportError:
    sys.exit("Run from the project folder (needs linguistic_reward.py).")

_CACHE = {}
_DEBUG = False   # set by `--debug`: print each claim's entailment verdict
_LLM_FALLBACK = {"calls": 0, "fallbacks": 0, "reported": False}
STATUS_SCORED = "scored"
STATUS_VAGUE_ONLY = "vague-only"
STATUS_UNSCOREABLE_NO_CLAIMS = "unscoreable:no-claims"
STATUS_UNSCOREABLE_NO_PREMISES = "unscoreable:no-premises"
STATUS_UNSCOREABLE_NO_CLAIMS_OR_PREMISES = "unscoreable:no-claims-or-premises"


def is_unscoreable_status(status):
    return status.startswith("unscoreable:")


def _result(F, unsupported, status, return_status):
    return (F, unsupported, status) if return_status else (F, unsupported)


def _warn_unscoreable(status, text, subject="", premises=None):
    why = {
        STATUS_UNSCOREABLE_NO_CLAIMS: "no claims survived split_claims()",
        STATUS_UNSCOREABLE_NO_PREMISES: "no premises were available",
        STATUS_UNSCOREABLE_NO_CLAIMS_OR_PREMISES: "no claims survived split_claims() and no premises were available",
    }.get(status, status)
    preview = " ".join(text.split())[:120]
    extra = f" subject={subject!r}" if subject else ""
    if premises is not None:
        extra += f" premises={len(premises)}"
    warnings.warn(
        f"nli_faithfulness is unscoreable ({why}); returning vacuous F=1.0.{extra} text={preview!r}",
        RuntimeWarning,
        stacklevel=2,
    )


def _report_llm_fallback_summary():
    if _LLM_FALLBACK["reported"] or _LLM_FALLBACK["calls"] == 0:
        return
    _LLM_FALLBACK["reported"] = True
    print(f"{_LLM_FALLBACK['fallbacks']}/{_LLM_FALLBACK['calls']} faithfulness calls fell back to lexical",
          file=sys.stderr)


atexit.register(_report_llm_fallback_summary)


def _faith_llm_fallback(text, case, exc):
    if FAITHFULNESS_STRICT:
        raise exc
    _LLM_FALLBACK["fallbacks"] += 1
    case_id = case.get("case_id", "?")
    warnings.warn(f"LLM faithfulness fell back to lexical for case {case_id}: {exc!r}",
                  RuntimeWarning, stacklevel=2)
    return _faith_lexical(text, case)


def _key(method, case, text):
    return (method, case.get("case_id", "?"), hashlib.sha1(text.encode("utf-8")).hexdigest())


# ====================================================================== #
# Shared: turn the structured fact base into text
# ====================================================================== #

def grounded_index(case):
    """Tokens + years the record actually supports (for the LEXICAL method)."""
    toks, years = set(), set()

    def add(s):
        s = str(s)
        for w in WORD_RE.findall(s):
            toks.add(w.lower())
        for y in re.findall(r"\d{3,4}", s):
            years.add(y)

    for ent in case["entities"].values():
        add(ent["name"])
        for a in ent["attributes"]:
            add(a["value"])
    for ev in case["events"]:
        add(ev["summary"])
        add(ev["date"]["value"])
    for src in case["sources"].values():
        add(src["kind"])
    return toks, years


def case_premise_facts(case):
    """The fact base as a LIST of atomic premise sentences (we own the facts).
    Each narrative claim is checked against each of these and supported if ANY
    one entails it (max-over-evidence, the FActScore/SummaC pattern). On real
    corpora this list becomes the retrieved source sentences."""
    facts = []
    for ent in case["entities"].values():
        nm = ent["name"]
        for a in ent["attributes"]:
            facts.append(f"The {a['key'].replace('_', ' ')} of {nm} is {a['value']}.")
    for ev in case["events"]:
        s = ev["summary"].rstrip(".")
        d = ev.get("date", {}).get("value")
        if d:
            s += f" in {d}"
        facts.append(s + ".")
    return facts


def case_premise_text(case):
    """The fact base as one concatenated premise (used by the LLM method)."""
    return " ".join(case_premise_facts(case))[:NLI_MAX_PREMISE_CHARS]


# ====================================================================== #
# Claim splitting (shared by nli + llm)
# ====================================================================== #

# Sentences ABOUT THE TEXT itself, not factual claims about the subject; we do
# not score these for grounding (they are entailment-neutral by construction).
_META = re.compile(r"\b(micro-?narrative|this (account|narrative|story|text)|"
                   r"the (narrative|account|record|story)|serves as|reflects|"
                   r"highlight|aims to|draws on|incorporat|acknowledg)\b", re.I)


# Clause boundaries — a "good" narrative packs many facts per sentence, which
# NLI reads as neutral (no single premise entails a 4-fact sentence). We split
# into ~atomic clauses so each can be entailed by one fact (FActScore-style).
_CLAUSE = re.compile(r",| who | which | where | and | but |;| until | after | before ", re.I)
# Pure source-attribution fragments ("the registry records", "according to family
# testimony") — provenance, not factual claims; drop so they don't penalise the
# careful renderings that cite sources more.
_PROV = re.compile(r"\b(according to|registry records?|family testimony|the records?|"
                   r"the registry|the catalogue|the caption|the letter|is recorded)\b", re.I)


def split_claims(text, subject=None):
    """Sentences -> ~atomic clauses, dropping meta and provenance fragments.
    `subject` (the focal entity name) is prepended to clauses that lost their
    subject, so 'worked at the workshop' becomes 'Marta Hellinger worked …'."""
    out = []
    for sent in re.split(r"(?<=[.;!?])\s+", text.strip()):
        s = sent.strip()
        if len(s.split()) < 4 or _META.search(s):
            continue
        for part in _CLAUSE.split(s):
            raw = part.strip(" ,.;:")
            words = raw.split()
            if len(words) < 3:                       # too short to be a claim
                continue
            if _PROV.search(raw) and len(words) <= 6:  # bare provenance fragment
                continue
            if words[-1].lower() in ("was", "were", "is", "are", "had", "has", "be", "been"):
                continue                              # clause with no predicate ("X was")
            p = f"{subject} {raw}" if (subject and not raw[:1].isupper()) else raw
            out.append(p.rstrip(" ,.") + ".")
    return out


# ====================================================================== #
# 1. LEXICAL (the original) — surface entity/year overlap
# ====================================================================== #

def _faith_lexical(text, case):
    toks, years = grounded_index(case)
    supported = unsupported = 0
    for sent in re.split(r"[.;:?!]\s+", text):
        words = sent.split()
        for i, w in enumerate(words):
            wt = w.strip(",.;:()[]'\"")
            if i > 0 and re.fullmatch(r"[A-Za-zÀ-ÿ]+", wt) and wt[0].isupper():
                if wt.lower() in STOPWORDS:
                    continue
                supported += (wt.lower() in toks)
                unsupported += (wt.lower() not in toks)
    for y in re.findall(r"\b\d{4}\b", text):
        supported += (y in years)
        unsupported += (y not in years)
    total = supported + unsupported
    F = supported / total if total else 1.0
    return F, unsupported


# ====================================================================== #
# 2. NLI — claim-level entailment against the fact base
# ====================================================================== #

_NLI = {"pipe": None, "ent_idx": None}


def _load_nli():
    if _NLI["pipe"] is not None:
        return _NLI["pipe"]
    try:
        import torch  # noqa
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, pipeline)
    except ImportError as e:
        sys.exit(f"NLI faithfulness needs transformers+torch: {e}\n"
                 f"Install: pip install -r requirements-local.txt  (also: sentencepiece)")
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    dev = 0 if (getattr(__import__("torch"), "cuda").is_available()) else -1
    _NLI["pipe"] = pipeline("text-classification", model=mdl, tokenizer=tok,
                            device=dev, top_k=None)
    # map label names regardless of the model's index order
    labels = [v.lower() for v in mdl.config.id2label.values()]
    _NLI["ent_label"] = next((v for v in labels if "entail" in v), "entailment")
    _NLI["con_label"] = next((v for v in labels if "contrad" in v), "contradiction")
    return _NLI["pipe"]


def _label_score(scores, label):
    for d in scores:
        if d["label"].lower() == label:
            return d["score"]
    return 0.0


def _has_invented_specific(claim, toks, years):
    """A capitalised non-stopword token or a 4-digit year NOT in the record =
    an invented specific (the lexical signal, reused to flag suspicious neutrals)."""
    for i, w in enumerate(claim.split()):
        wt = w.strip(",.;:()[]'\"")
        if i > 0 and re.fullmatch(r"[A-Za-zÀ-ÿ]+", wt) and wt[0].isupper() \
                and wt.lower() not in STOPWORDS and wt.lower() not in toks:
            return True
    return any(y not in years for y in re.findall(r"\b\d{4}\b", claim))


def nli_faithfulness(text, premises, subject="", grounded_toks=None, grounded_years=None,
                     return_status=False):
    """Core NLI grounding, reusable for ANY premise set (the synthetic fact base
    OR real retrieved source sentences). Per claim, max-over-premises entailment,
    three ways:
       entailed                       -> supported
       contradicted                   -> fabrication
       neutral + an invented specific -> fabrication (a name/year absent from the
                                         attested set `grounded_toks/years`)
       neutral + vague                -> ignored (bland, not false)
    F = supported / (supported + fabrications); vague-only text is vacuously
    faithful (F=1). Pass grounded_toks=None to skip the invented-specific check.
    With return_status=True, also return a status string so callers can detect an
    unscoreable vacuous case without breaking the default 2-tuple API."""
    claims = split_claims(text, subject=subject)
    if not claims or not premises:
        if not claims and not premises:
            status = STATUS_UNSCOREABLE_NO_CLAIMS_OR_PREMISES
        elif not claims:
            status = STATUS_UNSCOREABLE_NO_CLAIMS
        else:
            status = STATUS_UNSCOREABLE_NO_PREMISES
        _warn_unscoreable(status, text, subject=subject, premises=premises)
        return _result(1.0, 0, status, return_status)
    pipe = _load_nli()
    el, cl = _NLI["ent_label"], _NLI["con_label"]
    pairs = [{"text": p, "text_pair": c} for c in claims for p in premises]
    results = pipe(pairs, truncation=True, batch_size=32)
    nf = len(premises)
    supported = fabricated = 0
    for i, c in enumerate(claims):
        chunk = results[i * nf:(i + 1) * nf]
        ent = max(_label_score(sc, el) for sc in chunk)
        con = max(_label_score(sc, cl) for sc in chunk)
        if ent >= NLI_ENTAIL_THRESHOLD:
            tag, supported = "OK   ", supported + 1
        elif con >= NLI_CONTRADICT_THRESHOLD:
            tag, fabricated = "CONTR", fabricated + 1
        elif grounded_toks is not None and _has_invented_specific(c, grounded_toks, grounded_years or set()):
            tag, fabricated = "INVNT", fabricated + 1
        else:
            tag = "vague"
        if _DEBUG:
            print(f"      [{tag} e={ent:.2f} c={con:.2f}] {c}")
    denom = supported + fabricated
    status = STATUS_SCORED if denom else STATUS_VAGUE_ONLY
    return _result((supported / denom if denom else 1.0), fabricated, status, return_status)


def _faith_nli(text, case, return_status=False):
    """Synthetic path: premises = the fact base; attested set = grounded_index."""
    subject = case["entities"][case["focal"]]["name"]
    toks, years = grounded_index(case)
    return nli_faithfulness(text, case_premise_facts(case), subject, toks, years,
                            return_status=return_status)


# ====================================================================== #
# 3. LLM judge — list the unsupported claims (strongest, slowest)
# ====================================================================== #

def _faith_llm(text, case):
    _LLM_FALLBACK["calls"] += 1
    try:
        from llm_judge_reward import chat
    except Exception as exc:
        return _faith_llm_fallback(text, case, exc)
    premise = case_premise_text(case)
    sys_p = ("You verify grounding. Given RECORD facts and a NARRATIVE, count how "
             "many sentences in the narrative assert something NOT supported by the "
             "record. Reply ONLY as JSON: {\"unsupported\": int, \"total\": int}.")
    user_p = f"RECORD:\n{premise}\n\nNARRATIVE:\n{text}"
    try:
        raw = chat(sys_p, user_p, max_tokens=300)
        import json
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0))
        unsup, total = int(d["unsupported"]), max(1, int(d["total"]))
        return max(0.0, 1 - unsup / total), unsup
    except Exception as exc:
        return _faith_llm_fallback(text, case, exc)


# ====================================================================== #
# Dispatcher
# ====================================================================== #

_METHODS = {"lexical": _faith_lexical, "nli": _faith_nli, "llm": _faith_llm}


def faithfulness(text, case, method=None, return_status=False):
    """(F in [0,1], #unsupported). Method from arg or config.FAITHFULNESS_METHOD."""
    method = method or FAITHFULNESS_METHOD
    fn = _METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown FAITHFULNESS_METHOD {method!r}; use {list(_METHODS)}")
    if return_status:
        if method == "nli":
            return fn(text, case, return_status=True)
        F, unsupported = faithfulness(text, case, method=method)
        return F, unsupported, STATUS_SCORED
    k = _key(method, case, text)
    if k not in _CACHE:
        _CACHE[k] = fn(text, case)
    return _CACHE[k]


# ====================================================================== #
# Validation + the agreement diagnostic (answers: is faithfulness just
# the linguistic reward in disguise?)
# ====================================================================== #

def main():
    import argparse
    import statistics
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default=FAITHFULNESS_METHOD,
                    choices=list(_METHODS), help="which estimator to validate")
    ap.add_argument("--agreement", action="store_true",
                    help="correlate each faithfulness signal with the linguistic reward")
    ap.add_argument("--debug", action="store_true",
                    help="print each claim's entailment verdict for one case (nli)")
    args = ap.parse_args()

    from generate_synthetic_corpus import CASES, REGISTERS, PROFILES, render
    from linguistic_reward import linguistic_reward

    if args.debug:
        global _DEBUG
        _DEBUG = True
        case = CASES[0]
        print("FACT BASE (premises):")
        for f in case_premise_facts(case):
            print("   ", f)
        for p in PROFILES:
            print(f"\n--- {p} ({case['case_id']} / testimony) ---")
            F, nu = faithfulness(render(case, "testimony", p), case, method="nli")
            print(f"  => F={F:.2f}  unsupported={nu}")
        return

    print(f"Faithfulness method: {args.method}  (model: {NLI_MODEL if args.method=='nli' else '—'})\n")
    by_p = {p: {"F": [], "unsup": []} for p in PROFILES}
    for case in CASES:
        for reg in REGISTERS:
            for p in PROFILES:
                F, nu = faithfulness(render(case, reg, p), case, method=args.method)
                by_p[p]["F"].append(F)
                by_p[p]["unsup"].append(nu)
    print(f"{'profile':<12}{'mean F':>9}{'mean unsup':>12}")
    print("-" * 33)
    for p in PROFILES:
        print(f"{p:<12}{statistics.mean(by_p[p]['F']):>9.2f}{statistics.mean(by_p[p]['unsup']):>12.2f}")
    print("\nGood should sit high; fabricated should drop (low F, high unsup).")

    if args.agreement:
        # Pearson r between linguistic reward and faithfulness-F across the grid.
        xs, ys = [], []
        for case in CASES:
            for reg in REGISTERS:
                for p in PROFILES:
                    t = render(case, reg, p)
                    xs.append(linguistic_reward(t))
                    ys.append(faithfulness(t, case, method=args.method)[0])
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sx = sum((a - mx) ** 2 for a in xs) ** 0.5
        sy = sum((b - my) ** 2 for b in ys) ** 0.5
        r = cov / (sx * sy) if sx and sy else float("nan")
        print(f"\nAGREEMENT — corr(linguistic_reward, faithfulness-F[{args.method}]) = {r:+.2f}")
        print("Near +1 => faithfulness is tracking the SAME surface signal as the")
        print("linguistic reward (construct collapse / confound). Near 0 or negative")
        print("=> it measures something the surface reward cannot see (what we want).")


if __name__ == "__main__":
    main()
