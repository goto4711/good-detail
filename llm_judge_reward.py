#!/usr/bin/env python3
"""
llm_judge_reward.py
================================================================
Synthetic-human option #4: an LLM-AS-JUDGE reward (RLAIF) using the UvA LLM
proxy (https://llmproxy.uva.nl, OpenAI-compatible). The most realistic synthetic
stand-in for human judgement — and the bridge to validating against real
historians.

Backends (all OpenAI-compatible — pick with --backend):
  uva     UvA proxy (gpt-oss-120b / Qwen-2.5-VL / mistral-small-3.2). Needs UVA_LLM_API_KEY.
  ollama  LOCAL, no key, no network, no rate limits — easiest for laptop testing.
          install https://ollama.com ; then:  ollama pull llama3.2 && ollama serve
  gemini  Google Gemini free OpenAI endpoint. Needs GEMINI_API_KEY.
  vllm    LOCAL vLLM server (SURF Research Cloud VM / GPUs). Key optional. Pass the
          exact served --model.  vllm serve <model> --tensor-parallel-size 2 --port 8000

WHY it's an offline labeler, not a live GRPO reward
---------------------------------------------------
GRPO calls the reward once per completion per step (thousands of calls). An
API judge with rate limits can't sustain that. So the practical roles are:
  1. --validate : does the judge AGREE with the oracle? (do this before trust)
  2. --label    : write an LLM-judged preference set for DPO/KTO
                  (-> synthetic_corpus/preference_pairs_llm.jsonl, then
                   `python dpo_kto_train.py --pairs <that file>`)

Needs:  pip install requests python-dotenv
Examples:
  python llm_judge_reward.py --mock --validate                      # offline, no deps/network
  python llm_judge_reward.py --backend ollama --model llama3.2 --validate
  python llm_judge_reward.py --backend gemini --validate
  python llm_judge_reward.py --backend ollama --label --limit 4     # -> preference_pairs_llm.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from generate_synthetic_corpus import CASES, REGISTERS, PROFILES, render, compute_trace, oracle_score
    from grpo_train import _record_block
except ImportError as e:
    sys.exit(f"Run from the project folder: {e}")

HERE = Path(__file__).parent

# Backends, retry params, and the rubric live in config.py
from config import BACKENDS, MAX_RETRIES, RETRY_DELAY, REQUEST_DELAY, RUBRIC_SYSTEM, RUBRIC_USER
_BACKEND = "uva"   # set from --backend in main()
_DEBUG = False     # set from --debug in main()
CASE_BY_ID = {c["case_id"]: c for c in CASES}

# RUBRIC_SYSTEM / RUBRIC_USER: edit them in config.py


# ── UvA proxy client (mirrors your harness.py) ─────────────────────────────────

def _api_key(cfg):
    if cfg["key_env"] is None:
        return "ollama"   # local; the client wants a key but it's unused
    try:
        from dotenv import load_dotenv
        load_dotenv()  # local .env if present
    except ImportError:
        pass
    key = os.getenv(cfg["key_env"], "")
    if not key:
        if cfg.get("key_optional"):
            return "EMPTY"     # e.g. vLLM started without --api-key
        sys.exit(f"{cfg['key_env']} not set. Put it in .env or the environment "
                 f"(or use --mock to test the harness offline).")
    return key


def chat(model, messages, max_tokens=300, temperature=0.0):
    import requests
    cfg = BACKENDS[_BACKEND]
    chat_url = cfg["chat_url"]
    # vllm: override host/port via VLLM_URL (e.g. if 8000 is taken by Jupyter):
    #   VLLM_URL=http://localhost:8002 python llm_judge_reward.py --backend vllm ...
    if _BACKEND == "vllm" and os.getenv("VLLM_URL"):
        base = os.getenv("VLLM_URL").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        chat_url = base + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {_api_key(cfg)}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    # Retry on (a) network errors and (b) empty content for the known
    # empty-response quirk (uva gpt-oss-120b). Don't crash a long run on a blip.
    content, last_err = "", None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(chat_url, headers=headers, json=body, timeout=120)
            r.raise_for_status()
            content = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if content or model not in cfg["retry_models"]:
                return content
        except requests.exceptions.RequestException as e:
            last_err = e
        time.sleep(RETRY_DELAY)
    if last_err is not None:
        sys.exit(f"LLM request failed after {MAX_RETRIES} tries (backend={_BACKEND}, "
                 f"url={chat_url}): {last_err}\n"
                 f"Check the backend is up (vllm serving? right port via VLLM_URL? key set?).")
    return content


def _parse_score(raw):
    if not raw or not raw.strip():
        return 0.0, {"error": "empty response (try a bigger --max_tokens or a non-thinking model)"}
    text = re.sub(r"```(?:json)?", "", raw)               # strip code fences
    # try each {...} block, last first (the answer usually follows any reasoning)
    for block in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
        try:
            d = json.loads(block)
            if "score" in d:
                return float(d["score"]), d
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    m = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.I)
    if m:
        return float(m.group(1)), {"raw": raw[:300]}
    return 0.0, {"error": "no score parsed", "raw": raw[:300]}


# ── The judge ──────────────────────────────────────────────────────────────────

def judge(record, narrative, model="gpt-oss-120b"):
    messages = [{"role": "system", "content": RUBRIC_SYSTEM},
                {"role": "user", "content": RUBRIC_USER.format(record=record, narrative=narrative)}]
    raw = chat(model, messages, max_tokens=800)   # headroom for thinking models
    if _DEBUG:
        print(f"      [raw {len(raw)} chars] {raw[:200].replace(chr(10), ' ')!r}")
    time.sleep(REQUEST_DELAY)
    return _parse_score(raw)


def llm_judge_reward(text, case, model="gpt-oss-120b"):
    score, _ = judge(_record_block(case), text, model=model)
    return score


def llm_judge_reward_by_id(text, case_id, model="gpt-oss-120b"):
    return llm_judge_reward(text, CASE_BY_ID[case_id], model=model)


def _mock_score(text, case):
    """Deterministic offline stand-in (composite reward mapped to 0-10)."""
    from composite_reward import composite_reward
    c = composite_reward(text, case)
    return max(0.0, min(10.0, 5.0 + 2.5 * c))


# ── Modes ──────────────────────────────────────────────────────────────────────

def validate(args):
    """Does the LLM judge AGREE with the oracle? (run before trusting it)"""
    cells = [(c, "testimony") for c in CASES[:args.limit]]
    judge_name = "MOCK (offline heuristic)" if args.mock else f"{args.backend}:{args.model}"
    print(f"Judge: {judge_name}  | cases: {len(cells)}  (≈{len(cells)*3} API calls)\n")
    print(f"{'case':<18}{'profile':<12}{'LLM':>6}{'oracle':>9}")
    print("-" * 46)
    good_top = good_gt_fab = fab_penalised = ties = 0
    for case, reg in cells:
        scores = {}
        for p in PROFILES:
            text = render(case, reg, p)
            s = _mock_score(text, case) if args.mock else llm_judge_reward(text, case, args.model)
            scores[p] = s
            orc = oracle_score(compute_trace(case, p))
            print(f"{case['case_id']:<18}{p:<12}{s:>6.1f}{orc:>9.2f}")
        g, fl, fb = scores["good"], scores["flattened"], scores["fabricated"]
        good_top += (g > fl and g > fb)          # STRICTLY highest (ties don't count)
        good_gt_fab += (g > fb)
        fab_penalised += (fb < fl)               # hallucination scored worse than blandness
        ties += (len({g, fl, fb}) < 3)
        print()
    N = len(cells)
    print(f"good strictly highest : {good_top}/{N}   (CORE: agrees good is best)")
    print(f"good > fabricated     : {good_gt_fab}/{N}   (CORE: rejects hallucination)")
    print(f"fabricated < flattened: {fab_penalised}/{N}   (secondary: a CONTESTED ordering)")
    print(f"cells with tied scores: {ties}/{N}")
    core_ok = (good_top == N and good_gt_fab == N)
    if core_ok:
        print("\nVERDICT: trustworthy on the CORE ordering — good is best and fabrication is")
        print("rejected, agreeing with the oracle. The flattened-vs-fabricated tie/flip is a")
        print("*contested value* (is bland-but-vague better or worse than vivid-but-fabricated?),")
        print("the same pluralism the personas show — not a judge failure. OK to --label.")
    else:
        print("\nVERDICT: does NOT agree with the oracle on the CORE ordering (good should be")
        print("strictly highest AND above fabricated). Too weak — try a stronger / non-thinking")
        print("model: --model gemini-2.0-flash, or ollama qwen2.5:7b.")
    if args.mock:
        print("\n[MOCK] — composite reward stand-in. Drop --mock to use a real backend.")


def label(args):
    """Write an LLM-judged preference set for DPO/KTO."""
    out = HERE / "synthetic_corpus" / "preference_pairs_llm.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, cells = [], [(c, r) for c in CASES[:args.limit] for r in REGISTERS]
    print(f"Labelling {len(cells)} cells x 3 profiles with "
          f"{'MOCK' if args.mock else args.model} ...")
    for case, reg in cells:
        scored = []
        for p in PROFILES:
            t = render(case, reg, p)
            s = _mock_score(t, case) if args.mock else llm_judge_reward(t, case, args.model)
            scored.append((s, t))
        scored.sort(reverse=True)
        prompt = _record_block(case)
        best_text = scored[0][1]
        for s, t in scored[1:]:
            if t != best_text:
                rows.append({"prompt": prompt, "chosen": best_text, "rejected": t,
                             "case_id": case["case_id"], "judge": "mock" if args.mock else args.model})
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} judged preference pairs -> {out}")
    print(f"Train on them:  python dpo_kto_train.py --pairs {out} --method dpo")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(BACKENDS), default="uva",
                    help="uva | ollama (local, no key) | gemini")
    ap.add_argument("--model", default=None, help="model id (default: the backend's default)")
    ap.add_argument("--validate", action="store_true", help="check judge vs oracle agreement")
    ap.add_argument("--label", action="store_true", help="write an LLM-judged preference set")
    ap.add_argument("--mock", action="store_true", help="offline stand-in (no network/key)")
    ap.add_argument("--limit", type=int, default=2, help="how many cases (bounds API calls)")
    ap.add_argument("--debug", action="store_true", help="print each model's raw output")
    args = ap.parse_args()
    global _BACKEND, _DEBUG
    _BACKEND, _DEBUG = args.backend, args.debug
    if args.model is None:
        args.model = BACKENDS[args.backend]["default_model"]
    label(args) if args.label else validate(args)


if __name__ == "__main__":
    main()
