#!/usr/bin/env python3
"""
analyze_annotations.py
================================================================
The workshop payoff (P1): join historian ratings back to the reward scores and
answer the headline question — **which automated reward best predicts human
judgment of "good detail"?** — plus how much historians agree (the pluralism
question).

Inputs:
  candidates.json     from bestofn_demo.py --save (has each candidate's reward scores)
  responses_*.json    one per annotator, downloaded from the annotation sheet

  python analyze_annotations.py candidates.json responses_*.json
"""

import argparse
import json
import sys
from collections import defaultdict
from statistics import mean


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = sum((a - mx) ** 2 for a in xs) ** 0.5
    sy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):                       # average ties
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates")
    ap.add_argument("responses", nargs="+",
                    help="response JSON files, and/or a FOLDER of them (all *.json inside are read)")
    args = ap.parse_args()

    import glob
    import os
    resp_files = []
    for p in args.responses:
        if os.path.isdir(p):
            resp_files += sorted(glob.glob(os.path.join(p, "*.json")))
        else:
            resp_files.append(p)
    cand_abs = os.path.abspath(args.candidates)
    resp_files = [f for f in resp_files if os.path.abspath(f) != cand_abs]   # never treat candidates as a response
    if not resp_files:
        sys.exit("No response *.json files found (pass files or a folder containing them).")

    cand = json.load(open(args.candidates, encoding="utf-8"))
    # reward score per (record_id, candidate_i)
    rewards = ["composite", "linguistic", "F", "judge"]
    score = {}                              # (rid, i) -> {reward: val}
    present = set()
    for r in cand["records"]:
        for c in r["candidates"]:
            score[(r["id"], c["i"])] = c
            present.update(k for k in rewards if k in c)
    rewards = [r for r in rewards if r in present]
    rewards += []  # keep order

    # human ratings: (rid, i) -> [overall by annotator];  annotator -> {(rid,i): overall}
    by_item = defaultdict(list)
    by_annot = {}
    for path in resp_files:
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  (skipping {path}: {e})")
            continue
        if "responses" not in d:            # not an annotation file (e.g. stray JSON)
            continue
        a = d.get("annotator") or path
        by_annot[a] = {}
        for resp in d.get("responses", []):
            key = (resp["record_id"], resp["candidate_i"])
            by_item[key].append(resp["overall"])
            by_annot[a][key] = resp["overall"]
    if not by_item:
        sys.exit("No ratings found in the responses files.")

    print(f"Annotators: {len(by_annot)} ({', '.join(by_annot)})")
    print(f"Rated candidate-items: {len(by_item)}\n")

    # --- inter-annotator agreement: mean pairwise Spearman over shared items ---
    annots = list(by_annot)
    pair_corrs = []
    for i in range(len(annots)):
        for j in range(i + 1, len(annots)):
            shared = set(by_annot[annots[i]]) & set(by_annot[annots[j]])
            if len(shared) >= 3:
                xs = [by_annot[annots[i]][k] for k in shared]
                ys = [by_annot[annots[j]][k] for k in shared]
                pair_corrs.append(_spearman(xs, ys))
    if pair_corrs:
        print(f"Inter-annotator agreement (mean pairwise Spearman): {mean(pair_corrs):+.2f}")
        print("  (high = historians share a notion of good detail; low = genuine pluralism)\n")

    # --- which reward predicts the mean human rating? ---
    items = sorted(by_item)
    human = [mean(by_item[k]) for k in items]
    print("Reward vs. mean human rating (Spearman; higher = better predictor of historians):")
    ranking = []
    for rw in rewards:
        rv = [score[k].get(rw, float("nan")) for k in items]
        pairs = [(h, r) for h, r in zip(human, rv) if r == r]   # drop NaN
        rvals = [r for _, r in pairs]
        if len(pairs) < 3:
            print(f"  {rw:<11} rho =   n/a    (n={len(pairs)}; too few rated items)")
            continue
        if len(set(rvals)) <= 1:   # constant column → Spearman undefined, not a real predictor
            print(f"  {rw:<11} rho =   n/a    (n={len(pairs)}; reward is CONSTANT — zero discrimination, "
                  f"e.g. a saturated judge)")
            continue
        rho = _spearman([h for h, _ in pairs], rvals)
        ranking.append((rho, rw))
        print(f"  {rw:<11} rho = {rho:+.2f}   (n={len(pairs)})")
    if not ranking:
        sys.exit("\nNo reward varied enough to rank against the historians.")
    ranking.sort(key=lambda t: (t[0] != t[0], -t[0]))   # NaN last, else desc
    best = ranking[0][1]
    print(f"\nBest predictor of human 'good detail': **{best}**.")
    print("Reading: this is the validation — the reward that tracks historians is the one whose")
    print("operationalisation of 'good detail' the community's judgments support. A grounded")
    print("reward winning supports the situated thesis; the linguistic/judge winning would not.")

    # --- per-record: does each reward's top pick match the human top pick? ---
    print("\nTop-pick agreement with historians (per record):")
    rec_ids = sorted({k[0] for k in items})
    for rw in rewards:
        hits = 0
        tot = 0
        flat = 0
        for rid in rec_ids:
            ks = [k for k in items if k[0] == rid]
            if len(ks) < 2:
                continue
            tot += 1
            rvals = [score[k].get(rw, float("-inf")) for k in ks]
            if len(set(rvals)) <= 1:    # reward can't pick — it scores every candidate the same
                flat += 1
                continue
            human_top = max(ks, key=lambda k: mean(by_item[k]))
            rew_top = max(ks, key=lambda k: score[k].get(rw, float("-inf")))
            hits += (human_top == rew_top)
        if tot:
            note = f"  ({flat} record(s) the reward couldn't choose — all tied)" if flat else ""
            print(f"  {rw:<11} {hits}/{tot} records{note}")

    # --- per-ANNOTATOR: whose notion of good detail does each reward track? ---
    # The pooled mean hides pluralism. If agreement is low, a reward may match SOME
    # historians and not others — that pattern is the finding, not noise.
    print("\nPer-annotator reward correlation (Spearman of reward vs. that historian's ratings):")
    var_rewards = []
    for rw in rewards:
        rv_all = [score[k].get(rw, float("nan")) for k in items]
        if len({r for r in rv_all if r == r}) > 1:   # reward varies across items
            var_rewards.append(rw)
    if not var_rewards:
        print("  (no reward varies across items — nothing to correlate)")
    else:
        hdr = "  {:<22}".format("annotator") + "".join(f"{rw[:10]:>12}" for rw in var_rewards)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        per_rw_vals = {rw: [] for rw in var_rewards}
        for a in by_annot:
            shared = [k for k in items if k in by_annot[a]]
            if len(shared) < 3:
                continue
            hv = [by_annot[a][k] for k in shared]
            cells = []
            for rw in var_rewards:
                pairs = [(h, score[k].get(rw, float("nan"))) for h, k in zip(hv, shared)
                         if score[k].get(rw, float("nan")) == score[k].get(rw, float("nan"))]
                if len(pairs) >= 3 and len({r for _, r in pairs}) > 1 and len({h for h, _ in pairs}) > 1:
                    rho = _spearman([h for h, _ in pairs], [r for _, r in pairs])
                    per_rw_vals[rw].append(rho)
                    cells.append(f"{rho:+12.2f}")
                else:
                    cells.append(f"{'n/a':>12}")
            print(f"  {a[:22]:<22}" + "".join(cells))
        # spread across annotators = how pluralistic the reward's fit is
        print("  " + "-" * (len(hdr) - 2))
        mean_cells, range_cells = [], []
        for rw in var_rewards:
            vs = per_rw_vals[rw]
            if vs:
                mean_cells.append(f"{mean(vs):+12.2f}")
                range_cells.append(f"{(max(vs) - min(vs)):>12.2f}")
            else:
                mean_cells.append(f"{'n/a':>12}")
                range_cells.append(f"{'n/a':>12}")
        print(f"  {'mean across historians':<22}" + "".join(mean_cells))
        print(f"  {'spread (max-min)':<22}" + "".join(range_cells))
        print("  Reading: a HIGH mean = the reward tracks the community on average; a WIDE spread")
        print("  = it tracks some historians far better than others (pluralism — the reward encodes")
        print("  one situated notion of good detail, not a universal one).")


if __name__ == "__main__":
    main()
