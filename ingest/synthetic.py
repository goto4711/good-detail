"""
synthetic — adapter that maps the synthetic generator's `CASES` into `Record`s,
so the synthetic corpus flows through the same interface as real data. (The
synthetic *training/reward* path still uses the native `case` dicts; this adapter
is for symmetry and for corpus-agnostic inspection / generation.)
"""

from ingest import Record, adapter


@adapter("synthetic")
def load(base, spec):
    from generate_synthetic_corpus import CASES, render
    register = spec.get("register", "testimony")
    for case in CASES:
        focal = case["entities"][case["focal"]]["name"]
        ents = [{"name": e["name"], "type": e["type"].upper()}
                for e in case["entities"].values()]
        rels = [{"type": ev["type"], "head": focal, "tail": ev["summary"]}
                for ev in case["events"]]
        yield Record(id=case["case_id"], corpus="SYNTHETIC", register=register,
                     unit=focal, title=focal, entities=ents, relations=rels,
                     source_text=render(case, register, "good"),
                     subjects=[], provenance="synthetic")
