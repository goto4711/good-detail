"""
iob_jsonl — adapter for the relation-extraction output (the "IOB" data).
Reads merged_intermediate_data.jsonl: each line has `document_text`, typed
entity spans (`base_entities` + `new_entities`), and `relationships`. This is
the richest real-data grounding source — typed entities AND relations — the
real-data analogue of the synthetic fact base, and the format the DSPy
extraction front-end (NEXT_STEPS §3) is meant to produce.
"""

import json
from collections import Counter
from pathlib import Path

from ingest import Record, adapter


def _focal(entities, relations):
    """The subject to write about: a PERSON if present (the most frequent one),
    else the most-connected entity in the relation graph, else the first entity."""
    persons = [e["name"] for e in entities if e.get("type", "").upper().startswith("PER")]
    if persons:
        return Counter(persons).most_common(1)[0][0]
    if relations:
        deg = Counter()
        for r in relations:
            deg[r["head"]] += 1
            deg[r["tail"]] += 1
        if deg:
            return deg.most_common(1)[0][0]
    return entities[0]["name"] if entities else ""


@adapter("iob_jsonl")
def load(base, spec):
    path = base / spec["path"]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if not d.get("document_text"):          # skip conversation/training-only records
            continue
        ents = [{"name": e["entity_text"], "type": e.get("entity_type", "ENTITY")}
                for e in (d.get("base_entities", []) + d.get("new_entities", []))
                if e.get("entity_text")]
        rels = [{"type": r["relation_type"], "head": r["head_entity_text"],
                 "tail": r["tail_entity_text"]}
                for r in d.get("relationships", [])
                if r.get("head_entity_text") and r.get("tail_entity_text")]
        yield Record(id=str(d.get("id", "")), corpus="EHRI",
                     register=spec.get("register", "testimony"),
                     unit=_focal(ents, rels), title=str(d.get("id", "")),
                     entities=ents, relations=rels,
                     source_text=d["document_text"], subjects=[],
                     provenance="EHRI relation-extraction")
