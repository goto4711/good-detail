#!/usr/bin/env python3
"""
ingest_adapters.py
================================================================
The INGESTION side of the pipeline: read a TEI XML or EHRI-style IOB file
and reconstruct a `Record` (plain text + entity list with offsets) — the
shared "record unit" every corpus is mapped onto.

This is a first, tested version of the real ingestion adapters, exercised
end-to-end on the synthetic TEI/IOB files produced by emit_tei_iob.py.

What this test shows
--------------------
1. TEI ingest reconstructs the EXACT document text + entity spans (TEI keeps
   full text and offsets).
2. IOB ingest reconstructs the entities (surface + type); the document text
   is only an approximation, because IOB tokenisation discards original
   whitespace/punctuation spacing — an inherent property of the format.
3. The TAG-LOSS BOUNDARY: an ingested record carries text + entities but NOT
   the fact-base control tags (salience, grounding, certainty, sensitivity).
   So from TEI/IOB you can run the LINGUISTIC reward and the NER-grounding
   check, but NOT the ORACLE / preference labels — those need the fact base
   (i.e. human annotation in the real world). The pipeline's "human" arm
   cannot be reconstructed from format files alone; the linguistic arm can.

Usage:  python ingest_adapters.py
"""

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

try:
    from generate_synthetic_corpus import CASES, REGISTERS, render, linguistic_score
    from emit_tei_iob import (gold_candidates, find_mentions, gold_set, norm,
                              ELEM2CANON, SHORT2CANON, TOKEN_RE)
except ImportError as e:
    sys.exit(f"Run from the project folder (needs generate_synthetic_corpus.py + emit_tei_iob.py): {e}")

HERE = Path(__file__).parent
TEI_NS = "http://www.tei-c.org/ns/1.0"


@dataclass
class Record:
    doc_id: str
    text: str
    entities: list = field(default_factory=list)   # (surface, canonical_type, start, end)

    def entity_set(self):
        return {(norm(s), t) for s, t, _, _ in self.entities}


# ----------------------------------------------------------------------
# TEI -> Record  (full text + offsets preserved)
# ----------------------------------------------------------------------

def ingest_tei(path) -> Record:
    root = ET.parse(path).getroot()
    body = root.find(f".//{{{TEI_NS}}}body")
    parts, ents, pos = [], [], 0
    for p in body.findall(f"{{{TEI_NS}}}p"):
        if p.text:
            parts.append(p.text); pos += len(p.text)
        for child in p:
            local = child.tag.split("}")[-1]
            ctext = child.text or ""
            start = pos
            parts.append(ctext); pos += len(ctext)
            if local in ELEM2CANON:
                ents.append((ctext, ELEM2CANON[local], start, pos))
            if child.tail:
                parts.append(child.tail); pos += len(child.tail)
    return Record(Path(path).stem, "".join(parts), ents)


# ----------------------------------------------------------------------
# IOB -> Record  (entities exact; text is a detokenised approximation)
# ----------------------------------------------------------------------

def ingest_iob(path) -> Record:
    tokens, tags = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        word, tag = line.rsplit(" ", 1)
        tokens.append(word); tags.append(tag)

    text = " ".join(tokens)                      # approx: original spacing is lost
    # rebuild offsets in the detokenised text
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok))); pos += len(tok) + 1

    ents, i = [], 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            typ = tags[i][2:]
            start = offsets[i][0]; j = i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            end = offsets[j - 1][1]
            surface = text[start:end]
            ents.append((surface, SHORT2CANON[typ], start, end))
            i = j
        else:
            i += 1
    return Record(Path(path).stem, text, ents)


# ----------------------------------------------------------------------
# End-to-end test against the source of truth
# ----------------------------------------------------------------------

def main():
    tei_dir, iob_dir = HERE / "synthetic_corpus" / "tei", HERE / "synthetic_corpus" / "iob"
    if not tei_dir.exists():
        sys.exit("No synthetic_corpus/tei — run `python emit_tei_iob.py` first.")

    n = tei_diff = 0
    tei_text_ok = tei_ent_ok = iob_ent_ok = 0
    sample = None

    for case in CASES:
        for register in REGISTERS:
            src_text = render(case, register, "good")
            gold = gold_set(find_mentions(src_text, gold_candidates(case)))
            stem = f"{case['case_id']}_{register}"

            tei_rec = ingest_tei(tei_dir / f"{stem}.xml")
            iob_rec = ingest_iob(iob_dir / f"{stem}.txt")
            n += 1
            if tei_rec.text == src_text:
                tei_text_ok += 1
            if tei_rec.entity_set() == gold:
                tei_ent_ok += 1
            if iob_rec.entity_set() == gold:
                iob_ent_ok += 1
            if sample is None:
                sample = (stem, tei_rec, gold)

    print(f"Ingested {n} TEI and {n} IOB files.\n")
    print("Adapter fidelity (vs. the source fact base):")
    print(f"  TEI  exact document text : {tei_text_ok}/{n} {'PASS' if tei_text_ok == n else 'FAIL'}")
    print(f"  TEI  entities            : {tei_ent_ok}/{n} {'PASS' if tei_ent_ok == n else 'FAIL'}")
    print(f"  IOB  entities            : {iob_ent_ok}/{n} {'PASS' if iob_ent_ok == n else 'FAIL'}")
    print(f"  IOB  text is a detokenised approximation (whitespace not recoverable) — not asserted.")

    # --- demonstrate what the pipeline CAN and CANNOT do from an ingested record ---
    stem, rec, gold = sample
    ling, propers, nums = linguistic_score(rec.text)
    print(f"\nDownstream hook (example: {stem}):")
    print(f"  record has {len(rec.entities)} entities and {len(rec.text)} chars of text.")
    print(f"  LINGUISTIC reward computable from ingested text  -> {ling} (propers={propers}, nums={nums})")
    print(f"  ORACLE reward computable from ingested record    -> NO")
    print(f"     reason: TEI/IOB carry text + entities but NOT the fact-base control")
    print(f"     tags (salience, grounding, certainty, sensitivity). The linguistic arm")
    print(f"     survives ingestion; the human/oracle arm must come from annotation.")
    print(f"\nThis is the ingestion boundary, made concrete: the culture-blind reward")
    print(f"transfers through any format for free; the situated reward does not.")


if __name__ == "__main__":
    main()
