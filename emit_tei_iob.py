#!/usr/bin/env python3
"""
emit_tei_iob.py
================================================================
Serialise the synthetic fact base into two digital-humanities formats —
TEI XML and EHRI-style IOB — WITH gold entity annotations, and validate
each by a round trip (emit -> parse back -> assert recovered == gold).

Why this exists
---------------
The real-corpus ingestion adapters (TEI / IOB -> record units) are the
not-yet-built part of the pipeline. Because we OWN the synthetic fact base,
we can emit TEI/IOB whose annotations are gold by construction, then use
them to (a) exercise the adapters on realistic inputs and (b) prove the
adapters parse losslessly. The round trip here validates the *serialiser*;
the same parse functions are the seed of the real adapters.

Tag scheme matches prepare_data_latest/ehri_en.txt:
  IOB : space-delimited "TOKEN TAG", tags PERS / LOC / ORG / DATE (BIO).
  TEI : <persName> <placeName> <orgName> <date> inline in <text><body><p>,
        default TEI namespace, minimal <teiHeader>.

Everything is FICTIONAL (it comes from the synthetic cases).

Usage
-----
  python emit_tei_iob.py                      # both registers, validate
  python emit_tei_iob.py --register testimony
  python emit_tei_iob.py --out synthetic_corpus
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from generate_synthetic_corpus import CASES, REGISTERS, render
except ImportError:
    sys.exit("Run this from the same folder as generate_synthetic_corpus.py")

HERE = Path(__file__).parent
TEI_NS = "http://www.tei-c.org/ns/1.0"
ET.register_namespace("", TEI_NS)

# canonical type -> (TEI element, IOB short tag)
TYPE_MAP = {
    "person": ("persName", "PERS"),
    "place":  ("placeName", "LOC"),
    "org":    ("orgName",  "ORG"),
    "date":   ("date",     "DATE"),
}
CANON = {"person": "PERSON", "place": "LOCATION", "org": "ORGANIZATION", "date": "DATE"}
SHORT2CANON = {"PERS": "PERSON", "LOC": "LOCATION", "ORG": "ORGANIZATION", "DATE": "DATE"}
ELEM2CANON = {"persName": "PERSON", "placeName": "LOCATION", "orgName": "ORGANIZATION", "date": "DATE"}

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def norm(surface):
    """Tokenise + space-join, so surfaces compare equal regardless of
    punctuation spacing (used for both gold and recovered entities)."""
    return " ".join(TOKEN_RE.findall(surface))


def strip_article(name):
    return re.sub(r"^(the|a|an)\s+", "", name, flags=re.IGNORECASE)


# ----------------------------------------------------------------------
# Gold mention extraction (the ground truth, by construction)
# ----------------------------------------------------------------------

def gold_candidates(case):
    """(surface, base_type) pairs to look for in the rendered text."""
    out = []
    for ent in case["entities"].values():
        if ent["type"] in ("person", "place", "org"):
            out.append((strip_article(ent["name"]), ent["type"]))
    for ev in case["events"]:
        out.append((ev["date"]["value"], "date"))
    focal = case["entities"][case["focal"]]
    for a in focal["attributes"]:
        if a["key"] in ("birth_year", "built"):
            out.append((a["value"], "date"))
    # dedupe by surface, keep first type
    seen, uniq = set(), []
    for s, t in out:
        if s and s not in seen:
            seen.add(s)
            uniq.append((s, t))
    return uniq


def find_mentions(text, candidates):
    """Locate non-overlapping occurrences; longest candidate wins."""
    claimed, mentions = [], []
    for surface, base_type in sorted(candidates, key=lambda x: -len(x[0])):
        for m in re.finditer(re.escape(surface), text):
            s, e = m.start(), m.end()
            if any(s < ce and cs < e for cs, ce in claimed):
                continue
            claimed.append((s, e))
            mentions.append((s, e, surface, base_type))
    mentions.sort(key=lambda x: x[0])
    return mentions


def gold_set(mentions):
    return {(norm(surf), CANON[bt]) for _, _, surf, bt in mentions}


# ----------------------------------------------------------------------
# IOB serialise + parse
# ----------------------------------------------------------------------

def to_iob(text, mentions):
    lines = []
    for tok in TOKEN_RE.finditer(text):
        s, e, word = tok.start(), tok.end(), tok.group()
        tag = "O"
        for ms, me, _, bt in mentions:
            if s >= ms and e <= me:
                short = TYPE_MAP[bt][1]
                tag = f"B-{short}" if s == ms else f"I-{short}"
                break
        lines.append(f"{word} {tag}")
        if word in {".", "!", "?"}:
            lines.append("")            # blank line = sentence break
    return "\n".join(lines).rstrip() + "\n"


def parse_iob(path):
    """Reference IOB parser -> set of (norm surface, canonical type)."""
    ents, cur_toks, cur_type = set(), [], None

    def flush():
        nonlocal cur_toks, cur_type
        if cur_toks:
            ents.add((norm(" ".join(cur_toks)), SHORT2CANON[cur_type]))
        cur_toks, cur_type = [], None

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            flush()
            continue
        word, tag = line.rsplit(" ", 1)
        if tag == "O":
            flush()
        elif tag.startswith("B-"):
            flush()
            cur_type = tag[2:]
            cur_toks = [word]
        elif tag.startswith("I-"):
            if cur_type == tag[2:]:
                cur_toks.append(word)
            else:                       # malformed; treat as new
                flush()
                cur_type = tag[2:]
                cur_toks = [word]
    flush()
    return ents


# ----------------------------------------------------------------------
# TEI serialise + parse
# ----------------------------------------------------------------------

def q(tag):
    return f"{{{TEI_NS}}}{tag}"


def date_attrs(value):
    parts = re.split(r"[–-]", value)
    if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
        return {"from": parts[0].strip(), "to": parts[1].strip()}
    if value.strip().isdigit():
        return {"when": value.strip()}
    return {}


def build_p(text, mentions):
    p = ET.Element(q("p"))
    last = None
    cursor = 0

    def add_text(s):
        nonlocal last
        if not s:
            return
        if last is None:
            p.text = (p.text or "") + s
        else:
            last.tail = (last.tail or "") + s

    for ms, me, surf, bt in mentions:
        add_text(text[cursor:ms])
        elem_name, _ = TYPE_MAP[bt]
        child = ET.SubElement(p, q(elem_name))
        child.text = text[ms:me]
        if bt == "date":
            for k, v in date_attrs(surf).items():
                child.set(k, v)
        last = child
        cursor = me
    add_text(text[cursor:])
    return p


def to_tei(case, register, text, mentions):
    tei = ET.Element(q("TEI"))
    header = ET.SubElement(tei, q("teiHeader"))
    fd = ET.SubElement(header, q("fileDesc"))
    ts = ET.SubElement(fd, q("titleStmt"))
    ET.SubElement(ts, q("title")).text = \
        f"{case['entities'][case['focal']]['name']} — synthetic {register}"
    ps = ET.SubElement(fd, q("publicationStmt"))
    ET.SubElement(ps, q("p")).text = "Fictional synthetic data — not a real archival record."
    sd = ET.SubElement(fd, q("sourceDesc"))
    refs = ", ".join(s["archive_ref"] for s in case["sources"].values())
    ET.SubElement(sd, q("p")).text = f"Synthetic; nominal source refs: {refs}"
    body = ET.SubElement(ET.SubElement(tei, q("text")), q("body"))
    # Pretty-print the STRUCTURE first, then add the paragraph verbatim.
    # (ET.indent would inject significant whitespace into the <p> mixed
    # content; whitespace inside mixed content must be preserved exactly.)
    ET.indent(tei, space="  ")
    body.append(build_p(text, mentions))
    body.text = "\n      "
    body[0].tail = "\n    "
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(tei, encoding="unicode")


def parse_tei(path):
    """Reference TEI parser -> set of (norm surface, canonical type)."""
    root = ET.parse(path).getroot()
    body = root.find(f".//{q('body')}")
    ents = set()
    for elem in body.iter():
        local = elem.tag.split("}")[-1]
        if local in ELEM2CANON and elem.text:
            ents.add((norm(elem.text), ELEM2CANON[local]))
    return ents


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", choices=["testimony", "finding_aid", "both"], default="both")
    ap.add_argument("--out", type=Path, default=HERE / "synthetic_corpus")
    args = ap.parse_args()

    registers = REGISTERS if args.register == "both" else [args.register]
    tei_dir, iob_dir = args.out / "tei", args.out / "iob"
    tei_dir.mkdir(parents=True, exist_ok=True)
    iob_dir.mkdir(parents=True, exist_ok=True)

    total, tei_ok, iob_ok, ent_count = 0, 0, 0, 0
    failures = []

    for case in CASES:
        for register in registers:
            text = render(case, register, "good")          # annotate the grounded rendering
            mentions = find_mentions(text, gold_candidates(case))
            gold = gold_set(mentions)
            ent_count += len(gold)
            stem = f"{case['case_id']}_{register}"

            tei_str = to_tei(case, register, text, mentions)
            (tei_dir / f"{stem}.xml").write_text(tei_str, encoding="utf-8")
            iob_str = to_iob(text, mentions)
            (iob_dir / f"{stem}.txt").write_text(iob_str, encoding="utf-8")

            total += 1
            got_tei = parse_tei(tei_dir / f"{stem}.xml")
            got_iob = parse_iob(iob_dir / f"{stem}.txt")
            if got_tei == gold:
                tei_ok += 1
            else:
                failures.append((stem, "TEI", gold ^ got_tei))
            if got_iob == gold:
                iob_ok += 1
            else:
                failures.append((stem, "IOB", gold ^ got_iob))

    print(f"Wrote {total} TEI files -> {tei_dir}")
    print(f"Wrote {total} IOB files -> {iob_dir}")
    print(f"Gold entity mentions tagged: {ent_count}")
    print(f"\nRound-trip (emit -> parse -> compare to gold):")
    print(f"  TEI : {tei_ok}/{total} {'PASS' if tei_ok == total else 'FAIL'}")
    print(f"  IOB : {iob_ok}/{total} {'PASS' if iob_ok == total else 'FAIL'}")
    if failures:
        print("\nMismatches (symmetric difference gold ^ recovered):")
        for stem, fmt, diff in failures[:10]:
            print(f"  {stem} [{fmt}]: {diff}")
        sys.exit(1)
    print("\nAll round-trips pass: the serialisers are lossless, and parse_tei/")
    print("parse_iob are a validated starting point for the real ingestion adapters.")


if __name__ == "__main__":
    main()
