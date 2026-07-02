"""
tei_ehri — adapter for the EHRI early-testimony TEI/XML.
Each document = one `<hash>_header.xml` (entity inventory + provenance) + many
`<hash>_split_NNN.xml` (the testimony body, inline <term> keywords). Header →
attested entities; body → source_text (NLI premise).
"""

import re
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from ingest import Record, adapter

TEI = "{http://www.tei-c.org/ns/1.0}"


def _text(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def discover(xml_dir):
    docs = defaultdict(lambda: {"header": None, "splits": []})
    for p in sorted(Path(xml_dir).glob("*.xml")):
        doc_id = p.name.split("_")[0]
        if p.name.endswith("_header.xml"):
            docs[doc_id]["header"] = p
        elif "_split_" in p.name:
            docs[doc_id]["splits"].append(p)
    for d in docs.values():
        d["splits"].sort(key=lambda q: int(re.search(r"_split_(\d+)", q.name).group(1)))
    return dict(docs)


def parse_header(path):
    root = ET.parse(path).getroot()

    def first(tag):
        el = root.find(f".//{TEI}{tag}")
        return _text(el) if el is not None else ""

    persons = [_text(e) for e in root.findall(f".//{TEI}person/{TEI}persName")]
    places = []
    for pl in root.findall(f".//{TEI}place"):
        nm = pl.find(f"{TEI}placeName")
        if nm is not None:
            places.append(_text(nm))
    orgs = [_text(e) for e in root.findall(f".//{TEI}org/{TEI}orgName")]
    return {"title": first("title"), "repository": first("repository"),
            "collection": first("collection"),
            "persons": persons, "places": places, "orgs": orgs}


def parse_body(split_paths):
    paras, subjects = [], []
    for sp in split_paths:
        root = ET.parse(sp).getroot()
        for p in root.iter(f"{TEI}p"):
            t = _text(p)
            if t:
                paras.append(t)
        for term in root.iter(f"{TEI}term"):
            kw = _text(term)
            if kw:
                subjects.append(kw)
    return "\n\n".join(paras), subjects


def _focal(persons, title):
    for p in persons:
        if "narrator" in p.lower():
            return re.sub(r"\s*\(.*?\)\s*", "", p).strip()
    if persons:
        return re.sub(r"\s*\(.*?\)\s*", "", persons[0]).strip()
    return title.split(",")[0].strip()


@adapter("tei_ehri")
def load(base, spec):
    xml_dir = base / spec.get("path", "xml")
    for doc_id, d in discover(xml_dir).items():
        if not (d["header"] and d["splits"]):
            continue
        h = parse_header(d["header"])
        body, subjects = parse_body(d["splits"])
        entities = ([{"name": p, "type": "PERSON"} for p in h["persons"]]
                    + [{"name": p, "type": "PLACE"} for p in h["places"]]
                    + [{"name": o, "type": "ORG"} for o in h["orgs"]])
        prov = h["repository"] + (f", {h['collection']}" if h["collection"] else "")
        yield Record(id=doc_id, corpus="EHRI",
                     register=spec.get("register", "testimony"),
                     unit=_focal(h["persons"], h["title"]), title=h["title"],
                     entities=entities, relations=[], source_text=body,
                     subjects=subjects, provenance=prov)
