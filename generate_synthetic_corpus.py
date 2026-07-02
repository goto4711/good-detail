#!/usr/bin/env python3
"""
generate_synthetic_corpus.py
================================================================
Synthetic fact-base generator for the "good detail" reward study.

See synthetic_factbase_spec.md for the design. This v1:
  * holds a small set of HAND-AUTHORED, FICTIONAL fact bases (cases)
  * renders each case across the 2x3 grid:
        registers : testimony | finding_aid
        profiles  : good | flattened | fabricated
  * scores every rendering with two judges:
        ORACLE      -- reads the rendering's structured trace (ground truth:
                       coverage / specificity / source-attribution / calibration,
                       minus fabrication [gating] and sensationalism)
        LINGUISTIC  -- reads ONLY the surface text (proper-noun + number density);
                       blind to grounding by construction
  * writes cases.json, renderings.jsonl, preference_pairs.jsonl, report.md
  * runs the first hypothesis test (spec sec.4):
        ORACLE ranks good >> fabricated ; LINGUISTIC fails to separate them.

Rendering here is DETERMINISTIC / TEMPLATED on purpose: fully reproducible,
no model, no API, no drift. (LLM-rendering-with-verification is the v2 path.)

EVERYTHING IS FICTIONAL. Names, places and archive refs are invented.
"""

import json
import math
import os
import re
from pathlib import Path

OUTDIR = Path(__file__).parent / "synthetic_corpus"

# ----------------------------------------------------------------------
# 1. HAND-AUTHORED FICTIONAL CASES
# ----------------------------------------------------------------------
# Each event/attribute carries the five control tags from the spec.
# `fabrications` = false specifics the 'fabricated' profile injects;
#   sensational=True ones also trip the restraint dimension.

CASES = [
    {
        "case_id": "synth_case_001",
        "focal": "marta",
        "entities": {
            "marta": {"type": "person", "name": "Marta Hellinger", "fictional": True,
                      "attributes": [
                          {"key": "occupation", "value": "milliner", "specificity": "specific",
                           "grounded_by": ["S1"], "certainty": "attested"},
                          {"key": "birth_year", "value": "1911", "specificity": "specific",
                           "grounded_by": ["S1"], "certainty": "attested"}]},
            "adler": {"type": "org", "name": "the Adler & Son hat workshop", "fictional": True,
                      "attributes": [
                          {"key": "address", "value": "Brünnergasse 7, Brünn", "specificity": "specific",
                           "grounded_by": ["S1"], "certainty": "attested"}]},
            "josef": {"type": "person", "name": "Josef Hellinger", "fictional": True, "attributes": []},
            "rotterdam": {"type": "place", "name": "Rotterdam", "fictional": False, "attributes": []},
        },
        "events": [
            {"id": "E1", "type": "employment", "summary": "Marta worked at the Adler & Son hat workshop",
             "participants": [{"entity_id": "marta", "role": "employee"}], "place_id": "adler",
             "date": {"value": "1938", "precision": "year", "certainty": "attested"},
             "grounded_by": ["S1"], "salience": 3, "sensitivity": "low"},
            {"id": "E2", "type": "emigration", "summary": "Marta left for Rotterdam",
             "participants": [{"entity_id": "marta", "role": "emigrant"}], "place_id": "rotterdam",
             "date": {"value": "1939", "precision": "year", "certainty": "probable"},
             "grounded_by": ["S2"], "salience": 2, "sensitivity": "low"},
            {"id": "E3", "type": "detention", "summary": "Marta was detained during a round-up",
             "participants": [{"entity_id": "marta", "role": "detained"}], "place_id": None,
             "date": {"value": "1942", "precision": "year", "certainty": "uncertain"},
             "grounded_by": ["S2"], "salience": 3, "sensitivity": "high"},
        ],
        "relations": [
            {"head_id": "josef", "type": "sibling_of", "tail_id": "marta",
             "grounded_by": ["S2"], "certainty": "probable", "salience": 1}],
        "sources": {
            "S1": {"kind": "registry", "archive_ref": "WL/SYN/118",
                   "supports": ["marta.occupation", "marta.birth_year", "adler.address", "E1"]},
            "S2": {"kind": "testimony", "archive_ref": "WL/SYN/204", "supports": ["E2", "E3"]}},
        "fabrications": [
            {"text": "After the Gestapo seized the Adler workshop she fled to Rotterdam in March 1939.", "sensational": False},
            {"text": "In 1942 she was dragged from her home at dawn and deported to the east.", "sensational": True}],
    },
    {
        "case_id": "synth_case_002",
        "focal": "cafe",
        "entities": {
            "cafe": {"type": "place", "name": "Café Landauer", "fictional": True,
                     "attributes": [
                         {"key": "address", "value": "Krapfengasse 3, Brünn", "specificity": "specific",
                          "grounded_by": ["S1"], "certainty": "attested"}]},
            "hugo": {"type": "person", "name": "Hugo Landauer", "fictional": True,
                     "attributes": [
                         {"key": "role", "value": "proprietor", "specificity": "specific",
                          "grounded_by": ["S1"], "certainty": "attested"}]},
        },
        "events": [
            {"id": "E1", "type": "meeting_point",
             "summary": "Café Landauer served as an informal meeting point for emigration inquiries",
             "participants": [{"entity_id": "cafe", "role": "venue"}], "place_id": "cafe",
             "date": {"value": "1938–1939", "precision": "range", "certainty": "probable"},
             "grounded_by": ["S1"], "salience": 3, "sensitivity": "low"},
            {"id": "E2", "type": "closure", "summary": "Café Landauer was closed and Aryanised",
             "participants": [{"entity_id": "cafe", "role": "subject"}], "place_id": "cafe",
             "date": {"value": "1939", "precision": "year", "certainty": "uncertain"},
             "grounded_by": ["S2"], "salience": 2, "sensitivity": "medium"},
        ],
        "relations": [
            {"head_id": "hugo", "type": "proprietor_of", "tail_id": "cafe",
             "grounded_by": ["S1"], "certainty": "attested", "salience": 2}],
        "sources": {
            "S1": {"kind": "testimony", "archive_ref": "WL/SYN/301", "supports": ["cafe.address", "hugo.role", "E1"]},
            "S2": {"kind": "registry", "archive_ref": "WL/SYN/302", "supports": ["E2"]}},
        "fabrications": [
            {"text": "It was firebombed by stormtroopers on 9 November 1938 with patrons still inside.", "sensational": True}],
    },
    {
        "case_id": "synth_case_003",
        "focal": "committee",
        "entities": {
            "committee": {"type": "org", "name": "the Brünn Emigration Aid Committee", "fictional": True,
                          "attributes": [
                              {"key": "purpose", "value": "assistance with emigration papers", "specificity": "specific",
                               "grounded_by": ["S1"], "certainty": "attested"}]},
            "office": {"type": "place", "name": "Pekařská 12", "fictional": True, "attributes": []},
        },
        "events": [
            {"id": "E1", "type": "operation",
             "summary": "the Committee helped families obtain emigration documents from its office on Pekařská",
             "participants": [{"entity_id": "committee", "role": "actor"}], "place_id": "office",
             "date": {"value": "1939", "precision": "year", "certainty": "attested"},
             "grounded_by": ["S1"], "salience": 3, "sensitivity": "low"},
            {"id": "E2", "type": "dissolution", "summary": "the Committee ceased operating",
             "participants": [{"entity_id": "committee", "role": "subject"}], "place_id": None,
             "date": {"value": "1941", "precision": "year", "certainty": "probable"},
             "grounded_by": ["S2"], "salience": 2, "sensitivity": "medium"},
        ],
        "relations": [],
        "sources": {
            "S1": {"kind": "letter", "archive_ref": "WL/SYN/410", "supports": ["committee.purpose", "E1"]},
            "S2": {"kind": "registry", "archive_ref": "WL/SYN/411", "supports": ["E2"]}},
        "fabrications": [
            {"text": "Its entire leadership was secretly executed in 1941 after a betrayal.", "sensational": True}],
    },
    {
        "case_id": "synth_case_004",
        "focal": "ruth",
        "entities": {
            "ruth": {"type": "person", "name": "Ruth Salzmann", "fictional": True,
                     "attributes": [
                         {"key": "occupation", "value": "nurse", "specificity": "specific",
                          "grounded_by": ["S1"], "certainty": "attested"}]},
            "hospital": {"type": "org", "name": "the Jewish hospital on Ptašínského", "fictional": True, "attributes": []},
        },
        "events": [
            {"id": "E1", "type": "employment", "summary": "Ruth Salzmann nursed at the Jewish hospital on Ptašínského",
             "participants": [{"entity_id": "ruth", "role": "employee"}], "place_id": "hospital",
             "date": {"value": "1940", "precision": "year", "certainty": "attested"},
             "grounded_by": ["S1"], "salience": 3, "sensitivity": "low"},
            {"id": "E2", "type": "transfer", "summary": "Ruth Salzmann was sent to Theresienstadt",
             "participants": [{"entity_id": "ruth", "role": "deportee"}], "place_id": None,
             "date": {"value": "1942", "precision": "year", "certainty": "probable"},
             "grounded_by": ["S2"], "salience": 3, "sensitivity": "high"},
        ],
        "relations": [],
        "sources": {
            "S1": {"kind": "registry", "archive_ref": "WL/SYN/520", "supports": ["ruth.occupation", "E1"]},
            "S2": {"kind": "testimony", "archive_ref": "WL/SYN/521", "supports": ["E2"]}},
        "fabrications": [
            {"text": "She personally smuggled forty children to safety before being shot at the gate.", "sensational": True}],
    },
    {
        "case_id": "synth_case_005",
        "focal": "pavel",
        "entities": {
            "pavel": {"type": "person", "name": "Pavel Stern", "fictional": True,
                      "attributes": [
                          {"key": "school", "value": "the Hybešova primary school", "specificity": "specific",
                           "grounded_by": ["S1"], "certainty": "attested"},
                          {"key": "birth_year", "value": "1930", "specificity": "specific",
                           "grounded_by": ["S1"], "certainty": "attested"}]},
        },
        "events": [
            {"id": "E1", "type": "schooling", "summary": "Pavel Stern was enrolled at the Hybešova primary school",
             "participants": [{"entity_id": "pavel", "role": "pupil"}], "place_id": None,
             "date": {"value": "1938", "precision": "year", "certainty": "attested"},
             "grounded_by": ["S1"], "salience": 3, "sensitivity": "low"},
            {"id": "E2", "type": "exclusion", "summary": "Pavel Stern was removed from the school register",
             "participants": [{"entity_id": "pavel", "role": "subject"}], "place_id": None,
             "date": {"value": "1940", "precision": "year", "certainty": "probable"},
             "grounded_by": ["S1"], "salience": 2, "sensitivity": "medium"},
        ],
        "relations": [],
        "sources": {
            "S1": {"kind": "registry", "archive_ref": "WL/SYN/630",
                   "supports": ["pavel.school", "pavel.birth_year", "E1", "E2"]}},
        "fabrications": [
            {"text": "He was the last child to leave the burning schoolhouse in 1940.", "sensational": True}],
    },
    {
        "case_id": "synth_case_006",
        "focal": "synagogue",
        "entities": {
            "synagogue": {"type": "place", "name": "the Pferdgasse synagogue", "fictional": True,
                          "attributes": [
                              {"key": "built", "value": "1885", "specificity": "specific",
                               "grounded_by": ["S1"], "certainty": "attested"}]},
        },
        "events": [
            {"id": "E1", "type": "function", "summary": "the Pferdgasse synagogue served the city's eastern congregation",
             "participants": [{"entity_id": "synagogue", "role": "venue"}], "place_id": "synagogue",
             "date": {"value": "1885–1938", "precision": "range", "certainty": "attested"},
             "grounded_by": ["S1"], "salience": 3, "sensitivity": "low"},
            {"id": "E2", "type": "destruction", "summary": "the Pferdgasse synagogue was damaged and later demolished",
             "participants": [{"entity_id": "synagogue", "role": "subject"}], "place_id": "synagogue",
             "date": {"value": "1939", "precision": "year", "certainty": "uncertain"},
             "grounded_by": ["S2"], "salience": 3, "sensitivity": "high"},
        ],
        "relations": [],
        "sources": {
            "S1": {"kind": "catalogue_entry", "archive_ref": "WL/SYN/740", "supports": ["synagogue.built", "E1"]},
            "S2": {"kind": "caption", "archive_ref": "WL/SYN/741", "supports": ["E2"]}},
        "fabrications": [
            {"text": "A mob burned it to the ground in a single night while the congregation was at prayer.", "sensational": True}],
    },
    {
        "case_id": "synth_case_007",
        "focal": "heinrich",
        "entities": {
            "heinrich": {"type": "person", "name": "Heinrich Adler", "fictional": True,
                         "attributes": [
                             {"key": "role", "value": "owner of the Adler & Son hat workshop", "specificity": "specific",
                              "grounded_by": ["S1"], "certainty": "attested"}]},
        },
        "events": [
            {"id": "E1", "type": "ownership", "summary": "Heinrich Adler ran the family hat workshop on Brünnergasse",
             "participants": [{"entity_id": "heinrich", "role": "owner"}], "place_id": None,
             "date": {"value": "1925–1938", "precision": "range", "certainty": "attested"},
             "grounded_by": ["S1"], "salience": 3, "sensitivity": "low"},
            {"id": "E2", "type": "dispossession", "summary": "Heinrich Adler lost the workshop under forced transfer",
             "participants": [{"entity_id": "heinrich", "role": "subject"}], "place_id": None,
             "date": {"value": "1939", "precision": "year", "certainty": "probable"},
             "grounded_by": ["S2"], "salience": 3, "sensitivity": "medium"},
        ],
        "relations": [],
        "sources": {
            "S1": {"kind": "registry", "archive_ref": "WL/SYN/118", "supports": ["heinrich.role", "E1"]},
            "S2": {"kind": "letter", "archive_ref": "WL/SYN/850", "supports": ["E2"]}},
        "fabrications": [
            {"text": "He confronted the new owner at gunpoint before vanishing without trace in 1939.", "sensational": True}],
    },
    {
        "case_id": "synth_case_008",
        "focal": "suitcase",
        "entities": {
            "suitcase": {"type": "object", "name": "a brown leather suitcase", "fictional": True,
                         "attributes": [
                             {"key": "inventory_no", "value": "WL-OBJ-0099", "specificity": "specific",
                              "grounded_by": ["S1"], "certainty": "attested"}]},
            "owner": {"type": "person", "name": "Lena Kahn", "fictional": True, "attributes": []},
        },
        "events": [
            {"id": "E1", "type": "provenance",
             "summary": "the suitcase was donated to the archive as having belonged to Lena Kahn",
             "participants": [{"entity_id": "suitcase", "role": "object"}], "place_id": None,
             "date": {"value": "1947", "precision": "year", "certainty": "attested"},
             "grounded_by": ["S1"], "salience": 3, "sensitivity": "low"},
            {"id": "E2", "type": "attribution", "summary": "Lena Kahn is thought to have carried it into emigration",
             "participants": [{"entity_id": "owner", "role": "owner"}], "place_id": None,
             "date": {"value": "1939", "precision": "year", "certainty": "uncertain"},
             "grounded_by": ["S2"], "salience": 2, "sensitivity": "low"},
        ],
        "relations": [],
        "sources": {
            "S1": {"kind": "catalogue_entry", "archive_ref": "WL/SYN/960", "supports": ["suitcase.inventory_no", "E1"]},
            "S2": {"kind": "testimony", "archive_ref": "WL/SYN/961", "supports": ["E2"]}},
        "fabrications": [
            {"text": "It still contained a child's shoe and a farewell letter when opened in 1947.", "sensational": True}],
    },
]

# Experiment dials (registers, profiles, the fabrication pool) live in config.py
from config import REGISTERS, PROFILES, FALSE_NAMES, FALSE_PLACES, FALSE_DATES


def _false_clause(case_id, register):
    h = sum(ord(c) for c in case_id)
    name = FALSE_NAMES[h % len(FALSE_NAMES)]
    place = FALSE_PLACES[(h // 3) % len(FALSE_PLACES)]
    date = FALSE_DATES[(h // 5) % len(FALSE_DATES)]
    if register == "testimony":
        return f"{name} is said to have handled the matter at {place} on {date}."
    return f"Cross-referenced to {name}, {place}, {date} (confirmed)."

# ----------------------------------------------------------------------
# 2. RENDERING HELPERS
# ----------------------------------------------------------------------

def salient_grounded_events(case):
    return [e for e in case["events"] if e["salience"] >= 2 and e["grounded_by"]]

def focal_specific_attrs(case):
    f = case["entities"][case["focal"]]
    return [a for a in f["attributes"]
            if a["specificity"] == "specific" and isinstance(a["grounded_by"], list) and a["grounded_by"]]

def date_phrase(date, register, hedge):
    v, c = date["value"], date["certainty"]
    if register == "testimony":
        if c == "attested" or not hedge:
            return f"in {v}"
        if c == "probable":
            return f"probably in {v}"
        return f"at some point, perhaps {v}"
    else:  # finding_aid
        if c == "attested" or not hedge:
            return v
        if c == "probable":
            return f"{v} (probable)"
        return f"{v} (undated; unconfirmed)"

def attribution(src, register):
    if register == "testimony":
        return {"registry": "the registry records",
                "testimony": "according to family testimony",
                "letter": "a surviving letter notes",
                "catalogue_entry": "the catalogue records",
                "caption": "a photo caption records"}.get(src["kind"], "the record notes")
    return f"({src['kind']}, {src['archive_ref']})"

def terse(summary, focal_name):
    s = summary
    # strip a leading focal-name subject for finding-aid terseness
    for prefix in (focal_name + " ", focal_name.replace("the ", "") + " "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s[0].upper() + s[1:] if s else s

# ----------------------------------------------------------------------
# 3. RENDERERS  (text only; trace is computed separately)
# ----------------------------------------------------------------------

def render_flattened(case, register):
    f = case["entities"][case["focal"]]
    name, typ = f["name"], f["type"]
    if register == "testimony":
        if typ == "person":
            return f"{name} was an ordinary person whose life, like so many others, was caught up in the events of the period."
        return f"{name} was one of many such places and institutions in the city during those difficult years."
    else:
        label = name if name[0].isupper() else name.capitalize()
        return f"{label}. Of local significance; affected by wartime events. See file."

def render_good_or_fabricated(case, register, fabricate):
    f = case["entities"][case["focal"]]
    name = f["name"]
    hedge = not fabricate
    attrs = focal_specific_attrs(case)
    events = salient_grounded_events(case)

    if register == "testimony":
        out = []
        if attrs:
            a = attrs[0]
            src = case["sources"][a["grounded_by"][0]]
            keyphrase = {"occupation": f"was, {attribution(src, 'testimony')}, a {a['value']} by trade",
                         "role": f"served as {a['value']}, {attribution(src, 'testimony')}",
                         "address": f"stood at {a['value']}, {attribution(src, 'testimony')}",
                         "built": f"dated from {a['value']}, {attribution(src, 'testimony')}",
                         "school": f"attended {a['value']}, {attribution(src, 'testimony')}",
                         "purpose": f"existed for {a['value']}, {attribution(src, 'testimony')}",
                         "inventory_no": f"is held under inventory {a['value']}, {attribution(src, 'testimony')}",
                         }.get(a["key"], f"is recorded with {a['key']} {a['value']}, {attribution(src, 'testimony')}")
            out.append(f"{name} {keyphrase}.")
        for e in events:
            src = case["sources"][e["grounded_by"][0]]
            dp = date_phrase(e["date"], "testimony", hedge)
            if fabricate:
                out.append(f"{e['summary']} {dp}.")          # confident, no attribution
            else:
                out.append(f"{e['summary']} {dp}, {attribution(src, 'testimony')}.")
        if fabricate:
            out += [fb["text"] for fb in case.get("fabrications", [])]
            out.append(_false_clause(case["case_id"], "testimony"))
        else:
            # restraint marker for cases with a high-sensitivity event
            if any(e["sensitivity"] == "high" for e in case["events"]):
                out.append("Beyond that, the record falls silent.")
        return " ".join(out)

    # finding_aid
    parts = []
    # head line
    if f["type"] == "person":
        by = next((a["value"] for a in attrs if a["key"] == "birth_year"), None)
        occ = next((a["value"] for a in attrs if a["key"] in ("occupation", "role")), None)
        surname = name.split()[-1]
        given = " ".join(name.split()[:-1])
        head = f"{surname}, {given}"
        if by:
            head += f" (b. {by})"
        if occ:
            head += f", {occ}"
        parts.append(head + ".")
    else:
        addr = next((a["value"] for a in attrs if a["key"] in ("address", "built", "inventory_no", "purpose")), None)
        head = name if name[0].isupper() else name.capitalize()
        if addr:
            head += f", {addr}"
        parts.append(head + ".")
    for e in events:
        src = case["sources"][e["grounded_by"][0]]
        dp = date_phrase(e["date"], "finding_aid", hedge)
        phrase = terse(e["summary"], name)
        if fabricate:
            parts.append(f"{phrase}, {dp} (confirmed).")    # false certainty, no real provenance
        else:
            parts.append(f"{phrase}, {dp} ({src['kind']}, {src['archive_ref']}).")
    if fabricate:
        parts += [fb["text"] + " (confirmed)." if not fb["text"].rstrip().endswith(".")
                  else fb["text"] + " (confirmed)" for fb in case.get("fabrications", [])]
        parts.append(_false_clause(case["case_id"], "finding_aid"))
    else:
        if f["type"] == "person" and any(e["sensitivity"] == "high" for e in case["events"]):
            parts.append("Further fate undocumented.")
    return " ".join(parts)

def render(case, register, profile):
    if profile == "flattened":
        return render_flattened(case, register)
    return render_good_or_fabricated(case, register, fabricate=(profile == "fabricated"))

# ----------------------------------------------------------------------
# 4. TRACE  +  ORACLE  (ground-truth judge; register-invariant)
# ----------------------------------------------------------------------

def compute_trace(case, profile):
    events = salient_grounded_events(case)
    total = max(1, len(events))
    n_real_claims = len(events) + len(focal_specific_attrs(case))
    n_fab = len(case.get("fabrications", []))
    sens = any(fb.get("sensational") for fb in case.get("fabrications", []))

    if profile == "good":
        return dict(coverage=1.0, specific=1.0, attributed=1.0, calibration=1.0,
                    fabrications=0, sensational=False, total_salient_grounded=total)
    if profile == "flattened":
        return dict(coverage=0.0, specific=0.0, attributed=0.0, calibration=0.5,
                    fabrications=0, sensational=False, total_salient_grounded=total)
    # fabricated
    attributed = round(n_real_claims / (n_real_claims + n_fab), 3) if (n_real_claims + n_fab) else 0.0
    return dict(coverage=1.0, specific=1.0, attributed=attributed, calibration=0.0,
                fabrications=n_fab, sensational=sens, total_salient_grounded=total)

# weights are deliberately provisional; a workshop with historians sets the real ones.
W = dict(coverage=0.30, specific=0.20, attributed=0.20, calibration=0.15,
         fabrication_penalty=0.60, sensational_penalty=0.25)

def oracle_score(trace):
    s = (W["coverage"] * trace["coverage"]
         + W["specific"] * trace["specific"]
         + W["attributed"] * trace["attributed"]
         + W["calibration"] * trace["calibration"]
         - W["fabrication_penalty"] * trace["fabrications"]
         - W["sensational_penalty"] * (1.0 if trace["sensational"] else 0.0))
    return round(s, 3)

# ----------------------------------------------------------------------
# 5. LINGUISTIC JUDGE  (surface only; blind to grounding)
# ----------------------------------------------------------------------

def linguistic_score(text):
    """Proper-noun + number density of the NARRATIVE content. Goes UP with
    fabricated specifics. Parenthetical provenance like '(registry, WL/SYN/118)'
    is stripped first: a surface specificity metric measures the story's named
    detail, not citation IDs. (Note: if you DON'T strip them, citation-dense
    'good' renderings score high too -- a separate lesson in how brittle and
    game-dependent surface metrics are.)"""
    text = re.sub(r"\([^)]*\)", " ", text)
    sentences = re.split(r"[.;:]\s+", text)
    propers, nums = 0, 0
    for s in sentences:
        toks = s.split()
        for i, t in enumerate(toks):
            tt = t.strip(",.;:()[]'\"")
            if not tt:
                continue
            if i > 0 and re.match(r"^[A-Z]", tt):       # capitalised, not sentence-initial
                propers += 1
            if re.search(r"\d", tt):                    # contains a digit (date / number / ref)
                nums += 1
    ntok = max(1, len(text.split()))
    density = (propers + nums) / math.sqrt(ntok)
    return round(density, 3), propers, nums

# ----------------------------------------------------------------------
# 6. DRIVE: render grid, score, write outputs, run the hypothesis test
# ----------------------------------------------------------------------

INSTRUCTION = ("Write a short, archivally grounded micro-narrative about the unit below, "
               "in the style of {register}. Include specific, sourced detail; acknowledge "
               "uncertainty; avoid sensationalism.")

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    renderings, pref_pairs = [], []
    agg = {p: {"oracle": [], "ling": []} for p in PROFILES}

    for case in CASES:
        for register in REGISTERS:
            by_profile = {}
            for profile in PROFILES:
                text = render(case, register, profile)
                trace = compute_trace(case, profile)
                orc = oracle_score(trace)
                ling, propers, nums = linguistic_score(text)
                by_profile[profile] = dict(text=text, oracle=orc, ling=ling)
                renderings.append(dict(case_id=case["case_id"], focal=case["focal"],
                                       focal_type=case["entities"][case["focal"]]["type"],
                                       register=register, profile=profile, text=text,
                                       oracle_score=orc, linguistic_score=ling,
                                       linguistic_propers=propers, linguistic_numbers=nums,
                                       trace=trace))
                agg[profile]["oracle"].append(orc)
                agg[profile]["ling"].append(ling)
            # preference pairs: good is chosen over each bad sibling (same case+register)
            prompt = INSTRUCTION.format(register=register.replace("_", "-"))
            prompt += f"\nUnit: {case['entities'][case['focal']]['name']} (case {case['case_id']})."
            for bad in ("flattened", "fabricated"):
                pref_pairs.append(dict(prompt=prompt, chosen=by_profile["good"]["text"],
                                       rejected=by_profile[bad]["text"],
                                       case_id=case["case_id"], register=register,
                                       rejected_kind=bad))

    # ---- write artefacts ----
    (OUTDIR / "cases.json").write_text(json.dumps(CASES, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTDIR / "renderings.jsonl").open("w", encoding="utf-8") as fh:
        for r in renderings:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUTDIR / "preference_pairs.jsonl").open("w", encoding="utf-8") as fh:
        for p in pref_pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ---- hypothesis test (spec sec.4) ----
    def mean(xs):
        return round(sum(xs) / len(xs), 3)
    means = {p: dict(oracle=mean(agg[p]["oracle"]), ling=mean(agg[p]["ling"])) for p in PROFILES}

    oracle_ok = means["good"]["oracle"] > means["fabricated"]["oracle"] and \
                means["good"]["oracle"] > means["flattened"]["oracle"]
    ling_fooled = means["fabricated"]["ling"] >= means["good"]["ling"]

    # per-case check: does ORACLE put good top and LINGUISTIC put fabricated >= good?
    per_case_ling_fooled = 0
    for case in CASES:
        for register in REGISTERS:
            g = next(r for r in renderings if r["case_id"] == case["case_id"]
                     and r["register"] == register and r["profile"] == "good")
            fb = next(r for r in renderings if r["case_id"] == case["case_id"]
                      and r["register"] == register and r["profile"] == "fabricated")
            if fb["linguistic_score"] >= g["linguistic_score"]:
                per_case_ling_fooled += 1
    n_grid = len(CASES) * len(REGISTERS)

    report = []
    report.append("# Synthetic corpus — generation report\n")
    report.append(f"Cases: **{len(CASES)}**  |  grid cells: **{len(renderings)}** "
                  f"({len(CASES)}×{len(REGISTERS)}×{len(PROFILES)})  |  preference pairs: **{len(pref_pairs)}**\n")
    report.append("## Mean scores by detail profile\n")
    report.append("| profile | ORACLE (ground-truth) | LINGUISTIC (surface) |")
    report.append("|---|---|---|")
    for p in PROFILES:
        report.append(f"| {p} | {means[p]['oracle']} | {means[p]['ling']} |")
    report.append("")
    report.append("## Hypothesis test (spec §4)\n")
    report.append(f"- ORACLE ranks **good above both flattened and fabricated**: "
                  f"**{'PASS' if oracle_ok else 'FAIL'}** "
                  f"(good={means['good']['oracle']}, flattened={means['flattened']['oracle']}, "
                  f"fabricated={means['fabricated']['oracle']}).")
    report.append(f"- LINGUISTIC **fails to penalise fabrication** (fabricated ≥ good on the surface metric): "
                  f"**{'CONFIRMED' if ling_fooled else 'not observed'}** "
                  f"(good={means['good']['ling']}, fabricated={means['fabricated']['ling']}).")
    report.append(f"- Per grid cell, fabricated ≥ good on LINGUISTIC in **{per_case_ling_fooled}/{n_grid}** cells.")
    report.append("")
    report.append("**Reading:** the surface metric's ranking is driven by token density, which tracks "
                  "neither grounding nor truth. It rates fabricated detail at least as high as grounded "
                  "detail in the majority of cells and higher on average, while the grounded oracle "
                  "separates good from fabricated by ~1.1 points. The surface metric is not so much *fooled* "
                  "as *blind*. This is the existence proof that 'good detail' cannot be read off the text "
                  "surface — the empirical seed of the human-vs-linguistic comparison.\n")
    report.append("## One worked cell (case 001, testimony)\n")
    for p in PROFILES:
        r = next(r for r in renderings if r["case_id"] == "synth_case_001"
                 and r["register"] == "testimony" and r["profile"] == p)
        report.append(f"**{p}** — oracle {r['oracle_score']}, linguistic {r['linguistic_score']}\n")
        report.append(f"> {r['text']}\n")
    (OUTDIR / "report.md").write_text("\n".join(report), encoding="utf-8")

    # ---- console summary ----
    print(f"Wrote {len(renderings)} renderings, {len(pref_pairs)} preference pairs to {OUTDIR}")
    print("\nMean scores by profile:")
    print(f"{'profile':<12}{'ORACLE':>10}{'LINGUISTIC':>14}")
    for p in PROFILES:
        print(f"{p:<12}{means[p]['oracle']:>10}{means[p]['ling']:>14}")
    print(f"\nORACLE good > flattened & fabricated : {'PASS' if oracle_ok else 'FAIL'}")
    print(f"LINGUISTIC fooled by fabrication      : {'CONFIRMED' if ling_fooled else 'no'} "
          f"({per_case_ling_fooled}/{n_grid} cells)")


if __name__ == "__main__":
    main()
