"""
ingest — corpus-agnostic ingestion for the good-detail pipeline.
================================================================
ONE canonical `Record` is the contract: every source format (EHRI TEI/XML, the
IOB/relation extraction, the synthetic generator, …) is read by a small adapter
that emits `Record`s, and everything downstream (generation, the rewards,
scoring) consumes `Record`s without knowing which corpus it came from. That is
what makes "the workflow generalises across corpora" true rather than claimed.

Layout:
    data/<CORPUS>/corpus.json   — manifest: which adapter + path per source
    data/<CORPUS>/...           — the raw files
    ingest/<format>.py          — one adapter per raw format (@adapter-decorated)

Usage:
    from ingest import load_corpus, record_block
    recs = load_corpus("EHRI", source="iob", limit=3)
    block = record_block(recs[0])        # prompt context + structured grounding

CLI:  python -m ingest EHRI --source xml --limit 1     (inspect)
      python -m ingest EHRI --source iob --list        (count)
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class Record:
    """The canonical unit every adapter emits and every consumer reads."""
    id: str
    corpus: str = ""
    register: str = "testimony"          # testimony / finding_aid / …
    unit: str = ""                       # focal subject (narrator, place, …)
    title: str = ""
    entities: list = field(default_factory=list)   # [{"name":…, "type":…}]
    relations: list = field(default_factory=list)  # [{"type":…, "head":…, "tail":…}]
    source_text: str = ""                # the NLI faithfulness premise
    subjects: list = field(default_factory=list)
    provenance: str = ""


# --- adapter registry -----------------------------------------------------
_ADAPTERS = {}


def adapter(name):
    """Register a source-format reader: fn(base_path: Path, spec: dict) -> iter[Record]."""
    def deco(fn):
        _ADAPTERS[name] = fn
        return fn
    return deco


def available_adapters():
    return sorted(_ADAPTERS)


# --- helpers --------------------------------------------------------------
def _uniq(seq):
    seen, out = set(), []
    for s in seq:
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out


def record_block(rec, max_per_type=40, max_relations=20, max_subjects=20, with_relations=True):
    """Canonical prompt context + structured grounding (generalises the synthetic
    _record_block and the EHRI header block). Entities are the ATTESTED set: a
    name in a generated narrative that is absent here (and from source_text) is a
    candidate fabrication. Kept as inline prose, not a bulleted dump.
    `with_relations=False` drops the relations line — the ablation that isolates
    what the extracted relations contribute beyond the entity list."""
    lines = [f"Unit: {rec.unit or rec.title}"]
    if rec.title and rec.title != rec.unit:
        lines.append(f"Title: {rec.title}")
    if rec.provenance:
        lines.append(f"Source: {rec.provenance}")
    by_type = defaultdict(list)
    for e in rec.entities:
        by_type[e.get("type", "ENTITY")].append(e["name"])
    for t, names in by_type.items():
        lines.append(f"{t.title()}: " + "; ".join(_uniq(names)[:max_per_type]))
    if with_relations and rec.relations:
        rels = [f"{r['head']} [{r['type']}] {r['tail']}" for r in rec.relations[:max_relations]]
        lines.append("Relations: " + " | ".join(rels))
    if rec.subjects:
        lines.append("Subjects: " + "; ".join(_uniq(rec.subjects)[:max_subjects]))
    return "\n".join(lines)


def load_corpus(name, source=None, limit=None, data_dir=DATA):
    """Read data/<name>/corpus.json, dispatch to the chosen source's adapter,
    return a list of Records (capped at `limit`)."""
    manifest = json.loads((Path(data_dir) / name / "corpus.json").read_text(encoding="utf-8"))
    src = source or manifest.get("default_source")
    if src not in manifest.get("sources", {}):
        raise KeyError(f"corpus {name!r} has no source {src!r}; have {list(manifest.get('sources', {}))}")
    spec = manifest["sources"][src]
    fn = _ADAPTERS.get(spec["adapter"])
    if fn is None:
        raise KeyError(f"unknown adapter {spec['adapter']!r}; registered: {available_adapters()}")
    out = []
    for r in fn(Path(data_dir) / name, spec):
        out.append(r)
        if limit and len(out) >= limit:
            break
    return out


# Import the bundled adapters so they self-register on `import ingest`.
from . import tei_ehri, iob_jsonl, synthetic  # noqa: E402,F401
