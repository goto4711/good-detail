"""CLI: inspect a corpus through the ingest layer.
  python -m ingest EHRI --source xml --limit 1
  python -m ingest EHRI --source iob --list
"""

import argparse

from ingest import load_corpus, record_block


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--source", default=None, help="source key in corpus.json (else default)")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--list", action="store_true", help="count records only")
    ap.add_argument("--excerpt", type=int, default=400)
    a = ap.parse_args()

    recs = load_corpus(a.corpus, a.source, limit=None if a.list else a.limit)
    if a.list:
        print(f"{a.corpus}/{a.source or 'default'}: {len(recs)} records")
        return
    for r in recs:
        print(f"=== {r.id}  ({r.corpus}/{r.register})  unit: {r.unit} ===")
        print(record_block(r))
        print(f"\n[source_text {len(r.source_text)} chars] "
              f"{r.source_text[:a.excerpt].strip()} …\n")


if __name__ == "__main__":
    main()
