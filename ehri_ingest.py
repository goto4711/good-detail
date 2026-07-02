#!/usr/bin/env python3
"""DEPRECATED — the EHRI TEI parsing moved into the `ingest` package
(ingest/tei_ehri.py), behind the canonical Record + load_corpus(). Safe to delete.

    python -m ingest EHRI --source xml --limit 1
    from ingest import load_corpus, record_block
"""
import sys

sys.exit("ehri_ingest.py is deprecated -> use the `ingest` package:  "
         "python -m ingest EHRI --source xml --limit 1")
