#!/usr/bin/env python3
"""DEPRECATED — superseded by realdata_generate.py (corpus-agnostic, via the
`ingest` layer). Safe to delete this file.

    python realdata_generate.py --corpus EHRI --source xml|iob --adapter … --limit 3
"""
import sys

sys.exit("ehri_generate.py is deprecated -> use:  "
         "python realdata_generate.py --corpus EHRI --source xml|iob …")
