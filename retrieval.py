#!/usr/bin/env python3
"""
retrieval.py
================================================================
Optional EMBEDDING retrieval for the source-passage premise (P2 win). The
overnight run showed grounding hinges on focused source passages, and lexical
token-overlap is crude. `embed_retrieve` ranks sentences by sentence-embedding
cosine similarity to the query. Lazy-loaded and cached; falls back to None if
sentence-transformers isn't installed (callers then use lexical).

Selected via config.RETRIEVAL_METHOD = "embed"; model = config.EMBED_MODEL.
Install: pip install sentence-transformers
"""

from config import EMBED_MODEL

_MODEL = {"m": None, "ok": True}


def _model():
    if _MODEL["m"] is None and _MODEL["ok"]:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL["m"] = SentenceTransformer(EMBED_MODEL)
        except Exception:
            _MODEL["ok"] = False        # not installed / failed -> caller falls back
    return _MODEL["m"]


def embed_retrieve(sentences, query, k):
    """Top-k sentences by cosine similarity to query. Returns None if embeddings
    are unavailable (caller should fall back to lexical)."""
    m = _model()
    if m is None or not sentences:
        return None
    import numpy as np
    emb = m.encode([query] + sentences, normalize_embeddings=True)
    q, S = emb[0], emb[1:]
    sims = S @ q
    order = np.argsort(-sims)[:k]
    return [sentences[i] for i in order]
