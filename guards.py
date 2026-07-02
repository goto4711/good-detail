#!/usr/bin/env python3
"""
guards.py
================================================================
Shared reward-hacking guards used at training time and selection time.
Thresholds and weights live in config.py.
"""

import re

from config import ANTICOPY_N, ANTICOPY_TOL, TARGET_WORDS, W_COPY, W_FORMAT, W_LEN

_SCAFFOLD = re.compile(
    r"(?im)^\s*(focal|entities|events|sources|notes|record)\s*:"
    r"|\bsrc\s+s?\d"
    r"|\[\s*\d{3,4}\s*,"
    r"|^\s*[-*•]\s+",
    re.M,
)


def _ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def length_guard_penalty(text, target_words=None, weight=None):
    target_words = TARGET_WORDS if target_words is None else target_words
    weight = W_LEN if weight is None else weight
    n_words = len(text.split())
    return -weight * max(0.0, (n_words - target_words) / target_words)


def anti_copy_overlap(text, source_text, n=None):
    n = ANTICOPY_N if n is None else n
    comp = _ngrams(text.lower().split(), n)
    if not comp:
        return 0.0
    rec = _ngrams(source_text.lower().split(), n)
    return len(comp & rec) / len(comp)


def anti_copy_penalty(text, source_text, tolerance=None, weight=None, n=None):
    tolerance = ANTICOPY_TOL if tolerance is None else tolerance
    weight = W_COPY if weight is None else weight
    overlap = anti_copy_overlap(text, source_text, n=n)
    return -weight * max(0.0, overlap - tolerance)


def anti_copy_exceeds_tolerance(text, source_text, tolerance=None, n=None):
    tolerance = ANTICOPY_TOL if tolerance is None else tolerance
    return anti_copy_overlap(text, source_text, n=n) > tolerance


def format_guard_hits(text):
    hits = len(_SCAFFOLD.findall(text))
    extra_newlines = max(0, text.count("\n") - 1)
    return hits + 0.5 * extra_newlines


def format_guard_penalty(text, weight=None):
    weight = W_FORMAT if weight is None else weight
    return -weight * format_guard_hits(text)


def format_guard_violation(text):
    return format_guard_hits(text) > 0
