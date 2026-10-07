"""Tokenizer shared by BM25 and the keyword-coverage gate."""

from __future__ import annotations

import re

_STOPWORD_TEXT = """
a an and are as at be by can do does for from how i if in into is it its me my of on or
should so than that the their them then there these they this to was what when where
which who why will with would you your about tell explain mean means
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_TOKEN = re.compile(r"[a-z0-9]+(?:\(k\))?")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords; naive plural folding ('funds' -> 'fund')."""
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens
