"""Small text helpers shared by the index and the context builder.

Matching is intentionally simple and explainable: exact matches beat prefixes,
prefixes beat substrings, and a shared prefix of at least five characters
connects forms such as ``authenticate`` and ``authentication``.  No embedding
model is involved, so every hit can be traced back to a token and a file.
"""

from __future__ import annotations

import re

#: Words that carry no retrieval signal in a coding question.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "anything",
        "about",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "code",
        "could",
        "describe",
        "do",
        "does",
        "explain",
        "file",
        "files",
        "find",
        "for",
        "from",
        "how",
        "i",
        "implementation",
        "important",
        "in",
        "into",
        "is",
        "it",
        "main",
        "me",
        "modify",
        "my",
        "of",
        "on",
        "or",
        "please",
        "project",
        "show",
        "tell",
        "that",
        "the",
        "their",
        "this",
        "to",
        "understand",
        "use",
        "used",
        "using",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "work",
        "works",
    }
)

_SPLIT = re.compile(r"[^A-Za-z0-9_]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_MIN_TOKEN = 2
_MIN_PREFIX = 4


def split_identifier(text: str) -> tuple[str, ...]:
    """Split camelCase, snake_case, kebab-case and dotted names."""
    pieces: list[str] = []
    for chunk in _SPLIT.split(text):
        if not chunk:
            continue
        pieces.extend(_CAMEL.split(chunk))
    return tuple(piece.lower() for piece in pieces if piece)


def tokenize(text: str, *, keep_stopwords: bool = False) -> tuple[str, ...]:
    """Return unique, lower-cased search tokens in their original order."""
    seen: set[str] = set()
    tokens: list[str] = []
    for chunk in split_identifier(text):
        if len(chunk) < _MIN_TOKEN:
            continue
        if not keep_stopwords and chunk in STOPWORDS:
            continue
        if chunk in seen:
            continue
        seen.add(chunk)
        tokens.append(chunk)
    return tuple(tokens)


def match_score(query_token: str, text: str) -> float:
    """Score one query token against a piece of text (0–100)."""
    if not query_token or not text:
        return 0.0
    candidate = text.casefold()
    if query_token == candidate:
        return 100.0
    if candidate.startswith(query_token):
        return 70.0
    if query_token in candidate:
        return 45.0
    token_length = len(query_token)
    if token_length >= 3 and candidate in query_token:
        return 35.0
    shared = _shared_prefix(query_token, candidate)
    if shared >= _MIN_PREFIX and min(token_length, len(candidate)) >= _MIN_PREFIX:
        return 30.0
    return 0.0


def score_text(query_tokens: tuple[str, ...], text: str) -> float:
    """Best token match against a string, without double-counting."""
    best = 0.0
    for token in query_tokens:
        best = max(best, match_score(token, text))
    return best


def score_identifiers(query_tokens: tuple[str, ...], text: str) -> float:
    """Score against the whole string and each identifier it is made of."""
    if not query_tokens or not text:
        return 0.0
    best = score_text(query_tokens, text)
    for piece in split_identifier(text):
        best = max(best, score_text(query_tokens, piece))
    return best


def _shared_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index
