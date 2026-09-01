"""Tokenization and identifier extraction.

Ops text is full of tokens that must match *exactly* -- `ERR_5012`,
`db-prod-03`, `ORA-00060`, `HTTP 503`. Embeddings smear these together
(`db-prod-03` and `db-prod-07` are near-identical vectors), so they get their
own lane in the ranker instead.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")

STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those is are was were be been being
    to of in on at by for with from as it its we our you your they their he she i
    not no do does did done have has had can could should would will may might must
    about into over under after before during while when where which who whom what how why
    """.split()
)

# Patterns for tokens where an exact match is strong evidence and a fuzzy match
# is worthless. Order matters only for readability; all patterns are applied.
IDENTIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # ERR_5012, E_CONN_REFUSED, PG_LOCK_TIMEOUT
    ("error_code", re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:_[A-Z0-9]+){1,}\b")),
    # ORA-00060, CVE-2024-1234, INC-1842
    ("dashed_code", re.compile(r"\b[A-Z]{2,6}-\d{2,}\b")),
    # HTTP status codes, but only when introduced as such
    ("http_status", re.compile(r"\b(?:HTTP|status(?:\s+code)?)\s*[:=]?\s*([1-5]\d{2})\b", re.I)),
    # db-prod-03, kafka-broker-7, i-0abc123def, api-gw-eu-2
    ("host", re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){1,}\d*\b")),
    # java.sql.SQLTransientConnectionException, psycopg2.OperationalError
    ("exception", re.compile(r"\b(?:[a-z][a-z0-9]*\.){1,}[A-Z][A-Za-z0-9]*(?:Error|Exception)\b")),
    # bare exception class names
    ("exception_bare", re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception|Timeout)\b")),
)

# Host-shaped strings that are really just hyphenated English.
_HOST_NOISE = frozenset(
    {
        "read-only", "long-running", "well-known", "roll-back", "fail-over",
        "post-mortem", "on-call", "end-to-end", "back-off", "time-out",
        "high-cpu", "root-cause", "step-by-step", "up-to-date", "write-ahead",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed, keeping dotted/underscored
    identifiers intact (`psycopg2.operationalerror` stays one token)."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def expand_token(token: str) -> list[str]:
    """A token plus its sub-parts, so `connection_pool_exhausted` also matches
    documents that say `connection pool exhausted`."""
    parts = [p for p in re.split(r"[._-]", token) if len(p) > 1 and p not in STOPWORDS]
    return [token, *parts] if len(parts) > 1 else [token]


def extract_identifiers(text: str) -> list[str]:
    """Pull the exact-match tokens out of free text, normalized to lowercase.

    Returns them in first-seen order; duplicates removed.
    """
    found: dict[str, None] = {}
    for kind, pattern in IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1) if match.lastindex else match.group(0)
            token = raw.strip().lower()
            if kind == "http_status":
                token = f"http_{token}"
            elif kind == "host":
                if token in _HOST_NOISE or not any(c.isdigit() for c in token):
                    # Require a digit somewhere: `db-prod-03` is a host,
                    # `read-only` is not.
                    continue
            if len(token) < 3:
                continue
            found.setdefault(token, None)
    return list(found)


def keyword_overlap(query_keywords: list[str], doc_keywords: list[str]) -> tuple[float, list[str]]:
    """Fraction of the query's exact-match tokens present in the document.

    Scored against the *query* rather than the union, because an alert citing
    two error codes that both appear in an incident is a strong match even if
    that incident mentions ten other codes.
    """
    if not query_keywords:
        return 0.0, []
    doc = {k.lower() for k in doc_keywords}
    matched = [k for k in query_keywords if k.lower() in doc]
    return len(matched) / len(query_keywords), matched
