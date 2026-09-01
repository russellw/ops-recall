from ops_recall.retrieval.text import (
    expand_token,
    extract_identifiers,
    keyword_overlap,
    tokenize,
)


def test_extracts_error_codes_and_hosts():
    text = (
        "HikariPool-1 - Connection is not available on db-prod-03: ERR_5012, "
        "ORA-00060 deadlock, HTTP 503 from api-gw-eu-2, "
        "java.sql.SQLTransientConnectionException"
    )
    found = extract_identifiers(text)
    assert "err_5012" in found
    assert "ora-00060" in found
    assert "http_503" in found
    assert "db-prod-03" in found
    assert "api-gw-eu-2" in found
    assert "java.sql.sqltransientconnectionexception" in found


def test_hyphenated_english_is_not_a_host():
    found = extract_identifiers("a long-running read-only well-known transaction")
    assert found == []


def test_stopwords_and_short_tokens_are_dropped():
    assert tokenize("the pool is at a max") == ["pool", "max"]


def test_expand_token_keeps_whole_and_parts():
    assert expand_token("connection_pool_exhausted") == [
        "connection_pool_exhausted",
        "connection",
        "pool",
        "exhausted",
    ]
    assert expand_token("deadlock") == ["deadlock"]


def test_keyword_overlap_is_scored_against_the_query():
    score, matched = keyword_overlap(
        ["err_5012", "db-prod-03"], ["err_5012", "db-prod-03", "orders_db", "http_503"]
    )
    assert score == 1.0
    assert matched == ["err_5012", "db-prod-03"]

    score, matched = keyword_overlap(["err_5012", "missing"], ["err_5012"])
    assert score == 0.5
    assert matched == ["err_5012"]


def test_keyword_overlap_without_query_keywords_is_zero():
    assert keyword_overlap([], ["err_5012"]) == (0.0, [])
