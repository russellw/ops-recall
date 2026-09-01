from ops_recall.retrieval.sparse import BM25Encoder


CORPUS = [
    "connection pool exhausted on db-prod-03 with pending requests",
    "kafka consumer lag rising on orders-events partition 7",
    "connection refused by the api gateway during a deploy",
    "tls certificate expired at the edge gateway",
]


def _score(encoder: BM25Encoder, query: str, document: str) -> float:
    q = encoder.encode_query(query)
    d = encoder.encode_document(document)
    return sum(weight * d.get(index, 0.0) for index, weight in q.items())


def test_relevant_document_outscores_irrelevant_one():
    encoder = BM25Encoder().fit(CORPUS)
    relevant = _score(encoder, "connection pool exhausted", CORPUS[0])
    irrelevant = _score(encoder, "connection pool exhausted", CORPUS[1])
    assert relevant > irrelevant
    assert irrelevant == 0.0


def test_rare_terms_weigh_more_than_common_ones():
    encoder = BM25Encoder().fit(CORPUS)
    # "connection" appears in two documents, "exhausted" in one.
    assert encoder.idf("exhausted") > encoder.idf("connection")


def test_corpus_terms_only_drops_unmatchable_query_terms():
    encoder = BM25Encoder().fit(CORPUS)
    everything = encoder.encode_query("connection pool quantum flux capacitor")
    grounded = encoder.encode_query(
        "connection pool quantum flux capacitor", corpus_terms_only=True
    )
    assert len(grounded) < len(everything)
    # The unmatchable terms carry the highest idf, so they would otherwise
    # dominate the normalization ceiling.
    assert sum(grounded.values()) < sum(everything.values())


def test_extra_terms_are_added_to_the_query():
    encoder = BM25Encoder().fit(CORPUS)
    plain = encoder.encode_query("pool")
    with_extra = encoder.encode_query("pool", extra_terms=["db-prod-03"])
    assert len(with_extra) > len(plain)


def test_round_trip_serialization():
    encoder = BM25Encoder().fit(CORPUS)
    restored = BM25Encoder.from_dict(encoder.to_dict())
    assert restored.doc_count == encoder.doc_count
    assert restored.avg_len == encoder.avg_len
    assert _score(restored, "connection pool", CORPUS[0]) == _score(
        encoder, "connection pool", CORPUS[0]
    )


def test_empty_document_encodes_to_nothing():
    encoder = BM25Encoder().fit(CORPUS)
    assert encoder.encode_document("") == {}
