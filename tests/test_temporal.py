import math

from ops_recall.retrieval.temporal import decay_weight, describe_recency


def test_half_life_halves_the_decayable_part():
    assert decay_weight(0, 180) == 1.0
    assert math.isclose(decay_weight(180, 180), 0.5, abs_tol=1e-9)
    assert math.isclose(decay_weight(360, 180), 0.25, abs_tol=1e-9)


def test_floor_keeps_ancient_matches_reachable():
    weight = decay_weight(3650, half_life_days=180, floor=0.45)
    assert weight >= 0.45
    assert weight < 0.46


def test_decay_is_monotonic():
    weights = [decay_weight(days, 365, 0.6) for days in range(0, 2000, 100)]
    assert weights == sorted(weights, reverse=True)


def test_future_and_zero_ages_are_not_boosted():
    assert decay_weight(-5, 365) == 1.0
    assert decay_weight(0, 365) == 1.0


def test_describe_recency_reads_naturally():
    assert describe_recency(0.5) == "today"
    assert describe_recency(9) == "9 days ago"
    assert describe_recency(30) == "4 weeks ago"
    assert describe_recency(200) == "6 months ago"
    assert describe_recency(1095) == "3.0 years ago"
