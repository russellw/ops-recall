"""Time decay.

Ops knowledge rots. A fix from last quarter usually still applies; a fix from
three years ago was written for an architecture that may no longer exist. But
decay must never be absolute -- a three-year-old incident that matches an alert
exactly is still the single most useful thing in the corpus. So the multiplier
is exponential with a configurable half-life and a floor.
"""

from __future__ import annotations

import math
from datetime import datetime

from ops_recall.models import utcnow


def decay_weight(age_days: float, half_life_days: float, floor: float = 0.0) -> float:
    """Exponential decay in [floor, 1.0].

    >>> round(decay_weight(0, 180), 3)
    1.0
    >>> round(decay_weight(180, 180), 3)
    0.5
    """
    if age_days <= 0:
        return 1.0
    if half_life_days <= 0:
        return 1.0
    raw = math.pow(0.5, age_days / half_life_days)
    return floor + (1.0 - floor) * raw


def age_in_days(when: datetime, now: datetime | None = None) -> float:
    return max(0.0, ((now or utcnow()) - when).total_seconds() / 86400.0)


def describe_recency(age_days: float) -> str:
    """Human phrasing for the age of an incident, used in generated prose."""
    if age_days < 1:
        return "today"
    if age_days < 14:
        return f"{int(age_days)} days ago"
    if age_days < 60:
        return f"{int(age_days / 7)} weeks ago"
    if age_days < 730:
        return f"{int(age_days / 30)} months ago"
    return f"{age_days / 365:.1f} years ago"
