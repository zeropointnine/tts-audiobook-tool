"""Inclusive random-seed limits, independent of model and backend dependencies."""
from __future__ import annotations


def resolve_max_random_seed(*limits: int) -> int:
    """Intersect optional inclusive caps; -1 leaves the range unrestricted."""
    for limit in limits:
        if type(limit) is not int or limit < -1:
            raise ValueError("max_random_seed must be -1 or a nonnegative integer")
    return min((limit for limit in limits if limit != -1), default=-1)


def get_random_seed_max(default_max: int, max_random_seed: int = -1, *, min_seed: int = 0) -> int:
    """Bound a model's native inclusive maximum without widening its range.

    Callers retain their original RNG, lower bound, and exclusive/inclusive
    endpoint convention. This helper is used only when drawing a random seed.
    """
    maximum = resolve_max_random_seed(default_max, max_random_seed)
    if maximum < min_seed:
        raise ValueError(f"max_random_seed must be at least {min_seed} for this model's random seed range")
    return maximum
