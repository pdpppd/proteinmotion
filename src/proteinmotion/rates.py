"""Rate functions accepted by every play() call and animation.

A rate function (also called an easing) maps clip progress t in [0, 1] to animation
progress in [0, 1]. ``play(rate_func=...)``, ``play(easing=...)`` and the ``rate_func`` or
``easing`` argument of an animation take one of the functions below, its name as a string,
or any callable with that contract. resolve() turns a name into its function, and evaluate()
calls one and checks the result.
"""

import math


def resolve(easing):
    """Accept the same named or callable easing everywhere.

    A name such as ``"smooth"`` returns that function; a callable is returned unchanged.
    """
    if isinstance(easing, str):
        choices = {
            "linear": linear,
            "smooth": smooth,
            "ease_in_out_sine": ease_in_out_sine,
            "there_and_back": there_and_back,
        }
        if easing not in choices:
            raise ValueError(f"Unknown easing {easing!r}; choose from {', '.join(choices)}")
        return choices[easing]
    if not callable(easing):
        raise TypeError("easing must be a rate function or a named easing")
    return easing


def evaluate(easing, value):
    """Return ``easing(value)`` as a float, raising ValueError unless it is finite and in [0, 1]."""
    result = float(easing(float(value)))
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Rate functions must return a finite value in [0, 1]")
    return result


def linear(t):
    """Return progress ``t`` unchanged."""
    return t


def smooth(t):
    """Quintic smootherstep 10t³ − 15t⁴ + 6t⁵, with zero speed and acceleration at both ends."""
    # Mirror the upper half to avoid cancellation that can produce 1 + epsilon
    # near a clip boundary (e.g. a 1.8 s focus ending at a fractional frame time).
    u = 1 - t if t > 0.5 else t
    value = u * u * u * (10 + u * (-15 + 6 * u))
    return 1 - value if t > 0.5 else value


def ease_in_out_sine(t):
    """Cosine easing (1 − cos πt) / 2, slow at both ends."""
    return (1 - math.cos(math.pi * t)) / 2


def there_and_back(t):
    """Rise smoothly from 0 to 1 by t = 0.5, then return to 0 at t = 1."""
    # The clip ends at rate(1) = 0, so the next clip starts from the state this one started from.
    return smooth(2 * t if t < 0.5 else 2 * (1 - t))
