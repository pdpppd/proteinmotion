"""Rate functions accepted by every play() call and animation."""

import math


def linear(t):
    return t


def smooth(t):
    # Mirror the upper half to avoid cancellation that can produce 1 + epsilon
    # near a clip boundary (e.g. a 1.8 s focus ending at a fractional frame time).
    u = 1 - t if t > 0.5 else t
    value = u * u * u * (10 + u * (-15 + 6 * u))
    return 1 - value if t > 0.5 else value


def ease_in_out_sine(t):
    return (1 - math.cos(math.pi * t)) / 2


def there_and_back(t):
    return smooth(2 * t if t < 0.5 else 2 * (1 - t))
