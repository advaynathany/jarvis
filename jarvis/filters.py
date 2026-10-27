"""
One Euro filter (Casiez, Roussel & Vogel, CHI 2012).

A low-pass filter whose cutoff rises with speed: heavy smoothing when the
hand is nearly still (kills tremor), almost none when it moves fast (kills
lag). Two knobs instead of a smoothing/acceleration balancing act:
  min_cutoff  - Hz. Lower = steadier at rest, more lag when starting to move.
  beta        - how quickly the cutoff opens up with speed.
"""

import math


class LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x = LowPass()
        self.dx = LowPass()
        self.t_prev = None
        self.speed = 0.0

    def __call__(self, x, t, min_cutoff=None):
        if self.t_prev is None:
            self.t_prev = t
            self.dx.y = 0.0
            return self.x(x, 1.0)
        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t
        dx = self.dx((x - self.x.y) / dt, _alpha(self.d_cutoff, dt))
        self.speed = abs(dx)
        cutoff = (min_cutoff or self.min_cutoff) + self.beta * self.speed
        return self.x(x, _alpha(cutoff, dt))


class OneEuro2D:
    """Filters an (x, y) point; speed is reported as the vector magnitude."""

    def __init__(self, **kw):
        self.fx = OneEuroFilter(**kw)
        self.fy = OneEuroFilter(**kw)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x, y, t, min_cutoff=None):
        return self.fx(x, t, min_cutoff), self.fy(y, t, min_cutoff)

    @property
    def speed(self):
        return math.hypot(self.fx.speed, self.fy.speed)
