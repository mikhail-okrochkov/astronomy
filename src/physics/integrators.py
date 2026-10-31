"""Fixed-step symplectic integrators for ``x'' = a(x)``.

Symplectic methods don't let energy drift away over long runs; the energy
error stays bounded and oscillates. That makes them the right default for
orbits. They need a *fixed* step: use them when no close encounters happen
(planets around a star). For close encounters (three-body scattering) use an
adaptive integrator instead.

The acceleration function takes positions ``(N, D)`` and returns
accelerations ``(N, D)``; it must not depend on velocity. Typically it wraps
:func:`physics.gravity.accelerations` or :func:`physics.gravity.field_at`.

Available methods
-----------------
- ``"leapfrog"``: kick-drift-kick, 2nd order, 1 force evaluation per step.
- ``"yoshida4"``: Yoshida (1990) 4th-order composition of three leapfrog
  steps, 3 force evaluations per step. Error per orbit drops as dt**4, so it
  is usually much cheaper for a given accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["METHODS", "Trajectory", "integrate"]

FloatArray = NDArray[np.float64]
AccelFn = Callable[[FloatArray], FloatArray]


class Trajectory(NamedTuple):
    """Saved snapshots. ``pos`` and ``vel`` have shape ``(n_saved, N, D)``."""

    t: FloatArray
    pos: FloatArray
    vel: FloatArray


# Yoshida (1990) 4th-order coefficients.
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)
_W0 = -_CBRT2 * _W1

# Each method is a sequence of leapfrog sub-step fractions.
METHODS: dict[str, tuple[float, ...]] = {
    "leapfrog": (1.0,),
    "yoshida4": (_W1, _W0, _W1),
}


def integrate(
    pos: ArrayLike,
    vel: ArrayLike,
    accel: AccelFn,
    *,
    dt: float,
    n_steps: int,
    save_every: int = 1,
    t0: float = 0.0,
    method: str = "yoshida4",
) -> Trajectory:
    """Integrate from ``(pos, vel)`` for ``n_steps`` steps of size ``dt``.

    Snapshots are saved at the start and every ``save_every`` steps
    (the final state is always saved).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    if n_steps < 0 or save_every < 1:
        raise ValueError("n_steps must be >= 0 and save_every >= 1")

    x = np.array(pos, dtype=np.float64)
    v = np.array(vel, dtype=np.float64)
    if x.shape != v.shape:
        raise ValueError(f"pos and vel shapes differ: {x.shape} vs {v.shape}")

    substeps = tuple(w * dt for w in METHODS[method])
    a = accel(x)

    save_steps = list(range(0, n_steps + 1, save_every))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    n_saved = len(save_steps)
    ts = np.empty(n_saved)
    xs = np.empty((n_saved, *x.shape))
    vs = np.empty((n_saved, *x.shape))
    ts[0], xs[0], vs[0] = t0, x, v

    k = 1
    for step in range(1, n_steps + 1):
        for h in substeps:
            # Kick-drift-kick; the trailing acceleration is reused next substep.
            v += 0.5 * h * a
            x += h * v
            a = accel(x)
            v += 0.5 * h * a
        if k < n_saved and step == save_steps[k]:
            ts[k], xs[k], vs[k] = t0 + step * dt, x, v
            k += 1

    return Trajectory(ts, xs, vs)
