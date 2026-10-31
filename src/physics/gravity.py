"""Newtonian gravity between point masses, by direct summation.

This module only computes *forces and potentials* for a given configuration.
It knows nothing about time integration; integrators call into it.

Conventions
-----------
- Positions are plain float arrays of shape ``(N, D)``; ``D`` is usually 3,
  but 2-D problems work too. Masses have shape ``(N,)``.
- No units are attached. Pass ``G`` in whatever unit system the caller uses
  (see :mod:`physics.units`), and all outputs are in that same system.
- Softening uses the Plummer form: the separation ``r`` is replaced by
  ``sqrt(r**2 + eps**2)``. ``softening=0`` gives exact Newtonian gravity.

Cost is O(N * M) for N sources acting on M points. Work is done in chunks of
target points so memory stays bounded (roughly ``chunk_size * N * D`` floats).

Formulas (for source masses ``m_j`` at ``x_j``)::

    a(x)   =  G * sum_j m_j (x_j - x) / (|x_j - x|^2 + eps^2)^(3/2)
    phi(x) = -G * sum_j m_j / sqrt(|x_j - x|^2 + eps^2)
    U      = -G * sum_{i<j} m_i m_j / sqrt(|x_i - x_j|^2 + eps^2)
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["accelerations", "field_at", "potential_at", "potential_energy"]

FloatArray = NDArray[np.float64]

# Target number of (point, source) pairs per chunk. ~1e6 pairs * 3 dims * 8 bytes
# is ~24 MB per temporary array, which is comfortable on any laptop.
_DEFAULT_PAIRS_PER_CHUNK = 1_000_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def accelerations(
    pos: ArrayLike,
    mass: ArrayLike,
    *,
    G: float,
    softening: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray:
    """Gravitational acceleration on each body due to all the *other* bodies.

    Parameters
    ----------
    pos : (N, D) array
        Body positions.
    mass : (N,) array
        Body masses. Zero-mass bodies feel gravity but exert none.
    G : float
        Gravitational constant in the caller's unit system.
    softening : float
        Plummer softening length ``eps`` (>= 0).
    chunk_size : int, optional
        Number of bodies processed per chunk; defaults to a memory-friendly value.

    Returns
    -------
    (N, D) array of accelerations.
    """
    pos, mass = _validate_sources(pos, mass)
    return _field(pos, pos, mass, G, softening, chunk_size, exclude_self=True)


def field_at(
    points: ArrayLike,
    pos: ArrayLike,
    mass: ArrayLike,
    *,
    G: float,
    softening: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray:
    """Gravitational acceleration at arbitrary ``points`` due to the sources.

    Use this for massless test particles (asteroids, ring particles, tracer
    stars) and for sampling the field on a grid. Points are not sources, so the
    cost is O(N * M) rather than O((N + M)^2).

    With ``softening=0``, a point that coincides exactly with a source gets an
    infinite/NaN result, which is the honest answer.

    Returns
    -------
    (M, D) array of accelerations.
    """
    pos, mass = _validate_sources(pos, mass)
    points = _validate_points(points, pos.shape[1])
    return _field(points, pos, mass, G, softening, chunk_size, exclude_self=False)


def potential_at(
    points: ArrayLike,
    pos: ArrayLike,
    mass: ArrayLike,
    *,
    G: float,
    softening: float = 0.0,
    chunk_size: int | None = None,
) -> FloatArray:
    """Gravitational potential (energy per unit mass) at ``points``.

    Handy for contour plots of potential wells, and for checking that the
    field is the negative gradient of the potential.

    Returns
    -------
    (M,) array of potentials (negative for bound regions).
    """
    pos, mass = _validate_sources(pos, mass)
    points = _validate_points(points, pos.shape[1])
    return _potential(points, pos, mass, G, softening, chunk_size, exclude_self=False)


def potential_energy(
    pos: ArrayLike,
    mass: ArrayLike,
    *,
    G: float,
    softening: float = 0.0,
    chunk_size: int | None = None,
) -> float:
    """Total gravitational potential energy of the system, counting each pair once.

    Combine with kinetic energy to monitor energy conservation in integrators.
    """
    pos, mass = _validate_sources(pos, mass)
    phi = _potential(pos, pos, mass, G, softening, chunk_size, exclude_self=True)
    return float(0.5 * np.dot(mass, phi))


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _field(
    points: FloatArray,
    pos: FloatArray,
    mass: FloatArray,
    G: float,
    softening: float,
    chunk_size: int | None,
    *,
    exclude_self: bool,
) -> FloatArray:
    eps2 = _softening_squared(softening)
    out = np.empty_like(points)
    for start, stop in _chunks(len(points), len(pos), chunk_size):
        # d[k, j] = x_j - x_k, shape (c, N, D)
        d = pos[np.newaxis, :, :] - points[start:stop, np.newaxis, :]
        r2 = np.einsum("kjd,kjd->kj", d, d) + eps2
        with np.errstate(divide="ignore"):
            inv_r3 = r2**-1.5
        if exclude_self:
            _zero_diagonal(inv_r3, start)
        out[start:stop] = G * np.einsum("kj,j,kjd->kd", inv_r3, mass, d)
    return out


def _potential(
    points: FloatArray,
    pos: FloatArray,
    mass: FloatArray,
    G: float,
    softening: float,
    chunk_size: int | None,
    *,
    exclude_self: bool,
) -> FloatArray:
    eps2 = _softening_squared(softening)
    out = np.empty(len(points))
    for start, stop in _chunks(len(points), len(pos), chunk_size):
        d = pos[np.newaxis, :, :] - points[start:stop, np.newaxis, :]
        r2 = np.einsum("kjd,kjd->kj", d, d) + eps2
        with np.errstate(divide="ignore"):
            inv_r = r2**-0.5
        if exclude_self:
            _zero_diagonal(inv_r, start)
        out[start:stop] = -G * (inv_r @ mass)
    return out


def _zero_diagonal(a: FloatArray, start: int) -> None:
    """Zero the self-interaction entries of a (chunk, N) block starting at row ``start``."""
    k = np.arange(a.shape[0])
    a[k, start + k] = 0.0


def _chunks(n_points: int, n_sources: int, chunk_size: int | None) -> Iterator[tuple[int, int]]:
    if chunk_size is None:
        chunk_size = max(1, _DEFAULT_PAIRS_PER_CHUNK // max(n_sources, 1))
    elif chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    for start in range(0, n_points, chunk_size):
        yield start, min(start + chunk_size, n_points)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_sources(pos: ArrayLike, mass: ArrayLike) -> tuple[FloatArray, FloatArray]:
    pos = np.asarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if pos.ndim != 2:
        raise ValueError(f"pos must have shape (N, D), got {pos.shape}")
    if mass.shape != (pos.shape[0],):
        raise ValueError(f"mass must have shape ({pos.shape[0]},), got {mass.shape}")
    return pos, mass


def _validate_points(points: ArrayLike, dim: int) -> FloatArray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"points must have shape (M, {dim}), got {points.shape}")
    return points


def _softening_squared(softening: float) -> float:
    if softening < 0:
        raise ValueError(f"softening must be >= 0, got {softening}")
    return float(softening) ** 2
