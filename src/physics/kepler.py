"""Two-body (Kepler) orbits: orbital elements and exact propagation.

Use cases:
- turn a position/velocity state into classical orbital elements (a, e, i, ...)
  to report or plot orbit geometry;
- propagate a bound two-body orbit *exactly*, as ground truth for testing
  numerical integrators.

All functions are vectorized: ``r`` and ``v`` have shape ``(N, 3)`` and ``mu``
is a scalar or shape ``(N,)``. ``mu = G * (M + m)`` is the two-body
gravitational parameter, in the caller's unit system. Angles are in radians.

Only bound (elliptic, ``0 <= e < 1``) orbits are supported. The reference
plane is the x-y plane of the input coordinates (e.g. the ecliptic).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "OrbitalElements",
    "elements_to_state",
    "propagate",
    "solve_kepler",
    "state_to_elements",
]

FloatArray = NDArray[np.float64]


class OrbitalElements(NamedTuple):
    """Classical orbital elements; each field is an array of shape ``(N,)``."""

    a: FloatArray  #: semi-major axis
    e: FloatArray  #: eccentricity
    inc: FloatArray  #: inclination to the reference plane [rad]
    raan: FloatArray  #: longitude of the ascending node, Omega [rad]
    argp: FloatArray  #: argument of periapsis, omega [rad]
    mean_anomaly: FloatArray  #: mean anomaly, M [rad]

    def period(self, mu: ArrayLike) -> FloatArray:
        return 2 * np.pi * np.sqrt(self.a**3 / np.asarray(mu, dtype=np.float64))


def state_to_elements(r: ArrayLike, v: ArrayLike, mu: ArrayLike) -> OrbitalElements:
    """Convert Cartesian states to classical orbital elements.

    For equatorial orbits (i = 0) the node is undefined; it is set to 0 and
    the periapsis angle is measured from the x-axis. For circular orbits
    (e = 0) the periapsis is undefined; ``argp`` is set to 0 and the mean
    anomaly is measured from the node.
    """
    r, v = _as_states(r, v)
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (len(r),))

    rn = np.linalg.norm(r, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    h_hat = h / hn[:, None]

    energy = 0.5 * np.einsum("ij,ij->i", v, v) - mu / rn
    if np.any(energy >= 0):
        raise ValueError("only bound (elliptic) orbits are supported")
    a = -mu / (2 * energy)

    e_vec = np.cross(v, h) / mu[:, None] - r / rn[:, None]
    e = np.linalg.norm(e_vec, axis=1)

    inc = np.arccos(np.clip(h_hat[:, 2], -1.0, 1.0))

    # Node vector z_hat x h; fall back to the x-axis for equatorial orbits.
    node = np.column_stack([-h[:, 1], h[:, 0], np.zeros(len(r))])
    node_n = np.linalg.norm(node, axis=1)
    equatorial = node_n < 1e-15 * hn
    node[equatorial] = [1.0, 0.0, 0.0]
    node_n[equatorial] = 1.0
    node_hat = node / node_n[:, None]
    raan = np.mod(np.arctan2(node_hat[:, 1], node_hat[:, 0]), 2 * np.pi)

    # Periapsis direction; fall back to the node for circular orbits.
    circular = e < 1e-12
    peri_hat = np.where(circular[:, None], node_hat, e_vec / np.where(circular, 1, e)[:, None])

    argp = _signed_angle(node_hat, peri_hat, h_hat)
    nu = _signed_angle(peri_hat, r / rn[:, None], h_hat)

    big_e = np.arctan2(np.sqrt(1 - e**2) * np.sin(nu), e + np.cos(nu))
    mean_anomaly = np.mod(big_e - e * np.sin(big_e), 2 * np.pi)

    return OrbitalElements(a, e, inc, np.mod(raan, 2 * np.pi), argp, mean_anomaly)


def elements_to_state(el: OrbitalElements, mu: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Convert orbital elements to Cartesian position and velocity, each ``(N, 3)``."""
    a, e, inc, raan, argp, m = (np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in el)
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), a.shape)

    big_e = solve_kepler(m, e)
    cos_e, sin_e = np.cos(big_e), np.sin(big_e)
    sqrt_1me2 = np.sqrt(1 - e**2)

    # Position and velocity in the perifocal frame (x toward periapsis).
    rn = a * (1 - e * cos_e)
    x_p = a * (cos_e - e)
    y_p = a * sqrt_1me2 * sin_e
    vfac = np.sqrt(mu * a) / rn
    vx_p = -vfac * sin_e
    vy_p = vfac * sqrt_1me2 * cos_e

    # Rotate perifocal -> reference frame: R3(raan) R1(inc) R3(argp).
    p_hat, q_hat = _perifocal_axes(inc, raan, argp)
    r = x_p[:, None] * p_hat + y_p[:, None] * q_hat
    v = vx_p[:, None] * p_hat + vy_p[:, None] * q_hat
    return r, v


def propagate(
    r: ArrayLike, v: ArrayLike, mu: ArrayLike, dt: float
) -> tuple[FloatArray, FloatArray]:
    """Advance bound two-body orbits by time ``dt`` exactly (up to round-off)."""
    el = state_to_elements(r, v, mu)
    n = 2 * np.pi / el.period(mu)
    advanced = el._replace(mean_anomaly=np.mod(el.mean_anomaly + n * dt, 2 * np.pi))
    return elements_to_state(advanced, mu)


def solve_kepler(
    mean_anomaly: ArrayLike, e: ArrayLike, *, tol: float = 1e-14, max_iter: int = 50
) -> FloatArray:
    """Solve Kepler's equation ``E - e sin(E) = M`` for the eccentric anomaly E."""
    m = np.mod(np.asarray(mean_anomaly, dtype=np.float64), 2 * np.pi)
    e = np.broadcast_to(np.asarray(e, dtype=np.float64), m.shape)
    if np.any((e < 0) | (e >= 1)):
        raise ValueError("eccentricity must satisfy 0 <= e < 1")

    big_e = np.where(e < 0.8, m, np.pi)  # standard robust starting guess
    for _ in range(max_iter):
        f = big_e - e * np.sin(big_e) - m
        step = f / (1 - e * np.cos(big_e))
        big_e = big_e - step
        if np.all(np.abs(step) < tol):
            return big_e
    raise RuntimeError("Kepler's equation did not converge")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_states(r: ArrayLike, v: ArrayLike) -> tuple[FloatArray, FloatArray]:
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    if r.shape != v.shape or r.shape[1] != 3:
        raise ValueError(f"r and v must both have shape (N, 3), got {r.shape} and {v.shape}")
    return r, v


def _signed_angle(a: FloatArray, b: FloatArray, axis: FloatArray) -> FloatArray:
    """Angle from unit vector a to unit vector b, counter-clockwise about ``axis``, in [0, 2pi)."""
    sin = np.einsum("ij,ij->i", np.cross(a, b), axis)
    cos = np.einsum("ij,ij->i", a, b)
    return np.mod(np.arctan2(sin, cos), 2 * np.pi)


def _perifocal_axes(
    inc: FloatArray, raan: FloatArray, argp: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Unit vectors toward periapsis (P) and 90 degrees ahead in the orbit plane (Q)."""
    co, so = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(argp), np.sin(argp)
    ci, si = np.cos(inc), np.sin(inc)
    p_hat = np.column_stack([co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si])
    q_hat = np.column_stack([-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si])
    return p_hat, q_hat
