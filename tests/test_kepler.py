import numpy as np
import pytest

from physics.kepler import (
    OrbitalElements,
    elements_to_state,
    propagate,
    solve_kepler,
    state_to_elements,
)

MU = 1.0


def random_elements(rng, n=200):
    return OrbitalElements(
        a=rng.uniform(0.3, 30, n),
        e=rng.uniform(0.0, 0.95, n),
        inc=rng.uniform(0.01, np.pi - 0.01, n),
        raan=rng.uniform(0, 2 * np.pi, n),
        argp=rng.uniform(0, 2 * np.pi, n),
        mean_anomaly=rng.uniform(0, 2 * np.pi, n),
    )


def angle_diff(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


def test_solve_kepler_satisfies_equation():
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 2 * np.pi, 1000)
    e = rng.uniform(0, 0.99, 1000)
    big_e = solve_kepler(m, e)
    np.testing.assert_allclose(np.mod(big_e - e * np.sin(big_e), 2 * np.pi), m, atol=1e-12)


def test_elements_round_trip():
    rng = np.random.default_rng(1)
    el = random_elements(rng)
    back = state_to_elements(*elements_to_state(el, MU), MU)
    np.testing.assert_allclose(back.a, el.a, rtol=1e-11)
    np.testing.assert_allclose(back.e, el.e, atol=1e-11)
    np.testing.assert_allclose(back.inc, el.inc, atol=1e-11)
    for name in ("raan", "argp", "mean_anomaly"):
        assert np.all(angle_diff(getattr(back, name), getattr(el, name)) < 1e-8), name


def test_circular_orbit_in_plane():
    r = [[1.0, 0, 0]]
    v = [[0, 1.0, 0]]  # circular speed for mu=1, r=1
    el = state_to_elements(r, v, MU)
    assert el.a[0] == pytest.approx(1.0)
    assert el.e[0] == pytest.approx(0.0, abs=1e-14)
    assert el.inc[0] == pytest.approx(0.0, abs=1e-14)
    assert el.period(MU)[0] == pytest.approx(2 * np.pi)


def test_inclination_and_node_from_tilted_orbit():
    inc = np.radians(30)
    r = [[1.0, 0, 0]]  # starts at ascending node on +x
    v = [[0, np.cos(inc), np.sin(inc)]]
    el = state_to_elements(r, v, MU)
    assert el.inc[0] == pytest.approx(inc)
    assert angle_diff(el.raan[0], 0.0) < 1e-12


def test_propagate_full_period_returns_to_start():
    rng = np.random.default_rng(2)
    el = random_elements(rng, n=50)
    r, v = elements_to_state(el, MU)
    period = el.period(MU)
    for i in range(len(period)):
        r2, v2 = propagate(r[i : i + 1], v[i : i + 1], MU, period[i])
        np.testing.assert_allclose(r2, r[i : i + 1], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(v2, v[i : i + 1], rtol=1e-8, atol=1e-10)


def test_propagate_conserves_energy_and_angular_momentum():
    rng = np.random.default_rng(3)
    r, v = elements_to_state(random_elements(rng, n=100), MU)
    r2, v2 = propagate(r, v, MU, 12.345)

    def energy(r, v):
        return 0.5 * (v**2).sum(1) - MU / np.linalg.norm(r, axis=1)

    np.testing.assert_allclose(energy(r2, v2), energy(r, v), rtol=1e-11)
    np.testing.assert_allclose(np.cross(r2, v2), np.cross(r, v), rtol=1e-10, atol=1e-12)


def test_rejects_unbound_orbit():
    with pytest.raises(ValueError, match="bound"):
        state_to_elements([[1.0, 0, 0]], [[0, 2.0, 0]], MU)
