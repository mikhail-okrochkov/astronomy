import numpy as np
import pytest

from physics.gravity import accelerations
from physics.integrators import integrate
from physics.kepler import OrbitalElements, elements_to_state, propagate


def harmonic(x):
    return -x


def harmonic_error(method, dt, t_end=10.0):
    n = round(t_end / dt)
    traj = integrate([[1.0]], [[0.0]], harmonic, dt=dt, n_steps=n, method=method)
    return abs(traj.pos[-1, 0, 0] - np.cos(n * dt))


@pytest.mark.parametrize(("method", "order"), [("leapfrog", 2), ("yoshida4", 4)])
def test_convergence_order(method, order):
    e1 = harmonic_error(method, 0.02)
    e2 = harmonic_error(method, 0.01)
    measured = np.log2(e1 / e2)
    assert measured == pytest.approx(order, abs=0.15)


@pytest.mark.parametrize("method", ["leapfrog", "yoshida4"])
def test_energy_bounded_on_eccentric_orbit(method):
    # Two equal masses on an eccentric orbit, many periods: symplectic energy
    # error should stay small and not grow.
    mu = 2.0  # G=1, m1=m2=1
    el = OrbitalElements(*(np.array([x]) for x in (1.0, 0.5, 0.0, 0.0, 0.0, 0.0)))
    r_rel, v_rel = elements_to_state(el, mu)
    pos = np.vstack([-0.5 * r_rel, 0.5 * r_rel])
    vel = np.vstack([-0.5 * v_rel, 0.5 * v_rel])
    mass = np.ones(2)

    def acc(x):
        return accelerations(x, mass, G=1.0)

    def energy(x, v):
        return 0.5 * (mass[:, None] * v**2).sum() - 1.0 / np.linalg.norm(x[0] - x[1])

    period = 2 * np.pi * np.sqrt(1.0 / mu)
    steps_per_orbit, n_orbits = 400, 50
    traj = integrate(
        pos,
        vel,
        acc,
        dt=period / steps_per_orbit,
        n_steps=steps_per_orbit * n_orbits,
        save_every=4,
        method=method,
    )
    e0 = energy(traj.pos[0], traj.vel[0])
    rel_err = np.array(
        [abs(energy(x, v) / e0 - 1) for x, v in zip(traj.pos, traj.vel, strict=True)]
    )
    assert rel_err.max() < 1e-3
    # No secular growth: the worst error over the last 5 orbits is no worse than
    # over the first 5. (Sample densely: the error oscillates within each orbit,
    # and the numerical period differs slightly from the true one.)
    window = 5 * steps_per_orbit // 4
    assert rel_err[-window:].max() < 1.1 * rel_err[:window].max()


def test_yoshida_matches_exact_kepler():
    mu = 1.0
    el = OrbitalElements(*(np.array([x]) for x in (1.0, 0.2, 0.3, 1.0, 2.0, 0.5)))
    r, v = elements_to_state(el, mu)

    def acc(x):
        return -mu * x / np.linalg.norm(x, axis=1, keepdims=True) ** 3

    t_end = 10 * 2 * np.pi
    traj = integrate(r, v, acc, dt=t_end / 4000, n_steps=4000, method="yoshida4")
    r_exact, _ = propagate(r, v, mu, t_end)
    assert np.linalg.norm(traj.pos[-1] - r_exact) < 1e-5  # measured 5.9e-6


def test_saving_schedule():
    traj = integrate([[1.0]], [[0.0]], harmonic, dt=0.1, n_steps=10, save_every=4)
    np.testing.assert_allclose(traj.t, [0.0, 0.4, 0.8, 1.0])
    assert traj.pos.shape == (4, 1, 1)


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        integrate([[1.0]], [[0.0]], harmonic, dt=0.1, n_steps=1, method="euler")
