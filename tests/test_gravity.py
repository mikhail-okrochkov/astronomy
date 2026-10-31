"""Physics checks for physics.gravity, using cases with known answers."""

import numpy as np
import pytest

from physics.gravity import accelerations, field_at, potential_at, potential_energy

G = 1.0


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def random_system(rng, n=50, dim=3):
    pos = rng.normal(size=(n, dim))
    mass = rng.uniform(0.1, 2.0, size=n)
    return pos, mass


def fibonacci_sphere(n, radius):
    """Nearly uniform points on a sphere (low multipole error)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return radius * np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


def brute_force_acc(pos, mass, G, eps=0.0):
    n = len(pos)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = pos[j] - pos[i]
                acc[i] += G * mass[j] * d / (d @ d + eps**2) ** 1.5
    return acc


# --- Two-body / exact cases -------------------------------------------------


def test_two_body_matches_newton():
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mass = np.array([3.0, 5.0])
    acc = accelerations(pos, mass, G=G)
    # |a1| = G m2 / r^2 toward body 2; |a2| = G m1 / r^2 toward body 1
    np.testing.assert_allclose(acc[0], [5.0 / 4.0, 0, 0])
    np.testing.assert_allclose(acc[1], [-3.0 / 4.0, 0, 0])


def test_two_body_potential_energy():
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    mass = np.array([2.0, 6.0])
    assert potential_energy(pos, mass, G=2.0) == pytest.approx(-2.0 * 2.0 * 6.0 / 4.0)


def test_matches_brute_force(rng):
    pos, mass = random_system(rng, n=20)
    for eps in (0.0, 0.1):
        np.testing.assert_allclose(
            accelerations(pos, mass, G=G, softening=eps),
            brute_force_acc(pos, mass, G, eps),
            rtol=1e-12,
        )


def test_potential_energy_matches_pair_sum(rng):
    pos, mass = random_system(rng, n=30)
    expected = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            expected -= G * mass[i] * mass[j] / np.linalg.norm(pos[i] - pos[j])
    assert potential_energy(pos, mass, G=G) == pytest.approx(expected, rel=1e-12)


# --- Conservation / consistency ----------------------------------------------


def test_newtons_third_law_total_force_zero(rng):
    pos, mass = random_system(rng, n=200)
    acc = accelerations(pos, mass, G=G)
    total_force = (mass[:, None] * acc).sum(axis=0)
    scale = np.abs(mass[:, None] * acc).sum()
    assert np.all(np.abs(total_force) < 1e-12 * scale)


def test_field_is_minus_gradient_of_potential(rng):
    pos, mass = random_system(rng, n=10)
    points = rng.normal(size=(5, 3)) * 3 + 5  # away from sources
    h = 1e-5
    grad = np.zeros_like(points)
    for d in range(3):
        step = np.zeros(3)
        step[d] = h
        grad[:, d] = (
            potential_at(points + step, pos, mass, G=G)
            - potential_at(points - step, pos, mass, G=G)
        ) / (2 * h)
    np.testing.assert_allclose(field_at(points, pos, mass, G=G), -grad, rtol=1e-6)


def test_field_at_body_positions_equals_accelerations_with_softening(rng):
    # With softening, self-interaction contributes zero force, so evaluating the
    # field at the bodies' own positions must reproduce accelerations().
    pos, mass = random_system(rng, n=40)
    np.testing.assert_allclose(
        field_at(pos, pos, mass, G=G, softening=0.05),
        accelerations(pos, mass, G=G, softening=0.05),
        rtol=1e-12,
    )


def test_test_particle_scales_linearly_with_G(rng):
    pos, mass = random_system(rng, n=10)
    pts = rng.normal(size=(4, 3)) + 10
    np.testing.assert_allclose(
        field_at(pts, pos, mass, G=7.0), 7.0 * field_at(pts, pos, mass, G=1.0), rtol=1e-14
    )


# --- Shell theorem -----------------------------------------------------------


def test_shell_theorem_outside_looks_like_point_mass():
    n, R, M = 4000, 1.0, 10.0
    shell = fibonacci_sphere(n, R)
    mass = np.full(n, M / n)
    points = np.array([[5.0, 0, 0], [0, -3.0, 0], [2.0, 2.0, 2.0]])
    r = np.linalg.norm(points, axis=1)
    expected = -G * M * points / r[:, None] ** 3
    err = np.linalg.norm(field_at(points, shell, mass, G=G) - expected, axis=1)
    assert np.all(err < 1e-4 * np.linalg.norm(expected, axis=1))
    np.testing.assert_allclose(potential_at(points, shell, mass, G=G), -G * M / r, rtol=1e-4)


def test_shell_theorem_inside_field_vanishes():
    n, R, M = 4000, 1.0, 10.0
    shell = fibonacci_sphere(n, R)
    mass = np.full(n, M / n)
    points = np.array([[0.0, 0, 0], [0.3, 0.1, -0.2], [0, 0, 0.5]])
    acc = field_at(points, shell, mass, G=G)
    surface_scale = G * M / R**2
    assert np.all(np.linalg.norm(acc, axis=1) < 1e-2 * surface_scale)
    # Potential inside a shell is constant, -GM/R
    np.testing.assert_allclose(potential_at(points, shell, mass, G=G), -G * M / R, rtol=1e-3)


# --- Softening, dimensions, chunking -----------------------------------------


def test_softening_keeps_coincident_bodies_finite():
    pos = np.zeros((2, 3))
    mass = np.ones(2)
    acc = accelerations(pos, mass, G=G, softening=0.1)
    assert np.all(np.isfinite(acc))
    np.testing.assert_array_equal(acc, 0.0)
    assert potential_energy(pos, mass, G=G, softening=0.1) == pytest.approx(-1.0 / 0.1)


def test_softening_reduces_close_force():
    pos = np.array([[0.0, 0, 0], [0.01, 0, 0]])
    mass = np.ones(2)
    hard = accelerations(pos, mass, G=G)
    soft = accelerations(pos, mass, G=G, softening=0.1)
    assert np.linalg.norm(soft[0]) < np.linalg.norm(hard[0])


def test_works_in_two_dimensions(rng):
    pos, mass = random_system(rng, n=15, dim=2)
    acc = accelerations(pos, mass, G=G)
    assert acc.shape == (15, 2)
    np.testing.assert_allclose(acc, brute_force_acc(pos, mass, G), rtol=1e-12)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
def test_chunking_does_not_change_results(rng, chunk_size):
    pos, mass = random_system(rng, n=25)
    pts = rng.normal(size=(11, 3)) + 4
    ref_acc = accelerations(pos, mass, G=G)
    ref_U = potential_energy(pos, mass, G=G)
    np.testing.assert_allclose(accelerations(pos, mass, G=G, chunk_size=chunk_size), ref_acc)
    assert potential_energy(pos, mass, G=G, chunk_size=chunk_size) == pytest.approx(ref_U)
    np.testing.assert_allclose(
        field_at(pts, pos, mass, G=G, chunk_size=chunk_size), field_at(pts, pos, mass, G=G)
    )


def test_massless_bodies_feel_but_do_not_exert_gravity():
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    mass = np.array([1.0, 0.0])
    acc = accelerations(pos, mass, G=G)
    np.testing.assert_array_equal(acc[0], 0.0)
    np.testing.assert_allclose(acc[1], [-1.0, 0, 0])


# --- Validation ---------------------------------------------------------------


def test_rejects_bad_shapes():
    with pytest.raises(ValueError, match="pos"):
        accelerations(np.zeros(3), np.ones(1), G=G)
    with pytest.raises(ValueError, match="mass"):
        accelerations(np.zeros((3, 3)), np.ones(2), G=G)
    with pytest.raises(ValueError, match="points"):
        field_at(np.zeros((2, 2)), np.zeros((3, 3)), np.ones(3), G=G)


def test_rejects_negative_softening_and_bad_chunk():
    pos, mass = np.zeros((2, 3)), np.ones(2)
    with pytest.raises(ValueError, match="softening"):
        accelerations(pos, mass, G=G, softening=-1.0)
    with pytest.raises(ValueError, match="chunk_size"):
        accelerations(pos, mass, G=G, softening=0.1, chunk_size=0)
