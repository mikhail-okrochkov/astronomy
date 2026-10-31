import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from physics.units import GALACTIC, SI, SOLAR


def test_si_G_matches_codata():
    assert SI.G == pytest.approx(const.G.value)


def test_solar_G_is_close_to_four_pi_squared():
    # Not exact: the Julian year is not the sidereal period of a 1 AU orbit.
    assert SOLAR.G == pytest.approx(4 * np.pi**2, rel=1e-4)


def test_earth_orbital_speed_in_au_per_year():
    assert SOLAR.to_value(29.78 * u.km / u.s) == pytest.approx(2 * np.pi, rel=1e-3)


def test_galactic_velocity_unit():
    assert GALACTIC.to_value(1 * u.kpc / u.Myr) == pytest.approx(1.0)
    assert (1 * GALACTIC.velocity).to_value(u.km / u.s) == pytest.approx(977.79, rel=1e-4)


def test_round_trip_quantity():
    q = SOLAR.quantity([1.0, 2.0], SOLAR.velocity)
    np.testing.assert_allclose(SOLAR.to_value(q), [1.0, 2.0])


def test_to_value_works_on_arrays():
    masses = [1.0, 0.5] * u.M_sun
    np.testing.assert_allclose(SI.to_value(masses), [1.98840987e30, 0.99420494e30], rtol=1e-6)
