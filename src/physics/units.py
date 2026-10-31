"""Unit systems for simulations.

The physics kernels work on plain floats. A :class:`UnitSystem` fixes the base
units for length, mass and time, gives ``G`` in those units, and converts
astropy quantities in and out at the boundaries of a simulation.

Example
-------
>>> from astropy import units as u
>>> from physics.units import SOLAR
>>> v = SOLAR.to_value(29.78 * u.km / u.s)   # Earth's orbital speed, in AU/yr
>>> round(v, 2)
6.28
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from astropy import constants as const
from astropy import units as u
from numpy.typing import ArrayLike

__all__ = ["GALACTIC", "SI", "SOLAR", "UnitSystem"]


@dataclass(frozen=True)
class UnitSystem:
    """A set of base units for length, mass and time."""

    name: str
    length: u.UnitBase
    mass: u.UnitBase
    time: u.UnitBase

    @property
    def velocity(self) -> u.UnitBase:
        return self.length / self.time

    @property
    def acceleration(self) -> u.UnitBase:
        return self.length / self.time**2

    @property
    def energy(self) -> u.UnitBase:
        return self.mass * self.length**2 / self.time**2

    @cached_property
    def G(self) -> float:
        """Newton's constant expressed in this unit system (from astropy.constants)."""
        return self.to_value(const.G)

    def to_value(self, q: u.Quantity) -> float | np.ndarray:
        """Strip units from ``q`` after expressing it in this system's base units."""
        return q.decompose(bases=[self.length, self.mass, self.time]).value

    def quantity(self, value: ArrayLike, unit: u.UnitBase) -> u.Quantity:
        """Attach units to a plain value; ``unit`` should be built from this system,
        e.g. ``SOLAR.quantity(v, SOLAR.velocity)``."""
        return u.Quantity(value, unit)


SI = UnitSystem("SI", u.m, u.kg, u.s)
"""Metres, kilograms, seconds."""

SOLAR = UnitSystem("solar", u.AU, u.M_sun, u.yr)
"""AU, solar masses, Julian years. G is about 4*pi^2 (39.4769 vs 39.4784; the
difference comes from the Julian year not matching a 1 AU orbit exactly)."""

GALACTIC = UnitSystem("galactic", u.kpc, u.M_sun, u.Myr)
"""Kiloparsecs, solar masses, megayears. 1 kpc/Myr is about 978 km/s."""
