# Solar system: N-body vs Sun-only gravity

**Question:** how well does Newtonian gravity reproduce the real motion of the
planets, and how much do the planets pull on each other? Two models start
from the same real JPL positions and are compared with the JPL ephemeris:

- **N-body (default):** the Sun and eight planets all attract each other.
- **Sun-only (`--model kepler`):** each planet feels only the Sun, so every
  orbit is a fixed Kepler ellipse.

## Run

```bash
uv run simulations/solar_system/run.py                          # J2000, 25 years (~15 s)
uv run simulations/solar_system/run.py --epoch 2026-10-31 --years 50
uv run simulations/solar_system/run.py --model kepler           # Sun-only as primary
uv run simulations/solar_system/run.py --animate                # + inner_planets.gif
```

Initial states and reference positions come from the
[JPL Horizons API](https://ssd-api.jpl.nasa.gov/doc/horizons.html) (DE441
ephemeris) and are cached in `data/` (committed, so default runs work
offline). A new epoch or duration fetches and caches new data.

Outputs (`outputs/`, gitignored): `orbits.png/svg`, `orbits_3d.png/svg`,
`error_vs_jpl.png/svg`, `results.md`, and optionally `inner_planets.gif`.

## Model

- Ecliptic and equinox of J2000; AU and days (TDB). Results are reported
  heliocentric (relative to the Sun), like the JPL reference.
- **N-body:** 9 bodies with Newtonian gravity between all pairs
  (`physics.gravity.accelerations`), integrated in the barycentric frame.
  Heliocentric states from Horizons are shifted so total momentum is zero.
- **Sun-only:** heliocentric; planet *i* feels **a** = −μᵢ **r** / r³ with
  μᵢ = GM☉ + GMᵢ, the exact motion of an isolated Sun–planet pair.
- GM values from DE440 (Park et al. 2021). Bodies are planetary-system
  **barycenters** (planet + moons); "Earth" is the Earth–Moon barycenter.
- Integrator: 4th-order symplectic (Yoshida 1990), fixed step dt = 0.0625 d.

**Neglected (N-body model):** general relativity (Mercury's extra
43″/century perihelion precession), asteroids (Ceres, Vesta, ... mostly
affect Mars), the Moon as a separate body, the Sun's oblateness, and
solar mass loss.

## Validation

1. **Orbit geometry.** Osculating elements at J2000 match Standish's J2000
   mean elements: inclinations to ≤ 0.002°, inner-planet a and e to ~1e-4.
2. **Integrator error.** Estimated by rerunning at 2·dt: for a 4th-order
   method, error(dt) ≈ |x(dt) − x(2dt)| / 15. In the Sun-only model this
   estimate matches the *exact* Kepler solution (430 / 3.8 / 0.46 km vs exact
   427 / 3.85 / 0.458 km for Mercury / Venus / Earth at dt = 0.125 d).
   At the default dt = 0.0625 d, the integrator error after 25 years is 27 km
   for Mercury (about 1% of its physics error) and < 0.3 km for all others.
3. **Conservation.** Total energy of the N-body system is conserved to
   ~6e-13 over 25 years.
4. **Against JPL.** See below.

## Results (J2000 + 25 years)

| Planet | N-body error | N-body direction error | Sun-only error | Sun-only direction error |
|---|---|---|---|---|
| Mercury | 2,200 km | 6.6″ | 59,000 km | 3.1′ |
| Venus | 2,300 km | 4.4″ | 294,000 km | 9.4′ |
| Earth | 1,400 km | 2.0″ | 138,000 km | 3.2′ |
| Mars | 900 km | 0.8″ | 209,000 km | 3.0′ |
| Jupiter | 131 km | 0.04″ | 4.7 million km | 21′ |
| Saturn | 65 km | 0.01″ | 57 million km | 2.3° |
| Uranus | 25 km | < 0.01″ | 7.6 million km | 8.9′ |
| Neptune | 27 km | < 0.01″ | 11.8 million km | 6.4′ |

Direction error is the angle between simulated and true positions as seen
from the Sun.

## Findings worth writing about

- **Newtonian N-body gravity with nine bodies reproduces JPL to tens of km for
  the outer planets over 25 years**, and to about 1,000–2,000 km for the inner
  planets. That's a few arcseconds, invisible in any figure.
- **The planets' pull on each other matters a lot:** removing it raises the
  error 20–200× for the inner planets and 10⁴–10⁶× for the outer ones.
- **The Sun-only model's outer-planet error is mostly a frame effect.**
  Jupiter swings the Sun around by about a solar radius, so heliocentric
  velocities include that wobble. That makes Saturn's starting orbit look
  0.5% too big (a = 9.582 vs 9.537 AU) and its period 0.7% too long. A test
  with Kepler orbits in Jacobi coordinates (each planet orbits the
  barycenter of the Sun plus all planets interior to it) cut Saturn's
  25-year error from 57 to 12 million km.
- **The remaining inner-planet error points to missing physics.** Mercury's
  is likely dominated by general relativity, Mars's by asteroids; neither is
  tested yet.
- **Stability:** the integrator is symplectic, so energy error stays bounded
  however long it runs. On timescales beyond ~5–10 million years the real
  solar system is chaotic, so exact positions can't be predicted by *any*
  model. That's physics, not a numerical artefact.
