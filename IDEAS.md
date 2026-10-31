# Simulation ideas

Candidates that are interesting physics *and* produce good visuals. Each lists
the hook, the visual, a validation check, and rough difficulty.

## Three-body problem ⭐

The cleanest way to show deterministic chaos: three point masses, Newton's
law, no closed-form solution. One project can feed several posts.

- **Periodic choreographies.** The figure-eight (Chenciner–Montgomery 2000),
  Lagrange's rotating equilateral triangle, Euler's collinear solution, and
  the large catalogue of periodic orbits found numerically (Šuvakov & Dmitrašinović
  2013: "butterfly", "moth", "yin-yang", etc.).
  *Visual:* glowing, fading trails; a gallery grid of choreographies.
- **Chaos and sensitivity.** Run the figure-eight, nudge one body by 1e-6, and
  overlay the runs. Then fan out ~100 slightly perturbed copies and watch them
  split apart. Measure the Lyapunov time (how fast tiny differences grow).
  *Visual:* fans of colour-coded diverging trails.
- **Pythagorean problem (Burrau 1913).** Masses 3, 4, 5 start at rest on the
  corners of a 3-4-5 triangle. After a long chaotic dance, one body is
  ejected and the other two leave as a binary. A famous hard test case.
  *Visual:* the dance, then the escape.
- **Stability / escape-time maps.** Scan a 2-D grid of initial conditions
  (e.g. one body's starting position) and colour each pixel by time until a
  body escapes. Gives fractal, basin-like images.
  *Visual:* the best-looking output of this project; poster-quality.
- **Close encounters & binary–single scattering.** A star flies past a
  binary: capture, exchange or ionisation. Ties into how star clusters evolve.
- **Shape-sphere view.** Plot each configuration's triangle shape on the
  "shape sphere" (Montgomery). An unusual, elegant way to visualise the motion.

*Check:* energy and angular momentum conservation; known periods of the
catalogued periodic orbits; equilateral/collinear solutions should match
Lagrange's/Euler's analytic results.
*Numerics:* close encounters need adaptive steps or regularisation. Use
IAS15 (rebound) or an adaptive high-order integrator, not fixed-step leapfrog.
*Difficulty:* easy to start; the escape-time maps are medium (many runs, so
they need vectorizing or parallelism).

## Orbital dynamics

- **Lagrange points, horseshoe & tadpole orbits.** Seen in the frame rotating
  with the planet; Janus/Epimetheus swap orbits every 4 years.
  *Check:* the Jacobi constant. *Easy; a good first project.*
- **Kirkwood gaps.** Thousands of test asteroids plus Jupiter. Resonances
  (3:1, 5:2, 2:1) clear gaps over millions of years.
  *Check:* compare with the Minor Planet Center catalogue. *Medium; rebound.*
- **Ring systems & shepherd moons.** Sharp edges, gaps, wavy edges (Encke gap).
  *Medium–hard.*

## Galaxies

- **Galaxy collisions, tidal tails (Toomre & Toomre 1972).** Antennae, Mice.
  Restricted version (only the two cores attract; stars are tracers) runs on a
  laptop. *Medium; visually stunning.*
- **Spiral arms as density waves.** Stars move *through* the arms.
  *Medium; strong explainer.*
- **Galaxy rotation curves and dark matter.** The disk flies apart without a
  halo. *Easy.*

## Stars & compact objects

- **Black hole ray tracing (Schwarzschild).** Glowing gas disk, photon ring,
  lopsided brightness from the disk's motion.
  *Check:* shadow size √27 GM/c². *Medium–hard; showpiece.*
- **Tidal disruption event.** A star made of particles is torn into a stream.
  *Hard.*
- **Binary pulsar / gravitational-wave inspiral.** Approximate GR energy-loss
  formula; compare with the measured decay of Hulse–Taylor; make the chirp
  audible. *Easy–medium.*

## Planets & formation

- **Grand Tack / Nice model.** Giant-planet migration. *Hard.*
- **Exoplanet light curves with rings/moons.** Compare with Kepler/TESS data.
  *Easy.*
