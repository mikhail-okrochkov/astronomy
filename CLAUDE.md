# CLAUDE.md

This repo holds astronomical simulations: small, self-contained experiments
that are physically sound and look good enough to write about (blog posts,
articles, explainers).

## Goals

- **Correct physics first, pretty pictures second.** Each simulation should be
  checkable: conserved quantities, known analytic cases, or real data.
- **Explainable.** Every simulation should support a write-up: what question it
  answers, which assumptions it makes, and what the results show.
- **Reproducible.** Anyone with the repo should be able to rerun a simulation
  and get the same figures (fixed random seeds, pinned dependencies, no
  hand-edited outputs).

## Tooling

- Python 3.12+ managed with **uv** (`uv add <pkg>`, `uv run <script>`).
  Don't use bare `pip install`.
- Core stack: `numpy`, `scipy`, `matplotlib`, `astropy` (units, constants,
  coordinates, ephemerides). Add `numba` or `jax` only when performance
  actually needs it, and `rebound` for serious N-body work rather than
  writing a new integrator.
- Tests: `pytest` (`uv run pytest`). Lint/format: `ruff`
  (`uv run ruff check . && uv run ruff format .`).

## Layout (update this section as the repo grows)

```
src/physics/         shared physics library (installed package `physics`)
  gravity.py         Newtonian gravity by direct summation: accelerations,
                     field_at (test particles), potential_at, potential_energy
  units.py           UnitSystem + SI / SOLAR (AU, M☉, yr) / GALACTIC (kpc, M☉, Myr)
  integrators.py     fixed-step symplectic integrators: leapfrog, yoshida4
  kepler.py          orbital elements <-> state, exact two-body propagation
simulations/<name>/  one directory per simulation
  solar_system/      Sun + 8 planets from JPL Horizons; N-body vs Sun-only comparison
    horizons.py      cached JPL Horizons vector client (move to src/ when reused)
    data/            cached Horizons responses (committed for reproducibility)
  README.md          question, assumptions, how to run, results summary
  run.py             entry point that generates the outputs
  outputs/           generated figures/animations (gitignored unless published)
posts/<slug>/        drafts of write-ups, linking to simulation outputs
IDEAS.md             backlog of candidate simulations
tests/               tests for src/ and physics sanity checks
```

Gravity only computes forces; time integration belongs in a separate module.
Move simulation code into `src/` once a second simulation needs it, not before.

## Physics & numerical conventions

- **Units:** use `astropy.units` at the boundaries (inputs, outputs, plot
  labels). In hot loops, convert to plain floats in a documented unit system
  (e.g. AU / yr / M☉ with G = 4π², or SI) and say which in a comment.
- **Constants:** get them from `astropy.constants`. Don't hardcode values.
- **Integrators:** use symplectic integrators (leapfrog/Verlet, WHFast, IAS15
  via rebound) for orbital dynamics. Avoid plain Euler, and don't use RK4 for
  long-term orbits unless you discuss the drift.
- **Validation:** every simulation needs at least one sanity check: energy /
  angular momentum conservation, comparison to an analytic solution
  (Kepler orbit, virial theorem, etc.), or comparison to observed data.
  Record the check and its result in the simulation's README.
- **Randomness:** use `numpy.random.default_rng(seed)` with an explicit seed.
- Note approximations explicitly (softening length, neglected relativity,
  point masses, etc.). The write-up needs them.

## Figures & animations

- Produce figures from code only. Never edit outputs by hand.
- Label axes with units. Use colorblind-safe colormaps (`viridis`, `magma`,
  `cividis`) and a dark background where it suits the subject.
- Save publication figures as both PNG (≥150 dpi) and SVG/PDF. Animations go to
  MP4 (via ffmpeg) or GIF for short loops.
- Large outputs (videos, big data files) stay out of git. Commit only the
  final figures used in posts.

## Writing

- Drafts live in `posts/` as Markdown.
- Keep claims tied to what the simulation actually shows. Flag speculation and
  cite sources (papers, NASA/JPL data) for any real-world numbers.
- Explain for a curious non-specialist; put the math in an optional section.

## Workflow notes

- No GitHub remote yet. Git is local-only for now.
- Prefer small, runnable simulations over a large framework.
