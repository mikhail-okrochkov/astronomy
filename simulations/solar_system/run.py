"""Solar system simulation started from real JPL ephemerides.

Two models are run and compared against the real JPL ephemeris:

- ``nbody`` (default): the Sun and eight planets all attract each other.
- ``kepler``: each planet feels only the Sun, so every orbit is a fixed Kepler
  ellipse. The gap between the two shows how much the planets pull on each other.

Usage:
    uv run simulations/solar_system/run.py                    # N-body, J2000, 25 years
    uv run simulations/solar_system/run.py --epoch 2026-10-31 --years 50
    uv run simulations/solar_system/run.py --model kepler     # Sun-only as primary
    uv run simulations/solar_system/run.py --animate          # also write a GIF

Model and units
---------------
- Ecliptic and equinox of J2000; lengths in AU, time in days (TDB). Results are
  reported heliocentric (relative to the Sun), like the JPL reference.
- ``nbody``: 9 bodies integrated in the barycentric frame (initial states
  shifted so the total momentum is zero), Newtonian gravity between all pairs.
- ``kepler``: heliocentric; planet i feels a = -mu_i r / |r|^3 with
  mu_i = GM_sun + GM_i, the exact relative motion of an isolated Sun-planet pair.
- "Mass" arguments to physics.gravity are GM values, so G = 1.
- Bodies are planetary-system barycenters (planet + moons). For Earth that is
  the Earth-Moon barycenter, which follows a smooth orbit; Earth itself
  wobbles around it monthly.
- Integrator: 4th-order symplectic (Yoshida) with a fixed step. Its error is
  estimated by rerunning at 2*dt: error(dt) ~ |x(dt) - x(2 dt)| / 15.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.time import Time
from horizons import fetch_vectors
from matplotlib.animation import FuncAnimation, PillowWriter

from physics.gravity import accelerations, field_at, potential_energy
from physics.integrators import integrate
from physics.kepler import OrbitalElements, elements_to_state, state_to_elements

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"

GM_UNIT = u.km**3 / u.s**2
SIM_GM_UNIT = u.AU**3 / u.day**2
AU_KM = (1 * u.AU).to_value(u.km)

# GM values (km^3/s^2) from the JPL DE440 ephemeris: Park et al. 2021, AJ 161, 105.
GM_SUN = (132712440041.279419 * GM_UNIT).to_value(SIM_GM_UNIT)


@dataclass(frozen=True)
class Body:
    name: str
    horizons_id: str  # system barycenter
    gm_km3_s2: float  # DE440, planet + moons
    color: str
    # J2000 *mean* elements (a [AU], e, i [deg]) from Standish, "Keplerian
    # Elements for Approximate Positions of the Major Planets" (JPL SSD).
    # Used as an independent sanity check of the osculating elements we derive.
    ref_a: float
    ref_e: float
    ref_i: float

    @property
    def gm(self) -> float:
        return (self.gm_km3_s2 * GM_UNIT).to_value(SIM_GM_UNIT)


BODIES = [
    Body("Mercury", "1", 22031.868551, "#b9b4ad", 0.38709927, 0.20563593, 7.00497902),
    Body("Venus", "2", 324858.592000, "#e8c27a", 0.72333566, 0.00677672, 3.39467605),
    Body("Earth", "3", 403503.235625, "#4f9df7", 1.00000261, 0.01671123, -0.00001531),
    Body("Mars", "4", 42828.375816, "#e0603a", 1.52371034, 0.09339410, 1.84969142),
    Body("Jupiter", "5", 126712764.100000, "#d9a066", 5.20288700, 0.04838624, 1.30439695),
    Body("Saturn", "6", 37940584.841800, "#e9d58f", 9.53667594, 0.05386179, 2.48599187),
    Body("Uranus", "7", 5794556.400000, "#8fd8e0", 19.18916464, 0.04725744, 0.77263783),
    Body("Neptune", "8", 6836527.100580, "#5a7cf2", 30.06992276, 0.00859048, 1.77004347),
]
INNER = 4  # the first four bodies are the inner planets


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def initial_state(epoch_jd: float) -> tuple[np.ndarray, np.ndarray]:
    states = [fetch_vectors(b.horizons_id, DATA_DIR, tlist=[epoch_jd])[1][0] for b in BODIES]
    states = np.array(states)
    return states[:, :3], states[:, 3:]


def reference_positions(epoch_jd: float, days: float, step_days: int):
    """JPL positions at ``epoch + k * step_days``: returns (jd, pos[n_t, n_bodies, 3])."""
    jds, positions = None, []
    for b in BODIES:
        jd, state = fetch_vectors(
            b.horizons_id,
            DATA_DIR,
            start=f"JD {epoch_jd:.6f}",
            stop=f"JD {epoch_jd + days:.6f}",
            step=f"{step_days} d",
        )
        jds = jd if jds is None else jds
        if not np.array_equal(jd, jds):
            raise RuntimeError(f"Horizons returned mismatched times for {b.name}")
        positions.append(state[:, :3])
    return jds, np.stack(positions, axis=1)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


MODELS = ("nbody", "kepler")
MODEL_LABELS = {"nbody": "N-body", "kepler": "Sun-only"}
# Error ratio between steps 2*dt and dt for a 4th-order method is 2**4 = 16.
RICHARDSON_FACTOR = 2**4 - 1


class Run(NamedTuple):
    """Heliocentric planet trajectory, shape (n_t, n_planets, 3), plus diagnostics."""

    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    energy_drift: float  # max |E/E0 - 1| over the run (worst planet for kepler)


def simulate(model: str, r0, v0, *, dt: float, n_steps: int, save_every: int) -> Run:
    """Integrate from heliocentric planet states ``r0``, ``v0`` (each (n_planets, 3))."""
    gm = np.array([b.gm for b in BODIES])

    if model == "kepler":
        mu = GM_SUN + gm
        sun_pos, sun_gm = np.zeros((1, 3)), np.array([GM_SUN])
        scale = (mu / GM_SUN)[:, None]  # (GM_sun + GM_i) / GM_sun

        def accel(x):
            return field_at(x, sun_pos, sun_gm, G=1.0) * scale

        traj = integrate(r0, v0, accel, dt=dt, n_steps=n_steps, save_every=save_every)
        energy = 0.5 * (traj.vel**2).sum(-1) - mu / np.linalg.norm(traj.pos, axis=-1)
        drift = float(np.abs(energy / energy[0] - 1).max())
        return Run(traj.t, traj.pos, traj.vel, drift)

    if model == "nbody":
        masses = np.concatenate([[GM_SUN], gm])
        pos = np.vstack([np.zeros(3), r0])  # Sun first, at the heliocentric origin
        vel = np.vstack([np.zeros(3), v0])
        pos -= masses @ pos / masses.sum()  # move to the barycentric frame
        vel -= masses @ vel / masses.sum()

        def accel(x):
            return accelerations(x, masses, G=1.0)

        traj = integrate(pos, vel, accel, dt=dt, n_steps=n_steps, save_every=save_every)
        kinetic = 0.5 * np.einsum("b,tbi,tbi->t", masses, traj.vel, traj.vel)
        potential = np.array([potential_energy(x, masses, G=1.0) for x in traj.pos])
        energy = kinetic + potential
        drift = float(np.abs(energy / energy[0] - 1).max())
        helio_pos = traj.pos[:, 1:] - traj.pos[:, :1]
        helio_vel = traj.vel[:, 1:] - traj.vel[:, :1]
        return Run(traj.t, helio_pos, helio_vel, drift)

    raise ValueError(f"unknown model {model!r}")


def orbit_paths(elements: OrbitalElements, mu: np.ndarray, n: int = 720) -> np.ndarray:
    """Full orbit ellipse for each body; shape (n_bodies, n, 3)."""
    m = np.linspace(0, 2 * np.pi, n)
    paths = []
    for i in range(len(mu)):
        el = OrbitalElements(*(np.full(n, x[i]) for x in elements[:5]), m)
        paths.append(elements_to_state(el, mu[i])[0])
    return np.array(paths)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

plt.style.use("dark_background")
plt.rcParams.update({"font.size": 10, "axes.facecolor": "#05060a", "figure.facecolor": "#05060a"})


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def sun(ax, size=120, **kw):
    ax.scatter(0, 0, s=size, c="#ffd75e", edgecolors="#ffef9e", linewidths=0.5, zorder=5, **kw)


def plot_orbits(paths, r0, epoch_label):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, idx, lim, title in (
        (axes[0], range(INNER), 1.75, "Inner planets"),
        (axes[1], range(len(BODIES)), 32.5, "All planets"),
    ):
        for i in idx:
            b = BODIES[i]
            ax.plot(paths[i, :, 0], paths[i, :, 1], color=b.color, lw=1.0, alpha=0.75)
            ax.scatter(*r0[i, :2], s=30, color=b.color, zorder=6)
            if lim < 5 or i >= INNER:
                ax.annotate(
                    b.name,
                    r0[i, :2],
                    xytext=(6, 4),
                    textcoords="offset points",
                    color=b.color,
                    fontsize=9,
                )
        sun(ax, size=80 if lim < 5 else 30)
        ax.set(
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            aspect="equal",
            title=title,
            xlabel="x [AU] (toward vernal equinox)",
            ylabel="y [AU]",
        )
        ax.grid(alpha=0.12)
    fig.suptitle(f"Planet positions at {epoch_label}, ecliptic J2000 frame (top-down)", y=0.98)
    save(fig, "orbits")


def plot_orbits_3d(paths, r0, epoch_label, z_exaggeration):
    fig = plt.figure(figsize=(14, 6.5))
    for k, (idx, lim, title) in enumerate(
        (
            (range(INNER), 1.7, "Inner planets"),
            (range(INNER, len(BODIES)), 31.0, "Outer planets"),
        )
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.set_facecolor("#05060a")
        # Faint ecliptic-plane grid for reference.
        g = np.linspace(-lim, lim, 9)
        for c in g:
            ax.plot([c, c], [-lim, lim], 0, color="w", alpha=0.06, lw=0.6)
            ax.plot([-lim, lim], [c, c], 0, color="w", alpha=0.06, lw=0.6)
        for i in idx:
            b = BODIES[i]
            p = paths[i]
            above = p[:, 2] >= 0
            z = p[:, 2] * z_exaggeration
            ax.plot(p[:, 0], p[:, 1], np.where(above, z, np.nan), color=b.color, lw=1.3)
            ax.plot(
                p[:, 0],
                p[:, 1],
                np.where(~above, z, np.nan),
                color=b.color,
                lw=1.0,
                alpha=0.35,
                ls="--",
            )
            ax.scatter(*r0[i, :2], r0[i, 2] * z_exaggeration, s=25, color=b.color)
            ax.text(
                *r0[i, :2],
                r0[i, 2] * z_exaggeration + 0.06 * lim,
                b.name,
                color=b.color,
                fontsize=8,
            )
        ax.scatter(0, 0, 0, s=60, c="#ffd75e")
        ax.set(
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            zlim=(-0.35 * lim, 0.35 * lim),
            title=title,
            xlabel="x [AU]",
            ylabel="y [AU]",
        )
        ax.set_zlabel(f"z [AU] ×{z_exaggeration:g}")
        ax.set_box_aspect((1, 1, 0.35))
        ax.view_init(elev=14, azim=-60)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0.1))
    fig.suptitle(
        f"Orbit inclinations relative to the ecliptic at {epoch_label} "
        f"(vertical scale exaggerated ×{z_exaggeration:g}; dashed = below ecliptic)",
        y=0.95,
    )
    save(fig, "orbits_3d")


def plot_errors(years, err_km, int_err_km, primary, years_total):
    """Error vs JPL for both models; dotted = integrator-error estimate (primary model)."""
    from matplotlib.lines import Line2D

    other = next(m for m in MODELS if m != primary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, idx, title in (
        (axes[0], range(INNER), "Inner planets"),
        (axes[1], range(INNER, len(BODIES)), "Outer planets"),
    ):
        for i in idx:
            b = BODIES[i]
            ax.semilogy(years[1:], err_km[primary][1:, i], color=b.color, lw=1.5, label=b.name)
            ax.semilogy(years[1:], err_km[other][1:, i], color=b.color, lw=1.0, alpha=0.45, ls="--")
            ax.semilogy(
                years[1:],
                np.maximum(int_err_km[1:, i], 1e-6),
                color=b.color,
                lw=0.8,
                ls=":",
                alpha=0.8,
            )
        ax.axhline(384_400, color="w", lw=0.6, alpha=0.4)
        ax.text(0.2, 384_400 * 1.25, "Earth–Moon distance", fontsize=8, alpha=0.6)
        ax.set(title=title, xlabel="years since epoch", xlim=(0, years_total))
        ax.grid(alpha=0.12, which="both")
        planets = ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.add_artist(planets)
    styles = [
        Line2D([], [], color="w", lw=1.5, label=f"{MODEL_LABELS[primary]} vs JPL"),
        Line2D(
            [], [], color="w", lw=1.0, ls="--", alpha=0.45, label=f"{MODEL_LABELS[other]} vs JPL"
        ),
        Line2D([], [], color="w", lw=0.8, ls=":", label="integrator error (estimate)"),
    ]
    axes[0].legend(handles=styles, loc="lower left", fontsize=8, frameon=False)
    axes[0].set_ylabel("position error [km]")
    fig.suptitle(
        f"{MODEL_LABELS['nbody']} vs {MODEL_LABELS['kepler']} gravity, compared with the "
        "JPL ephemeris (DE441)",
        y=0.98,
    )
    save(fig, "error_vs_jpl")


def animate_inner(model, r0, v0, dt, epoch_jd, days=730, frame_days=2):
    traj = simulate(
        model, r0, v0, dt=dt, n_steps=round(days / dt), save_every=round(frame_days / dt)
    )
    pos = traj.pos[:, :INNER]
    trail = 45  # frames

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    lim = 1.75
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    ax.set_axis_off()
    sun(ax, size=160)
    lines = [ax.plot([], [], color=BODIES[i].color, lw=1.4, alpha=0.8)[0] for i in range(INNER)]
    dots = [ax.plot([], [], "o", ms=6, color=BODIES[i].color)[0] for i in range(INNER)]
    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=10, alpha=0.85)

    def update(f):
        lo = max(0, f - trail)
        for i in range(INNER):
            lines[i].set_data(pos[lo : f + 1, i, 0], pos[lo : f + 1, i, 1])
            dots[i].set_data([pos[f, i, 0]], [pos[f, i, 1]])
        label.set_text(Time(epoch_jd + traj.t[f], format="jd", scale="tdb").strftime("%Y-%m-%d"))
        return [*lines, *dots, label]

    anim = FuncAnimation(fig, update, frames=len(pos), blit=True)
    anim.save(OUT_DIR / "inner_planets.gif", writer=PillowWriter(fps=30), dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--epoch", default="2000-01-01 12:00", help="start time, TDB (default J2000)"
    )
    parser.add_argument("--years", type=float, default=25.0, help="duration to simulate")
    parser.add_argument("--model", choices=MODELS, default="nbody", help="primary model")
    parser.add_argument("--dt", type=float, default=0.0625, help="time step [days]")
    parser.add_argument("--sample-days", type=int, default=10, help="comparison interval [days]")
    parser.add_argument("--z-exaggeration", type=float, default=5.0)
    parser.add_argument("--animate", action="store_true", help="write inner-planet GIF")
    args = parser.parse_args()

    save_every = round(args.sample_days / args.dt)
    if save_every % 2 or not np.isclose(save_every * args.dt, args.sample_days):
        parser.error("--sample-days must be an even multiple of --dt")
    OUT_DIR.mkdir(exist_ok=True)

    epoch = Time(args.epoch, scale="tdb")
    epoch_jd = float(epoch.jd)
    epoch_label = epoch.strftime("%Y-%m-%d")
    n_samples = int(args.years * 365.25 // args.sample_days)
    days = n_samples * args.sample_days
    n_steps = round(days / args.dt)

    print(f"Fetching JPL Horizons states for {epoch_label} (cached in {DATA_DIR}) ...")
    r0, v0 = initial_state(epoch_jd)
    ref_jd, ref_pos = reference_positions(epoch_jd, days, args.sample_days)

    runs = {}
    for model in MODELS:
        print(
            f"Integrating {MODEL_LABELS[model]} model for {days / 365.25:.1f} years "
            f"({n_steps} steps of {args.dt} d) ..."
        )
        runs[model] = simulate(model, r0, v0, dt=args.dt, n_steps=n_steps, save_every=save_every)
        if not np.allclose(runs[model].t, ref_jd - epoch_jd):
            raise RuntimeError("simulation and reference times do not line up")
    primary = runs[args.model]

    print("Estimating integrator error (rerun at 2*dt) ...")
    coarse = simulate(
        args.model, r0, v0, dt=2 * args.dt, n_steps=n_steps // 2, save_every=save_every // 2
    )
    int_err_km = np.linalg.norm(primary.pos - coarse.pos, axis=-1) / RICHARDSON_FACTOR * AU_KM

    err_km, ang_arcsec = {}, {}
    for model, run in runs.items():
        err_km[model] = np.linalg.norm(run.pos - ref_pos, axis=-1) * AU_KM
        # Angle between simulated and true direction as seen from the Sun.
        cosang = np.einsum("tbi,tbi->tb", run.pos, ref_pos) / (
            np.linalg.norm(run.pos, axis=-1) * np.linalg.norm(ref_pos, axis=-1)
        )
        ang_arcsec[model] = np.degrees(np.arccos(np.clip(cosang, -1, 1))) * 3600

    mu = GM_SUN + np.array([b.gm for b in BODIES])
    el = state_to_elements(r0, v0, mu)
    years = primary.t / 365.25

    print("Writing figures ...")
    paths = orbit_paths(el, mu)
    plot_orbits(paths, r0, epoch_label)
    plot_orbits_3d(paths, r0, epoch_label, args.z_exaggeration)
    plot_errors(years, err_km, int_err_km, args.model, days / 365.25)
    if args.animate:
        print("Rendering animation ...")
        animate_inner(args.model, r0, v0, args.dt, epoch_jd)

    nb, kp = "nbody", "kepler"
    lines = [
        f"# Solar system: results from {epoch_label} over {days / 365.25:.1f} years",
        "",
        f"Primary model: {MODEL_LABELS[args.model]}. Integrator: yoshida4, dt = {args.dt} d. "
        "Reference: JPL Horizons (DE441).",
        f"Max relative energy drift: N-body {runs[nb].energy_drift:.1e}, "
        f"Sun-only {runs[kp].energy_drift:.1e}.",
        "Osculating elements at epoch vs Standish J2000 mean elements (expected to agree to",
        "roughly 1e-3 in a and e and ~0.01 deg in i, since osculating elements oscillate).",
        "Errors are at the end of the run; integrator error is for the primary model.",
        "",
        "| Planet | a [AU] | a (ref) | e | e (ref) | i [deg] | i (ref) "
        "| N-body error [km] | N-body angle [arcsec] "
        "| Sun-only error [km] | Sun-only angle [arcsec] | integrator error [km] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, b in enumerate(BODIES):
        lines.append(
            f"| {b.name} | {el.a[i]:.5f} | {b.ref_a:.5f} | {el.e[i]:.5f} | {b.ref_e:.5f} "
            f"| {np.degrees(el.inc[i]):.4f} | {b.ref_i:.4f} "
            f"| {err_km[nb][-1, i]:,.0f} | {ang_arcsec[nb][-1, i]:.2f} "
            f"| {err_km[kp][-1, i]:,.0f} | {ang_arcsec[kp][-1, i]:.0f} "
            f"| {int_err_km[-1, i]:.2g} |"
        )
    (OUT_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
