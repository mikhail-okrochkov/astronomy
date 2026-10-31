"""Minimal client for the JPL Horizons API (state vectors only), with on-disk cache.

API docs: https://ssd-api.jpl.nasa.gov/doc/horizons.html

Vectors are heliocentric (center = Sun body center, ``500@10``), in the
ecliptic-and-equinox-of-J2000 frame, in AU and AU/day, times in TDB.
Responses are cached as JSON so reruns are offline and reproducible. Delete
the cache file to force a re-fetch.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

BASE_PARAMS = {
    "format": "json",
    "OBJ_DATA": "NO",
    "MAKE_EPHEM": "YES",
    "EPHEM_TYPE": "VECTORS",
    "CENTER": "500@10",
    "REF_PLANE": "ECLIPTIC",
    "REF_SYSTEM": "ICRF",
    "OUT_UNITS": "AU-D",
    "VEC_TABLE": "2",
    "CSV_FORMAT": "YES",
    "TIME_TYPE": "TDB",
}


def fetch_vectors(
    command: str,
    cache_dir: Path,
    *,
    tlist: list[float] | None = None,
    start: str | None = None,
    stop: str | None = None,
    step: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(jd_tdb, state)`` with ``state`` of shape ``(n, 6)`` = x, y, z, vx, vy, vz.

    Give either ``tlist`` (Julian dates, TDB) or ``start``/``stop``/``step``
    (Horizons time strings, e.g. ``"2000-01-01 12:00"``, ``"10 d"``).
    """
    params = dict(BASE_PARAMS, COMMAND=f"'{command}'")
    if tlist is not None:
        params["TLIST"] = " ".join(f"'{jd:.9f}'" for jd in tlist)
    else:
        params.update(START_TIME=f"'{start}'", STOP_TIME=f"'{stop}'", STEP_SIZE=f"'{step}'")

    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    cache_file = Path(cache_dir) / f"horizons_{command}_{key}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
    else:
        cached = {"params": params, **_query(params)}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cached, indent=1))
    return np.array(cached["jd_tdb"]), np.array(cached["state"])


def _query(params: dict[str, str]) -> dict:
    url = API_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)
    if "error" in payload:
        raise RuntimeError(f"Horizons error: {payload['error']}")
    result = payload["result"]
    target = next(
        (
            ln.split(":", 1)[1].split("{")[0].strip()
            for ln in result.splitlines()
            if ln.startswith("Target body name")
        ),
        "?",
    )
    jd, state = parse_vectors(result)
    return {"target": target, "source": payload.get("signature"), "jd_tdb": jd, "state": state}


def parse_vectors(result: str) -> tuple[list[float], list[list[float]]]:
    """Parse the CSV block between ``$$SOE`` and ``$$EOE`` of a VEC_TABLE=2 result."""
    try:
        body = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    except IndexError as exc:
        raise RuntimeError(f"No ephemeris block in Horizons result:\n{result[:2000]}") from exc
    jd, state = [], []
    for line in body.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        # JDTDB, calendar date, X, Y, Z, VX, VY, VZ, (trailing empty)
        jd.append(float(fields[0]))
        state.append([float(f) for f in fields[2:8]])
    return jd, state
