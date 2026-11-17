"""Croatia-wide terrain & LiDAR raster analytics engine for BetterCroLidar."""

from __future__ import annotations

import base64
import csv
import io
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cruise import DMT20_PATH, load_dmt20_region
from view_lidar_3d import find_material_matches, load_materials

APP_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = APP_DIR / "sample_istra_sensor.csv"

OVERLAY_MODES: dict[str, dict[str, str]] = {
    "density": {"label": "Object Density / Anomaly", "default_cmap": "turbo"},
    "slope": {"label": "Terrain Slope (deg)", "default_cmap": "inferno"},
    "svf": {"label": "Sky-View Factor (SVF)", "default_cmap": "cividis"},
    "openness": {"label": "Terrain Openness", "default_cmap": "viridis"},
    "multi_hillshade": {"label": "Multi-Azimuth Hillshade", "default_cmap": "gray"},
    "lrm": {"label": "Local Relief Model (LRM)", "default_cmap": "coolwarm"},
    "elevation": {"label": "Elevation (m a.s.l.)", "default_cmap": "terrain"},
}

COLORMAPS = ["turbo", "viridis", "inferno", "magma", "cividis", "gray", "terrain", "coolwarm"]


def _compute_lrm(z_arr: np.ndarray, radius: int = 4) -> np.ndarray:
    """Compute a fast Local Relief Model (elevation minus local mean)."""
    from scipy.ndimage import uniform_filter

    valid = np.isfinite(z_arr)
    filled = np.where(valid, z_arr, 0.0)
    weights = valid.astype(np.float64)
    smooth_sum = uniform_filter(filled, size=radius * 2 + 1, mode="nearest")
    smooth_w = np.maximum(uniform_filter(weights, size=radius * 2 + 1, mode="nearest"), 1e-6)
    local_mean = smooth_sum / smooth_w
    lrm = np.where(valid, z_arr - local_mean, np.nan)
    return lrm


def _load_metric(
    bbox: tuple[float, float, float, float],
    mode: str,
    max_dim: int = 650,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load elevation and compute the requested terrain/LiDAR metric over bbox."""
    if not DMT20_PATH.exists():
        raise FileNotFoundError(f"DMT20 dataset not found at {DMT20_PATH}")

    if mode in ("lrm", "elevation"):
        _, _, z_arr, _, _ = load_dmt20_region(
            DMT20_PATH, bbox, max_dim=max_dim, color_by="density", window=5, hide_sea=True
        )
        if mode == "elevation":
            return z_arr, z_arr, "Elevation (m)"
        return z_arr, _compute_lrm(z_arr), "Local Relief Model (m)"

    _, _, z_arr, metric, label = load_dmt20_region(
        DMT20_PATH, bbox, max_dim=max_dim, color_by=mode, window=5, hide_sea=True
    )
    return z_arr, metric, label


def render_overlay(
    bbox: tuple[float, float, float, float],
    mode: str = "density",
    cmap_name: str | None = None,
    opacity: float = 0.65,
    max_dim: int = 650,
    material_target: float | None = None,
    material_tol: float = 0.5,
) -> dict[str, Any]:
    """Render a transparent PNG data-URI overlay and optional material matches for bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    mode = mode if mode in OVERLAY_MODES else "density"
    cmap_use = cmap_name if cmap_name in COLORMAPS else OVERLAY_MODES[mode]["default_cmap"]

    _, metric, scalar_label = _load_metric(bbox, mode=mode, max_dim=max_dim)
    valid = np.isfinite(metric)
    if not np.any(valid):
        return {
            "ok": False,
            "error": "No valid terrain pixels in selected region.",
            "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        }

    vmin = float(np.nanpercentile(metric, 2))
    vmax = float(np.nanpercentile(metric, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = np.clip((metric - vmin) / (vmax - vmin), 0.0, 1.0)
    cm = plt.get_cmap(cmap_use)
    rgba = cm(np.where(valid, norm, 0.0))
    rgba[..., 3] = np.where(valid, float(np.clip(opacity, 0.05, 1.0)), 0.0)

    img = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    palette = [mcolors.to_hex(cm(t)) for t in np.linspace(0.0, 1.0, 10)]

    matches: list[dict[str, float]] = []
    match_count = 0
    if material_target is not None:
        mask, match_count, _ = find_material_matches(
            metric, float(material_target), float(material_tol), mark_label=str(material_target)
        )
        if match_count > 0:
            h, w = metric.shape
            rows, cols = np.where(mask)
            if len(rows) > 400:
                step = int(np.ceil(len(rows) / 400))
                rows, cols = rows[::step], cols[::step]
            for r, c in zip(rows, cols):
                mlat = lat_max - (float(r) / max(h - 1, 1)) * (lat_max - lat_min)
                mlon = lon_min + (float(c) / max(w - 1, 1)) * (lon_max - lon_min)
                matches.append({"lat": round(mlat, 5), "lon": round(mlon, 5), "val": round(float(metric[r, c]), 3)})

    return {
        "ok": True,
        "image_data": f"data:image/png;base64,{b64}",
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "vmin": round(vmin, 3),
        "vmax": round(vmax, 3),
        "label": scalar_label,
        "mode": mode,
        "cmap": cmap_use,
        "palette": palette,
        "match_count": int(match_count),
        "matches": matches,
    }


def query_point(lat: float, lon: float) -> dict[str, Any]:
    """Query elevation, slope, density, and openness at any coordinate in Croatia."""
    pad = 0.008
    bbox = (lon - pad, lat - pad, lon + pad, lat + pad)
    try:
        _, _, z_arr, den_arr, _ = load_dmt20_region(
            DMT20_PATH, bbox, max_dim=48, color_by="density", window=5, hide_sea=False
        )
        _, _, _, slp_arr, _ = load_dmt20_region(
            DMT20_PATH, bbox, max_dim=48, color_by="slope", window=5, hide_sea=False
        )
        _, _, _, opn_arr, _ = load_dmt20_region(
            DMT20_PATH, bbox, max_dim=48, color_by="openness", window=5, hide_sea=False
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    h, w = z_arr.shape
    r, c = h // 2, w // 2
    elev = float(z_arr[r, c]) if np.isfinite(z_arr[r, c]) else None
    den = float(den_arr[r, c]) if np.isfinite(den_arr[r, c]) else None
    slp = float(slp_arr[r, c]) if np.isfinite(slp_arr[r, c]) else None
    opn = float(opn_arr[r, c]) if np.isfinite(opn_arr[r, c]) else None
    return {
        "ok": True,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "elevation_m": round(elev, 2) if elev is not None else None,
        "density": round(den, 3) if den is not None else None,
        "slope_deg": round(slp, 2) if slp is not None else None,
        "openness": round(opn, 3) if opn is not None else None,
    }


def profile_line(
    lat1: float, lon1: float, lat2: float, lon2: float, samples: int = 100
) -> dict[str, Any]:
    """Extract an elevation + density cross-section between two points anywhere in Croatia."""
    pad = 0.01
    bbox = (
        min(lon1, lon2) - pad,
        min(lat1, lat2) - pad,
        max(lon1, lon2) + pad,
        max(lat1, lat2) + pad,
    )
    lon_min, lat_min, lon_max, lat_max = bbox
    _, _, z_arr, den_arr, _ = load_dmt20_region(
        DMT20_PATH, bbox, max_dim=260, color_by="density", window=5, hide_sea=False
    )
    h, w = z_arr.shape

    # Haversine distance in meters
    r_earth = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    total_dist_m = 2 * r_earth * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    pts: list[dict[str, float | None]] = []
    for i in range(samples):
        t = i / max(samples - 1, 1)
        la = lat1 + t * (lat2 - lat1)
        lo = lon1 + t * (lon2 - lon1)
        rr = int(np.clip(round((lat_max - la) / max(lat_max - lat_min, 1e-9) * (h - 1)), 0, h - 1))
        cc = int(np.clip(round((lo - lon_min) / max(lon_max - lon_min, 1e-9) * (w - 1)), 0, w - 1))
        zv = float(z_arr[rr, cc]) if np.isfinite(z_arr[rr, cc]) else None
        dv = float(den_arr[rr, cc]) if np.isfinite(den_arr[rr, cc]) else None
        pts.append(
            {
                "dist_m": round(t * total_dist_m, 1),
                "elev_m": round(zv, 2) if zv is not None else None,
                "density": round(dv, 3) if dv is not None else None,
                "lat": round(la, 5),
                "lon": round(lo, 5),
            }
        )

    return {"ok": True, "total_dist_m": round(total_dist_m, 1), "points": pts}


def get_materials_list() -> list[dict[str, Any]]:
    """Return known material signatures from materials.json."""
    mats = load_materials()
    out: list[dict[str, Any]] = []
    for name, info in mats.items():
        if isinstance(info, dict) and "density" in info:
            out.append({"name": name, "density": float(info["density"]), "unit": info.get("unit", "g/cm3")})
    return out


def get_sample_sensor_points() -> list[dict[str, Any]]:
    """Load sample sensor CSV points if present."""
    pts: list[dict[str, Any]] = []
    if not SAMPLE_CSV.exists():
        return pts
    with SAMPLE_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pts.append(
                    {
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "depth": float(row["depth"]),
                        "density": float(row["density"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return pts
