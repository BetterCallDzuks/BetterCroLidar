#!/usr/bin/env python3
"""
Interactive 3D LiDAR & DTM/DEM Viewer.

Displays elevation in 3D (Z-axis) and colors the surface/points by object/point
density (or slope/elevation) using a cold-to-warm colormap (blue = less dense,
yellow/orange/red = denser).

Supported inputs:
  - Point clouds: .las, .laz (uses local 2D spatial point density)
  - Elevation rasters: .hgt, .tif, .tiff (uses local roughness/object-density
    proxy, or (DSM - DTM) object height/density if --dsm is also provided)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

MATERIALS_FILE = Path(__file__).resolve().parent / "materials.json"


def load_materials() -> dict[str, float]:
    """Load saved material density/target values from materials.json."""
    if not MATERIALS_FILE.exists():
        return {}
    try:
        with MATERIALS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            str(k).lower(): float(v)
            for k, v in data.items()
            if np.isfinite(float(v))
        }
    except Exception:
        return {}


def save_material(name: str, value: float) -> None:
    """Save or update a material target value in materials.json."""
    val = float(value)
    if not np.isfinite(val):
        raise ValueError(f"Invalid non-finite material value: {value}")
    materials = load_materials()
    materials[name.lower()] = val
    MATERIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MATERIALS_FILE.open("w", encoding="utf-8") as f:
        json.dump(materials, f, indent=2, sort_keys=True)
        f.write("\n")


def find_material_matches(
    metric: np.ndarray,
    mark_value: float,
    mark_tolerance: float,
    mark_label: str | None = None,
) -> tuple[np.ndarray, int, str]:
    """Return boolean mask, match count, and display label for target value."""
    metric = np.asarray(metric, dtype=np.float64)
    mask = np.isfinite(metric) & (np.abs(metric - mark_value) <= mark_tolerance)
    count = int(np.sum(mask))
    label = mark_label or f"Target ({mark_value:.2f})"
    print(f"Material match [{label} = {mark_value} +/- {mark_tolerance}]: {count} matches")
    return mask, count, label


def warn_material_ambiguity(
    mark_material: str | None,
    mark_value: float,
    mark_tolerance: float,
    scalar_label: str,
) -> None:
    """Warn about overlapping materials or non-physical metrics when matching."""
    if "sensor" not in scalar_label.lower() and "g/cm3" not in scalar_label.lower():
        print(
            f"Note: Active metric is '{scalar_label}', not physical material density. "
            "Matches reflect terrain/point geometry values, not chemical composition.",
            file=sys.stderr,
        )

    materials = load_materials()
    selected = mark_material.lower() if mark_material else None
    overlaps = [
        f"{name} ({val:.2f})"
        for name, val in sorted(materials.items())
        if name != selected and abs(val - mark_value) <= mark_tolerance
    ]
    if overlaps:
        print(
            f"Warning: Target {mark_value:.2f} +/- {mark_tolerance} also overlaps with: "
            + ", ".join(overlaps),
            file=sys.stderr,
        )


COLORMAPS = {
    "turbo": "Turbo",
    "coolwarm": "RdBu_r",
    "jet": "Jet",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "viridis": "Viridis",
    "terrain": "earth",
}

DETAIL_MAX_DIM = {
    "low": 300,
    "medium": 600,
    "high": 1000,
    "ultra": 1500,
    "native": 2500,
}


def compute_point_density(
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float | None = None,
) -> np.ndarray:
    """Compute per-point 2D spatial density (points per grid cell)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(len(x), len(y))
    if n == 0:
        return np.empty(0, dtype=np.float32)
    x = x[:n]
    y = y[:n]

    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return np.zeros(n, dtype=np.float32)

    x_valid = x[valid]
    y_valid = y[valid]
    x_min, x_max = float(x_valid.min()), float(x_valid.max())
    y_min, y_max = float(y_valid.min()), float(y_valid.max())

    try:
        cell_size_val = float(cell_size) if cell_size is not None else None
        if cell_size_val is not None and (not np.isfinite(cell_size_val) or cell_size_val <= 0):
            cell_size_val = None
    except (TypeError, ValueError):
        cell_size_val = None

    if cell_size_val is None:
        span = max(x_max - x_min, y_max - y_min, 1e-6)
        cell_size = max(span / 200.0, 1.0)
    else:
        cell_size = max(cell_size_val, 0.1)

    nx = max(int(np.ceil((x_max - x_min) / cell_size)), 1)
    ny = max(int(np.ceil((y_max - y_min) / cell_size)), 1)

    ix = np.clip(((x_valid - x_min) / cell_size).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y_valid - y_min) / cell_size).astype(np.int64), 0, ny - 1)

    grid = np.zeros((nx, ny), dtype=np.float32)
    np.add.at(grid, (ix, iy), 1.0)
    density = np.zeros(n, dtype=np.float32)
    density[valid] = grid[ix, iy]
    return density


def compute_raster_metric(
    z: np.ndarray,
    color_by: str,
    window: int = 5,
    dsm: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Compute scalar coloring metric (density, slope, elevation, lrm, hillshade, svf) for a 2D grid."""
    from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter

    z = np.asarray(z, dtype=np.float64)
    color_by = color_by.lower()

    label_map = {
        "elevation": "Elevation (m)",
        "slope": "Slope (deg)",
        "lrm": "Local Relief Model (m)",
        "hillshade": "Hillshade Relief (0-1)",
        "svf": "Sky-View Factor (0-1)",
        "multi_hillshade": "Multi-Directional Hillshade (0-1)",
        "openness": "Topographic Openness Diff (deg)",
        "density": "Object Height Density (m)" if dsm is not None else "Object Density (Cold=Low, Warm=High)",
    }

    if z.size == 0 or z.ndim != 2:
        return z.copy(), label_map.get(color_by, "Object Density (Cold=Low, Warm=High)")

    if color_by in {"slope", "hillshade", "multi_hillshade"} and min(z.shape) < 2:
        return np.zeros_like(z), label_map.get(color_by, "Object Density (Cold=Low, Warm=High)")

    if dsm is not None and color_by == "density":
        dsm = np.asarray(dsm, dtype=np.float64)
        if dsm.shape == z.shape:
            obj_height = np.clip(dsm - z, 0.0, None)
            metric = uniform_filter(obj_height, size=window)
            return metric, "Object Height Density (m)"

    if color_by == "elevation":
        return z.copy(), "Elevation (m)"

    if color_by == "slope":
        gy, gx = np.gradient(z)
        slope_deg = np.rad2deg(np.arctan(np.hypot(gx, gy)))
        return slope_deg, "Slope (deg)"

    if color_by == "lrm":
        sigma = max(float(window) * 1.5, 2.0)
        trend = gaussian_filter(z, sigma=sigma)
        return z - trend, "Local Relief Model (m)"

    if color_by == "hillshade":
        gy, gx = np.gradient(z)
        slope = np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        azimuth = np.deg2rad(315.0)
        altitude = np.deg2rad(45.0)
        hs = (
            np.sin(altitude) * np.cos(slope)
            + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
        )
        return np.clip(hs, 0.0, 1.0), "Hillshade Relief (0-1)"

    if color_by == "svf":
        # Fast local Sky-View Factor / positive openness proxy
        radius = max(int(window), 3)
        local_max = maximum_filter(z, size=radius * 2 + 1)
        relief = np.clip(local_max - z, 0.0, None)
        horizon_angle = np.arctan(relief / max(float(radius), 1.0))
        svf = 1.0 - np.sin(horizon_angle)
        return np.clip(svf, 0.0, 1.0), "Sky-View Factor (0-1)"

    if color_by == "multi_hillshade":
        gy, gx = np.gradient(z)
        slope = np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        altitude = np.deg2rad(45.0)
        acc = np.zeros_like(z, dtype=np.float64)
        for az_deg in (315.0, 45.0, 135.0, 225.0):
            az = np.deg2rad(az_deg)
            acc += (
                np.sin(altitude) * np.cos(slope)
                + np.cos(altitude) * np.sin(slope) * np.cos(az - aspect)
            )
        return np.clip(acc / 4.0, 0.0, 1.0), "Multi-Directional Hillshade (0-1)"

    if color_by == "openness":
        from scipy.ndimage import minimum_filter

        radius = max(int(window), 3)
        local_max = maximum_filter(z, size=radius * 2 + 1)
        local_min = minimum_filter(z, size=radius * 2 + 1)
        pos_open = 90.0 - np.rad2deg(np.arctan(np.clip(local_max - z, 0.0, None) / max(float(radius), 1.0)))
        neg_open = 90.0 - np.rad2deg(np.arctan(np.clip(z - local_min, 0.0, None) / max(float(radius), 1.0)))
        return pos_open - neg_open, "Topographic Openness Diff (deg)"

    # Default: local terrain/object density proxy (local elevation std-dev)
    local_mean = uniform_filter(z, size=window)
    local_sq = uniform_filter(z**2, size=window)
    density = np.sqrt(np.clip(local_sq - local_mean**2, 0.0, None))
    return density, "Object Density (Cold=Low, Warm=High)"


def cluster_material_matches(
    metric: np.ndarray,
    z: np.ndarray,
    mark_value: float,
    mark_tolerance: float,
    bbox_lonlat: tuple[float, float, float, float] | None = None,
    min_pixels: int = 1,
    min_depth: float | None = None,
    max_depth: float | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Group matching points into discrete target clusters with center coordinates & confidence %."""
    from scipy.ndimage import label as nd_label

    metric = np.asarray(metric, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if x is not None:
        x = np.asarray(x, dtype=np.float64)
    if y is not None:
        y = np.asarray(y, dtype=np.float64)

    if metric.ndim == 1:
        z = z.ravel()
        n_pts = min(len(metric), len(z))
        if x is not None:
            x = x.ravel()
            n_pts = min(n_pts, len(x))
        if y is not None:
            y = y.ravel()
            n_pts = min(n_pts, len(y))
        metric = metric[:n_pts]
        z = z[:n_pts]
        if x is not None:
            x = x[:n_pts]
        if y is not None:
            y = y[:n_pts]
    elif metric.ndim == 2:
        if metric.shape != z.shape:
            return []
        if x is not None and x.shape != metric.shape:
            x = None
        if y is not None and y.shape != metric.shape:
            y = None
    else:
        return []

    try:
        tol_val = float(mark_tolerance)
        mark_tolerance = tol_val if np.isfinite(tol_val) and tol_val >= 0 else 0.5
    except (TypeError, ValueError):
        mark_tolerance = 0.5

    if min_depth is not None:
        try:
            dmin_val = float(min_depth)
            min_depth = dmin_val if np.isfinite(dmin_val) and dmin_val >= 0 else None
        except (TypeError, ValueError):
            min_depth = None

    if max_depth is not None:
        try:
            dmax_val = float(max_depth)
            max_depth = dmax_val if np.isfinite(dmax_val) and dmax_val >= 0 else None
        except (TypeError, ValueError):
            max_depth = None

    mask = np.isfinite(metric) & (np.abs(metric - mark_value) <= mark_tolerance) & np.isfinite(z)
    if x is not None:
        mask &= np.isfinite(x)
    if y is not None:
        mask &= np.isfinite(y)
    if min_depth is not None:
        mask &= (np.abs(z) >= min_depth)
    if max_depth is not None:
        mask &= (np.abs(z) <= max_depth)
    if not np.any(mask):
        return []

    def _conf(mean_val: float) -> float:
        score = 100.0 * (1.0 - min(abs(mean_val - mark_value) / max(mark_tolerance, 1e-6), 1.0))
        return round(max(score, 0.0), 1)

    clusters: list[dict[str, float | int]] = []
    if metric.ndim == 2:
        labeled, num = nd_label(mask)
        h, w = metric.shape
        for cid in range(1, num + 1):
            pts = labeled == cid
            count = int(np.sum(pts))
            if count < min_pixels:
                continue
            rs, cs = np.where(pts)
            r_mean = float(np.mean(rs))
            c_mean = float(np.mean(cs))
            vals = metric[pts]
            elevs = z[pts]
            item: dict[str, float | int] = {
                "id": len(clusters) + 1,
                "pixels": count,
                "row": r_mean,
                "col": c_mean,
                "elevation": float(np.nanmean(elevs)),
                "mean_value": float(np.nanmean(vals)),
                "max_value": float(np.nanmax(vals)),
            }
            if x is not None and y is not None:
                item["x"] = float(np.nanmean(x[pts]))
                item["y"] = float(np.nanmean(y[pts]))
            else:
                item["x"] = float(c_mean)
                item["y"] = float(r_mean)
            if bbox_lonlat is not None and all(np.isfinite(float(coord)) for coord in bbox_lonlat):
                lon_min, lat_min, lon_max, lat_max = (float(coord) for coord in bbox_lonlat)
                item["lat"] = lat_max - (r_mean / max(h - 1, 1)) * (lat_max - lat_min)
                item["lon"] = lon_min + (c_mean / max(w - 1, 1)) * (lon_max - lon_min)
            clusters.append(item)
    elif x is not None and y is not None and np.sum(mask) > 1:
        # Spatial clustering for 1D underground CSV / point-cloud scans
        mx, my, mz, mv = x[mask], y[mask], z[mask], metric[mask]
        span_x = float(mx.max() - mx.min())
        span_y = float(my.max() - my.min())
        max_span = max(span_x, span_y, 1.0)
        # Adapt cell resolution so occupation grid is compact and bounded (<= 2000 cells)
        cell = max(max_span / 500.0, 1.0)
        gx = np.clip(np.floor((mx - mx.min()) / cell).astype(int), 0, 1999)
        gy = np.clip(np.floor((my - my.min()) / cell).astype(int), 0, 1999)
        occ = np.zeros((int(gx.max()) + 1, int(gy.max()) + 1), dtype=bool)
        occ[gx, gy] = True
        labeled, num = nd_label(occ)
        pt_labels = labeled[gx, gy]
        for cid in range(1, num + 1):
            pts = pt_labels == cid
            count = int(np.sum(pts))
            if count < min_pixels:
                continue
            clusters.append(
                {
                    "id": len(clusters) + 1,
                    "pixels": count,
                    "x": float(np.mean(mx[pts])),
                    "y": float(np.mean(my[pts])),
                    "elevation": float(np.nanmean(mz[pts])),
                    "mean_value": float(np.nanmean(mv[pts])),
                    "max_value": float(np.nanmax(mv[pts])),
                }
            )
    else:
        idxs = np.where(mask)[0]
        if min_pixels <= 1:
            for idx in idxs[:500]:
                item_1d: dict[str, float | int] = {
                    "id": len(clusters) + 1,
                    "pixels": 1,
                    "elevation": float(z[idx]),
                    "mean_value": float(metric[idx]),
                    "max_value": float(metric[idx]),
                }
                if x is not None and y is not None:
                    item_1d["x"] = float(x[idx])
                    item_1d["y"] = float(y[idx])
                clusters.append(item_1d)
    clusters.sort(key=lambda c: int(c["pixels"]), reverse=True)
    for i, c in enumerate(clusters, start=1):
        c["id"] = i
        c["confidence"] = _conf(float(c["mean_value"]))
    return clusters


def export_matches(
    clusters: list[dict[str, float | int]],
    out_path: Path,
    label: str = "Target",
) -> None:
    """Export clustered target coordinates to .gpx, .geojson, or .csv."""
    import csv
    import xml.etree.ElementTree as ET

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext == ".gpx":
        gpx = ET.Element("gpx", version="1.1", creator="lidar-cruiser")
        geo_count = 0
        for c in clusters:
            if "lat" not in c or "lon" not in c:
                continue
            geo_count += 1
            wpt = ET.SubElement(
                gpx, "wpt", lat=f"{float(c['lat']):.6f}", lon=f"{float(c['lon']):.6f}"
            )
            ET.SubElement(wpt, "ele").text = f"{float(c['elevation']):.2f}"
            ET.SubElement(wpt, "name").text = f"{label} #{c['id']} ({c['pixels']}px)"
            ET.SubElement(wpt, "desc").text = f"mean={float(c['mean_value']):.3f}"
        if geo_count == 0 and len(clusters) > 0:
            print(
                f"Warning: GPX export requires geographic coordinates (lat/lon). "
                f"Skipped {len(clusters)} cartesian cluster(s); consider exporting to .geojson or .csv.",
                file=sys.stderr,
            )
        ET.ElementTree(gpx).write(out_path, encoding="utf-8", xml_declaration=True)
        print(f"Exported {geo_count} target clusters to: {out_path}")
    elif ext in {".geojson", ".json"}:
        features = []
        for c in clusters:
            try:
                has_geo = (
                    "lat" in c
                    and "lon" in c
                    and np.isfinite(float(c["lat"]))
                    and np.isfinite(float(c["lon"]))
                )
            except (TypeError, ValueError):
                has_geo = False
            try:
                has_xy = (
                    "x" in c
                    and "y" in c
                    and np.isfinite(float(c["x"]))
                    and np.isfinite(float(c["y"]))
                )
            except (TypeError, ValueError):
                has_xy = False
            if not has_geo and not has_xy:
                continue
            if has_geo:
                clat = float(c["lat"])
                clon = float(c["lon"])
                cele = float(c.get("elevation", 0.0))
                cele = cele if np.isfinite(cele) else 0.0
                coords = [clon, clat, cele]
            else:
                cx = float(c["x"])
                cy = float(c["y"])
                cele = float(c.get("elevation", 0.0))
                cele = cele if np.isfinite(cele) else 0.0
                coords = [cx, cy, cele]
            props = {
                k: (v if (not isinstance(v, float) or np.isfinite(v)) else None)
                for k, v in c.items()
            }
            props.update({
                "label": label,
                "coordinate_type": "geographic" if has_geo else "cartesian",
            })
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": coords,
                    },
                    "properties": props,
                }
            )
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
        print(f"Exported {len(features)} target clusters to: {out_path}")
    else:
        base_fields = ["id", "confidence", "pixels"]
        geo_fields = ["lat", "lon"] if any("lat" in c for c in clusters) else []
        cart_fields = ["x", "y"] if any("x" in c for c in clusters) else []
        grid_fields = ["row", "col"] if any("row" in c for c in clusters) else []
        val_fields = ["elevation", "mean_value", "max_value"]
        fieldnames = base_fields + geo_fields + cart_fields + grid_fields + val_fields
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(clusters)
        print(f"Exported {len(clusters)} target clusters to: {out_path}")


def compute_elevation_profile(
    z: np.ndarray,
    metric: np.ndarray,
    r0: float,
    c0: float,
    r1: float,
    c1: float,
    bbox_lonlat: tuple[float, float, float, float] | None = None,
    n_samples: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample elevation and metric along a line between two grid points (r0,c0)->(r1,c1)."""
    z = np.asarray(z, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    if z.size == 0 or metric.size == 0 or z.ndim != 2 or metric.ndim != 2 or z.shape != metric.shape:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    try:
        n_samples_val = int(n_samples)
        if n_samples_val < 2:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    except (TypeError, ValueError):
        n_samples_val = 150
    n_samples = n_samples_val

    r0, c0, r1, c1 = float(r0), float(c0), float(r1), float(c1)
    if not (np.isfinite(r0) and np.isfinite(c0) and np.isfinite(r1) and np.isfinite(c1)):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    h, w = z.shape
    rs = np.clip(np.round(np.linspace(r0, r1, n_samples)).astype(int), 0, h - 1)
    cs = np.clip(np.round(np.linspace(c0, c1, n_samples)).astype(int), 0, w - 1)
    elevs = z[rs, cs]
    vals = metric[rs, cs]

    has_finite_bbox = False
    if bbox_lonlat is not None:
        try:
            bbox_coords = [float(c) for c in bbox_lonlat]
            if len(bbox_coords) == 4 and all(np.isfinite(c) for c in bbox_coords):
                has_finite_bbox = True
        except (TypeError, ValueError):
            has_finite_bbox = False

    if has_finite_bbox:
        assert bbox_lonlat is not None
        lon_min, lat_min, lon_max, lat_max = (float(c) for c in bbox_lonlat)
        lat0 = lat_max - (r0 / max(h - 1, 1)) * (lat_max - lat_min)
        lon0 = lon_min + (c0 / max(w - 1, 1)) * (lon_max - lon_min)
        lat1 = lat_max - (r1 / max(h - 1, 1)) * (lat_max - lat_min)
        lon1 = lon_min + (c1 / max(w - 1, 1)) * (lon_max - lon_min)
        dy_m = (lat1 - lat0) * 110_540.0
        dx_m = (lon1 - lon0) * 111_320.0 * np.cos(np.deg2rad((lat0 + lat1) / 2.0))
        total_m = float(np.hypot(dx_m, dy_m))
    else:
        total_m = float(np.hypot(r1 - r0, c1 - c0))
    dists = np.linspace(0.0, total_m, n_samples)
    return dists, elevs, vals


def compute_point_cloud_profile(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    metric: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    corridor_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a 2D cross-section transect from 3D points (x, y, z, metric) between (x0, y0) and (x1, y1).

    Returns:
        (distances_along_transect, elevations, metric_values)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    n_pts = min(len(x), len(y), len(z), len(metric))
    if n_pts == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    x = x[:n_pts]
    y = y[:n_pts]
    z = z[:n_pts]
    metric = metric[:n_pts]
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    finite_pts = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(metric)
    if not np.any(finite_pts):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    x = x[finite_pts]
    y = y[finite_pts]
    z = z[finite_pts]
    metric = metric[finite_pts]

    dx = x1 - x0
    dy = y1 - y0
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    rx = x - x0
    ry = y - y0

    dist = rx * ux + ry * uy
    perp = np.abs(rx * nx + ry * ny)

    try:
        cw_val = float(corridor_width) if corridor_width is not None else None
        if cw_val is not None and (not np.isfinite(cw_val) or cw_val <= 0):
            cw_val = None
    except (TypeError, ValueError):
        cw_val = None

    corridor_width = max(length * 0.02, 1.0) if cw_val is None else max(cw_val, 0.1)

    along_mask = (dist >= 0.0) & (dist <= length)
    if not np.any(along_mask):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    mask = along_mask & (perp <= corridor_width)
    if not np.any(mask):
        along_indices = np.where(along_mask)[0]
        max_search_width = max(corridor_width * 5.0, 20.0)
        close_along = along_indices[perp[along_indices] <= max_search_width]
        if len(close_along) == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        nearest_idx = close_along[np.argsort(perp[close_along])[:min(60, len(close_along))]]
        mask = np.zeros(len(x), dtype=bool)
        mask[nearest_idx] = True

    d_sub = dist[mask]
    z_sub = z[mask]
    m_sub = metric[mask]

    order = np.argsort(d_sub)
    return d_sub[order], z_sub[order], m_sub[order]


def load_gps_track(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[tuple[float, float, str]]]:
    """Load GPS track & waypoints from .gpx, .kml, .geojson, or .csv."""
    path = Path(path)
    if not path.exists():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), None, []
    ext = path.suffix.lower()
    lats: list[float] = []
    lons: list[float] = []
    elevs: list[float] = []
    waypoints: list[tuple[float, float, str]] = []

    try:
        if ext == ".gpx":
            import gpxpy

            with path.open("r", encoding="utf-8") as f:
                gpx = gpxpy.parse(f)
            for trk in gpx.tracks:
                for seg in trk.segments:
                    for pt in seg.points:
                        lats.append(pt.latitude)
                        lons.append(pt.longitude)
                        elevs.append(pt.elevation if pt.elevation is not None else 0.0)
            for rte in gpx.routes:
                for pt in rte.points:
                    lats.append(pt.latitude)
                    lons.append(pt.longitude)
                    elevs.append(pt.elevation if pt.elevation is not None else 0.0)
            for wpt in gpx.waypoints:
                waypoints.append((wpt.latitude, wpt.longitude, wpt.name or "Waypoint"))
        elif ext in {".geojson", ".json", ".kml", ".shp"}:
            import geopandas as gpd

            gdf = gpd.read_file(path)
            if gdf.crs is not None:
                gdf = gdf.to_crs(epsg=4326)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                if geom.geom_type == "Point":
                    waypoints.append((geom.y, geom.x, str(row.get("name", "Point"))))
                elif geom.geom_type == "LineString":
                    for c in geom.coords:
                        lons.append(c[0])
                        lats.append(c[1])
                        elevs.append(c[2] if len(c) > 2 else 0.0)
        else:
            import pandas as pd

            df = pd.read_csv(path)
            if len(df.columns) >= 2:
                cols = {str(c).lower().strip(): str(c) for c in df.columns}
                lat_col = cols.get("lat") or cols.get("latitude") or list(df.columns)[0]
                lon_col = cols.get("lon") or cols.get("longitude") or list(df.columns)[1]
                s_lat = pd.to_numeric(df[lat_col], errors="coerce").dropna()
                s_lon = pd.to_numeric(df[lon_col], errors="coerce").dropna()
                n = min(len(s_lat), len(s_lon))
                lats = s_lat.iloc[:n].tolist()
                lons = s_lon.iloc[:n].tolist()
                elev_col = cols.get("ele") or cols.get("elev") or cols.get("elevation")
                if elev_col and elev_col in df.columns:
                    s_ele = pd.to_numeric(df[elev_col], errors="coerce").fillna(0.0)
                    elevs = s_ele.iloc[:n].tolist()
                else:
                    elevs = [0.0] * n
    except Exception as exc:
        print(f"Warning: Failed to parse GPS file {path}: {exc}", file=sys.stderr)
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), None, []

    return (
        np.asarray(lats, dtype=np.float64),
        np.asarray(lons, dtype=np.float64),
        np.asarray(elevs, dtype=np.float64) if elevs else None,
        waypoints,
    )


def _col_label(idx: int) -> str:
    """Convert 0-based column index to spreadsheet-style letters (A, B, ..., AA)."""
    out = ""
    try:
        n = max(int(idx), 0)
    except (TypeError, ValueError):
        n = 0
    while True:
        out = chr(ord("A") + (n % 26)) + out
        n = n // 26 - 1
        if n < 0:
            return out


def compute_region_grid(
    bbox_lonlat: tuple[float, float, float, float],
    cell_km: float = 5.0,
) -> tuple[list[dict[str, object]], int, int]:
    """Divide bbox_lonlat into ~cell_km x cell_km squares labeled A1, B1, ... (NW to SE)."""
    try:
        cell_km_val = float(cell_km)
        if not np.isfinite(cell_km_val) or cell_km_val <= 0:
            cell_km_val = 5.0
    except (TypeError, ValueError):
        cell_km_val = 5.0
    cell_km = max(cell_km_val, 0.1)

    try:
        lon0, lat0, lon1, lat1 = (float(c) for c in bbox_lonlat)
        if not all(np.isfinite(c) for c in (lon0, lat0, lon1, lat1)):
            lon0, lat0, lon1, lat1 = (13.48, 44.75, 14.25, 45.55)
    except Exception:
        lon0, lat0, lon1, lat1 = (13.48, 44.75, 14.25, 45.55)

    lon_min, lon_max = min(lon0, lon1), max(lon0, lon1)
    lat_min, lat_max = min(lat0, lat1), max(lat0, lat1)
    lat_mid = (lat_min + lat_max) / 2.0
    dlat = cell_km / 110.54
    dlon = cell_km / (111.32 * max(float(np.cos(np.deg2rad(lat_mid))), 0.1))

    n_rows = max(int(np.ceil((lat_max - lat_min) / dlat)), 1)
    n_cols = max(int(np.ceil((lon_max - lon_min) / dlon)), 1)

    cells: list[dict[str, object]] = []
    for r in range(n_rows):
        c_lat_max = lat_max - r * dlat
        c_lat_min = max(lat_min, c_lat_max - dlat)
        for c in range(n_cols):
            c_lon_min = lon_min + c * dlon
            c_lon_max = min(lon_max, c_lon_min + dlon)
            cell_id = f"{_col_label(c)}{r + 1}"
            cells.append(
                {
                    "id": cell_id,
                    "row": r,
                    "col": c,
                    "bbox": (c_lon_min, c_lat_min, c_lon_max, c_lat_max),
                    "center": ((c_lon_min + c_lon_max) / 2.0, (c_lat_min + c_lat_max) / 2.0),
                }
            )
    return cells, n_rows, n_cols


def render_folium_map(
    metric: np.ndarray,
    bbox_lonlat: tuple[float, float, float, float],
    title: str,
    out_html: Path,
    cmap: str = "turbo",
    gpx_path: Path | None = None,
    auto_open: bool = True,
    scalar_label: str = "Value",
    mark_value: float | None = None,
    mark_tolerance: float = 0.5,
    mark_label: str | None = None,
    grid_cells: list[dict[str, object]] | None = None,
    grid_km: float = 5.0,
    grid_region: str = "istra",
) -> None:
    """Render 2D ArcGIS/Esri Satellite + OpenTopoMap + Cold-to-Warm overlay + GPS tracks."""
    import html
    import webbrowser

    import branca.colormap as bcm
    import folium
    from folium import plugins
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None, control_scale=True)
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

    # ArcGIS / Esri + OpenStreetMap / OpenTopoMap basemaps
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="ArcGIS Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Topo",
        name="ArcGIS Topo (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="OpenTopoMap",
        overlay=False,
        control=True,
    ).add_to(m)

    # Render cold-to-warm metric as RGBA image overlay + colorbar legend
    valid = np.isfinite(metric)
    if metric.ndim == 2 and np.any(valid):
        vmin = float(np.nanpercentile(metric, 2))
        vmax = float(np.nanpercentile(metric, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = np.clip((metric - vmin) / (vmax - vmin), 0.0, 1.0)
        cm = plt.get_cmap(cmap if cmap in plt.colormaps() else "turbo")
        rgba = cm(np.where(valid, norm, 0.0))
        rgba[..., 3] = np.where(valid, 0.65, 0.0)
        rgba_u8 = (rgba * 255).astype(np.uint8)

        folium.raster_layers.ImageOverlay(
            image=rgba_u8,
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            name=f"{scalar_label}",
            interactive=True,
            cross_origin=False,
        ).add_to(m)

        palette = [mcolors.to_hex(cm(t)) for t in np.linspace(0.0, 1.0, 12)]
        bcm.LinearColormap(
            colors=palette,
            vmin=round(vmin, 2),
            vmax=round(vmax, 2),
            caption=scalar_label,
        ).add_to(m)

    if mark_value is not None and metric.ndim == 2:
        mask, count, label = find_material_matches(
            metric, mark_value, mark_tolerance, mark_label=mark_label
        )
        if count > 0:
            h, w = metric.shape
            rows, cols = np.where(mask)
            max_markers = 500
            if count > max_markers:
                step = int(np.ceil(count / max_markers))
                rows, cols = rows[::step], cols[::step]
                print(f"2D map: showing {len(rows)} of {count} markers (subsampled for speed)")

            fg = folium.FeatureGroup(name=f"{label} ({count} matches)")
            for r, c in zip(rows, cols):
                mlat = lat_max - (float(r) / max(h - 1, 1)) * (lat_max - lat_min)
                mlon = lon_min + (float(c) / max(w - 1, 1)) * (lon_max - lon_min)
                mval = float(metric[r, c])
                folium.CircleMarker(
                    location=[mlat, mlon],
                    radius=5,
                    color="#ffd700",
                    fill=True,
                    fill_color="#ffd700",
                    fill_opacity=0.85,
                    popup=folium.Popup(
                        f"<b>{html.escape(label)}</b><br>"
                        f"Value: {mval:.3f}<br>"
                        f"Lat: {mlat:.5f}, Lon: {mlon:.5f}",
                        max_width=240,
                    ),
                ).add_to(fg)
            fg.add_to(m)

    if grid_cells:
        import os

        bridge_url = os.environ.get("LIDAR_BRIDGE_URL", "").rstrip("/")
        grid_group = folium.FeatureGroup(name=f"Grid ({grid_km:g} km)")
        for cell in grid_cells:
            c_lon_min, c_lat_min, c_lon_max, c_lat_max = cell["bbox"]  # type: ignore[misc]
            c_lon, c_lat = cell["center"]  # type: ignore[misc]
            cid = str(cell["id"])
            cmd = f"./cruise --region {grid_region} --grid-cell {cid} --detail ultra"
            bridge_btn = ""
            if bridge_url:
                open_url = f"{bridge_url}/open?region={grid_region}&cell={cid}"
                bridge_btn = (
                    f"<br><button onclick=\"fetch('{html.escape(open_url)}')\" "
                    'style="margin-top:6px;padding:4px 8px;background:#0284c7;color:#fff;'
                    'border:none;border-radius:4px;cursor:pointer;">Open in 3D App</button>'
                )
            folium.Rectangle(
                bounds=[[c_lat_min, c_lon_min], [c_lat_max, c_lon_max]],
                color="#38bdf8",
                weight=1.2,
                fill=True,
                fill_color="#38bdf8",
                fill_opacity=0.05,
                tooltip=f"Cell {cid} ({c_lat:.4f}N, {c_lon:.4f}E)",
                popup=folium.Popup(
                    f"<b>Grid Square {html.escape(cid)}</b> ({grid_km:g} km)<br>"
                    f"Center: {c_lat:.5f}N, {c_lon:.5f}E<br>"
                    f"<code>{html.escape(cmd)}</code>"
                    f"{bridge_btn}",
                    max_width=320,
                ),
            ).add_to(grid_group)
            folium.Marker(
                location=[c_lat, c_lon],
                icon=folium.DivIcon(
                    html=(
                        '<div style="font-size:10px;font-weight:700;color:#e0f2fe;'
                        'text-shadow:0 0 3px #000,0 0 3px #000;pointer-events:none;">'
                        f"{html.escape(cid)}</div>"
                    ),
                    icon_size=(30, 14),
                    icon_anchor=(15, 7),
                ),
            ).add_to(grid_group)
        grid_group.add_to(m)

    if gpx_path is not None and gpx_path.exists():
        lats, lons, _, waypoints = load_gps_track(gpx_path)
        gpx_group = folium.FeatureGroup(name=f"GPS: {gpx_path.name}")
        if len(lats) > 1:
            folium.PolyLine(
                list(zip(lats, lons)),
                color="yellow",
                weight=4,
                opacity=0.9,
                tooltip=gpx_path.name,
            ).add_to(gpx_group)
        for wlat, wlon, wname in waypoints:
            folium.Marker([wlat, wlon], popup=wname).add_to(gpx_group)
        gpx_group.add_to(m)

    title_html = (
        '<div style="position: fixed; top: 10px; left: 54px; z-index: 9999; '
        'background: rgba(15, 23, 42, 0.85); color: #f8fafc; padding: 6px 12px; '
        'border-radius: 6px; font-family: sans-serif; font-size: 13px; '
        'font-weight: 600; box-shadow: 0 2px 6px rgba(0,0,0,0.35);">'
        f"{html.escape(title)}</div>"
    )
    m.get_root().html.add_child(folium.Element(title_html))

    plugins.MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Coords:",
        num_digits=5,
    ).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)

    folium.LatLngPopup().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(f"Saved 2D ArcGIS/GPS map to: {out_html}")
    if auto_open:
        webbrowser.open(out_html.resolve().as_uri())


def load_csv_scan(
    path: Path,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load a CSV sensor scan with columns x,y,z,density (or value), optionally filtered by |depth|."""
    import pandas as pd

    try:
        df = pd.read_csv(path)
    except Exception:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            "Density (g/cm3 / sensor)",
        )

    if len(df.columns) == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            "Density (g/cm3 / sensor)",
        )

    cols = {str(c).lower().strip(): str(c) for c in df.columns}
    x_col = cols.get("x") or cols.get("lon") or (list(df.columns)[0] if len(df.columns) > 0 else "x")
    y_col = cols.get("y") or cols.get("lat") or (list(df.columns)[1] if len(df.columns) > 1 else x_col)
    z_col = cols.get("z") or cols.get("depth") or cols.get("elevation") or (list(df.columns)[2] if len(df.columns) > 2 else y_col)
    d_col = cols.get("density") or cols.get("value") or cols.get("intensity") or (list(df.columns)[3] if len(df.columns) > 3 else z_col)

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64) if x_col in df.columns else np.empty(0, dtype=np.float64)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64) if y_col in df.columns else np.empty(0, dtype=np.float64)
    z = pd.to_numeric(df[z_col], errors="coerce").to_numpy(dtype=np.float64) if z_col in df.columns else np.empty(0, dtype=np.float64)
    metric = pd.to_numeric(df[d_col], errors="coerce").to_numpy(dtype=np.float64) if d_col in df.columns else np.empty(0, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(metric)
    x, y, z, metric = x[valid], y[valid], z[valid], metric[valid]
    if min_depth is not None or max_depth is not None:
        try:
            dmin_val = float(min_depth) if min_depth is not None else None
            if dmin_val is not None and (not np.isfinite(dmin_val) or dmin_val < 0):
                dmin_val = None
        except (TypeError, ValueError):
            dmin_val = None
        try:
            dmax_val = float(max_depth) if max_depth is not None else None
            if dmax_val is not None and (not np.isfinite(dmax_val) or dmax_val < 0):
                dmax_val = None
        except (TypeError, ValueError):
            dmax_val = None

        if dmin_val is not None or dmax_val is not None:
            keep = np.isfinite(z)
            if dmin_val is not None:
                keep &= np.abs(z) >= dmin_val
            if dmax_val is not None:
                keep &= np.abs(z) <= dmax_val
            x, y, z, metric = x[keep], y[keep], z[keep], metric[keep]
            print(
                f"Depth filter ({dmin_val or 0:g}-{dmax_val if dmax_val is not None else 'inf'} m): "
                f"kept {int(np.sum(keep))} of {len(keep)} points"
            )
    if len(x) == 0:
        return x, y, z, metric, f"{d_col} (g/cm3 / sensor)"
    x_mean = float(np.nanmean(x)) if np.any(np.isfinite(x)) else 0.0
    y_mean = float(np.nanmean(y)) if np.any(np.isfinite(y)) else 0.0
    return x - x_mean, y - y_mean, z, metric, f"{d_col} (g/cm3 / sensor)"


def generate_test_underground_csv(out_path: Path) -> Path:
    """Create a synthetic 0-2.8 m underground sensor CSV for testing depth slices & clustering."""
    import csv

    rng = np.random.default_rng(42)
    rows: list[tuple[float, float, float, float]] = []
    # Background soil (1.8 - 2.6 g/cm3)
    for _ in range(180):
        rows.append(
            (
                round(float(rng.uniform(-10.0, 10.0)), 2),
                round(float(rng.uniform(-10.0, 10.0)), 2),
                round(-float(rng.uniform(0.1, 2.5)), 2),
                round(float(rng.uniform(1.8, 2.7)), 2),
            )
        )
    # 3 underground anomalies within 0-2m + 1 deeper anomaly at -2.7m
    anomalies = [
        (-4.0, 3.0, -0.35, 19.30, 8),   # Shallow (0-0.5m)
        (2.5, -2.0, -0.80, 19.15, 6),   # Mid (0.5-1.0m)
        (6.0, 5.0, -1.55, 7.85, 7),     # Deep (1.0-2.0m, steel-like)
        (-6.0, -5.0, -2.70, 19.32, 5),  # Below 2m (filtered out by 0-2m)
    ]
    for cx, cy, cz, val, n in anomalies:
        for _ in range(n):
            rows.append(
                (
                    round(cx + float(rng.normal(0.0, 0.2)), 2),
                    round(cy + float(rng.normal(0.0, 0.2)), 2),
                    round(cz + float(rng.normal(0.0, 0.05)), 2),
                    round(val + float(rng.normal(0.0, 0.08)), 2),
                )
            )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "depth", "density"])
        w.writerows(rows)
    return out_path


def load_point_cloud(
    path: Path,
    max_points: int | None,
    cell_size: float | None,
    color_by: str = "density",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load .las/.laz point cloud and compute density or elevation (max_points=0/None loads all)."""
    import laspy

    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    c_by = (color_by or "density").lower()
    if c_by == "elevation":
        metric = z.copy()
        label = "Elevation (m)"
    elif c_by == "intensity":
        if hasattr(las, "intensity"):
            metric = np.asarray(las.intensity, dtype=np.float64)
            label = "LiDAR Intensity"
        else:
            print("Warning: Point cloud does not contain intensity data; using point density.", file=sys.stderr)
            metric = compute_point_density(x, y, cell_size=cell_size)
            label = "Point Density (pts/cell)"
    elif c_by == "classification":
        if hasattr(las, "classification"):
            metric = np.asarray(las.classification, dtype=np.float64)
            label = "ASPRS Classification"
        else:
            print("Warning: Point cloud does not contain classification data; using point density.", file=sys.stderr)
            metric = compute_point_density(x, y, cell_size=cell_size)
            label = "Point Density (pts/cell)"
    else:
        metric = compute_point_density(x, y, cell_size=cell_size)
        label = "Point Density (pts/cell)"

    if max_points is not None and max_points > 0 and len(x) > max_points:
        step = max(len(x) // max_points, 1)
        x, y, z, metric = x[::step], y[::step], z[::step], metric[::step]

    if len(x) == 0:
        return x, y, z, metric, label

    x_mean = float(np.nanmean(x)) if np.any(np.isfinite(x)) else 0.0
    y_mean = float(np.nanmean(y)) if np.any(np.isfinite(y)) else 0.0
    return x - x_mean, y - y_mean, z, metric, label


def load_raster(
    path: Path,
    stride: int,
    dsm_path: Path | None = None,
    color_by: str = "density",
    window: int = 5,
    hide_sea: bool = False,
    max_dim: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load .hgt/.tif raster and compute object density, slope, or elevation."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        h, w = src.height, src.width
        auto_stride = max(int(np.ceil(max(h, w) / max_dim)), 1)
        eff_stride = max(stride, auto_stride)
        out_h = max(h // eff_stride, 1)
        out_w = max(w // eff_stride, 1)
        z = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype(np.float64)
        transform = src.transform * src.transform.scale(w / out_w, h / out_h)
        crs = src.crs

    z_filled = np.where(np.ma.getmaskarray(z), np.nan, np.asarray(z))
    valid = ~np.isnan(z_filled)
    if not np.any(valid):
        raise ValueError(f"Raster {path} contains no valid elevation values.")

    median_z = float(np.nanmedian(z_filled))
    z_sub = np.where(valid, z_filled, median_z)

    dsm_sub = None
    if dsm_path is not None:
        with rasterio.open(dsm_path) as dsm_src:
            dsm = dsm_src.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.bilinear,
                masked=True,
            ).astype(np.float64)
        dsm_sub = np.where(np.ma.getmaskarray(dsm), z_sub, np.asarray(dsm))

    metric, label = compute_raster_metric(
        z_sub, color_by=color_by, window=window, dsm=dsm_sub
    )

    if hide_sea:
        metric = np.where(z_sub <= 0.5, np.nan, metric)

    rows, cols = np.meshgrid(
        np.arange(out_h),
        np.arange(out_w),
        indexing="ij",
    )
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.asarray(xs, dtype=np.float64).reshape(z_sub.shape)
    ys = np.asarray(ys, dtype=np.float64).reshape(z_sub.shape)

    if crs is not None and crs.is_geographic:
        lat_mid = float(np.mean(ys))
        xs = (xs - xs.mean()) * 111_320.0 * np.cos(np.deg2rad(lat_mid))
        ys = (ys - ys.mean()) * 110_540.0
    else:
        xs = xs - xs.mean()
        ys = ys - ys.mean()

    return xs, ys, z_sub, metric, label


def _attach_navigation_controls(plotter, zoom_to_pointer: bool = True) -> None:
    """Attach zoom-to-pointer, F-to-focus, and WASD/Arrow movement in PyVista."""
    import vtk

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.005)

    def _pick_under_pointer() -> np.ndarray | None:
        mx, my = plotter.iren.get_event_position()
        if picker.Pick(mx, my, 0, plotter.renderer):
            return np.array(picker.GetPickPosition(), dtype=np.float64)
        return None

    def _pan_camera(dx: float, dy: float) -> None:
        cam = plotter.camera
        pos = np.array(cam.position, dtype=np.float64)
        foc = np.array(cam.focal_point, dtype=np.float64)
        forward = foc - pos
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            forward = np.array([0.0, 1.0, 0.0])
        else:
            forward /= norm
        right = np.array([forward[1], -forward[0], 0.0])
        step = max(np.linalg.norm(foc - pos) * 0.08, 100.0)
        delta = (forward * dy + right * dx) * step
        cam.position = tuple(pos + delta)
        cam.focal_point = tuple(foc + delta)
        plotter.render()

    if zoom_to_pointer:
        def _on_wheel(factor: float):
            def _handler(_obj, _event) -> None:
                target = _pick_under_pointer()
                cam = plotter.camera
                pos = np.array(cam.position, dtype=np.float64)
                foc = np.array(cam.focal_point, dtype=np.float64)
                if target is None:
                    target = foc
                # Move both camera position and focal point toward/away from pointer target
                new_pos = target + (pos - target) * factor
                new_foc = target + (foc - target) * factor
                cam.position = tuple(new_pos)
                cam.focal_point = tuple(new_foc)
                plotter.render()
            return _handler

        iren = plotter.iren.interactor
        iren.RemoveObservers("MouseWheelForwardEvent")
        iren.RemoveObservers("MouseWheelBackwardEvent")
        iren.AddObserver("MouseWheelForwardEvent", _on_wheel(0.82), 1.0)
        iren.AddObserver("MouseWheelBackwardEvent", _on_wheel(1.22), 1.0)

    def _focus_pointer() -> None:
        target = _pick_under_pointer()
        if target is None:
            return
        cam = plotter.camera
        pos = np.array(cam.position, dtype=np.float64)
        foc = np.array(cam.focal_point, dtype=np.float64)
        offset = pos - foc
        cam.focal_point = tuple(target)
        cam.position = tuple(target + offset)
        plotter.render()

    plotter.add_key_event("f", _focus_pointer)
    plotter.add_key_event("w", lambda: _pan_camera(0.0, 1.0))
    plotter.add_key_event("s", lambda: _pan_camera(0.0, -1.0))
    plotter.add_key_event("a", lambda: _pan_camera(-1.0, 0.0))
    plotter.add_key_event("d", lambda: _pan_camera(1.0, 0.0))
    plotter.add_key_event("Up", lambda: _pan_camera(0.0, 1.0))
    plotter.add_key_event("Down", lambda: _pan_camera(0.0, -1.0))
    plotter.add_key_event("Left", lambda: _pan_camera(-1.0, 0.0))
    plotter.add_key_event("Right", lambda: _pan_camera(1.0, 0.0))


def render_pyvista(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    metric: np.ndarray,
    is_grid: bool,
    z_scale: float,
    title: str,
    cmap: str = "turbo",
    scalar_label: str = "Object Density (Cold=Low, Warm=High)",
    contours: bool = False,
    screenshot: Path | None = None,
    zoom_to_pointer: bool = True,
    mark_value: float | None = None,
    mark_tolerance: float = 0.5,
    mark_label: str | None = None,
) -> None:
    """Render 3D elevation colored by cold-to-warm density using PyVista."""
    import pyvista as pv

    pv.set_plot_theme("dark")
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(title=title, off_screen=screenshot is not None)

    if is_grid:
        if z.size > 0:
            grid = pv.StructuredGrid(x, y, z * z_scale)
            grid[scalar_label] = metric.ravel(order="F")
            plotter.add_mesh(
                grid,
                scalars=scalar_label,
                cmap=cmap,
                smooth_shading=True,
                nan_color="#0b132b",
                scalar_bar_args={"title": scalar_label},
            )
            if contours:
                grid["Elevation (m)"] = (z * z_scale).ravel(order="F")
                iso = grid.contour(isosurfaces=12, scalars="Elevation (m)")
                plotter.add_mesh(iso, color="white", line_width=1.0, opacity=0.45)
    else:
        if len(x) > 0:
            pts = np.column_stack([x, y, z * z_scale])
            cloud = pv.PolyData(pts)
            cloud[scalar_label] = metric
            plotter.add_mesh(
                cloud,
                scalars=scalar_label,
                cmap=cmap,
                point_size=3.0,
                render_points_as_spheres=False,
                scalar_bar_args={"title": scalar_label},
            )

    if mark_value is not None:
        mask, count, label = find_material_matches(
            metric, mark_value, mark_tolerance, mark_label=mark_label
        )
        if count > 0:
            mx = x[mask]
            my = y[mask]
            mz = (z[mask] * z_scale)
            match_pts = pv.PolyData(np.column_stack([mx, my, mz]))
            plotter.add_mesh(
                match_pts,
                color="gold",
                point_size=12.0,
                render_points_as_spheres=True,
                label=f"{label} ({count})",
            )
            plotter.add_legend()

    plotter.add_text(
        "Drag: Rotate | Scroll: Zoom to Pointer | Shift+Drag or WASD/Arrows: Move | F: Focus Pointer | R: Reset | Q: Quit",
        position="lower_left",
        font_size=9,
        color="lightgray",
    )
    plotter.add_axes()

    if screenshot is not None:
        screenshot = Path(screenshot)
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot))
        print(f"Saved screenshot to: {screenshot}")
    else:
        _attach_navigation_controls(plotter, zoom_to_pointer=zoom_to_pointer)
        plotter.show()


def render_plotly(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    metric: np.ndarray,
    is_grid: bool,
    z_scale: float,
    title: str,
    out_html: Path,
    cmap: str = "turbo",
    scalar_label: str = "Density (Cold=Low, Warm=High)",
    contours: bool = False,
    auto_open: bool = True,
    mark_value: float | None = None,
    mark_tolerance: float = 0.5,
    mark_label: str | None = None,
) -> None:
    """Render interactive 3D HTML map using Plotly."""
    import plotly.graph_objects as go

    colorscale = COLORMAPS.get(cmap, "Turbo")

    if is_grid:
        contours_cfg = (
            {"z": {"show": True, "usecolormap": False, "color": "rgba(255,255,255,0.4)"}}
            if contours
            else {}
        )
        traces = [
            go.Surface(
                x=x,
                y=y,
                z=z * z_scale,
                surfacecolor=metric,
                customdata=z,
                colorscale=colorscale,
                colorbar={"title": scalar_label},
                contours=contours_cfg,
                hovertemplate=(
                    "X: %{x:.1f} m<br>Y: %{y:.1f} m<br>"
                    "Elevation: %{customdata:.2f} m<br>"
                    f"{scalar_label}: %{{surfacecolor:.3f}}<extra></extra>"
                ),
            )
        ]
    else:
        traces = [
            go.Scatter3d(
                x=x,
                y=y,
                z=z * z_scale,
                customdata=np.column_stack([z, metric]),
                mode="markers",
                marker={
                    "size": 2,
                    "color": metric,
                    "colorscale": colorscale,
                    "colorbar": {"title": scalar_label},
                },
                hovertemplate=(
                    "X: %{x:.1f} m<br>Y: %{y:.1f} m<br>"
                    "Elevation: %{customdata[0]:.2f} m<br>"
                    f"{scalar_label}: %{{customdata[1]:.3f}}<extra></extra>"
                ),
            )
        ]

    if mark_value is not None:
        mask, count, label = find_material_matches(
            metric, mark_value, mark_tolerance, mark_label=mark_label
        )
        if count > 0:
            traces.append(
                go.Scatter3d(
                    x=x[mask],
                    y=y[mask],
                    z=z[mask] * z_scale,
                    customdata=np.column_stack([z[mask], metric[mask]]),
                    mode="markers",
                    name=f"{label} ({count})",
                    marker={"size": 7, "color": "gold", "symbol": "diamond"},
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "X: %{x:.1f} m<br>Y: %{y:.1f} m<br>"
                        "Elevation: %{customdata[0]:.2f} m<br>"
                        f"{scalar_label}: %{{customdata[1]:.3f}}<extra></extra>"
                    ),
                )
            )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        uirevision="constant",
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": f"Elevation (m, x{z_scale})",
            "aspectmode": "data",
        },
        template="plotly_dark",
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, auto_open=auto_open)
    print(f"Saved interactive 3D map to: {out_html}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="3D Elevation + Cold-to-Warm Object Density Map Viewer"
    )
    parser.add_argument("input", type=Path, help="Path to .las/.laz, .tif/.hgt, or .csv")
    parser.add_argument(
        "--dsm",
        type=Path,
        default=None,
        help="Optional DSM/DMP raster to compute (DSM - DTM) object density",
    )
    parser.add_argument(
        "--backend",
        choices=["pyvista", "plotly"],
        default="pyvista",
        help="3D viewer backend (default: pyvista)",
    )
    parser.add_argument(
        "--color-by",
        choices=[
            "density",
            "slope",
            "elevation",
            "lrm",
            "hillshade",
            "svf",
            "multi_hillshade",
            "openness",
            "intensity",
            "classification",
        ],
        default="density",
        help="What drives the cold-to-warm color scale (default: density)",
    )
    parser.add_argument(
        "--cmap",
        choices=list(COLORMAPS.keys()),
        default="turbo",
        help="Color palette (default: turbo, cold blue -> warm red)",
    )
    parser.add_argument(
        "--z-scale",
        type=float,
        default=2.0,
        help="Vertical exaggeration factor (default: 2.0)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Raster subsampling stride for smooth 3D rendering (default: 4)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Neighborhood window (pixels) for density calculation (default: 5)",
    )
    parser.add_argument(
        "--hide-sea",
        action="store_true",
        help="Mask sea-level pixels (<= 0.5 m) so land density contrast pops",
    )
    parser.add_argument(
        "--contours",
        action="store_true",
        help="Overlay elevation contour lines on the 3D terrain",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=400_000,
        help="Max points to render for .las/.laz clouds (default: 400000)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=None,
        help="Grid cell size in meters for point cloud density calculation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("lidar_3d_map.html"),
        help="Output HTML path when using --backend plotly",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Save a PNG screenshot (PyVista backend) and exit",
    )
    parser.add_argument(
        "--no-zoom-to-pointer",
        action="store_true",
        help="Disable zoom-to-pointer and use default center zoom",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically launch web browser for plotly backend",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="Minimum |depth| in meters for underground CSV scans (e.g. 0.5)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Filter underground CSV scans to |depth| <= MAX_DEPTH meters (e.g. 2.0)",
    )
    parser.add_argument(
        "--mark-material",
        type=str,
        default=None,
        help="Highlight objects matching a saved material name (e.g. gold, steel)",
    )
    parser.add_argument(
        "--mark-value",
        type=float,
        default=None,
        help="Highlight objects matching a numeric density/sensor value",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Match tolerance (+/-) for --mark-material / --mark-value (default: 0.5)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1

    mark_val: float | None = None
    mark_label: str | None = None
    if args.mark_material:
        mats = load_materials()
        mat_key = args.mark_material.lower()
        if mat_key in mats:
            mark_val = mats[mat_key]
            mark_label = args.mark_material.title()
        else:
            print(f"Warning: Unknown material '{args.mark_material}'", file=sys.stderr)
    elif args.mark_value is not None:
        mark_val = args.mark_value
        mark_label = f"Target ({mark_val:g})"

    ext = args.input.suffix.lower()
    if ext in {".las", ".laz"}:
        try:
            x, y, z, metric, label = load_point_cloud(
                args.input,
                max_points=args.max_points,
                cell_size=args.cell_size,
                color_by=args.color_by,
            )
        except Exception as exc:
            print(f"Error: Failed to load point cloud '{args.input}': {exc}", file=sys.stderr)
            return 1
        is_grid = False
    elif ext == ".csv":
        try:
            x, y, z, metric, label = load_csv_scan(
                args.input,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
        except Exception as exc:
            print(f"Error: Failed to load CSV scan '{args.input}': {exc}", file=sys.stderr)
            return 1
        is_grid = False
    elif ext in {".hgt", ".tif", ".tiff"}:
        try:
            x, y, z, metric, label = load_raster(
                args.input,
                stride=max(args.stride, 1),
                dsm_path=args.dsm,
                color_by=args.color_by,
                window=max(args.window, 3),
                hide_sea=args.hide_sea,
            )
        except Exception as exc:
            print(f"Error: Failed to load raster '{args.input}': {exc}", file=sys.stderr)
            return 1
        is_grid = True
    else:
        print(f"Error: Unsupported file format '{ext}'. Supported: .las, .laz, .tif, .hgt, .csv", file=sys.stderr)
        return 1

    title = f"3D Elevation & {args.color_by.title()} — {args.input.name}"
    if args.backend == "pyvista":
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and args.screenshot is None:
            print(
                "Notice: Headless environment detected without DISPLAY/WAYLAND. "
                "Use --screenshot <path.png> or --backend plotly.",
                file=sys.stderr,
            )
            return 0
        render_pyvista(
            x,
            y,
            z,
            metric,
            is_grid,
            args.z_scale,
            title,
            cmap=args.cmap,
            scalar_label=label,
            contours=args.contours,
            screenshot=args.screenshot,
            zoom_to_pointer=not args.no_zoom_to_pointer,
            mark_value=mark_val,
            mark_tolerance=args.tolerance,
            mark_label=mark_label,
        )
    else:
        render_plotly(
            x,
            y,
            z,
            metric,
            is_grid,
            args.z_scale,
            title,
            args.out,
            cmap=args.cmap,
            scalar_label=label,
            contours=args.contours,
            auto_open=not args.no_open,
            mark_value=mark_val,
            mark_tolerance=args.tolerance,
            mark_label=mark_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
