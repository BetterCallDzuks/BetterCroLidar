#!/usr/bin/env python3
"""
Interactive 3D map cruiser for Istra & Primorsko-goranska (and any .tif/.hgt/.las/.laz).

Shows elevation in 3D (Z-axis) and object/terrain density in cold-to-warm colors
(blue = less dense/flat, yellow/orange/red = denser/complex).

Examples:
  ./cruise                           # Launch 3D window for all of Istra + PGZ
  ./cruise --menu                    # Interactive menu to pick region & options
  ./cruise --list                    # List available tiles and preset regions
  ./cruise --region ucka --detail high --hide-sea --contours
  ./cruise --region istra --browser  # Open interactive 3D map in web browser
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

from view_lidar_3d import (
    COLORMAPS,
    DETAIL_MAX_DIM,
    cluster_material_matches,
    compute_raster_metric,
    compute_region_grid,
    export_matches,
    load_csv_scan,
    load_materials,
    load_point_cloud,
    load_raster,
    render_folium_map,
    render_plotly,
    render_pyvista,
    save_material,
    warn_material_ambiguity,
)


REGIONS: dict[str, tuple[tuple[float, float, float, float], str]] = {
    "all": ((13.45, 44.45, 15.05, 45.65), "Full Istra & Primorsko-goranska"),
    "istra": ((13.48, 44.75, 14.25, 45.55), "Istria Peninsula"),
    "rijeka": ((14.05, 45.15, 14.65, 45.50), "Rijeka, Opatija & Ucka"),
    "gorski": ((14.45, 45.20, 15.05, 45.65), "Gorski Kotar (Delnice, Cabar)"),
    "islands": ((14.25, 44.45, 14.95, 45.25), "Kvarner Islands (Krk, Cres, Losinj, Rab)"),
    "ucka": ((14.08, 45.18, 14.28, 45.38), "Ucka Mountain & Opatija Riviera"),
    "pula": ((13.72, 44.78, 14.02, 45.00), "Pula & Southern Istria"),
    "rovinj": ((13.55, 45.00, 13.82, 45.22), "Rovinj & Limski Kanal"),
    "krk": ((14.42, 44.93, 14.82, 45.24), "Krk Island"),
    "cres": ((14.28, 44.50, 14.55, 45.12), "Cres & Losinj Islands"),
}


def load_merged_region(
    maps_dir: Path,
    bbox: tuple[float, float, float, float],
    max_dim: int = 600,
    color_by: str = "density",
    window: int = 5,
    hide_sea: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Merge all .tif/.hgt tiles in maps_dir cropped to bbox and compute metric."""
    import rasterio
    from rasterio.merge import merge

    files = sorted(
        list(maps_dir.glob("*.tif"))
        + list(maps_dir.glob("*.tiff"))
        + list(maps_dir.glob("*.hgt"))
    )
    if not files:
        raise FileNotFoundError(f"No .tif/.hgt map files found in {maps_dir}")

    srcs = [rasterio.open(f) for f in files]
    try:
        mosaic, transform = merge(srcs, bounds=bbox)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()

    z = mosaic[0].astype(np.float64)
    z = np.where(np.isnan(z) | (z < -500), 0.0, z)

    h, w = z.shape
    stride = max(int(np.ceil(max(h, w) / max_dim)), 1)
    z_sub = z[::stride, ::stride]

    metric, label = compute_raster_metric(z_sub, color_by=color_by, window=window)
    if hide_sea:
        metric = np.where(z_sub <= 0.5, np.nan, metric)

    rows, cols = np.meshgrid(
        np.arange(0, h, stride),
        np.arange(0, w, stride),
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


DMT20_PATH = Path(__file__).resolve().parent / "DMT Croatia 20m" / "DTM Croatia 20m v1 by Sonny.tif"


def load_dmt20_region(
    dmt_path: Path,
    bbox: tuple[float, float, float, float],
    max_dim: int = 600,
    color_by: str = "density",
    window: int = 5,
    hide_sea: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load cropped window from projected 20m Croatia DTM (EPSG:32633) and compute metric."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    if not dmt_path.exists():
        raise FileNotFoundError(f"DMT 20m file not found: {dmt_path}")

    with rasterio.open(dmt_path) as src:
        proj_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*proj_bounds, transform=src.transform).round_offsets().round_lengths()
        try:
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        except Exception:
            raise ValueError("Selected region is outside DMT Croatia 20m coverage.")
        if win.width <= 0 or win.height <= 0:
            raise ValueError("Selected region is outside DMT Croatia 20m coverage.")

        scale = max(int(np.ceil(max(win.height, win.width) / max_dim)), 1)
        out_h = max(int(win.height // scale), 2)
        out_w = max(int(win.width // scale), 2)
        z_sub = src.read(1, window=win, out_shape=(out_h, out_w)).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            z_sub = np.where(z_sub == nodata, 0.0, z_sub)
        z_sub = np.where(np.isnan(z_sub) | (z_sub < -500), 0.0, z_sub)

        win_transform = src.window_transform(win)
        res_x = win_transform.a * (win.width / out_w)
        res_y = abs(win_transform.e) * (win.height / out_h)

    metric, label = compute_raster_metric(z_sub, color_by=color_by, window=window)
    if hide_sea:
        metric = np.where(z_sub <= 0.5, np.nan, metric)

    xs_1d = (np.arange(out_w, dtype=np.float64) - out_w / 2.0) * res_x
    ys_1d = (out_h / 2.0 - np.arange(out_h, dtype=np.float64)) * res_y
    xs, ys = np.meshgrid(xs_1d, ys_1d)
    return xs, ys, z_sub, metric, label


def list_maps_and_regions(maps_dir: Path) -> None:
    """Print available map files, point cloud scans, and preset regions."""
    print("=== Map Files in", maps_dir, "===")
    files = sorted(
        list(maps_dir.glob("*.tif"))
        + list(maps_dir.glob("*.tiff"))
        + list(maps_dir.glob("*.hgt"))
    )
    if not files:
        print("  (none)")
    else:
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")

    las_dir = maps_dir.parent / "las_scans"
    if las_dir.exists():
        las_files = sorted(list(las_dir.glob("*.las")) + list(las_dir.glob("*.laz")))
        if las_files:
            print("\n=== Point Cloud Scans in", las_dir, "===")
            for lf in las_files:
                size_mb = lf.stat().st_size / (1024 * 1024)
                print(f"  - {lf.name} ({size_mb:.1f} MB)")

    if DMT20_PATH.exists():
        print("\n=== High-Resolution DTM Layer (--layer dmt20) ===")
        size_mb = DMT20_PATH.stat().st_size / (1024 * 1024)
        print(f"  - {DMT20_PATH.name} ({size_mb:.1f} MB)")

    print("\n=== Preset Regions (--region <name>) ===")
    for name, (_, desc) in REGIONS.items():
        print(f"  - {name:10s} : {desc}")

    print("\n=== Saved Material Values (materials.json) ===")
    for mat, val in sorted(load_materials().items()):
        print(f"  - {mat:12s} : {val:.2f} g/cm3")


def run_interactive_menu(args: argparse.Namespace) -> None:
    """Prompt the user interactively for region and rendering options."""
    region_keys = list(REGIONS.keys())
    print("\n--- 3D LiDAR / Elevation Map Cruiser ---")
    for i, key in enumerate(region_keys, 1):
        print(f"  {i:2d}) {key:10s} - {REGIONS[key][1]}")
    choice = input(f"Select region [1-{len(region_keys)}, default=1 (all)]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(region_keys):
        args.region = region_keys[int(choice) - 1]

    color_choice = input("Color by [density / lrm / hillshade / svf / slope / elevation] (default=density): ").strip().lower()
    if color_choice in {"density", "lrm", "hillshade", "svf", "slope", "elevation"}:
        args.color_by = color_choice

    detail_choice = input("Detail [low / medium / high / ultra / native] (default=medium): ").strip().lower()
    if detail_choice in DETAIL_MAX_DIM:
        args.detail = detail_choice

    sea_choice = input("Hide sea level so land density pops? [y/N]: ").strip().lower()
    args.hide_sea = sea_choice == "y"

    contour_choice = input("Show elevation contour lines? [y/N]: ").strip().lower()
    args.contours = contour_choice == "y"

    viewer_choice = input("Open in [1] 3D Window, [2] 3D Web Browser, or [3] 2D ArcGIS/GPS Map? (default=1): ").strip()
    args.browser = viewer_choice == "2"
    args.map2d = viewer_choice == "3"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cruise Istra & Primorsko-goranska 3D Elevation + Object Density Maps"
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=None,
        help="Optional specific .las/.laz/.tif/.hgt/.csv file to open",
    )
    parser.add_argument(
        "--file",
        dest="file_opt",
        type=Path,
        default=None,
        help="Optional specific .las/.laz/.tif/.hgt/.csv file to open",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available map files and preset regions, then exit",
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Launch an interactive terminal menu to pick region and options",
    )
    parser.add_argument(
        "--region",
        choices=list(REGIONS.keys()),
        default="all",
        help="Preset region to cruise (default: all)",
    )
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help="Custom high-detail sector as LON,LAT,RADIUS_KM (e.g. 13.85,44.87,3)",
    )
    parser.add_argument(
        "--detail",
        choices=list(DETAIL_MAX_DIM.keys()),
        default="medium",
        help="Mesh resolution level (default: medium; use 'native' for 1:1 sectors)",
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
        help="Color scale metric (default: density)",
    )
    parser.add_argument(
        "--layer",
        choices=["dsm", "dmt20"],
        default="dsm",
        help="Terrain data layer: dsm (Copernicus 30m maps/) or dmt20 (DMT Croatia 20m)",
    )
    parser.add_argument(
        "--dsm",
        type=Path,
        default=None,
        help="Optional DSM raster for (DSM - DTM) object height when using --file",
    )
    parser.add_argument(
        "--export-matches",
        type=Path,
        default=None,
        help="Export clustered --mark-material / --mark-value targets to .gpx, .geojson, or .csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output HTML path for --browser or --map2d",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically launch web browser for --browser or --map2d",
    )
    parser.add_argument(
        "--cmap",
        choices=list(COLORMAPS.keys()),
        default="turbo",
        help="Color palette (default: turbo, cold=low -> warm=high)",
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
        "--z-scale",
        type=float,
        default=4.0,
        help="Vertical exaggeration for 3D relief (default: 4.0)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open interactive 3D map in your web browser instead of PyVista window",
    )
    parser.add_argument(
        "--map2d",
        action="store_true",
        help="Open 2D ArcGIS/Esri Satellite + OpenTopoMap + Density overlay in browser",
    )
    parser.add_argument(
        "--gpx",
        type=Path,
        default=None,
        help="Overlay a GPS track/waypoints (.gpx, .kml, .geojson, .csv) on the map",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Save a PNG screenshot (3D window mode) and exit",
    )
    parser.add_argument(
        "--no-zoom-to-pointer",
        action="store_true",
        help="Disable zoom-to-pointer and use default center zoom",
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
    parser.add_argument(
        "--save-material",
        nargs=2,
        metavar=("NAME", "VALUE"),
        default=None,
        help="Save a custom material density/target value to materials.json and exit",
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Launch the desktop GUI app",
    )
    parser.add_argument(
        "--grid-km",
        type=float,
        default=None,
        help="Overlay a N-km grid on --map2d (e.g. 5 for 5 km squares)",
    )
    parser.add_argument(
        "--grid-cell",
        type=str,
        default=None,
        help="Zoom into a specific grid square ID (e.g. F10; defaults to 5 km grid)",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Speak a voice command using the hardware microphone",
    )
    parser.add_argument(
        "--mic-device",
        type=str,
        default="default",
        help="ALSA capture device for --mic (default: default)",
    )
    parser.add_argument(
        "--mic-lang",
        type=str,
        default="en-US",
        help="Speech recognition language for --mic (default: en-US)",
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
    args = parser.parse_args()
    if args.file is None and args.file_opt is not None:
        args.file = args.file_opt

    if args.app:
        from app import main as app_main

        return app_main()

    if args.mic:
        from llm_assistant import VALID_FILTERS, query_lmstudio
        from mic_input import MicRecorder, transcribe_audio

        rec = MicRecorder()
        print(f"Listening on mic '{args.mic_device}'... Press Enter to stop and run.")
        rec.start(device=args.mic_device)
        try:
            input()
        finally:
            wav = rec.stop()
        try:
            spoken = transcribe_audio(wav, language=args.mic_lang)
        finally:
            wav.unlink(missing_ok=True)
        print(f"Heard: {spoken}")
        cmd, src = query_lmstudio(spoken, load_materials())
        print(f"Parsed ({src}): {cmd}")
        if cmd.get("region") in REGIONS:
            args.region = str(cmd["region"])
        if cmd.get("grid_cell"):
            args.grid_km = args.grid_km or 5.0
            args.grid_cell = str(cmd["grid_cell"])
        elif cmd.get("source_mode") == "grid":
            args.grid_km = args.grid_km or 5.0
        if isinstance(cmd.get("sector"), dict):
            sec = cmd["sector"]
            if {"lon", "lat", "radius_km"} <= set(sec.keys()):
                args.sector = f"{sec['lon']},{sec['lat']},{sec['radius_km']}"
        if cmd.get("color_by") in VALID_FILTERS:
            args.color_by = str(cmd["color_by"])
        if cmd.get("cmap") in COLORMAPS:
            args.cmap = str(cmd["cmap"])
        if cmd.get("detail") in DETAIL_MAX_DIM:
            args.detail = str(cmd["detail"])
        if isinstance(cmd.get("hide_sea"), bool):
            args.hide_sea = bool(cmd["hide_sea"])
        if isinstance(cmd.get("contours"), bool):
            args.contours = bool(cmd["contours"])
        if cmd.get("mark_material"):
            args.mark_material = str(cmd["mark_material"])
        if cmd.get("view_mode") == "map2d":
            args.map2d = True
        elif cmd.get("view_mode") == "browser":
            args.browser = True

    maps_dir = Path(__file__).resolve().parent / "maps"
    if args.save_material is not None:
        try:
            mat_name, mat_val = args.save_material[0], float(args.save_material[1])
            if not np.isfinite(mat_val):
                raise ValueError
        except ValueError:
            print(
                f"Error: Invalid numeric value '{args.save_material[1]}' for material '{args.save_material[0]}'.",
                file=sys.stderr,
            )
            return 1
        save_material(mat_name, mat_val)
        print(f"Saved material '{mat_name.lower()}' = {mat_val} to materials.json")
        return 0

    mark_val = args.mark_value
    mark_label = None
    if args.mark_material is not None:
        materials = load_materials()
        key = args.mark_material.lower()
        if key not in materials:
            print(f"Error: Unknown material '{args.mark_material}'. Use --list or --save-material.", file=sys.stderr)
            return 1
        mark_val = materials[key]
        mark_label = f"{key} ({mark_val:.2f})"

    if args.list:
        list_maps_and_regions(maps_dir)
        return 0

    if args.menu and args.file is None:
        run_interactive_menu(args)

    if args.file is not None:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
        ext = args.file.suffix.lower()
        if ext == ".csv":
            try:
                x, y, z, metric, label = load_csv_scan(
                    args.file, min_depth=args.min_depth, max_depth=args.max_depth
                )
            except Exception as exc:
                print(f"Error: Failed to load CSV scan '{args.file}': {exc}", file=sys.stderr)
                return 1
            is_grid = False
        elif ext in {".las", ".laz"}:
            pts_by_detail = {
                "low": 200_000,
                "medium": 500_000,
                "high": 1_500_000,
                "ultra": 4_000_000,
                "native": 0,
            }
            max_pts = pts_by_detail.get(args.detail, 500_000)
            try:
                x, y, z, metric, label = load_point_cloud(
                    args.file, max_pts, None, color_by=args.color_by
                )
            except Exception as exc:
                print(f"Error: Failed to load point cloud '{args.file}': {exc}", file=sys.stderr)
                return 1
            is_grid = False
        elif ext in {".hgt", ".tif", ".tiff"}:
            stride = max(6 - list(DETAIL_MAX_DIM.keys()).index(args.detail) * 2, 1)
            try:
                x, y, z, metric, label = load_raster(
                    args.file,
                    stride=stride,
                    dsm_path=args.dsm,
                    color_by=args.color_by,
                    window=max(args.window, 3),
                    hide_sea=args.hide_sea,
                )
            except Exception as exc:
                print(f"Error: Failed to load raster '{args.file}': {exc}", file=sys.stderr)
                return 1
            is_grid = True
        else:
            print(
                f"Error: Unsupported file format '{ext}'. Supported: .las, .laz, .tif, .hgt, .csv",
                file=sys.stderr,
            )
            return 1
        bbox = (0.0, 0.0, 1.0, 1.0)
        grid_cells_overlay = None
        grid_km_val = args.grid_km
        title = f"3D Elevation & {args.color_by.title()} — {args.file.name}"
    else:
        if args.sector is not None:
            try:
                lon_s, lat_s, rad_s = (float(v.strip()) for v in args.sector.split(","))
                if not (np.isfinite(lon_s) and np.isfinite(lat_s) and np.isfinite(rad_s)) or rad_s <= 0:
                    raise ValueError
            except ValueError:
                print(
                    "Error: --sector must be LON,LAT,RADIUS_KM with positive radius (e.g. 13.85,44.87,3)",
                    file=sys.stderr,
                )
                return 1
            dlat = rad_s / 110.54
            dlon = rad_s / (111.32 * max(float(np.cos(np.deg2rad(lat_s))), 0.1))
            bbox = (lon_s - dlon, lat_s - dlat, lon_s + dlon, lat_s + dlat)
            desc = f"Sector ({lat_s:.4f}N, {lon_s:.4f}E, r={rad_s:g}km)"
        else:
            bbox, desc = REGIONS[args.region]

        grid_cells_overlay = None
        grid_km_val = args.grid_km
        if grid_km_val is not None:
            try:
                val = float(grid_km_val)
                grid_km_val = val if np.isfinite(val) and val > 0 else 5.0
            except (TypeError, ValueError):
                grid_km_val = 5.0
        if args.grid_cell is not None:
            grid_km_val = grid_km_val or 5.0
            cells, n_rows, n_cols = compute_region_grid(bbox, cell_km=grid_km_val)
            target_id = args.grid_cell.strip().upper()
            match = next((c for c in cells if str(c["id"]).upper() == target_id), None)
            if match is None:
                print(
                    f"Error: Unknown grid cell '{args.grid_cell}' for region '{args.region}' "
                    f"({n_cols} cols x {n_rows} rows, e.g. A1..{cells[-1]['id']}).",
                    file=sys.stderr,
                )
                return 1
            bbox = match["bbox"]  # type: ignore[assignment]
            desc = f"{desc} [Grid {match['id']} ({grid_km_val:g} km)]"
        elif grid_km_val is not None:
            grid_cells_overlay, n_rows, n_cols = compute_region_grid(bbox, cell_km=grid_km_val)
            print(f"Grid ({grid_km_val:g} km): {n_cols} cols x {n_rows} rows ({len(grid_cells_overlay)} squares)")

        max_dim = DETAIL_MAX_DIM[args.detail]
        if args.layer == "dmt20":
            try:
                x, y, z, metric, label = load_dmt20_region(
                    DMT20_PATH,
                    bbox,
                    max_dim=max_dim,
                    color_by=args.color_by,
                    window=max(args.window, 3),
                    hide_sea=args.hide_sea,
                )
            except Exception as exc:
                print(f"Error: Failed to load DMT 20m layer: {exc}", file=sys.stderr)
                return 1
            desc = f"{desc} [DMT 20m]"
        else:
            try:
                x, y, z, metric, label = load_merged_region(
                    maps_dir,
                    bbox,
                    max_dim=max_dim,
                    color_by=args.color_by,
                    window=max(args.window, 3),
                    hide_sea=args.hide_sea,
                )
            except Exception as exc:
                print(f"Error: Failed to load terrain data: {exc}", file=sys.stderr)
                return 1
        is_grid = True
        title = f"{desc} — 3D Elevation & {args.color_by.title()}"

    if mark_val is not None:
        warn_material_ambiguity(args.mark_material, mark_val, args.tolerance, label)
        if args.export_matches is not None:
            clusters = cluster_material_matches(
                metric,
                z,
                mark_val,
                args.tolerance,
                bbox_lonlat=bbox if args.file is None else None,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                x=x,
                y=y,
            )
            export_matches(clusters, args.export_matches, label=mark_label or "Target")
    elif args.export_matches is not None:
        print(
            "Notice: --export-matches requires --mark-material or --mark-value to define target criteria.",
            file=sys.stderr,
        )

    if args.map2d:
        if args.file is not None:
            print("Note: --map2d uses the selected --region / --sector bounds.", file=sys.stderr)
            bbox, desc = REGIONS[args.region]
            grid_cells_overlay = None
            grid_km_val = args.grid_km
        out_html = (
            args.out
            if args.out is not None
            else Path(__file__).resolve().parent / f"map_2d_{args.region}_{args.color_by}.html"
        )
        render_folium_map(
            metric,
            bbox,
            title,
            out_html,
            cmap=args.cmap,
            gpx_path=args.gpx,
            scalar_label=label,
            auto_open=not args.no_open,
            mark_value=mark_val,
            mark_tolerance=args.tolerance,
            mark_label=mark_label,
            grid_cells=grid_cells_overlay,
            grid_km=grid_km_val or 5.0,
            grid_region=args.region,
        )
    elif args.browser:
        stem = args.file.stem if args.file is not None else args.region
        out_html = (
            args.out
            if args.out is not None
            else Path(__file__).resolve().parent / f"map_3d_{stem}_{args.color_by}.html"
        )
        render_plotly(
            x,
            y,
            z,
            metric,
            is_grid,
            args.z_scale,
            title,
            out_html,
            cmap=args.cmap,
            scalar_label=label,
            contours=args.contours,
            auto_open=not args.no_open,
            mark_value=mark_val,
            mark_tolerance=args.tolerance,
            mark_label=mark_label,
        )
    else:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and args.screenshot is None:
            print(
                "Notice: Headless environment detected without DISPLAY/WAYLAND. "
                "Use --screenshot <path.png>, --browser, or --map2d.",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
