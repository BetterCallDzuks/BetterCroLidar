"""Croatian region presets and 5 km grid utilities for BetterCroLidar."""

from __future__ import annotations

import math

# (lon_min, lat_min, lon_max, lat_max), description
CROATIA_REGIONS: dict[str, tuple[tuple[float, float, float, float], str]] = {
    "croatia": ((13.45, 42.38, 19.45, 46.56), "All Croatia (Full Country)"),
    "istra": ((13.48, 44.75, 14.25, 45.55), "Istria Peninsula"),
    "kvarner": ((14.05, 44.45, 15.05, 45.65), "Kvarner, Rijeka & Gorski Kotar"),
    "zagreb": ((15.55, 45.60, 16.35, 46.05), "Zagreb & Medvednica"),
    "zagorje": ((15.60, 45.95, 16.95, 46.56), "Hrvatsko Zagorje & Međimurje"),
    "slavonia": ((17.20, 45.00, 19.45, 45.95), "Slavonia & Baranja"),
    "lika": ((14.85, 44.20, 16.20, 45.15), "Lika & Velebit"),
    "dalmatia_north": ((14.70, 43.45, 16.45, 44.45), "Northern Dalmatia (Zadar, Šibenik)"),
    "dalmatia_central": ((16.05, 43.05, 17.55, 43.80), "Central Dalmatia (Split, Brač, Hvar)"),
    "dalmatia_south": ((17.30, 42.38, 18.55, 43.15), "Southern Dalmatia (Dubrovnik, Pelješac)"),
}


def build_grid_cells(
    bbox: tuple[float, float, float, float], step_km: float = 5.0
) -> list[dict[str, object]]:
    """Generate a lightweight labeled 5 km grid over bbox (capped for full-country view)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_mid = 0.5 * (lat_min + lat_max)
    dlat = step_km / 111.32
    dlon = step_km / (111.32 * max(math.cos(math.radians(lat_mid)), 0.2))

    # Avoid generating thousands of DOM rectangles on full-country view
    cols_est = max(1, int(math.ceil((lon_max - lon_min) / dlon)))
    rows_est = max(1, int(math.ceil((lat_max - lat_min) / dlat)))
    if cols_est * rows_est > 350:
        scale = math.sqrt((cols_est * rows_est) / 250.0)
        dlat *= scale
        dlon *= scale

    cells: list[dict[str, object]] = []
    r = 0
    lat = lat_min
    while lat < lat_max and r < 26:
        c = 0
        lon = lon_min
        row_letter = chr(ord("A") + r)
        while lon < lon_max and c < 40:
            c += 1
            c1 = min(lon + dlon, lon_max)
            r1 = min(lat + dlat, lat_max)
            cells.append(
                {
                    "id": f"{row_letter}{c}",
                    "bbox": [round(lat, 5), round(lon, 5), round(r1, 5), round(c1, 5)],
                    "center": [round(0.5 * (lat + r1), 5), round(0.5 * (lon + c1), 5)],
                }
            )
            lon += dlon
        lat += dlat
        r += 1
    return cells
