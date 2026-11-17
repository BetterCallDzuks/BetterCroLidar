"""FastAPI backend server for BetterCroLidar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import (
    COLORMAPS,
    OVERLAY_MODES,
    get_materials_list,
    get_sample_sensor_points,
    profile_line,
    query_point,
    render_overlay,
)
from .regions import CROATIA_REGIONS, build_grid_cells

PKG_DIR = Path(__file__).resolve().parent
WEB_DIR = PKG_DIR / "web"
ROOT_DIR = PKG_DIR.parent
NOTES_PATH = ROOT_DIR / "better_cro_lidar_notes.json"

app = FastAPI(title="BetterCroLidar", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class NotesPayload(BaseModel):
    tasks: list[dict[str, Any]] = []
    notes: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/info")
def api_info() -> dict[str, Any]:
    regions = [
        {"key": k, "bbox": list(v[0]), "desc": v[1]}
        for k, v in CROATIA_REGIONS.items()
    ]
    return {
        "app": "BetterCroLidar",
        "regions": regions,
        "modes": OVERLAY_MODES,
        "colormaps": COLORMAPS,
        "materials": get_materials_list(),
        "sensor_points": get_sample_sensor_points(),
    }


@app.get("/api/grid")
def api_grid(
    lon_min: float = Query(...),
    lat_min: float = Query(...),
    lon_max: float = Query(...),
    lat_max: float = Query(...),
) -> dict[str, Any]:
    cells = build_grid_cells((lon_min, lat_min, lon_max, lat_max), step_km=5.0)
    return {"cells": cells}


@app.get("/api/overlay")
def api_overlay(
    lon_min: float = Query(...),
    lat_min: float = Query(...),
    lon_max: float = Query(...),
    lat_max: float = Query(...),
    mode: str = Query("density"),
    cmap: str = Query("turbo"),
    opacity: float = Query(0.65),
    max_dim: int = Query(650),
    material_target: float | None = Query(None),
    material_tol: float = Query(0.5),
) -> JSONResponse:
    res = render_overlay(
        bbox=(lon_min, lat_min, lon_max, lat_max),
        mode=mode,
        cmap_name=cmap,
        opacity=opacity,
        max_dim=max(120, min(max_dim, 1000)),
        material_target=material_target,
        material_tol=material_tol,
    )
    return JSONResponse(res)


@app.get("/api/query")
def api_query(lat: float = Query(...), lon: float = Query(...)) -> dict[str, Any]:
    return query_point(lat, lon)


@app.get("/api/profile")
def api_profile(
    lat1: float = Query(...),
    lon1: float = Query(...),
    lat2: float = Query(...),
    lon2: float = Query(...),
    samples: int = Query(100),
) -> dict[str, Any]:
    return profile_line(lat1, lon1, lat2, lon2, samples=max(20, min(samples, 250)))


@app.get("/api/notes")
def api_get_notes() -> dict[str, Any]:
    if NOTES_PATH.exists():
        try:
            return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"tasks": [], "notes": ""}


@app.post("/api/notes")
def api_save_notes(payload: NotesPayload) -> dict[str, Any]:
    data = {"tasks": payload.tasks, "notes": payload.notes}
    NOTES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}
