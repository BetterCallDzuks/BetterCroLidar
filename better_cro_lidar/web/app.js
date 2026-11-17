(() => {
  const map = L.map("map", { zoomControl: false, attributionControl: false }).setView([44.55, 16.40], 7);

  const BASEMAPS = {
    "ArcGIS Satellite (No Watermark)": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19 }),
    "Esri Clarity Satellite (Sharp)": L.tileLayer("https://clarity.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19 }),
    "Esri Hillshade (Terrain Relief)": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}", { maxZoom: 16 }),
    "Dark Matter (Radar / Scan)": L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { maxZoom: 19 }),
    "Esri Dark Gray (Tactical)": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", { maxZoom: 16 }),
    "Esri World Topo": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19 }),
    "OpenTopoMap (Contours)": L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", { maxZoom: 17 })
  };

  let activeBasemapName = "ArcGIS Satellite (No Watermark)";
  BASEMAPS[activeBasemapName].addTo(map);

  const WMS_OVERLAYS = {
    "DGU 25cm LiDAR Orthophoto (WMS)": L.tileLayer.wms("https://geoportal.dgu.hr/services/inspire/orthophoto_lidar_2022_2023/wms", {
      layers: "OI.OrthoimageCoverage", format: "image/jpeg", transparent: false, opacity: 0.85
    }),
    "DGU HOK 1:5000 (Croatian Topo WMS)": L.tileLayer.wms("https://geoportal.dgu.hr/services/hok/wms", {
      layers: "HOK", format: "image/png", transparent: true, opacity: 0.75
    }),
    "ZRC-SAZU 0.5m DFM (Relief)": L.tileLayer.wms("https://lidar.zrc-sazu.si/geoserver/wms", {
      layers: "ALS_Europe:CFVGS_DFM_05m_CVAT", format: "image/png", transparent: true, opacity: 0.65
    })
  };

  let meta = { regions: [], modes: {}, colormaps: [], materials: [], sensor_points: [] };
  let activeMode = null;
  let rasterOverlay = null;
  const matchGroup = L.layerGroup().addTo(map);
  const gridGroup = L.layerGroup();
  const sensorGroup = L.layerGroup();
  const customGroup = L.layerGroup().addTo(map);
  const profileGroup = L.layerGroup().addTo(map);
  const measureGroup = L.layerGroup().addTo(map);

  let measureActive = false;
  let measurePts = [];
  let profileActive = false;
  let profilePts = [];
  let notesState = { tasks: [], notes: "" };

  function getCurrentBBox() {
    const b = map.getBounds();
    return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  }

  // Sidebar collapse & tabs
  document.getElementById("left-toggle").onclick = function () {
    const sb = document.getElementById("left-sidebar");
    const c = sb.classList.toggle("collapsed");
    this.textContent = c ? "»" : "«";
  };
  document.getElementById("right-toggle").onclick = function () {
    const sb = document.getElementById("right-sidebar");
    const c = sb.classList.toggle("collapsed");
    this.textContent = c ? "«" : "»";
  };
  document.querySelectorAll(".tbtn").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".tbtn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("pane-" + btn.dataset.tab).classList.add("active");
    };
  });

  // Quick tools
  document.getElementById("btn-zoom-in").onclick = () => map.zoomIn();
  document.getElementById("btn-zoom-out").onclick = () => map.zoomOut();
  document.getElementById("btn-fit-region").onclick = () => jumpToRegion(document.getElementById("region-select").value);
  document.getElementById("btn-measure").onclick = function () {
    measureActive = !measureActive;
    profileActive = false;
    this.classList.toggle("active", measureActive);
    document.getElementById("hud-measure").style.display = measureActive ? "block" : "none";
    if (!measureActive) {
      measurePts = [];
      measureGroup.clearLayers();
      document.getElementById("hud-measure").textContent = "Dist: 0 m";
    }
  };
  document.getElementById("btn-clear-overlays").onclick = clearAllOverlays;

  function clearAllOverlays() {
    activeMode = null;
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    if (rasterOverlay) { map.removeLayer(rasterOverlay); rasterOverlay = null; }
    matchGroup.clearLayers();
    document.getElementById("hud-legend").style.display = "none";
    Object.values(WMS_OVERLAYS).forEach((l) => { if (map.hasLayer(l)) map.removeLayer(l); });
    document.querySelectorAll("#wms-list input[type=checkbox]").forEach((cb) => { cb.checked = false; });
    if (map.hasLayer(gridGroup)) map.removeLayer(gridGroup);
    if (map.hasLayer(sensorGroup)) map.removeLayer(sensorGroup);
    document.querySelectorAll("#vector-list input[type=checkbox]").forEach((cb) => { cb.checked = false; });
  }

  function jumpToRegion(key) {
    const r = meta.regions.find((x) => x.key === key);
    if (!r) return;
    const [w, s, e, n] = r.bbox;
    map.fitBounds([[s, w], [n, e]]);
  }

  // Image filters
  function applyImageFilters() {
    const b = document.getElementById("f-bright").value;
    const c = document.getElementById("f-contrast").value;
    const s = document.getElementById("f-sat").value;
    const h = document.getElementById("f-hue").value;
    const inv = document.getElementById("f-invert").checked ? 1 : 0;
    const gr = document.getElementById("f-gray").checked ? 1 : 0;
    document.getElementById("v-bright").textContent = b + "%";
    document.getElementById("v-contrast").textContent = c + "%";
    document.getElementById("v-sat").textContent = s + "%";
    document.getElementById("v-hue").textContent = h + "°";
    const pane = map.getPane("tilePane");
    if (pane) {
      pane.style.filter = `brightness(${b}%) contrast(${c}%) saturate(${s}%) hue-rotate(${h}deg) invert(${inv}) grayscale(${gr})`;
    }
  }
  ["f-bright", "f-contrast", "f-sat", "f-hue", "f-invert", "f-gray"].forEach((id) => {
    document.getElementById(id).addEventListener("input", applyImageFilters);
    document.getElementById(id).addEventListener("change", applyImageFilters);
  });
  document.querySelectorAll(".pbtn[data-preset]").forEach((btn) => {
    btn.onclick = () => {
      const p = btn.dataset.preset;
      const sets = {
        default: [100, 100, 100, 0, false, false],
        night: [85, 130, 60, 190, true, false],
        contrast: [105, 165, 120, 0, false, false],
        edge: [110, 180, 190, 270, false, false]
      }[p];
      if (!sets) return;
      document.getElementById("f-bright").value = sets[0];
      document.getElementById("f-contrast").value = sets[1];
      document.getElementById("f-sat").value = sets[2];
      document.getElementById("f-hue").value = sets[3];
      document.getElementById("f-invert").checked = sets[4];
      document.getElementById("f-gray").checked = sets[5];
      applyImageFilters();
    };
  });

  // Render dynamic overlay
  async function renderDynamicOverlay(mode, withMaterial = false) {
    activeMode = mode;
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    const [w, s, e, n] = getCurrentBBox();
    const cmap = document.getElementById("cmap-select").value;
    const op = Number(document.getElementById("f-opacity").value) / 100.0;
    const params = new URLSearchParams({
      lon_min: w, lat_min: s, lon_max: e, lat_max: n,
      mode, cmap, opacity: op.toFixed(2), max_dim: "680"
    });
    if (withMaterial) {
      const matVal = document.getElementById("mat-select").value;
      const tol = document.getElementById("f-tol").value;
      if (matVal) {
        params.set("material_target", matVal);
        params.set("material_tol", tol);
      }
    }
    const hudQ = document.getElementById("hud-query");
    hudQ.textContent = `Rendering ${mode}...`;
    const res = await fetch("/api/overlay?" + params.toString()).then((r) => r.json());
    if (!res.ok) {
      hudQ.textContent = res.error || "Overlay render failed";
      return;
    }
    if (rasterOverlay) map.removeLayer(rasterOverlay);
    rasterOverlay = L.imageOverlay(res.image_data, res.bounds, { opacity: 1.0, interactive: false }).addTo(map);
    hudQ.textContent = `${res.label} rendered`;

    // Legend
    const leg = document.getElementById("hud-legend");
    leg.style.display = "flex";
    document.getElementById("legend-label").textContent = res.label + ":";
    document.getElementById("legend-min").textContent = res.vmin;
    document.getElementById("legend-max").textContent = res.vmax;
    document.getElementById("legend-bar").style.background = `linear-gradient(90deg, ${res.palette.join(",")})`;

    // Material matches
    matchGroup.clearLayers();
    if (withMaterial) {
      document.getElementById("mat-status").textContent = `Found ${res.match_count} matches in current view (showing ${res.matches.length}).`;
      res.matches.forEach((pt) => {
        L.circleMarker([pt.lat, pt.lon], {
          radius: 4, color: "#ffd700", fillColor: "#ffd700", fillOpacity: 0.88, weight: 1
        }).bindPopup(`<b>Material Hit</b><br>Val: ${pt.val}<br>${pt.lat}, ${pt.lon}`).addTo(matchGroup);
      });
    }
  }

  document.getElementById("btn-render-view").onclick = () => renderDynamicOverlay(activeMode || "density", false);
  document.getElementById("btn-scan-material").onclick = () => renderDynamicOverlay(activeMode || "density", true);
  document.getElementById("f-opacity").oninput = function () {
    document.getElementById("v-opacity").textContent = this.value + "%";
    if (rasterOverlay) rasterOverlay.setOpacity(Number(this.value) / 100.0);
  };
  document.getElementById("f-tol").oninput = function () {
    document.getElementById("v-tol").textContent = Number(this.value).toFixed(2);
  };

  // Profile tool
  document.getElementById("btn-start-profile").onclick = () => {
    profileActive = true;
    measureActive = false;
    document.getElementById("btn-measure").classList.remove("active");
    profilePts = [];
    profileGroup.clearLayers();
    document.getElementById("profile-stats").textContent = "Click point A and point B on map...";
  };
  document.getElementById("btn-clear-profile").onclick = () => {
    profileActive = false;
    profilePts = [];
    profileGroup.clearLayers();
    document.getElementById("profile-stats").textContent = "";
    const ctx = document.getElementById("profile-canvas").getContext("2d");
    ctx.clearRect(0, 0, 276, 170);
  };

  async function runProfile(p1, p2) {
    L.polyline([p1, p2], { color: "#38bdf8", weight: 2.5, dashArray: "5,5" }).addTo(profileGroup);
    const params = new URLSearchParams({ lat1: p1.lat, lon1: p1.lng, lat2: p2.lat, lon2: p2.lng, samples: 100 });
    const data = await fetch("/api/profile?" + params.toString()).then((r) => r.json());
    if (!data.ok) return;
    document.getElementById("profile-stats").textContent = `Length: ${data.total_dist_m.toFixed(0)} m (${data.points.length} samples)`;
    drawProfileChart(data.points);
  }

  function drawProfileChart(pts) {
    const cvs = document.getElementById("profile-canvas");
    const ctx = cvs.getContext("2d");
    const W = cvs.width, H = cvs.height;
    ctx.clearRect(0, 0, W, H);
    const elevs = pts.map((p) => p.elev_m).filter((v) => v !== null);
    if (!elevs.length) return;
    const eMin = Math.min(...elevs), eMax = Math.max(...elevs, eMin + 1);
    const dens = pts.map((p) => p.density).filter((v) => v !== null);
    const dMin = dens.length ? Math.min(...dens) : 0;
    const dMax = dens.length ? Math.max(...dens, dMin + 0.1) : 1;

    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => {
      if (p.elev_m === null) return;
      const x = (i / Math.max(pts.length - 1, 1)) * (W - 20) + 10;
      const y = H - 18 - ((p.elev_m - eMin) / (eMax - eMin)) * (H - 36);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    if (dens.length) {
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      pts.forEach((p, i) => {
        if (p.density === null) return;
        const x = (i / Math.max(pts.length - 1, 1)) * (W - 20) + 10;
        const y = H - 18 - ((p.density - dMin) / (dMax - dMin)) * (H - 36);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px sans-serif";
    ctx.fillText(`Elev: ${eMin.toFixed(0)}–${eMax.toFixed(0)}m (cyan) | Density (amber)`, 8, 12);
  }

  // Map mousemove & click
  map.on("mousemove", (e) => {
    document.getElementById("hud-coords").textContent = `Lat: ${e.latlng.lat.toFixed(5)} | Lon: ${e.latlng.lng.toFixed(5)}`;
  });

  map.on("click", async (e) => {
    if (measureActive) {
      measurePts.push(e.latlng);
      L.circleMarker(e.latlng, { radius: 4, color: "#38bdf8", fillOpacity: 1 }).addTo(measureGroup);
      if (measurePts.length > 1) {
        L.polyline(measurePts, { color: "#38bdf8", weight: 2 }).addTo(measureGroup);
        let dist = 0;
        for (let i = 1; i < measurePts.length; i++) dist += measurePts[i - 1].distanceTo(measurePts[i]);
        document.getElementById("hud-measure").textContent = dist >= 1000 ? `Dist: ${(dist / 1000).toFixed(2)} km` : `Dist: ${dist.toFixed(1)} m`;
      }
      return;
    }
    if (profileActive) {
      profilePts.push(e.latlng);
      L.circleMarker(e.latlng, { radius: 5, color: "#38bdf8", fillOpacity: 1 }).addTo(profileGroup);
      if (profilePts.length === 2) {
        profileActive = false;
        runProfile(profilePts[0], profilePts[1]);
      }
      return;
    }
    const q = await fetch(`/api/query?lat=${e.latlng.lat}&lon=${e.latlng.lng}`).then((r) => r.json());
    if (q.ok) {
      document.getElementById("hud-query").textContent =
        `Elev: ${q.elevation_m ?? "—"} m | Slope: ${q.slope_deg ?? "—"}° | Density: ${q.density ?? "—"} | Openness: ${q.openness ?? "—"}`;
    }
  });

  // Upload CSV / GeoJSON
  document.getElementById("file-upload").addEventListener("change", (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const txt = String(reader.result || "");
      if (file.name.endsWith(".geojson") || file.name.endsWith(".json")) {
        L.geoJSON(JSON.parse(txt), { style: { color: "#22d3ee", weight: 2 } }).addTo(customGroup);
      } else {
        const lines = txt.trim().split(/\r?\n/);
        const hdr = lines[0].split(",").map((s) => s.trim().toLowerCase());
        const iLat = hdr.indexOf("lat"), iLon = hdr.indexOf("lon");
        if (iLat >= 0 && iLon >= 0) {
          lines.slice(1).forEach((ln) => {
            const cols = ln.split(",");
            const la = parseFloat(cols[iLat]), lo = parseFloat(cols[iLon]);
            if (Number.isFinite(la) && Number.isFinite(lo)) {
              L.circleMarker([la, lo], { radius: 5, color: "#22d3ee", fillOpacity: 0.85 }).bindPopup(ln).addTo(customGroup);
            }
          });
        }
      }
    };
    reader.readAsText(file);
  });
  document.getElementById("btn-clear-pins").onclick = () => customGroup.clearLayers();
  document.getElementById("btn-export-geojson").onclick = () => {
    const blob = new Blob([JSON.stringify(customGroup.toGeoJSON(), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "better_cro_lidar_export.geojson";
    a.click();
  };

  // Tasks & Field Notes
  async function saveNotes() {
    await fetch("/api/notes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(notesState)
    });
  }
  function renderTasks() {
    const ul = document.getElementById("task-list");
    ul.innerHTML = "";
    notesState.tasks.forEach((t, idx) => {
      const li = document.createElement("li");
      if (t.done) li.classList.add("done");
      const lbl = document.createElement("label");
      lbl.style.cssText = "display:flex;align-items:center;gap:6px;flex:1;cursor:pointer;";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !!t.done;
      cb.onchange = () => { t.done = cb.checked; renderTasks(); saveNotes(); };
      const sp = document.createElement("span");
      sp.textContent = t.text;
      lbl.append(cb, sp);
      const del = document.createElement("button");
      del.className = "task-del";
      del.textContent = "×";
      del.onclick = () => { notesState.tasks.splice(idx, 1); renderTasks(); saveNotes(); };
      li.append(lbl, del);
      ul.appendChild(li);
    });
  }
  document.getElementById("btn-add-task").onclick = () => {
    const inp = document.getElementById("task-input");
    const v = inp.value.trim();
    if (!v) return;
    notesState.tasks.push({ text: v, done: false });
    inp.value = "";
    renderTasks();
    saveNotes();
  };
  document.getElementById("field-notes").addEventListener("input", function () {
    notesState.notes = this.value;
    saveNotes();
  });

  // Init
  async function init() {
    meta = await fetch("/api/info").then((r) => r.json());
    const regSel = document.getElementById("region-select");
    meta.regions.forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r.key;
      opt.textContent = r.desc;
      regSel.appendChild(opt);
    });
    regSel.value = "croatia";
    regSel.onchange = () => jumpToRegion(regSel.value);

    // Basemaps
    const bmList = document.getElementById("basemap-list");
    Object.keys(BASEMAPS).forEach((name) => {
      const lbl = document.createElement("label");
      const rb = document.createElement("input");
      rb.type = "radio";
      rb.name = "basemap";
      rb.checked = name === activeBasemapName;
      rb.onchange = () => {
        map.removeLayer(BASEMAPS[activeBasemapName]);
        activeBasemapName = name;
        BASEMAPS[name].addTo(map);
      };
      lbl.append(rb, document.createTextNode(name));
      bmList.appendChild(lbl);
    });

    // Analytics modes
    const modeBox = document.getElementById("analytics-modes");
    Object.entries(meta.modes).forEach(([k, info]) => {
      const btn = document.createElement("button");
      btn.className = "mode-btn";
      btn.dataset.mode = k;
      btn.textContent = info.label;
      btn.onclick = () => {
        document.getElementById("cmap-select").value = info.default_cmap;
        renderDynamicOverlay(k, false);
      };
      modeBox.appendChild(btn);
    });

    // WMS overlays (all unchecked on fresh start)
    const wmsBox = document.getElementById("wms-list");
    Object.entries(WMS_OVERLAYS).forEach(([name, layer]) => {
      const lbl = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = false;
      cb.onchange = () => { if (cb.checked) layer.addTo(map); else map.removeLayer(layer); };
      lbl.append(cb, document.createTextNode(name));
      wmsBox.appendChild(lbl);
    });

    // Vector overlays (all unchecked on fresh start)
    const vecBox = document.getElementById("vector-list");
    const gridLbl = document.createElement("label");
    const gridCb = document.createElement("input");
    gridCb.type = "checkbox";
    gridCb.onchange = async () => {
      if (gridCb.checked) {
        gridGroup.clearLayers();
        const [w, s, e, n] = getCurrentBBox();
        const g = await fetch(`/api/grid?lon_min=${w}&lat_min=${s}&lon_max=${e}&lat_max=${n}`).then((r) => r.json());
        g.cells.forEach((c) => {
          L.rectangle([[c.bbox[0], c.bbox[1]], [c.bbox[2], c.bbox[3]]], { color: "#38bdf8", weight: 1, fillOpacity: 0.03 })
            .bindTooltip(`Cell ${c.id}`).addTo(gridGroup);
        });
        gridGroup.addTo(map);
      } else {
        map.removeLayer(gridGroup);
      }
    };
    gridLbl.append(gridCb, document.createTextNode("5 km Grid (Current View)"));
    vecBox.appendChild(gridLbl);

    const senLbl = document.createElement("label");
    const senCb = document.createElement("input");
    senCb.type = "checkbox";
    senCb.onchange = () => {
      if (senCb.checked) {
        sensorGroup.clearLayers();
        meta.sensor_points.forEach((pt) => {
          const isGold = Math.abs(pt.density - 19.32) <= 0.5;
          L.circleMarker([pt.lat, pt.lon], {
            radius: isGold ? 7 : 5,
            color: "#020617",
            weight: 1.5,
            fillColor: isGold ? "#ffd700" : "#38bdf8",
            fillOpacity: 0.95
          }).bindPopup(`Density: ${pt.density} g/cm³<br>Depth: ${pt.depth} m`).addTo(sensorGroup);
        });
        sensorGroup.addTo(map);
      } else {
        map.removeLayer(sensorGroup);
      }
    };
    senLbl.append(senCb, document.createTextNode("Sample Sensor CSV"));
    vecBox.appendChild(senLbl);

    // Colormaps & Materials
    const cmapSel = document.getElementById("cmap-select");
    meta.colormaps.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c; opt.textContent = c;
      cmapSel.appendChild(opt);
    });
    const matSel = document.getElementById("mat-select");
    meta.materials.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.density; opt.textContent = `${m.name} (${m.density} ${m.unit})`;
      matSel.appendChild(opt);
    });

    // Load Notes
    notesState = await fetch("/api/notes").then((r) => r.json());
    document.getElementById("field-notes").value = notesState.notes || "";
    renderTasks();
  }

  init();
})();
