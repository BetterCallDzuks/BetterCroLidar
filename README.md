# BetterCroLidar

Croatia-wide LiDAR & Terrain Studio native desktop application.

## Windows Setup (After Cloning)
1. Install **Python 3.10+** (check *"Add Python to PATH"* during installation).
2. Clone the repository:
   ```cmd
   git clone https://github.com/BetterCallDzuks/BetterCroLidar.git
   cd BetterCroLidar
   ```
3. *(Optional for offline DMT 20m terrain analytics)* Place `DTM Croatia 20m v1 by Sonny.tif` inside:
   ```text
   BetterCroLidar\DMT Croatia 20m\DTM Croatia 20m v1 by Sonny.tif
   ```
   *(Online basemaps, DGU 25cm LiDAR WMS, DGU HOK 1:5000, ZRC-SAZU 0.5m DFM, image filters, measure, CSV/GeoJSON upload, and notes work even without the local GeoTIFF.)*
4. Double-click **`run_windows.bat`** (or run it from Command Prompt). On first launch it creates `.venv`, installs dependencies from `requirements.txt`, and opens **BetterCroLidar** as a native desktop window via Microsoft Edge / Chrome / Brave.

## Linux Launch
```bash
./better-cro-lidar
```
