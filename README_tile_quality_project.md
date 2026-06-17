# Tile-Based Plantation UAV Image Quality Project

This is the recommended version of the quality gate. It evaluates each large orthomosaic by tiles, predicts bad-quality areas, then summarizes the result at image level.

## Files

- `config.py`: edit your folder paths, output paths, tile size, and thresholds here.
- `quality_core.py`: shared functions for image loading, masking, metric extraction, modeling, and heatmaps.
- `01_extract_metrics.py`: extracts image-level and tile-level metrics from labeled `good` and `bad` folders.
- `02_train_model.py`: trains a tile quality model.
- `03_predict_quality.py`: scores new image folders or a single image.
- `04_make_heatmaps.py`: creates visual quality heatmaps from predictions.

## Run Order

Run these commands from the `outputs` folder.

1. Install dependencies if needed:

```powershell
pip install -r requirements.txt
```

2. Edit `config.py`.
3. Run extraction:

```powershell
python 01_extract_metrics.py
```

4. Train:

```powershell
python 02_train_model.py
```

5. Predict new images:

```powershell
python 03_predict_quality.py
```

6. Make heatmaps:

```powershell
python 04_make_heatmaps.py
```

## My Recommendation

Use this as a pre-canopy-detector quality gate:

- `image_predictions.csv` decides whether the whole orthomosaic should proceed, be reviewed, or be reprocessed.
- `tile_predictions.csv` shows which areas are risky.
- Heatmaps help the drone/GIS team see where blur, exposure, shadow, or stitch-like artifacts are hurting the image.

The most important improvement over a single whole-image classifier is that one orthomosaic can be partly usable and partly bad. Tile-based scoring captures that.
