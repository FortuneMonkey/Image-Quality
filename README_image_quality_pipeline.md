# Plantation UAV Image Quality Assessment

This starter pipeline scores orthomosaic/UAV JPGs before they enter canopy/tree detection. The idea is correct: build a labeled image-quality dataset, extract objective metrics, train a classifier, then reject or review images likely to damage downstream tree counting.

## Recommended Workflow

1. Put known examples into two folders:
   - `good`: images where canopy detection works acceptably.
   - `bad`: blurry, shadow-heavy, overbright, or stitch-problem images.

2. Extract metrics:

```powershell
python image_quality_pipeline.py extract --good "D:\uav\good" --bad "D:\uav\bad" --out "D:\uav\metrics.csv" --db "D:\uav\metrics.sqlite"
```

3. Train and evaluate:

```powershell
python image_quality_pipeline.py train --metrics "D:\uav\metrics.csv" --model "D:\uav\image_quality_model.joblib"
```

4. Predict new images:

```powershell
python image_quality_pipeline.py predict --model "D:\uav\image_quality_model.joblib" --images "D:\uav\new_images" --out "D:\uav\predictions.csv"
```

## Metrics Extracted

- Blur/sharpness: Laplacian variance, absolute Laplacian mean, Tenengrad/Sobel energy, Brenner focus score, high-frequency FFT ratio.
- Exposure: mean brightness, contrast, clipped dark pixels, clipped bright pixels, too-bright ratio.
- Shadow: global shadow ratio plus tile-level shadow max/mean.
- Background/color: yellow, white, black, green pixel ratios, saturation mean/std.
- Mosaic footprint: border-connected white/yellow/black background ratios, valid-area ratio, and canopy-area ratio.
- Masked canopy quality: blur/brightness/contrast metrics inside valid plantation/canopy pixels, so rotated mosaic background does not dominate the model.
- Uneven quality: tile-level sharpness mean, p10, p25, min, coefficient of variation.
- Stitch proxies: long straight line count and row/column brightness jumps.
- Metadata: original size, megapixels, file size, EXIF ISO/exposure/f-number where available.

## Practical Notes

- Start with your 50 good and 50 bad images, but treat this as a first model. More labels will improve reliability, especially if the bad class has different causes.
- Use the model output as a quality gate before canopy detection: `good` goes to detection, `bad` goes to manual review or reprocessing.
- For very large 400 MB JPGs, the script analyzes a resized image by default with `--max-dim 3000`. Increase this only if fine blur differences are missed.
- For rotated orthomosaics like the sample, prioritize `valid_*`, `canopy_*`, and `tile_*` metrics over raw global metrics because white/yellow/black background can falsely look like exposure or shadow problems.
- Yellow background is handled conservatively: yellow is removed only when it is common along the image border, because yellow roads and soil inside the plantation should remain part of the valid image.
- Keep a reason label if possible, such as `blur`, `shadow`, `overbright`, or `stitch`. A future version can train separate defect classifiers instead of one generic bad/good model.

## Best Next Improvements

- Add image tiles as separate training samples so the model can localize bad areas, not only classify the whole orthomosaic.
- Compare quality score against canopy detector performance, for example tree count error or missed-canopy rate. This makes the quality model optimize for the real business outcome.
- Add visual QA outputs: heatmaps showing blurry/shadow/overbright zones for the drone team.
