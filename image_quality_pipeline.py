"""
Plantation UAV image quality assessment pipeline.

Usage:
  python image_quality_pipeline.py extract --good "D:/images/good" --bad "D:/images/bad" --out metrics.csv --db metrics.sqlite
  python image_quality_pipeline.py train --metrics metrics.csv --model model.joblib
  python image_quality_pipeline.py predict --model model.joblib --images "D:/new_images" --out predictions.csv

The extractor uses a resized copy of each image plus a grid of tiles, so it can
handle very large orthomosaic JPGs without loading full-resolution analysis data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DROP_COLUMNS = {"path", "filename", "label"}


@dataclass
class ExtractConfig:
    max_dim: int = 3000
    tile_grid: int = 6
    jpeg_quality_sample_max_dim: int = 1200


def iter_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        yield input_path
        return
    for path in input_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_resized_bgr(path: Path, max_dim: int) -> tuple[np.ndarray, dict]:
    with Image.open(path) as img:
        exif = dict(img.getexif() or {})
        width, height = img.size
        scale = min(1.0, max_dim / max(width, height))
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        img = img.convert("RGB")
        rgb = np.asarray(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    meta = {
        "orig_width": width,
        "orig_height": height,
        "megapixels": (width * height) / 1_000_000,
        "analysis_scale": scale,
        "exif_iso": exif.get(34855),
        "exif_exposure_time": str(exif.get(33434)) if exif.get(33434) else None,
        "exif_fnumber": str(exif.get(33437)) if exif.get(33437) else None,
    }
    return bgr, meta


def safe_stat(values: np.ndarray, fn, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    value = fn(values)
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(1.0, hist.sum())
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def high_frequency_ratio(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    f = np.fft.fftshift(np.fft.fft2(small.astype(np.float32)))
    mag = np.abs(f)
    h, w = mag.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    high = mag[radius > min(h, w) * 0.18].sum()
    total = mag.sum() + 1e-9
    return float(high / total)


def tile_metrics(gray: np.ndarray, hsv: np.ndarray, grid: int, valid_mask: np.ndarray | None = None) -> dict:
    h, w = gray.shape
    lap_values = []
    bright_values = []
    shadow_values = []
    color_values = []

    for y0 in np.linspace(0, h, grid + 1, dtype=int)[:-1]:
        y1 = min(h, y0 + math.ceil(h / grid))
        for x0 in np.linspace(0, w, grid + 1, dtype=int)[:-1]:
            x1 = min(w, x0 + math.ceil(w / grid))
            tile_g = gray[y0:y1, x0:x1]
            tile_hsv = hsv[y0:y1, x0:x1]
            tile_mask = None if valid_mask is None else valid_mask[y0:y1, x0:x1]
            if tile_g.size < 100:
                continue
            if tile_mask is not None and np.mean(tile_mask) < 0.2:
                continue
            lap_tile = cv2.Laplacian(tile_g, cv2.CV_64F)
            if tile_mask is not None:
                lap_values.append(lap_tile[tile_mask].var())
                bright_values.append(tile_g[tile_mask].mean())
                shadow_values.append(np.mean(tile_hsv[:, :, 2][tile_mask] < 55))
                color_values.append(tile_hsv[:, :, 1][tile_mask].mean())
            else:
                lap_values.append(lap_tile.var())
                bright_values.append(tile_g.mean())
                shadow_values.append(np.mean(tile_hsv[:, :, 2] < 55))
                color_values.append(tile_hsv[:, :, 1].mean())

    lap = np.asarray(lap_values, dtype=np.float64)
    bright = np.asarray(bright_values, dtype=np.float64)
    shadow = np.asarray(shadow_values, dtype=np.float64)
    sat = np.asarray(color_values, dtype=np.float64)

    return {
        "tile_lap_mean": safe_stat(lap, np.mean),
        "tile_lap_p10": safe_stat(lap, lambda x: np.percentile(x, 10)),
        "tile_lap_p25": safe_stat(lap, lambda x: np.percentile(x, 25)),
        "tile_lap_min": safe_stat(lap, np.min),
        "tile_lap_cv": safe_stat(lap, lambda x: np.std(x) / (np.mean(x) + 1e-9)),
        "tile_brightness_std": safe_stat(bright, np.std),
        "tile_brightness_range": safe_stat(bright, lambda x: np.max(x) - np.min(x)),
        "tile_shadow_mean": safe_stat(shadow, np.mean),
        "tile_shadow_max": safe_stat(shadow, np.max),
        "tile_saturation_std": safe_stat(sat, np.std),
    }


def seam_proxy_metrics(gray: np.ndarray) -> dict:
    """Approximate stitch issues using strong long-line and brightness jumps."""
    small = cv2.resize(gray, (1400, 1400), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=180,
        minLineLength=small.shape[0] // 3,
        maxLineGap=15,
    )
    line_count = 0 if lines is None else len(lines)

    row_means = small.mean(axis=1)
    col_means = small.mean(axis=0)
    row_jump = np.max(np.abs(np.diff(row_means))) if len(row_means) > 1 else 0.0
    col_jump = np.max(np.abs(np.diff(col_means))) if len(col_means) > 1 else 0.0
    return {
        "long_line_count": float(line_count),
        "max_row_brightness_jump": float(row_jump),
        "max_col_brightness_jump": float(col_jump),
    }


def border_fraction(mask: np.ndarray) -> float:
    edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(np.mean(edges))


def border_connected(mask: np.ndarray) -> np.ndarray:
    candidate = mask.astype(np.uint8)
    h, w = candidate.shape
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    filled = candidate.copy()
    connected = np.zeros_like(candidate, dtype=np.uint8)

    for x in range(w):
        for seed in ((x, 0), (x, h - 1)):
            if filled[seed[1], seed[0]] == 1 and connected[seed[1], seed[0]] == 0:
                tmp = filled.copy()
                cv2.floodFill(tmp, flood.copy(), seedPoint=seed, newVal=2)
                region = tmp == 2
                connected[region] = 1
                filled[region] = 0
    for y in range(h):
        for seed in ((0, y), (w - 1, y)):
            if filled[seed[1], seed[0]] == 1 and connected[seed[1], seed[0]] == 0:
                tmp = filled.copy()
                cv2.floodFill(tmp, flood.copy(), seedPoint=seed, newVal=2)
                region = tmp == 2
                connected[region] = 1
                filled[region] = 0

    return connected.astype(bool)


def border_connected_background(
    white_candidate: np.ndarray,
    yellow_candidate: np.ndarray,
    black_candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Find no-data/background colors connected to the image border.

    Orthomosaic footprints often have white/yellow/black fill outside the real
    mapped area. We only remove candidate background connected to an edge, so
    internal roads, bright soil, or real shadows are less likely to be erased.
    """
    edge_stats = {
        "background_white_border_fraction": border_fraction(white_candidate),
        "background_yellow_border_fraction": border_fraction(yellow_candidate),
        "background_black_border_fraction": border_fraction(black_candidate),
    }
    white_bg = border_connected(white_candidate) if edge_stats["background_white_border_fraction"] > 0.05 else np.zeros_like(white_candidate)
    yellow_bg = border_connected(yellow_candidate) if edge_stats["background_yellow_border_fraction"] > 0.25 else np.zeros_like(yellow_candidate)
    black_bg = border_connected(black_candidate) if edge_stats["background_black_border_fraction"] > 0.05 else np.zeros_like(black_candidate)
    connected = white_bg | yellow_bg | black_bg
    kernel = np.ones((5, 5), np.uint8)
    connected = cv2.morphologyEx(connected.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return (
        connected,
        white_bg,
        yellow_bg,
        black_bg,
        edge_stats,
    )


def extract_metrics(path: Path, label: str | None, cfg: ExtractConfig) -> dict:
    bgr, meta = load_resized_bgr(path, cfg.max_dim)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = sobel_x**2 + sobel_y**2

    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    h = hsv[:, :, 0]
    clipped_dark = np.mean(v <= 8)
    clipped_bright = np.mean(v >= 247)
    shadow = np.mean(v < 55)
    too_bright = np.mean(v > 225)
    low_contrast = float(gray.std())

    yellow_mask = ((h >= 18) & (h <= 42) & (s > 45) & (v > 45))
    white_mask = ((s < 35) & (v > 165))
    black_mask = v < 45
    green_mask = ((h >= 35) & (h <= 95) & (s > 35) & (v > 35))
    background_white_candidate = (s < 28) & (v > 225)
    background_yellow_candidate = ((h >= 12) & (h <= 45) & (s > 35) & (v > 120))
    background_black_candidate = v < 35
    (
        background_mask,
        background_white_mask,
        background_yellow_mask,
        background_black_mask,
        background_edge_stats,
    ) = border_connected_background(
        background_white_candidate,
        background_yellow_candidate,
        background_black_candidate,
    )
    valid_mask = ~background_mask
    valid_mask = cv2.erode(valid_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    canopy_mask = green_mask & valid_mask

    def masked_value(name: str, image: np.ndarray, mask: np.ndarray, fn) -> tuple[str, float]:
        values = image[mask]
        return name, safe_stat(values.astype(np.float64), fn)

    masked_metrics = dict(
        [
            masked_value("valid_gray_mean", gray, valid_mask, np.mean),
            masked_value("valid_gray_std", gray, valid_mask, np.std),
            masked_value("valid_laplacian_var", lap, valid_mask, np.var),
            masked_value("canopy_gray_mean", gray, canopy_mask, np.mean),
            masked_value("canopy_gray_std", gray, canopy_mask, np.std),
            masked_value("canopy_laplacian_var", lap, canopy_mask, np.var),
            masked_value("canopy_tenengrad_mean", tenengrad, canopy_mask, np.mean),
        ]
    )

    record = {
        "path": str(path),
        "filename": path.name,
        "label": label,
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        **meta,
        "gray_mean": float(gray.mean()),
        "gray_std": low_contrast,
        "gray_entropy": entropy(gray),
        "laplacian_var": float(lap.var()),
        "laplacian_abs_mean": float(np.abs(lap).mean()),
        "tenengrad_mean": float(tenengrad.mean()),
        "tenengrad_p10": float(np.percentile(tenengrad, 10)),
        "brenner": float(np.mean((gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)) ** 2)),
        "high_freq_ratio": high_frequency_ratio(gray),
        "clipped_dark_ratio": float(clipped_dark),
        "clipped_bright_ratio": float(clipped_bright),
        "shadow_ratio": float(shadow),
        "too_bright_ratio": float(too_bright),
        "saturation_mean": float(s.mean()),
        "saturation_std": float(s.std()),
        "yellow_ratio": float(yellow_mask.mean()),
        "white_ratio": float(white_mask.mean()),
        "background_ratio": float(background_mask.mean()),
        "background_white_ratio": float(background_white_mask.mean()),
        "background_yellow_ratio": float(background_yellow_mask.mean()),
        "background_black_ratio": float(background_black_mask.mean()),
        **background_edge_stats,
        "valid_area_ratio": float(valid_mask.mean()),
        "black_ratio": float(black_mask.mean()),
        "green_ratio": float(green_mask.mean()),
        "canopy_area_ratio": float(canopy_mask.mean()),
        **masked_metrics,
    }
    record.update(tile_metrics(gray, hsv, cfg.tile_grid, valid_mask))
    record.update(seam_proxy_metrics(gray))
    return record


def write_sqlite(csv_path: Path, db_path: Path) -> None:
    df = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("image_quality_metrics", conn, if_exists="replace", index=False)


def extract_command(args: argparse.Namespace) -> None:
    cfg = ExtractConfig(max_dim=args.max_dim, tile_grid=args.tile_grid)
    jobs: list[tuple[Path, str | None]] = []
    if args.good:
        jobs += [(p, "good") for p in iter_images(Path(args.good))]
    if args.bad:
        jobs += [(p, "bad") for p in iter_images(Path(args.bad))]
    if args.images:
        jobs += [(p, None) for p in iter_images(Path(args.images))]
    if not jobs:
        raise SystemExit("No images found. Provide --good/--bad folders or --images.")

    rows = []
    for i, (path, label) in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] extracting {path}")
        try:
            rows.append(extract_metrics(path, label, cfg))
        except Exception as exc:
            rows.append({"path": str(path), "filename": path.name, "label": label, "error": repr(exc)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.db:
        write_sqlite(out, Path(args.db))
    print(f"Saved metrics: {out}")


def train_command(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.metrics)
    df = df[df["label"].isin(["good", "bad"])].copy()
    if df.empty:
        raise SystemExit("Metrics file has no labeled rows with label good/bad.")
    df["target"] = (df["label"] == "bad").astype(int)
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS | {"target", "error"}]
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["target"]

    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
        model_name = "xgboost"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        )
        model_name = "random_forest_fallback"

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    n_splits = min(5, y.value_counts().min())
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
        pred = (proba >= args.threshold).astype(int)
        print(classification_report(y, pred, target_names=["good", "bad"]))
        print("Confusion matrix [[good_ok, good_flagged], [bad_missed, bad_flagged]]:")
        print(confusion_matrix(y, pred))
        try:
            print(f"ROC AUC: {roc_auc_score(y, proba):.3f}")
        except Exception:
            pass
    else:
        print("Not enough samples per class for cross-validation; training final model only.")

    model.fit(x, y)
    package = {
        "model": model,
        "model_name": model_name,
        "feature_cols": feature_cols,
        "threshold": args.threshold,
    }
    joblib.dump(package, args.model)
    print(f"Saved model: {args.model}")

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        top = sorted(zip(feature_cols, importances), key=lambda item: item[1], reverse=True)[:15]
        print("Top features:")
        for name, value in top:
            print(f"  {name}: {value:.4f}")


def predict_command(args: argparse.Namespace) -> None:
    package = joblib.load(args.model)
    model = package["model"]
    feature_cols = package["feature_cols"]
    threshold = args.threshold if args.threshold is not None else package.get("threshold", 0.5)

    tmp_metrics = Path(args.out).with_suffix(".metrics.csv")
    extract_args = argparse.Namespace(
        good=None,
        bad=None,
        images=args.images,
        out=str(tmp_metrics),
        db=None,
        max_dim=args.max_dim,
        tile_grid=args.tile_grid,
    )
    extract_command(extract_args)

    df = pd.read_csv(tmp_metrics)
    x = df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").fillna(0)
    proba_bad = model.predict_proba(x)[:, 1]
    df["bad_quality_probability"] = proba_bad
    df["quality_prediction"] = np.where(proba_bad >= threshold, "bad", "good")
    df.to_csv(args.out, index=False)
    print(f"Saved predictions: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UAV plantation image quality assessment")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract")
    p.add_argument("--good")
    p.add_argument("--bad")
    p.add_argument("--images")
    p.add_argument("--out", default="metrics.csv")
    p.add_argument("--db")
    p.add_argument("--max-dim", type=int, default=3000)
    p.add_argument("--tile-grid", type=int, default=6)
    p.set_defaults(func=extract_command)

    p = sub.add_parser("train")
    p.add_argument("--metrics", required=True)
    p.add_argument("--model", default="image_quality_model.joblib")
    p.add_argument("--threshold", type=float, default=0.5)
    p.set_defaults(func=train_command)

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--threshold", type=float)
    p.add_argument("--max-dim", type=int, default=3000)
    p.add_argument("--tile-grid", type=int, default=6)
    p.set_defaults(func=predict_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
