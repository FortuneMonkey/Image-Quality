from __future__ import annotations

import csv
import math
import sqlite3
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
DROP_COLUMNS = {
    "path",
    "filename",
    "label",
    "image_id",
    "tile_id",
    "x0",
    "y0",
    "x1",
    "y1",
    "error",
}


@dataclass
class ExtractSettings:
    max_analysis_dim: int
    tile_size: int
    tile_stride: int
    min_valid_tile_ratio: float
    white_border_fraction_threshold: float
    yellow_border_fraction_threshold: float
    black_border_fraction_threshold: float


def iter_images(input_path: str | Path) -> Iterable[Path]:
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        yield path
        return
    if not path.exists():
        return
    for image_path in path.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


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
        "analysis_width": bgr.shape[1],
        "analysis_height": bgr.shape[0],
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
    size = min(512, gray.shape[0], gray.shape[1])
    if size < 64:
        return 0.0
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    f = np.fft.fftshift(np.fft.fft2(small.astype(np.float32)))
    mag = np.abs(f)
    h, w = mag.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    high = mag[radius > min(h, w) * 0.18].sum()
    return float(high / (mag.sum() + 1e-9))


def border_fraction(mask: np.ndarray) -> float:
    edges = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(np.mean(edges))


def border_connected(mask: np.ndarray) -> np.ndarray:
    candidate = mask.astype(np.uint8)
    h, w = candidate.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    connected = np.zeros_like(candidate, dtype=np.uint8)
    visited = candidate.copy()

    seeds = []
    for x in range(w):
        seeds.append((x, 0))
        seeds.append((x, h - 1))
    for y in range(h):
        seeds.append((0, y))
        seeds.append((w - 1, y))

    for x, y in seeds:
        if visited[y, x] == 1 and connected[y, x] == 0:
            tmp = visited.copy()
            cv2.floodFill(tmp, flood_mask.copy(), seedPoint=(x, y), newVal=2)
            region = tmp == 2
            connected[region] = 1
            visited[region] = 0
    return connected.astype(bool)


def background_masks(hsv: np.ndarray, settings: ExtractSettings) -> dict:
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    white_candidate = (s < 28) & (v > 225)
    yellow_candidate = ((h >= 12) & (h <= 45) & (s > 35) & (v > 120))
    black_candidate = v < 35

    white_edge = border_fraction(white_candidate)
    yellow_edge = border_fraction(yellow_candidate)
    black_edge = border_fraction(black_candidate)

    white_bg = (
        border_connected(white_candidate)
        if white_edge > settings.white_border_fraction_threshold
        else np.zeros_like(white_candidate)
    )
    yellow_bg = (
        border_connected(yellow_candidate)
        if yellow_edge > settings.yellow_border_fraction_threshold
        else np.zeros_like(yellow_candidate)
    )
    black_bg = (
        border_connected(black_candidate)
        if black_edge > settings.black_border_fraction_threshold
        else np.zeros_like(black_candidate)
    )
    background = white_bg | yellow_bg | black_bg
    background = cv2.morphologyEx(background.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)

    return {
        "background_mask": background,
        "background_white_mask": white_bg,
        "background_yellow_mask": yellow_bg,
        "background_black_mask": black_bg,
        "background_white_border_fraction": white_edge,
        "background_yellow_border_fraction": yellow_edge,
        "background_black_border_fraction": black_edge,
    }


def base_feature_record(gray: np.ndarray, hsv: np.ndarray, valid_mask: np.ndarray, canopy_mask: np.ndarray) -> dict:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = sobel_x**2 + sobel_y**2
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    h = hsv[:, :, 0]

    yellow_mask = ((h >= 18) & (h <= 42) & (s > 45) & (v > 45))
    white_mask = ((s < 35) & (v > 165))
    black_mask = v < 45
    green_mask = ((h >= 35) & (h <= 95) & (s > 35) & (v > 35))

    valid_lap = lap[valid_mask]
    canopy_lap = lap[canopy_mask]
    canopy_ten = tenengrad[canopy_mask]
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_entropy": entropy(gray),
        "laplacian_var": float(lap.var()),
        "laplacian_abs_mean": float(np.abs(lap).mean()),
        "tenengrad_mean": float(tenengrad.mean()),
        "brenner": float(np.mean((gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)) ** 2)) if gray.shape[1] > 2 else 0.0,
        "high_freq_ratio": high_frequency_ratio(gray),
        "clipped_dark_ratio": float(np.mean(v <= 8)),
        "clipped_bright_ratio": float(np.mean(v >= 247)),
        "shadow_ratio": float(np.mean(v < 55)),
        "too_bright_ratio": float(np.mean(v > 225)),
        "saturation_mean": float(s.mean()),
        "saturation_std": float(s.std()),
        "yellow_ratio": float(yellow_mask.mean()),
        "white_ratio": float(white_mask.mean()),
        "black_ratio": float(black_mask.mean()),
        "green_ratio": float(green_mask.mean()),
        "valid_area_ratio": float(valid_mask.mean()),
        "canopy_area_ratio": float(canopy_mask.mean()),
        "valid_gray_mean": safe_stat(gray[valid_mask].astype(np.float64), np.mean),
        "valid_gray_std": safe_stat(gray[valid_mask].astype(np.float64), np.std),
        "valid_laplacian_var": safe_stat(valid_lap.astype(np.float64), np.var),
        "canopy_gray_mean": safe_stat(gray[canopy_mask].astype(np.float64), np.mean),
        "canopy_gray_std": safe_stat(gray[canopy_mask].astype(np.float64), np.std),
        "canopy_laplacian_var": safe_stat(canopy_lap.astype(np.float64), np.var),
        "canopy_tenengrad_mean": safe_stat(canopy_ten.astype(np.float64), np.mean),
    }


def seam_proxy_metrics(gray: np.ndarray) -> dict:
    if min(gray.shape) < 128:
        return {"long_line_count": 0.0, "max_row_brightness_jump": 0.0, "max_col_brightness_jump": 0.0}
    small = cv2.resize(gray, (min(1400, gray.shape[1]), min(1400, gray.shape[0])), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 60, 160)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=180, minLineLength=max(80, min(small.shape) // 3), maxLineGap=15)
    row_means = small.mean(axis=1)
    col_means = small.mean(axis=0)
    return {
        "long_line_count": float(0 if lines is None else len(lines)),
        "max_row_brightness_jump": float(np.max(np.abs(np.diff(row_means))) if len(row_means) > 1 else 0.0),
        "max_col_brightness_jump": float(np.max(np.abs(np.diff(col_means))) if len(col_means) > 1 else 0.0),
    }


def tile_windows(width: int, height: int, tile_size: int, stride: int) -> Iterable[tuple[int, int, int, int]]:
    for y0 in range(0, max(1, height - tile_size + 1), stride):
        for x0 in range(0, max(1, width - tile_size + 1), stride):
            yield x0, y0, min(width, x0 + tile_size), min(height, y0 + tile_size)
    if height > tile_size:
        y0 = height - tile_size
        for x0 in range(0, max(1, width - tile_size + 1), stride):
            yield x0, y0, min(width, x0 + tile_size), height
    if width > tile_size:
        x0 = width - tile_size
        for y0 in range(0, max(1, height - tile_size + 1), stride):
            yield x0, y0, width, min(height, y0 + tile_size)


def extract_one_image(path: Path, label: str | None, settings: ExtractSettings) -> tuple[dict, list[dict]]:
    bgr, meta = load_resized_bgr(path, settings.max_analysis_dim)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bg = background_masks(hsv, settings)
    valid_mask = ~bg["background_mask"]
    valid_mask = cv2.erode(valid_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    canopy_mask = ((h_chan >= 35) & (h_chan <= 95) & (s_chan > 35) & (v_chan > 35)) & valid_mask

    image_id = path.stem
    image_record = {
        "image_id": image_id,
        "path": str(path),
        "filename": path.name,
        "label": label,
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        **meta,
        "background_ratio": float(bg["background_mask"].mean()),
        "background_white_ratio": float(bg["background_white_mask"].mean()),
        "background_yellow_ratio": float(bg["background_yellow_mask"].mean()),
        "background_black_ratio": float(bg["background_black_mask"].mean()),
        "background_white_border_fraction": bg["background_white_border_fraction"],
        "background_yellow_border_fraction": bg["background_yellow_border_fraction"],
        "background_black_border_fraction": bg["background_black_border_fraction"],
        **base_feature_record(gray, hsv, valid_mask, canopy_mask),
        **seam_proxy_metrics(gray),
    }

    tiles: list[dict] = []
    seen = set()
    for tile_index, (x0, y0, x1, y1) in enumerate(tile_windows(gray.shape[1], gray.shape[0], settings.tile_size, settings.tile_stride)):
        key = (x0, y0, x1, y1)
        if key in seen:
            continue
        seen.add(key)
        tile_valid = valid_mask[y0:y1, x0:x1]
        if tile_valid.size == 0 or float(tile_valid.mean()) < settings.min_valid_tile_ratio:
            continue
        tile_gray = gray[y0:y1, x0:x1]
        tile_hsv = hsv[y0:y1, x0:x1]
        tile_canopy = canopy_mask[y0:y1, x0:x1]
        rec = {
            "image_id": image_id,
            "tile_id": f"{image_id}_{tile_index:05d}",
            "path": str(path),
            "filename": path.name,
            "label": label,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "tile_valid_ratio": float(tile_valid.mean()),
            "tile_canopy_ratio": float(tile_canopy.mean()),
            **base_feature_record(tile_gray, tile_hsv, tile_valid, tile_canopy),
        }
        tiles.append(rec)

    image_record["tile_count"] = len(tiles)
    image_record["tile_lap_mean"] = safe_stat(np.asarray([t["canopy_laplacian_var"] for t in tiles]), np.mean)
    image_record["tile_lap_p10"] = safe_stat(np.asarray([t["canopy_laplacian_var"] for t in tiles]), lambda x: np.percentile(x, 10))
    image_record["tile_lap_cv"] = safe_stat(np.asarray([t["canopy_laplacian_var"] for t in tiles]), lambda x: np.std(x) / (np.mean(x) + 1e-9))
    return image_record, tiles


def extract_dataset(jobs: list[tuple[Path, str | None]], settings: ExtractSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_rows = []
    tile_rows = []
    for index, (path, label) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] extracting {path}")
        try:
            image_record, tiles = extract_one_image(path, label, settings)
            image_rows.append(image_record)
            tile_rows.extend(tiles)
        except Exception as exc:
            image_rows.append({"path": str(path), "filename": path.name, "label": label, "error": repr(exc)})
    return pd.DataFrame(image_rows), pd.DataFrame(tile_rows)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def save_sqlite(image_df: pd.DataFrame, tile_df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        image_df.to_sql("image_metrics", conn, if_exists="replace", index=False)
        tile_df.to_sql("tile_metrics", conn, if_exists="replace", index=False)


def train_tile_model(tile_metrics_csv: Path, model_path: Path, threshold: float) -> None:
    df = pd.read_csv(tile_metrics_csv)
    df = df[df["label"].isin(["good", "bad"])].copy()
    if df.empty:
        raise SystemExit("No labeled tile rows found. Run extract_metrics.py first.")
    df["target"] = (df["label"] == "bad").astype(int)
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS | {"target"}]
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["target"]
    groups = df["image_id"].astype(str)

    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
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

        model = RandomForestClassifier(n_estimators=500, max_depth=8, class_weight="balanced", random_state=42)
        model_name = "random_forest_fallback"

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    unique_groups = groups.nunique()
    min_class_groups = df.groupby("target")["image_id"].nunique().min()
    n_splits = min(5, unique_groups, min_class_groups)
    if n_splits >= 2:
        cv = GroupKFold(n_splits=n_splits)
        proba = cross_val_predict(model, x, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
        pred = (proba >= threshold).astype(int)
        print(classification_report(y, pred, target_names=["good", "bad"]))
        print("Confusion matrix [[good_ok, good_flagged], [bad_missed, bad_flagged]]:")
        print(confusion_matrix(y, pred))
        try:
            print(f"ROC AUC: {roc_auc_score(y, proba):.3f}")
        except Exception:
            pass
    else:
        print("Not enough labeled images per class for grouped cross-validation. Training final model only.")

    model.fit(x, y)
    package = {"model": model, "model_name": model_name, "feature_cols": feature_cols, "threshold": threshold}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, model_path)
    print(f"Saved model: {model_path}")

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        print("Top features:")
        for name, value in sorted(zip(feature_cols, importances), key=lambda item: item[1], reverse=True)[:20]:
            print(f"  {name}: {value:.4f}")


def predict_tiles(tile_df: pd.DataFrame, model_path: Path, threshold: float | None = None) -> pd.DataFrame:
    package = joblib.load(model_path)
    model = package["model"]
    feature_cols = package["feature_cols"]
    active_threshold = package.get("threshold", 0.5) if threshold is None else threshold
    x = tile_df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").fillna(0)
    proba = model.predict_proba(x)[:, 1]
    out = tile_df.copy()
    out["bad_quality_probability"] = proba
    out["tile_quality_prediction"] = np.where(proba >= active_threshold, "bad", "good")
    return out


def summarize_image_predictions(tile_predictions: pd.DataFrame, bad_tile_threshold: float, mean_prob_threshold: float) -> pd.DataFrame:
    rows = []
    for image_id, group in tile_predictions.groupby("image_id"):
        bad_ratio = float((group["tile_quality_prediction"] == "bad").mean())
        mean_prob = float(group["bad_quality_probability"].mean())
        p90 = float(group["bad_quality_probability"].quantile(0.90))
        decision = "bad" if bad_ratio >= bad_tile_threshold or mean_prob >= mean_prob_threshold else "good"
        rows.append(
            {
                "image_id": image_id,
                "filename": group["filename"].iloc[0],
                "path": group["path"].iloc[0],
                "tile_count": len(group),
                "bad_tile_ratio": bad_ratio,
                "mean_bad_probability": mean_prob,
                "p90_bad_probability": p90,
                "quality_prediction": decision,
            }
        )
    return pd.DataFrame(rows).sort_values(["quality_prediction", "mean_bad_probability"], ascending=[True, False])


def make_heatmaps(tile_predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_id, group in tile_predictions.groupby("image_id"):
        path = Path(group["path"].iloc[0])
        bgr, _ = load_resized_bgr(path, max(int(group["x1"].max()), int(group["y1"].max())))
        overlay = bgr.copy()
        for _, row in group.iterrows():
            prob = float(row["bad_quality_probability"])
            x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
            color = (0, int(255 * (1 - prob)), int(255 * prob))
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)
        blended = cv2.addWeighted(overlay, 0.38, bgr, 0.62, 0)
        for _, row in group.iterrows():
            x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
            cv2.rectangle(blended, (x0, y0), (x1, y1), (30, 30, 30), thickness=1)
        out_path = output_dir / f"{image_id}_quality_heatmap.jpg"
        cv2.imwrite(str(out_path), blended)

