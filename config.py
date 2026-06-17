"""
Edit this file for your project paths and quality thresholds.

Use raw strings on Windows paths, for example:
GOOD_IMAGE_DIR = r"D:\plantation_quality\good"
"""

from pathlib import Path


# 1) Training folders. Put your labeled images here.
GOOD_IMAGE_DIR = r"D:\plantation_quality\good"
BAD_IMAGE_DIR = r"D:\plantation_quality\bad"

# 2) Folder or single image to score after training.
PREDICT_IMAGE_PATH = r"D:\plantation_quality\new_images"

# 3) Output folder.
OUTPUT_DIR = Path(__file__).resolve().parent / "quality_outputs"

IMAGE_METRICS_CSV = OUTPUT_DIR / "image_metrics.csv"
TILE_METRICS_CSV = OUTPUT_DIR / "tile_metrics.csv"
QUALITY_MODEL_PATH = OUTPUT_DIR / "tile_quality_model.joblib"
TILE_PREDICTIONS_CSV = OUTPUT_DIR / "tile_predictions.csv"
IMAGE_PREDICTIONS_CSV = OUTPUT_DIR / "image_predictions.csv"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"

# 4) Extraction settings.
# Large 400 MB JPGs are analyzed on a resized copy. Increase only if blur is
# difficult to detect after resizing.
MAX_ANALYSIS_DIM = 3000
TILE_SIZE = 512
TILE_STRIDE = 512
MIN_VALID_TILE_RATIO = 0.35

# 5) Background settings.
# White/black background is easy to detect at image borders. Yellow is handled
# more conservatively because roads/soil can be yellow inside the plantation.
WHITE_BORDER_FRACTION_THRESHOLD = 0.05
BLACK_BORDER_FRACTION_THRESHOLD = 0.05
YELLOW_BORDER_FRACTION_THRESHOLD = 0.25

# 6) Model decision settings.
# Higher threshold means fewer false alarms but more missed bad tiles.
BAD_TILE_PROBABILITY_THRESHOLD = 0.50

# Image-level decision rules after tile prediction.
IMAGE_BAD_TILE_RATIO_THRESHOLD = 0.25
IMAGE_MEAN_BAD_PROBABILITY_THRESHOLD = 0.45

