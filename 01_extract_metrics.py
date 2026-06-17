from pathlib import Path

import config
from quality_core import ExtractSettings, extract_dataset, iter_images, save_dataframe, save_sqlite


def main() -> None:
    settings = ExtractSettings(
        max_analysis_dim=config.MAX_ANALYSIS_DIM,
        tile_size=config.TILE_SIZE,
        tile_stride=config.TILE_STRIDE,
        min_valid_tile_ratio=config.MIN_VALID_TILE_RATIO,
        white_border_fraction_threshold=config.WHITE_BORDER_FRACTION_THRESHOLD,
        yellow_border_fraction_threshold=config.YELLOW_BORDER_FRACTION_THRESHOLD,
        black_border_fraction_threshold=config.BLACK_BORDER_FRACTION_THRESHOLD,
    )

    jobs = []
    jobs.extend((path, "good") for path in iter_images(config.GOOD_IMAGE_DIR))
    jobs.extend((path, "bad") for path in iter_images(config.BAD_IMAGE_DIR))
    if not jobs:
        raise SystemExit("No training images found. Edit GOOD_IMAGE_DIR and BAD_IMAGE_DIR in config.py.")

    image_df, tile_df = extract_dataset(jobs, settings)
    save_dataframe(image_df, config.IMAGE_METRICS_CSV)
    save_dataframe(tile_df, config.TILE_METRICS_CSV)
    save_sqlite(image_df, tile_df, Path(config.OUTPUT_DIR) / "quality_metrics.sqlite")

    print(f"Saved image metrics: {config.IMAGE_METRICS_CSV}")
    print(f"Saved tile metrics: {config.TILE_METRICS_CSV}")


if __name__ == "__main__":
    main()

