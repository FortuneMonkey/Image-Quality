import config
from quality_core import (
    ExtractSettings,
    extract_dataset,
    iter_images,
    predict_tiles,
    save_dataframe,
    summarize_image_predictions,
)


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

    jobs = [(path, None) for path in iter_images(config.PREDICT_IMAGE_PATH)]
    if not jobs:
        raise SystemExit("No prediction images found. Edit PREDICT_IMAGE_PATH in config.py.")

    _, tile_df = extract_dataset(jobs, settings)
    tile_predictions = predict_tiles(tile_df, config.QUALITY_MODEL_PATH, config.BAD_TILE_PROBABILITY_THRESHOLD)
    image_predictions = summarize_image_predictions(
        tile_predictions,
        config.IMAGE_BAD_TILE_RATIO_THRESHOLD,
        config.IMAGE_MEAN_BAD_PROBABILITY_THRESHOLD,
    )

    save_dataframe(tile_predictions, config.TILE_PREDICTIONS_CSV)
    save_dataframe(image_predictions, config.IMAGE_PREDICTIONS_CSV)
    print(f"Saved tile predictions: {config.TILE_PREDICTIONS_CSV}")
    print(f"Saved image predictions: {config.IMAGE_PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()

