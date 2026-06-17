import config
from quality_core import train_tile_model


def main() -> None:
    train_tile_model(
        tile_metrics_csv=config.TILE_METRICS_CSV,
        model_path=config.QUALITY_MODEL_PATH,
        threshold=config.BAD_TILE_PROBABILITY_THRESHOLD,
    )


if __name__ == "__main__":
    main()

