import pandas as pd

import config
from quality_core import make_heatmaps


def main() -> None:
    tile_predictions = pd.read_csv(config.TILE_PREDICTIONS_CSV)
    make_heatmaps(tile_predictions, config.HEATMAP_DIR)
    print(f"Saved heatmaps: {config.HEATMAP_DIR}")


if __name__ == "__main__":
    main()

