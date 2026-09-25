from pathlib import Path
from types import SimpleNamespace
import tomllib

from build_data import build_experiment_loaders
from training.hyper_search import grid_search
from training.train_loop import train_model


CONFIG_PATH = Path(__file__).with_name("config.toml")


def load_config(path: Path):
    with path.open("rb") as f:
        raw = tomllib.load(f)

    param_grid = raw.pop("grid", None)
    if not param_grid:
        raise ValueError("config.toml must contain a non-empty [grid] section.")

    flat = {}
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"Section [{section}] must be a TOML table.")
        for key, value in values.items():
            if key in flat:
                raise ValueError(f"Duplicate config key: {key}")
            flat[key] = value

    for key, values in param_grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid.{key} must be a non-empty TOML array.")

    return SimpleNamespace(**flat), param_grid


def main():
    cfg, param_grid = load_config(CONFIG_PATH)

    def train_fn_for_search(local_cfg, _train_loader_ignored, _test_loaders_ignored):
        train_loader, val_loaders, test_loaders = build_experiment_loaders(local_cfg)
        return train_model(
            local_cfg,
            train_loader,
            test_loaders,
            val_loaders=val_loaders,
        )

    best_params, best_metrics = grid_search(
        base_cfg=cfg,
        train_loader=None,
        test_loaders=None,
        train_fn=train_fn_for_search,
        param_grid=param_grid,
        log_file=cfg.grid_search_log,
        selection_metric=cfg.selection_metric,
    )

    print("Best params:", best_params)
    print("Best metrics:", best_metrics)


if __name__ == "__main__":
    main()
