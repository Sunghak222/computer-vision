import itertools
from pathlib import Path

from experiment_config import ExperimentConfig
from train import run_experiment


def get_rf_param_grid():
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "random_state": [42],
    }


def expand_grid(param_grid: dict):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    configs = []
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        configs.append(params)

    return configs


def main():
    feature_set = ["hsv"]          # final selected feature set
    model_name = "rf"              # final selected model
    augmentation = True            # keep same best setting

    param_grid = get_rf_param_grid()
    param_list = expand_grid(param_grid)

    best_summary = None

    print(f"Total tuning trials: {len(param_list)}")

    for i, params in enumerate(param_list, start=1):
        print(f"\n[{i}/{len(param_list)}] Tuning params: {params}")

        config = ExperimentConfig(
            model_name=model_name,
            feature_set=feature_set,
            augmentation=augmentation,
            dataset_tag="tuned",
            notes="RF tuning run",
            model_params=params
        )

        summary = run_experiment(config)

        if best_summary is None or summary["val_macro_f1"] > best_summary["val_macro_f1"]:
            best_summary = summary

    print("\nBest tuning result:")
    print(best_summary)


if __name__ == "__main__":
    main()