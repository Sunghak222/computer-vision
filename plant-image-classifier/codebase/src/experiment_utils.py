import json
from datetime import datetime
from pathlib import Path


EXPERIMENTS_ROOT = Path("results/experiments")
MODELS_ROOT = Path("models/experiments")


def make_experiment_output_paths(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}__{config.experiment_tag()}"

    result_dir = EXPERIMENTS_ROOT / run_name
    model_dir = MODELS_ROOT / run_name

    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_name": run_name,
        "result_dir": result_dir,
        "model_dir": model_dir,
        "model_path": model_dir / "model.pkl",
        "config_path": result_dir / "config.json",
        "metrics_path": result_dir / "metrics.json",
        "report_path": result_dir / "classification_report.json",
        "summary_path": result_dir / "summary.json",
    }


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(data, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")