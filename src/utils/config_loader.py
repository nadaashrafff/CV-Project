from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(Path(config_path)) as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_id: str, config_path: str = "configs/experiment_config.yaml") -> dict:
    cfg = load_config(config_path)
    return cfg["experiments"][experiment_id]
