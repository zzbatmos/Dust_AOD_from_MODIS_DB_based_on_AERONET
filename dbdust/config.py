"""Load the single versioned scientific configuration."""
from pathlib import Path
from importlib.resources import files
import yaml


def config_path():
    path = Path(__file__).resolve().parents[1] / "configs/dbdust_v2.0.0.yaml"
    return path if path.exists() else files('dbdust.configs').joinpath('dbdust_v2.0.0.yaml')


def load_config(path=None):
    path = config_path() if path is None else Path(path)
    with path.open() as stream:
        return yaml.safe_load(stream)


CONFIG = load_config()
