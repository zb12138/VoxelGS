import torch
from pathlib import Path
from importlib import import_module
def find_dataset_type(cfg):
    cfg.path = Path(cfg.path)
    assert cfg.path.exists(), f"Dataset root {cfg.path} does not exist!"
    if (cfg.path / "sparse").exists():
        dataset_type = "colmap"
    elif (cfg.path / "transforms_train.json").exists():
        dataset_type = "blender"
    else:
        raise ValueError(f"Could not recognize dataset type for {cfg.path}!")
    return dataset_type

def load_dataset(cfg):
    model = import_module(f"datasets.{find_dataset_type(cfg)}")
    return model.Dataset(cfg)