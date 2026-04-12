from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    circor_root: str
    sample_rate: int = 16000
    max_sec: float = 10.0


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    epochs: int = 10
    seed: int = 42


@dataclass
class ModelConfig:
    num_classes: int = 2
    encoder_name: str = "m2d"
    freeze_encoder: bool = True
