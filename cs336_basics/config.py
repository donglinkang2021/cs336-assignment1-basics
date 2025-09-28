from dataclasses import dataclass, field
from typing import Any, Optional

# Hydra will detect this file as a structured config
# and allow for type-safe access and validation.

@dataclass
class DataConfig:
    """Data configuration."""
    path: str = "data/tinystories"
    tokenizer_path: str = "hf_tokenizer/tinystories/tokenizer.json"

@dataclass
class ModelConfig:
    """Transformer Language Model configuration."""
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344  # From model/default.yaml
    rope_theta: float = 10000.0

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    max_l2_norm: float = 1.0 # For gradient clipping

@dataclass
class TrainingConfig:
    """Training loop configuration."""
    seed: int = 1337
    device: str = "cuda" # "cuda" or "cpu"
    batch_size: int = 256
    max_iters: int = 5000
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 200
    resume_from: Optional[str] = None
    out_dir: str = "outputs" # From training/default.yaml

@dataclass
class LoggerConfig:
    """Logger configuration (wandb, tensorboard, etc.)."""
    # These fields are common across different loggers
    type: str = "wandb"
    project_name: str = "cs336-assignment1"
    run_name: str = "gpt-training-run"
    # The actual config loaded will be one of the YAMLs in conf/logger

@dataclass
class HydraRunConfig:
    dir: str = "${training.out_dir}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}"

@dataclass
class HydraSweepConfig:
    dir: str = "${training.out_dir}/multiruns/${now:%Y-%m-%d}_${now:%H-%M-%S}"
    subdir: str = "${hydra.job.num}"

@dataclass
class HydraConfig:
    """Hydra-specific configuration."""
    defaults: list[Any] = field(default_factory=lambda: [
        {"override hydra_logging": "colorlog"},
        {"override job_logging": "colorlog"},
    ])
    run: HydraRunConfig = field(default_factory=HydraRunConfig)
    sweep: HydraSweepConfig = field(default_factory=HydraSweepConfig)
    job_logging: dict[str, Any] = field(default_factory=lambda: {
        "handlers": {
            "file": {
                "filename": "${hydra.runtime.output_dir}/${hydra.job.name}.log"
            }
        }
    })


@dataclass
class TrainConfig:
    """
    The main configuration object, composed of all sub-configs.
    The `defaults` list is used by Hydra to compose the final config.
    """
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"hydra": "default"},
            {"data": "default"},
            {"model": "default"},
            {"optimizer": "default"},
            {"training": "default"},
            {"logger": "wandb"},
        ]
    )

    # Sub-configs are defined with default_factory to be instantiated correctly
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logger: Any = field(default_factory=LoggerConfig) # Use Any for flexibility with different loggers
    hydra: HydraConfig = field(default_factory=HydraConfig)