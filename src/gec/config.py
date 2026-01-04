from typing import Optional
from datetime import datetime
from pathlib import Path
from typing import Self

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings

from .paths import get_path_to_configs


class GECConfig(BaseSettings):
    seed: int
    resume_from_checkpoint: Optional[str] = None

    # Language Model parameters
    model_name: str
    max_seq_length: int
    system_prompt: str

    # Dataset
    dataset_name: str
    dataset_size: Optional[int] = None

    # Training hyperparameters
    learning_rate: float
    warmup_steps: int
    max_steps: int
    gradient_accumulation_steps: int = 8

    # vLLM inference
    per_device_train_batch_size: int
    num_generations: int
    generation_batch_size: int
    max_completion_length: int
    use_vllm: bool
    vllm_mode: str
    vllm_gpu_memory_utilization: float

    # GRPO specific
    beta: float = 0.04  # KL penalty
    temperature: float = 1.0  # Sampling temperature for generations

    # Reward weights (should sum to 1.0)
    greco_weight: float = 0.6
    semantic_weight: float = 0.3
    laziness_weight: float = 0.1
    gain_epsilon: float = 0.02

    # Reward model paths
    greco_model_name: str = "mrqorib/grammaticality"
    mpnet_model: str = "sentence-transformers/all-mpnet-base-v2"

    # Experiment tracking
    wandb_enabled: bool
    wandb_project_name: str
    wandb_experiment_name: Optional[str] = None
    logging_steps: int
    push_to_hf: Optional[bool] = True

    # Checkpointing
    save_steps: int = 100
    save_total_limit: int = 3

    # LoRA parameters
    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    use_rslora: bool = False
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    @classmethod
    def from_yaml(cls, file_name: str) -> Self:
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @model_validator(mode="after")
    def set_experiment_name(self):
        if self.wandb_experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.wandb_experiment_name = f"{model_short}-gec-{timestamp}"
        return self
