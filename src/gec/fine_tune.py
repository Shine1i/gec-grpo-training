import os

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import wandb

from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from .config import GECConfig
from .dataset import load_gec_dataset, make_gec_messages
from .rewards import GECRewardModel, build_gec_reward_func
from .paths import get_path_model_checkpoints, get_path_model_checkpoints_local

# Modal setup
app = get_modal_app("gec-fine-tune-with-grpo")
image = get_docker_image()
hf_models_volume = get_volume("hf-model-cache")
model_checkpoints_volume = get_volume("gec-fine-tune-with-grpo")


def create_peft_config(config: GECConfig) -> LoraConfig | None:
    """Creates LoRA config from GECConfig."""
    if not config.use_peft:
        return None

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
        use_rslora=config.use_rslora,
    )


def _apply_local_overrides(config: GECConfig) -> GECConfig:
    dataset_size = config.dataset_size or 64
    overrides = {
        "dataset_size": min(dataset_size, 64),
        "max_steps": min(config.max_steps, 20),
        "per_device_train_batch_size": min(config.per_device_train_batch_size, 1),
        "generation_batch_size": config.num_generations,
        "logging_steps": min(config.logging_steps, 1),
    }
    print(f"Applying local overrides: {overrides}")
    return config.model_copy(update=overrides)


def _build_train_dataset(
    source_texts: list[str],
    system_prompt: str,
    references: list[str] | None = None,
) -> Dataset:
    prompts = [make_gec_messages(src, system_prompt) for src in source_texts]
    data = {"prompt": prompts, "source": source_texts}
    if references is not None:
        data["reference"] = references
    return Dataset.from_dict(data)


def run_fine_tune(config: GECConfig, local: bool = False) -> None:
    """Fine-tune GEC model using GRPO with neural rewards."""
    if config.wandb_enabled:
        print(f"Initializing WandB: {config.wandb_experiment_name}")
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_experiment_name,
            config=config.__dict__,
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # Initialize reward model
    print("Loading reward models...")
    reward_model = GECRewardModel(
        greco_model_name=config.greco_model_name,
        mpnet_model=config.mpnet_model,
        greco_weight=config.greco_weight,
        semantic_weight=config.semantic_weight,
        laziness_weight=config.laziness_weight,
    )
    reward_func = build_gec_reward_func(reward_model)

    # Load dataset
    print(f"Loading dataset from {config.dataset_name}")
    dataset = load_gec_dataset(config.dataset_name, config.dataset_size)
    source_texts = dataset["prompt"]
    references = dataset["reference"] if "reference" in dataset.column_names else None

    # Prepare training dataset
    train_dataset = _build_train_dataset(
        source_texts=source_texts,
        system_prompt=config.system_prompt,
        references=references,
    )

    output_dir = (
        get_path_model_checkpoints_local(config.wandb_experiment_name)
        if local
        else get_path_model_checkpoints(config.wandb_experiment_name)
    )

    print("Creating GRPOConfig...")
    grpo_config = GRPOConfig(
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        generation_batch_size=config.generation_batch_size,
        max_completion_length=config.max_completion_length,
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        output_dir=output_dir,
        logging_steps=config.logging_steps,
        report_to="wandb" if config.wandb_enabled else "none",
        remove_unused_columns=False,
    )

    print("Setting up GRPOTrainer...")
    peft_config = create_peft_config(config)

    if peft_config:
        print(f"LoRA enabled: r={config.lora_r}, alpha={config.lora_alpha}")

    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=[reward_func],
        train_dataset=train_dataset,
        args=grpo_config,
        peft_config=peft_config,
    )

    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

    if config.push_to_hf:
        print("Pushing model to HuggingFace Hub")
        trainer.push_to_hub()


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/hf_model_cache": hf_models_volume,
        "/model_checkpoints": model_checkpoints_volume,
    },
    secrets=get_secrets(),
    timeout=2 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1,
)
def fine_tune(config: GECConfig) -> None:
    run_fine_tune(config, local=False)


@app.local_entrypoint()
def main(config_file_name: str, local: bool = False):
    config = GECConfig.from_yaml(file_name=config_file_name)

    try:
        if local:
            print("Running local GEC fine-tune (no Modal)...")
            config = _apply_local_overrides(config)
            run_fine_tune(config, local=True)
            print("Local GEC fine-tuning completed successfully!")
        else:
            fine_tune.remote(config=config)
            print("GEC fine-tuning completed successfully!")
    except Exception as e:
        print(f"GEC fine-tuning failed: {e}")
        raise e


if __name__ == "__main__":
    main()
