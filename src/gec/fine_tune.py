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
from .dataset import load_gec_dataset, make_gec_prompt
from .rewards import GECRewardModel, reward_completion
from .paths import get_path_model_checkpoints

# Modal setup
app = get_modal_app("gec-fine-tune-with-grpo")
image = get_docker_image()
hf_models_volume = get_volume("hf-model-cache")
model_checkpoints_volume = get_volume("gec-fine-tune-with-grpo")


def rollout_func(
    prompts: list[str],
    trainer: GRPOTrainer,
    reward_model: GECRewardModel,
    system_prompt: str,
    source_texts: list[str],
) -> dict[str, list]:
    """
    GEC rollout: single-step generation per prompt.

    Unlike browser control (multi-step episodes), GEC is one-shot:
    - Input: incorrect sentence
    - Output: corrected sentence
    - Reward: GRECO + semantic + laziness composite
    """
    from trl.experimental.openenv import generate_rollout_completions

    episode_prompt_ids: list[list[int]] = []
    episode_completion_ids: list[list[int]] = []
    episode_logprobs: list[list[float]] = []
    completion_rewards: list[float] = []

    tokenizer = trainer.processing_class

    print(f"\n[DEBUG] GEC rollout_func called with {len(prompts)} prompts")

    # Get source texts for this batch
    batch_sources = source_texts[: len(prompts)]

    # Format prompts with chat template
    formatted_prompts = [
        make_gec_prompt(src, system_prompt, tokenizer) for src in batch_sources
    ]

    # Single-step generation (no episode loop)
    for i, (prompt_text, source_text) in enumerate(
        zip(formatted_prompts, batch_sources)
    ):
        print(f"[DEBUG] Processing prompt {i + 1}/{len(prompts)}")

        # Generate completion
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]

        episode_prompt_ids.append(rollout_outputs["prompt_ids"])
        episode_completion_ids.append(rollout_outputs["completion_ids"])
        episode_logprobs.append(rollout_outputs["logprobs"])

        # Decode completion
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Compute reward
        reward = reward_model.compute_rewards([source_text], [completion_text])[0]
        completion_rewards.append(reward)

        print(f"  Source: {source_text[:60]}...")
        print(f"  Correction: {completion_text[:60]}...")
        print(f"  Reward: {reward:.4f}")

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "completion_reward": completion_rewards,
    }


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

    # Load dataset
    print(f"Loading dataset from {config.dataset_name}")
    dataset = load_gec_dataset(config.dataset_name, config.dataset_size)
    source_texts = dataset["prompt"]

    # Prepare training dataset
    train_dataset = Dataset.from_dict({"prompt": source_texts})

    output_dir = get_path_model_checkpoints(config.wandb_experiment_name)

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
    )

    print("Setting up GRPOTrainer...")
    peft_config = create_peft_config(config)

    if peft_config:
        print(f"LoRA enabled: r={config.lora_r}, alpha={config.lora_alpha}")

    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=[reward_completion],
        train_dataset=train_dataset,
        args=grpo_config,
        peft_config=peft_config,
        rollout_func=lambda prompts, trainer: rollout_func(
            prompts=prompts,
            trainer=trainer,
            reward_model=reward_model,
            system_prompt=config.system_prompt,
            source_texts=source_texts,
        ),
    )

    trainer_stats = trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

    if config.push_to_hf:
        print("Pushing model to HuggingFace Hub")
        trainer.push_to_hub()


@app.local_entrypoint()
def main(config_file_name: str):
    config = GECConfig.from_yaml(file_name=config_file_name)

    try:
        fine_tune.remote(config=config)
        print("GEC fine-tuning completed successfully!")
    except Exception as e:
        print(f"GEC fine-tuning failed: {e}")
        raise e


if __name__ == "__main__":
    main()
