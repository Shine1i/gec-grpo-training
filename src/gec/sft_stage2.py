"""
Stage 2 SFT: Domain-Specific Fine-tuning with Unsloth
Full fine-tuning on moogin/typix-hq-grammar (62k samples)
"""

import modal

app = modal.App("gec-sft-stage2")

# Modal image with unsloth and dependencies
# Install order: torch+triton -> flash-attn -> unsloth
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ninja-build")
    .pip_install(
        "torch==2.5.1",
        "triton",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    )
    .pip_install(
        "unsloth",
        "xformers",
    )
    .pip_install(
        "datasets",
        "wandb",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Volume for checkpoints
volume = modal.Volume.from_name("gec-sft-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=14400,  # 4 hours
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
)
def train_stage2(resume: bool = False):
    import os
    import torch
    from pathlib import Path
    # Import unsloth FIRST for optimizations
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    # Then other imports
    from datasets import load_dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer, SFTConfig
    import wandb

    # Config
    HF_TOKEN = os.environ.get("HF_TOKEN")
    MODEL_NAME = "moogin/typix-grammar-1.2b-stage1"
    DATASET_NAME = "moogin/typix-hq-grammar"
    OUTPUT_NAME = "moogin/typix-grammar-1.2b-stage2"
    MAX_SEQ_LENGTH = 768

    # Training hyperparameters
    PER_DEVICE_BATCH = 16
    GRAD_ACCUM = 8  # Effective batch = 128
    LEARNING_RATE = 5e-6
    NUM_EPOCHS = 1
    WARMUP_RATIO = 0.05
    EVAL_STEPS = 50
    SAVE_STEPS = 50  # ~9 evals for 62k dataset
    EARLY_STOPPING_PATIENCE = 3

    # Find latest checkpoint if resuming
    checkpoint_dir = Path("/checkpoints/stage2")
    resume_from = None
    if resume and checkpoint_dir.exists():
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"Resuming from checkpoint: {resume_from}")

    # Init wandb (resume if checkpoint exists)
    wandb.init(
        project="gec-sft-stage2",
        name="lfm2-1.2b-stage2",
        resume="allow" if resume_from else None,
    )

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # Full precision for full fine-tuning
        load_in_8bit=False,
        full_finetuning=True,
        token=HF_TOKEN,
    )

    # Apply chatml chat template (LFM2 uses chatml)
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, token=HF_TOKEN)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Format dataset - apply chat template
    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # Remove BOS token to avoid double BOS
            texts.append(text.removeprefix(tokenizer.bos_token or ""))
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_func, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(formatting_func, batched=True, num_proc=4)

    # Calculate total steps for logging
    total_steps = (len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM)) * NUM_EPOCHS
    print(f"Total training steps: {total_steps}")

    # Training config
    training_args = SFTConfig(
        output_dir="/checkpoints/stage2",
        dataset_text_field="text",
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        # Evaluation
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        # Saving
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=5,  # Keep last 5 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=10,
        report_to="wandb",
        # Performance
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
    )

    # Create trainer (TRL 0.24+ uses processing_class instead of tokenizer)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Train on completions only (mask system + user, only train on assistant)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Add early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=0.0,
    )
    trainer.add_callback(early_stopping)

    # Verify masking works
    sample_idx = 0
    print("\n=== Verifying train_on_responses_only masking ===")
    print("Full text:")
    print(tokenizer.decode(trainer.train_dataset[sample_idx]["input_ids"]))
    print("\nMasked (only assistant response):")
    labels = trainer.train_dataset[sample_idx]["labels"]
    print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in labels]).replace(tokenizer.pad_token, " "))
    print("=" * 50)

    # Train (resume from checkpoint if available)
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    # Save best model
    print("\nSaving best model...")
    trainer.save_model("/checkpoints/stage2/best")

    # Push to HF
    print(f"\nPushing to HuggingFace: {OUTPUT_NAME}")
    model.push_to_hub(OUTPUT_NAME, token=HF_TOKEN)
    tokenizer.push_to_hub(OUTPUT_NAME, token=HF_TOKEN)

    # Commit volume
    volume.commit()

    print("\nTraining complete!")
    wandb.finish()

    return {"status": "complete", "output_model": OUTPUT_NAME}


@app.local_entrypoint()
def main(resume: bool = False):
    print(f"Starting Stage 2 SFT (resume={resume})")
    result = train_stage2.remote(resume=resume)
    print(f"Training result: {result}")
