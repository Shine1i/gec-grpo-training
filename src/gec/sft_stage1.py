"""
Stage 1 SFT: General Task Adaptation with Unsloth
Full fine-tuning LFM2-1.2B on moogin/typix-hq-grammar-lang8 (1M samples)
"""

import modal

app = modal.App("gec-sft-stage1")

# Modal image with unsloth and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "xformers",
        "trl>=0.12.0",
        "peft",
        "accelerate",
        "bitsandbytes",
        "transformers>=4.46.0",
        "datasets",
        "wandb",
        "huggingface_hub",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Volume for checkpoints
volume = modal.Volume.from_name("gec-sft-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu=modal.gpu.A100(size="40GB"),
    timeout=7200,  # 2 hours
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
)
def train_stage1():
    import os
    import torch
    from datasets import load_dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    import wandb

    # Config
    HF_TOKEN = os.environ.get("HF_TOKEN")
    MODEL_NAME = "unsloth/LFM2-1.2B"
    DATASET_NAME = "moogin/typix-hq-grammar-lang8"
    OUTPUT_NAME = "moogin/typix-grammar-1.2b-stage1"
    MAX_SEQ_LENGTH = 768

    # Training hyperparameters
    PER_DEVICE_BATCH = 16
    GRAD_ACCUM = 8  # Effective batch = 128
    LEARNING_RATE = 5e-6
    NUM_EPOCHS = 1
    WARMUP_RATIO = 0.05
    EVAL_STEPS = 500
    SAVE_STEPS = 1000
    EARLY_STOPPING_PATIENCE = 3

    # Init wandb
    wandb.init(project="gec-sft-stage1", name="lfm2-1.2b-full-ft")

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
            texts.append(text)
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_func, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(formatting_func, batched=True, num_proc=4)

    # Calculate total steps for logging
    total_steps = (len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM)) * NUM_EPOCHS
    print(f"Total training steps: {total_steps}")

    # Training config
    training_args = SFTConfig(
        output_dir="/checkpoints/stage1",
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
        max_seq_length=MAX_SEQ_LENGTH,
        # Evaluation
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        # Saving
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=10,
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

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
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

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save best model
    print("\nSaving best model...")
    trainer.save_model("/checkpoints/stage1/best")

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
def main():
    result = train_stage1.remote()
    print(f"Training result: {result}")
