"""
Stage 2 SFT: Domain-Specific Fine-tuning with Unsloth
Full fine-tuning on mixed grammar + rewriting dataset
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
    from datasets import load_dataset, concatenate_datasets
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer, SFTConfig
    import wandb

    # Config
    HF_TOKEN = os.environ.get("HF_TOKEN")
    MODEL_NAME = "unsloth/LFM2-1.2B"
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

    # ==================== DATASET LOADING ====================
    print("Loading datasets...")

    # Load all datasets
    dataset_grammar = load_dataset("moogin/typix-hq-grannar")
    dataset_rewriting = load_dataset("HuggingFaceTB/smoltalk", "explore-instruct-rewriting")
    dataset_rewrite = load_dataset("HuggingFaceTB/smoltalk", "smol-rewrite")

    # Define the new system prompt
    NEW_SYSTEM_PROMPT = "You are a writing assistant that edits text. Follow the user's instruction exactly. Preserve the original meaning unless the user asks to change it. Output only the revised text."

    # Define the system prompts for each tone
    PROFESSIONAL_PROMPT = "You're an AI assistant for text re-writing. Rewrite the input text to make it more professional and formal while retaining its essential content."
    CONCISE_PROMPT = "You're an AI assistant for text re-writing. Rewrite the input text to make it more concise while preserving its core meaning."
    FRIENDLY_PROMPT = "You're an AI assistant for text re-writing. Rewrite the input text to make it more friendly and approachable while maintaining its main points."

    def transform_rewriting_conversation(example):
        """Transform explore-instruct-rewriting format"""
        messages = example["messages"]
        transformed_messages = []

        for msg in messages:
            if msg["role"] == "system":
                transformed_messages.append({
                    "role": "system",
                    "content": NEW_SYSTEM_PROMPT
                })
            elif msg["role"] == "user":
                content = msg["content"]
                if "\n" in content:
                    split_pos = content.index("\n") + 1
                    prefix = content[:split_pos]
                    text = content[split_pos:]
                    new_content = f"{prefix}<Text>\n{text}\n</Text>"
                else:
                    new_content = f"<Text>\n{content}\n</Text>"

                transformed_messages.append({
                    "role": "user",
                    "content": new_content
                })
            else:
                transformed_messages.append(msg)

        return {"messages": transformed_messages}

    def transform_rewrite_conversation(example):
        """Transform smol-rewrite format"""
        messages = example["messages"]
        transformed_messages = []

        # Extract the instruction from system prompt
        system_content = None
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
                break

        # Remove the prefix and use as instruction
        instruction = system_content.replace("You're an AI assistant for text re-writing. ", "")

        for msg in messages:
            if msg["role"] == "system":
                transformed_messages.append({
                    "role": "system",
                    "content": NEW_SYSTEM_PROMPT
                })
            elif msg["role"] == "user":
                # Add instruction as prefix, wrap original content
                new_content = f"{instruction}\n<Text>\n{msg['content']}\n</Text>"
                transformed_messages.append({
                    "role": "user",
                    "content": new_content
                })
            else:
                transformed_messages.append(msg)

        return {"messages": transformed_messages}

    def filter_by_system_prompt(example, target_prompt):
        """Filter for specific system prompt"""
        messages = example["messages"]
        for msg in messages:
            if msg["role"] == "system" and msg["content"] == target_prompt:
                return True
        return False

    # Transform explore-instruct-rewriting
    train_rewriting = dataset_rewriting["train"].map(transform_rewriting_conversation)
    eval_rewriting = dataset_rewriting["test"].map(transform_rewriting_conversation)

    # Filter and transform smol-rewrite by tone
    train_professional = dataset_rewrite["train"].filter(
        lambda x: filter_by_system_prompt(x, PROFESSIONAL_PROMPT)
    ).map(transform_rewrite_conversation)

    train_concise = dataset_rewrite["train"].filter(
        lambda x: filter_by_system_prompt(x, CONCISE_PROMPT)
    ).map(transform_rewrite_conversation)

    train_friendly = dataset_rewrite["train"].filter(
        lambda x: filter_by_system_prompt(x, FRIENDLY_PROMPT)
    ).map(transform_rewrite_conversation)

    eval_professional = dataset_rewrite["test"].filter(
        lambda x: filter_by_system_prompt(x, PROFESSIONAL_PROMPT)
    ).map(transform_rewrite_conversation)

    eval_concise = dataset_rewrite["test"].filter(
        lambda x: filter_by_system_prompt(x, CONCISE_PROMPT)
    ).map(transform_rewrite_conversation)

    eval_friendly = dataset_rewrite["test"].filter(
        lambda x: filter_by_system_prompt(x, FRIENDLY_PROMPT)
    ).map(transform_rewrite_conversation)

    # Calculate required samples from actual dataset sizes
    GRAMMAR_TRAIN = len(dataset_grammar["train"])
    GRAMMAR_EVAL = len(dataset_grammar["validation"])

    # Train calculations
    TOTAL_TRAIN = int(GRAMMAR_TRAIN / 0.6)
    REWRITE_TRAIN_TOTAL = TOTAL_TRAIN - GRAMMAR_TRAIN
    SMOL_REWRITE_TRAIN = int(REWRITE_TRAIN_TOTAL * 0.70)
    EXPLORE_TRAIN = REWRITE_TRAIN_TOTAL - SMOL_REWRITE_TRAIN

    PROFESSIONAL_TRAIN = int(SMOL_REWRITE_TRAIN * 0.50)
    CONCISE_TRAIN = int(SMOL_REWRITE_TRAIN * 0.35)
    FRIENDLY_TRAIN = SMOL_REWRITE_TRAIN - PROFESSIONAL_TRAIN - CONCISE_TRAIN

    # Eval calculations
    TOTAL_EVAL = int(GRAMMAR_EVAL / 0.6)
    REWRITE_EVAL_TOTAL = TOTAL_EVAL - GRAMMAR_EVAL
    SMOL_REWRITE_EVAL = int(REWRITE_EVAL_TOTAL * 0.70)
    EXPLORE_EVAL = REWRITE_EVAL_TOTAL - SMOL_REWRITE_EVAL

    PROFESSIONAL_EVAL = int(SMOL_REWRITE_EVAL * 0.50)
    CONCISE_EVAL = int(SMOL_REWRITE_EVAL * 0.35)
    FRIENDLY_EVAL = SMOL_REWRITE_EVAL - PROFESSIONAL_EVAL - CONCISE_EVAL

    print("=== Sample Requirements ===")
    print(f"\nTRAIN:")
    print(f"  Grammar: {GRAMMAR_TRAIN:,} (60%)")
    print(f"  Rewrite/Tone: {REWRITE_TRAIN_TOTAL:,} (40%)")
    print(f"    - smol-rewrite: {SMOL_REWRITE_TRAIN:,} (70% of rewrite)")
    print(f"      • Professional/formal: {PROFESSIONAL_TRAIN:,} (50%)")
    print(f"      • Concise: {CONCISE_TRAIN:,} (35%)")
    print(f"      • Friendly: {FRIENDLY_TRAIN:,} (15%)")
    print(f"    - explore-instruct-rewriting: {EXPLORE_TRAIN:,} (30% of rewrite)")
    print(f"  TOTAL: {TOTAL_TRAIN:,}")

    print(f"\nEVAL:")
    print(f"  Grammar: {GRAMMAR_EVAL:,} (60%)")
    print(f"  Rewrite/Tone: {REWRITE_EVAL_TOTAL:,} (40%)")
    print(f"    - smol-rewrite: {SMOL_REWRITE_EVAL:,} (70% of rewrite)")
    print(f"      • Professional/formal: {PROFESSIONAL_EVAL:,} (50%)")
    print(f"      • Concise: {CONCISE_EVAL:,} (35%)")
    print(f"      • Friendly: {FRIENDLY_EVAL:,} (15%)")
    print(f"    - explore-instruct-rewriting: {EXPLORE_EVAL:,} (30% of rewrite)")
    print(f"  TOTAL: {TOTAL_EVAL:,}")

    print("\n=== Available Samples ===")
    print(f"\nTRAIN:")
    print(f"  Professional: {len(train_professional):,}")
    print(f"  Concise: {len(train_concise):,}")
    print(f"  Friendly: {len(train_friendly):,}")
    print(f"  Explore-instruct-rewriting: {len(train_rewriting):,}")

    print(f"\nEVAL:")
    print(f"  Professional: {len(eval_professional):,}")
    print(f"  Concise: {len(eval_concise):,}")
    print(f"  Friendly: {len(eval_friendly):,}")
    print(f"  Explore-instruct-rewriting: {len(eval_rewriting):,}")

    # Sample the required amounts (shuffle first for randomness)
    train_professional_sampled = train_professional.shuffle(seed=42).select(range(min(PROFESSIONAL_TRAIN, len(train_professional))))
    train_concise_sampled = train_concise.shuffle(seed=42).select(range(min(CONCISE_TRAIN, len(train_concise))))
    train_friendly_sampled = train_friendly.shuffle(seed=42).select(range(min(FRIENDLY_TRAIN, len(train_friendly))))
    train_rewriting_sampled = train_rewriting.shuffle(seed=42).select(range(min(EXPLORE_TRAIN, len(train_rewriting))))

    eval_professional_sampled = eval_professional.shuffle(seed=42).select(range(min(PROFESSIONAL_EVAL, len(eval_professional))))
    eval_concise_sampled = eval_concise.shuffle(seed=42).select(range(min(CONCISE_EVAL, len(eval_concise))))
    eval_friendly_sampled = eval_friendly.shuffle(seed=42).select(range(min(FRIENDLY_EVAL, len(eval_friendly))))
    eval_rewriting_sampled = eval_rewriting.shuffle(seed=42).select(range(min(EXPLORE_EVAL, len(eval_rewriting))))

    # Combine all datasets
    train_dataset = concatenate_datasets([
        dataset_grammar["train"],
        train_professional_sampled,
        train_concise_sampled,
        train_friendly_sampled,
        train_rewriting_sampled
    ]).shuffle(seed=42)

    eval_dataset = concatenate_datasets([
        dataset_grammar["validation"],
        eval_professional_sampled,
        eval_concise_sampled,
        eval_friendly_sampled,
        eval_rewriting_sampled
    ]).shuffle(seed=42)

    print("\n=== Final Dataset ===")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Eval samples: {len(eval_dataset):,}")

    # Show samples from each source
    print("\n=== Sample Examples ===")
    print("\n--- Grammar sample ---")
    print(dataset_grammar["train"][0]["messages"])
    print("\n--- Professional tone sample ---")
    print(train_professional_sampled[0]["messages"])
    print("\n--- Explore-instruct-rewriting sample ---")
    print(train_rewriting_sampled[0]["messages"])
    # ==================== END DATASET LOADING ====================

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
