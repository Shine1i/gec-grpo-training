import modal


def get_modal_app(name: str) -> modal.App:
    return modal.App(name)


def get_docker_image() -> modal.Image:
    docker_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_pip_install(
            # Base training dependencies
            "trl[vllm]",
            "transformers>=4.55",
            "wandb",
            "peft>=0.13.0",
            # GEC-specific dependencies
            "sentence-transformers>=2.2.0",
            "huggingface_hub",
        )
        .env({"HF_HOME": "/hf_model_cache"})
        .add_local_dir("greco", "/root/greco")
    )
    return docker_image


def get_volume(name: str) -> modal.Volume:
    return modal.Volume.from_name(name, create_if_missing=True)


def get_retries(max_retries: int) -> modal.Retries:
    return modal.Retries(initial_delay=0.0, max_retries=max_retries)


def get_secrets() -> list[modal.Secret]:
    wandb_secret = modal.Secret.from_name("wandb-secret")
    return [wandb_secret]
