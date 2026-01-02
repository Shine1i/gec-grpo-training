from pathlib import Path


def get_path_to_configs() -> str:
    path = str(Path(__file__).parent.parent.parent / "configs")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_path_model_checkpoints(experiment_name: str) -> str:
    path = Path("/model_checkpoints") / experiment_name.replace("/", "--")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_path_model_checkpoints_local(experiment_name: str) -> str:
    base_dir = Path(__file__).parent.parent.parent / "local_checkpoints"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / experiment_name.replace("/", "--")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return str(path)
