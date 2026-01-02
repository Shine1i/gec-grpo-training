fine-tune:
	uv run modal run -m src.gec.fine_tune --config-file-name $(config)

local:
	uv run python -m src.gec.fine_tune --config-file-name $(config) --local

test:
	uv run python test_reward_models.py
