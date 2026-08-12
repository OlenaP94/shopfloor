.PHONY: data clean-data tensor lint format test check

data:  ## download, verify and unpack the UCI hydraulic dataset
	uv run python scripts/download_data.py

clean-data:  ## remove the downloaded dataset
	rm -rf data/raw/hydraulic

lint:  ## everything pre-commit runs — ruff plus whitespace and file hygiene
	uv run pre-commit run --all-files

format:  ## apply formatting
	uv run ruff format src tests scripts

test:  ## run the test suite
	uv run pytest

check: format lint test  ## format, then everything CI runs

tensor:  ## build the numpy tensor from the raw dataset
	uv run python -m shopfloor.arrays
