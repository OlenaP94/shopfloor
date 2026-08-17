.PHONY: data clean-data tensor features eda lint format test check

# --- pipeline, in the order it has to run ---------------------------------

data:  ## download, verify and unpack the UCI hydraulic dataset
	uv run python scripts/download_data.py

tensor:  ## resample 17 sensors into a 24-channel tensor
	uv run python -m shopfloor.arrays

features:  ## build the per-window feature table
	uv run python -m shopfloor.features

eda:  ## plot which channels respond to which fault
	uv run python scripts/eda.py

clean-data:  ## remove the downloaded dataset
	rm -rf data/raw/hydraulic

# --- development ----------------------------------------------------------

format:  ## apply formatting
	uv run ruff format src tests scripts

lint:  ## everything pre-commit runs — ruff plus whitespace and file hygiene
	uv run pre-commit run --all-files

test:  ## run the test suite
	uv run pytest

check: format lint test  ## format, then everything CI runs
