"""Project configuration, read from the environment or a .env file."""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Everything configurable in one place."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SHOPFLOOR_",
        extra="ignore",
    )

    data_dir: Path = Path("data/raw/hydraulic")
    """Where the dataset should live. Whether it is actually there is checked on use."""

    processed_dir: Path = Path("data/processed")
    """Where derived arrays are written. Created on first write."""

    seed: int = 42
    test_size: float = Field(default=0.2, gt=0, lt=1)
    val_size: float = Field(default=0.15, gt=0, lt=1)
    anthropic_api_key: str | None = None

    @model_validator(mode="after")
    def splits_must_leave_training_data(self) -> Self:
        used = self.test_size + self.val_size
        if used >= 1:
            raise ValueError(
                f"test_size ({self.test_size}) + val_size ({self.val_size}) = {used:.2f}, "
                "which leaves nothing for training"
            )
        return self


settings = Settings()
