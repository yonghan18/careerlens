from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    sample_dir: Path = ROOT / "data" / "sample"
    upload_dir: Path = ROOT / "data" / "uploads"
    index_dir: Path = ROOT / "data" / "index"
    chunks_file: Path = ROOT / "data" / "index" / "chunks.json"
    qdrant_dir: Path = ROOT / "data" / "index" / "qdrant"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    budget_usd: float = float(os.getenv("OPENAI_BUDGET_USD", "4.50"))
    input_usd_per_million: float = float(
        os.getenv("OPENAI_INPUT_USD_PER_MILLION", "0.40")
    )
    output_usd_per_million: float = float(
        os.getenv("OPENAI_OUTPUT_USD_PER_MILLION", "1.60")
    )
    embedding_usd_per_million: float = float(
        os.getenv("OPENAI_EMBEDDING_USD_PER_MILLION", "0.02")
    )
    max_output_tokens: int = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "500"))
    reasoning_effort: str = os.getenv("OPENAI_REASONING_EFFORT", "")
    origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("APP_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.sample_dir,
            self.upload_dir,
            self.index_dir,
            self.root / "data" / "budget",
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
