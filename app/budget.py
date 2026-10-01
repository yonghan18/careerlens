from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class BudgetExceeded(RuntimeError):
    """Raised before a model call that could exceed the configured budget."""


@dataclass(frozen=True)
class PriceBook:
    input_per_million: float
    output_per_million: float
    embedding_per_million: float

    def generation_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000

    def embedding_cost(self, tokens: int) -> float:
        return tokens * self.embedding_per_million / 1_000_000


class BudgetLedger:
    """Conservative, persistent accounting for calls made by this application."""

    def __init__(self, path: Path, cap_usd: float, prices: PriceBook):
        self.path = path
        self.cap_usd = cap_usd
        self.prices = prices
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"version": 1, "entries": []})

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # A token cannot encode less than one UTF-8 byte. Counting bytes therefore
        # intentionally over-reserves for English and remains conservative for
        # multilingual or unusual input.
        return max(1, len(text.encode("utf-8")))

    def estimate_generation(self, text: str, max_output_tokens: int) -> float:
        return self.prices.generation_cost(
            self.estimate_tokens(text) + 256, max_output_tokens
        )

    def estimate_embedding(self, texts: list[str]) -> float:
        return self.prices.embedding_cost(
            sum(self.estimate_tokens(text) + 8 for text in texts)
        )

    def _read(self) -> dict[str, object]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, object]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def status(self) -> dict[str, float | int]:
        with self._lock:
            entries = self._read()["entries"]
            spent = sum(float(entry["cost_usd"]) for entry in entries)
            return {
                "cap_usd": round(self.cap_usd, 4),
                "spent_usd": round(spent, 6),
                "remaining_usd": round(max(0.0, self.cap_usd - spent), 6),
                "recorded_calls": len(entries),
            }

    def reserve(self, operation: str, model: str, estimated_cost: float) -> str:
        with self._lock:
            data = self._read()
            entries = data["entries"]
            spent = sum(float(entry["cost_usd"]) for entry in entries)
            if estimated_cost > self.cap_usd - spent:
                raise BudgetExceeded(
                    f"The estimated ${estimated_cost:.6f} call would exceed the "
                    f"${self.cap_usd:.2f} application budget."
                )
            reservation_id = str(uuid4())
            entries.append(
                {
                    "id": reservation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "operation": operation,
                    "model": model,
                    "status": "reserved",
                    "cost_usd": round(estimated_cost, 8),
                }
            )
            self._write(data)
            return reservation_id

    def commit(
        self,
        reservation_id: str,
        actual_cost: float,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> None:
        with self._lock:
            data = self._read()
            entry = next(
                item for item in data["entries"] if item["id"] == reservation_id
            )
            entry.update(
                {
                    "status": "committed",
                    "cost_usd": round(actual_cost, 8),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )
            self._write(data)

    def mark_uncertain(self, reservation_id: str, reason: str) -> None:
        # Retain the conservative reservation because a failed request may still be billable.
        with self._lock:
            data = self._read()
            entry = next(
                item for item in data["entries"] if item["id"] == reservation_id
            )
            entry.update({"status": "uncertain", "reason": reason[:160]})
            self._write(data)
