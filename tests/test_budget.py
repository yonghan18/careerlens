from pathlib import Path

import pytest

from app.budget import BudgetExceeded, BudgetLedger, PriceBook


def test_budget_reservation_and_commit(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "ledger.json", 0.01, PriceBook(0.2, 1.2, 0.02))
    reservation = ledger.reserve("answer", "test-model", 0.005)
    ledger.commit(reservation, 0.001, 1_000, 500)
    status = ledger.status()
    assert status["spent_usd"] == 0.001
    assert status["remaining_usd"] == 0.009


def test_budget_blocks_overspend(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "ledger.json", 0.001, PriceBook(0.2, 1.2, 0.02))
    with pytest.raises(BudgetExceeded):
        ledger.reserve("answer", "test-model", 0.0011)
